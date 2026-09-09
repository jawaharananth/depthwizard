"""Localise geometry defects to individual buildings, repair them, and iterate.

The existing checks (geometry_validate, float_check) answer "is this mesh
broken?" with one verdict for the whole class. That is enough to refuse a build
and not enough to fix one: knowing that 96 triangles are oversized does not say
WHICH buildings to redo. This module answers the second question, repairs those
buildings specifically, and re-checks.

CONVERGENCE IS THE HARD PART, not detection.

"Repeat until perfect" is not implementable as written. Two things go wrong:

  * Some defects cannot be repaired at all. A footprint over a street canyon
    where monocular depth has no signal will fail a plausibility test no matter
    how many times it is re-extracted, because the information is absent from
    the image. A loop that insists on zero defects never exits.

  * A repair can create a defect elsewhere. Splitting one oversized footprint
    into three can produce two that now overlap a neighbour. Without a progress
    requirement the loop oscillates between two states forever.

So the loop is bounded by three rules:

  1. Every pass must strictly reduce the defect count. If it does not, the
     repair strategy escalates rather than repeating a move that did not work.
  2. Strategies escalate through a fixed ladder ending in "drop the building".
     The ladder terminates, so the loop terminates.
  3. A hard iteration cap backstops both.

What survives is reported, not hidden. A build that ends with four unrepairable
buildings says so, and says which ones and why.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

import city_model


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# A footprint wider than this is a merged block, not a building. Expressed as a
# fraction of the scene span so it scales with the tile: on a 400 m scene a
# genuine building can be 60 m across, on a 4 km scene it can be far larger.
MAX_SPAN_FRAC = 0.22
MAX_SPAN_ABS_M = 200.0

# Below this a ring is a sprawling blob rather than one structure.
MIN_SOLIDITY = 0.35

# A roof is meant to be flat or gently pitched. Vertical spread across a single
# roof beyond this is a fitted plane that ran away.
MAX_ROOF_SPREAD = 8.0

# How far a base may sit above the terrain that is actually drawn.
MAX_FLOAT_M = 0.30

# Two footprints sharing more than this fraction of the smaller one are the same
# structure counted twice; the duplicate renders as z-fighting shells.
MAX_OVERLAP_FRAC = 0.45

# A footprint this elongated with a short side this narrow is a sliver: the
# leftover strip between two regions, not a structure. Extruded it renders as a
# thin blade standing on edge, which is the most visible of the small defects.
MAX_ASPECT = 8.0
MIN_SHORT_SIDE_M = 2.5

# A ring vertex sharper than this produces a needle triangle -- near-zero area,
# no reliable normal, and a bright shading artefact along its spine.
#
# Set at 8 deg, not 18: at 18 this fired on 272 of 1039 footprints and, because
# a fixed-epsilon re-trace could not remove the spike, escalated 210 of them to
# "drop" -- discarding 31% of the scene to remove a shading artefact. Genuine
# buildings have acute corners. Only a true needle qualifies, and the repair
# below now actually removes it rather than deferring to deletion.
MIN_CORNER_DEG = 8.0

# A building whose height disagrees with every neighbour by this many robust
# deviations is reading a roof that is not there (a shadow, a tree, a crane).
HEIGHT_OUTLIER_SIGMAS = 4.0
NEIGHBOUR_RADIUS_M = 120.0
MIN_NEIGHBOURS = 6

MAX_ITERATIONS = 8

# Each defect gets the repair that addresses IT, then escalates along its own
# ladder. A single global strategy applied to every defect was measurably wrong:
# re-tracing a duplicate footprint produces the same duplicate, so overlap counts
# sat at 73-75 across six passes while the crossing-ring count fell 65 -> 9.
#
# Every ladder ends in "drop", which is what guarantees termination.
REPAIRS = {
    "self_intersecting":  ["resimplify", "drop"],
    "blob":               ["resimplify", "split", "drop"],
    "oversized":          ["split", "resimplify", "drop"],
    "degenerate_ring":    ["drop"],
    # A duplicate is not malformed -- it is a second copy of a structure already
    # in the scene. Re-tracing it cannot help; the copy has to go.
    "duplicate_overlap":  ["drop"],
    "roof_runaway":       ["flatten_roof", "drop"],
    "floating":           ["flatten_roof", "drop"],
    # A sliver is a leftover strip between two regions. Re-tracing keeps it a
    # strip, so there is nothing to repair into.
    "sliver":             ["drop"],
    # A needle corner is a simplification artefact: re-tracing at a coarser
    # tolerance removes the spike while keeping the building.
    "needle_corner":      ["resimplify", "desimplify", "drop"],
    # An outlying height usually means the roof sample caught something that is
    # not the roof. Re-deriving the roof from a robust percentile is the fix;
    # if it still disagrees with every neighbour, the building goes.
}

# Order matters when one footprint carries several defects: the most destructive
# repair wins, so a ring that is both a duplicate and self-intersecting is
# dropped rather than re-traced into a duplicate that is merely well formed.
SEVERITY = ["degenerate_ring", "duplicate_overlap", "sliver", "oversized",
            "blob", "self_intersecting", "needle_corner",
            "roof_runaway", "floating"]


# ---------------------------------------------------------------------------
# Defect detection
# ---------------------------------------------------------------------------

def _span_m(poly: np.ndarray, gsd: float) -> float:
    if len(poly) < 2:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(max(x.max() - x.min(), y.max() - y.min()) * gsd)


def _solidity(poly: np.ndarray) -> float:
    a = cv2.contourArea(poly.astype(np.float32))
    h = cv2.contourArea(cv2.convexHull(poly.astype(np.float32)))
    return float(a / h) if h > 0 else 0.0


def _aspect(poly: np.ndarray, gsd: float):
    """(aspect ratio, short side in metres) of the minimum-area rectangle."""
    if len(poly) < 3:
        return 1.0, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(poly.astype(np.float32))
    lo, hi = sorted((float(w), float(h)))
    if lo <= 1e-6:
        return 1e6, 0.0
    return hi / lo, lo * gsd


def _sharpest_corner_deg(poly: np.ndarray) -> float:
    """Smallest interior angle on the ring, in degrees."""
    n = len(poly)
    if n < 3:
        return 180.0
    p = poly.astype(np.float64)
    prev, cur, nxt = np.roll(p, 1, 0), p, np.roll(p, -1, 0)
    a, b = prev - cur, nxt - cur
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-9) & (nb > 1e-9)
    if not ok.any():
        return 180.0
    cosang = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).min())


def _height_outliers(binfo: Dict, gsd: float) -> List[int]:
    """Buildings whose height disagrees with their local neighbourhood.

    Compared against NEIGHBOURS rather than the scene, because building height
    is strongly spatially correlated: a 40 m block beside other 40 m blocks is
    ordinary, while the same 40 m beside a street of 8 m houses is usually a
    roof reading taken off a crane, a tree, or a shadow edge.

    Robust statistics throughout -- a neighbourhood containing two bad heights
    would have its mean and standard deviation dragged far enough to accept
    both.
    """
    builds = binfo.get("buildings") or []
    if len(builds) < MIN_NEIGHBOURS + 1:
        return []
    cent = np.array([b["centroid_px"] for b in builds], dtype=np.float64) * gsd
    hs = np.array([b.get("height_m", 0.0) for b in builds], dtype=np.float64)

    out = []
    r2 = NEIGHBOUR_RADIUS_M ** 2
    for i in range(len(builds)):
        d2 = np.sum((cent - cent[i]) ** 2, axis=1)
        sel = (d2 <= r2)
        sel[i] = False
        if sel.sum() < MIN_NEIGHBOURS:
            continue
        nb = hs[sel]
        med = float(np.median(nb))
        mad = float(np.median(np.abs(nb - med)))
        sigma = mad * 1.4826
        if sigma < 0.5:                  # a uniform neighbourhood: use a floor
            sigma = 0.5
        if abs(hs[i] - med) > HEIGHT_OUTLIER_SIGMAS * sigma:
            pi = builds[i].get("poly_index")
            if pi is not None:
                out.append(int(pi))
    return out


def _overlap_pairs(footprints: Sequence[np.ndarray], shape,
                   max_frac: float = MAX_OVERLAP_FRAC) -> List[int]:
    """Indices of footprints substantially covered by an earlier footprint.

    Rasterised rather than computed analytically: the rings are already integer
    pixel polygons, and a raster test needs no robust-predicate machinery to be
    correct on the degenerate cases that matter here.
    """
    H, W = shape
    scale = 4                       # coarse grid: this is a duplicate test
    hh, ww = max(H // scale, 1), max(W // scale, 1)
    occupancy = np.zeros((hh, ww), np.int32)
    areas, masks = [], []
    for poly in footprints:
        m = np.zeros((hh, ww), np.uint8)
        cv2.fillPoly(m, [(poly / scale).astype(np.int32)], 1)
        masks.append(m.astype(bool))
        areas.append(int(m.sum()))

    dup = []
    for i, m in enumerate(masks):
        if areas[i] == 0:
            continue
        prior = occupancy > 0
        inter = int((m & prior).sum())
        if inter / areas[i] > max_frac:
            dup.append(i)
        else:
            occupancy[m] += 1
    return dup


def find_defects(footprints: Sequence[np.ndarray],
                 verts: np.ndarray, faces: np.ndarray, binfo: Dict,
                 gsd: float, shape,
                 ground_small: Optional[np.ndarray] = None,
                 cell_m: Optional[float] = None) -> Dict[int, List[str]]:
    """Map footprint index -> list of defect names.

    Keyed by ``poly_index``, never by position: prisms are skipped for legitimate
    reasons, so the n-th building is not the n-th footprint.
    """
    H, W = shape
    scene_span = max(H, W) * gsd
    max_span = min(MAX_SPAN_FRAC * scene_span, MAX_SPAN_ABS_M)

    defects: Dict[int, List[str]] = {}

    def add(i, name):
        defects.setdefault(int(i), []).append(name)

    # -- footprint-only defects, checkable before extrusion -----------------
    for i, poly in enumerate(footprints):
        if len(poly) < 3:
            add(i, "degenerate_ring")
            continue
        if _span_m(poly, gsd) > max_span:
            add(i, "oversized")
        if city_model._self_intersects(poly):
            add(i, "self_intersecting")
        if _solidity(poly) < MIN_SOLIDITY:
            add(i, "blob")
        ar, short_m = _aspect(poly, gsd)
        if ar > MAX_ASPECT and short_m < MIN_SHORT_SIDE_M:
            add(i, "sliver")
        if _sharpest_corner_deg(poly) < MIN_CORNER_DEG:
            add(i, "needle_corner")

    for i in _overlap_pairs(footprints, shape):
        add(i, "duplicate_overlap")

    # -- defects visible only in the built prism ----------------------------
    if binfo and binfo.get("buildings") is not None and len(verts):
        for b in binfo["buildings"]:
            pi = b.get("poly_index")
            if pi is None:
                continue
            off, cnt = b["vertex_offset"], b["vertex_count"]
            if off + cnt > len(verts):
                continue
            block = verts[off:off + cnt]
            n = b["n_sides"]
            roof = block[n:]           # upper ring
            if len(roof):
                spread = float(roof[:, 1].max() - roof[:, 1].min())
                if spread > MAX_ROOF_SPREAD:
                    add(pi, "roof_runaway")

            if ground_small is not None and cell_m:
                import float_check as fc
                base = block[:n]
                if len(base):
                    gy = fc.sample_ground(ground_small, cell_m,
                                          base[:, 0], base[:, 2])
                    if float(np.median(base[:, 1] - gy)) > MAX_FLOAT_M:
                        add(pi, "floating")

        # HEIGHT OUTLIER DETECTION IS DELIBERATELY NOT ENABLED.
        #
        # It was implemented and measured, and it made the model worse. Flagging
        # buildings whose height disagrees with their neighbours assumes height
        # is locally smooth. In a downtown it is not: a tower beside low-rise
        # stock is the normal case, not an error. Measured on the uploaded
        # Jacksonville tile it removed the tallest buildings outright --
        # max height fell from 40 to 22 units and the model-vs-image height
        # correlation collapsed from 0.723 to 0.402, turning a GOOD audit into
        # a POOR one.
        #
        # It also would not settle: dropping outliers changes the neighbourhood
        # medians, which creates new outliers, so the count sat at 88 for two
        # passes and then produced 19 fresh ones after the drop.
        #
        # _height_outliers is kept because the failure is worth being able to
        # reproduce, but a height is not evidence of a defect on its own. Making
        # this useful needs an independent signal that the roof SAMPLE is wrong
        # -- low photometric confidence over the footprint -- which the upload
        # path does not currently carry.
    return defects


# ---------------------------------------------------------------------------
# Repairs
# ---------------------------------------------------------------------------

def _rasterise(poly: np.ndarray, shape):
    H, W = shape
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    pad = 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, (0, 0)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32) - [x0, y0]], 1)
    return m, (x0, y0)


def _resimplify(poly: np.ndarray, shape, eps_frac: float = 0.02) -> List[np.ndarray]:
    """Rasterise and re-trace: resolves crossings, ragged edges and spikes.

    ``eps_frac`` is the Douglas-Peucker tolerance as a fraction of perimeter. A
    spike survives a fine tolerance by construction -- it is a real feature of
    the traced outline -- so removing one needs a coarser trace, not a repeat of
    the same one.
    """
    m, (x0, y0) = _rasterise(poly, shape)
    if m is None or m.sum() < 20:
        return []
    # Opening knocks off one-pixel spurs, which is what most needles are.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if m.sum() < 20:
        return []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    eps = eps_frac * cv2.arcLength(c, True)
    p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32) + [x0, y0]
    return [p] if len(p) >= 3 else []


def _split(poly: np.ndarray, shape) -> List[np.ndarray]:
    """Break a merged block into its constituent structures.

    Distance transform + watershed, the same separation the footprint extractor
    uses. A merged block is exactly the case watershed handles: several compact
    lobes joined by narrow necks, which the distance transform separates at the
    necks.
    """
    m, (x0, y0) = _rasterise(poly, shape)
    if m is None or m.sum() < 200:
        return []

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    _, sure = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure = sure.astype(np.uint8)
    ncomp, markers = cv2.connectedComponents(sure)
    if ncomp <= 2:                      # nothing to split into
        return []

    unknown = cv2.subtract(m * 255, sure)
    markers = markers + 1
    markers[unknown == 255] = 0
    rgb = cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(rgb, markers.astype(np.int32))

    out = []
    for lab in range(2, ncomp + 1):
        part = (markers == lab).astype(np.uint8)
        if part.sum() < 60:
            continue
        cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        eps = 0.02 * cv2.arcLength(c, True)
        p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32) + [x0, y0]
        if len(p) >= 3:
            out.append(p)
    return out if len(out) >= 2 else []


def _apply(strategy: str, poly: np.ndarray, shape) -> List[np.ndarray]:
    if strategy == "resimplify":
        return _resimplify(poly, shape)
    if strategy == "desimplify":
        # Coarser trace, used for spikes: a fine one reproduces them.
        return _resimplify(poly, shape, eps_frac=0.05)
    if strategy == "split":
        got = _split(poly, shape)
        return got if got else _resimplify(poly, shape)
    if strategy == "flatten_roof":
        # Handled at extrusion time by disabling roof plane fitting; the
        # footprint itself is returned unchanged.
        return [poly]
    return []                            # "drop"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _kind_counts(defects: Dict[int, List[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for names in defects.values():
        for n in names:
            out[n] = out.get(n, 0) + 1
    return out


# Defects that need no extrusion to detect. These are the overwhelming majority,
# and separating them is what makes the loop affordable: extruding 1344 prisms
# on a 2560px tile costs ~45 s, so a loop that re-extrudes every pass turned a
# 46 s build into ten minutes. Footprint defects are pure polygon work.
FOOTPRINT_DEFECTS = {"oversized", "self_intersecting", "blob",
                     "degenerate_ring", "duplicate_overlap",
                     "sliver", "needle_corner"}


def _repair_pass(fps, defects, attempts, stalled, shape, flat_roof):
    """Apply one repair to every defective footprint. Returns (fps, attempts, dropped)."""
    new_fps, next_attempts, dropped = [], {}, 0
    for i, poly in enumerate(fps):
        names = defects.get(i)
        if not names:
            next_attempts[len(new_fps)] = attempts.get(i, 0)
            new_fps.append(poly)
            continue

        worst = next((d for d in SEVERITY if d in names), names[0])
        ladder = REPAIRS.get(worst, ["resimplify", "drop"])
        step = attempts.get(i, 0) + (1 if stalled else 0)
        strategy = ladder[min(step, len(ladder) - 1)]

        if strategy == "drop":
            dropped += 1
            continue
        if strategy == "flatten_roof":
            idx = len(new_fps)
            flat_roof.add(idx)
            next_attempts[idx] = step + 1
            new_fps.append(poly)
            continue
        for rep in _apply(strategy, poly, shape):
            idx = len(new_fps)
            next_attempts[idx] = step + 1
            new_fps.append(rep)
    return new_fps, next_attempts, dropped


def repair_build(footprints: Sequence[np.ndarray], dsm: np.ndarray,
                 ground: np.ndarray, gsd: float,
                 image_np: Optional[np.ndarray] = None,
                 min_height_m: float = 1.5,
                 ground_small: Optional[np.ndarray] = None,
                 cell_m: Optional[float] = None,
                 verbose: bool = True) -> Dict:
    """Repair footprints, then extrude, then repair what only extrusion reveals.

    Two phases, because the two defect families cost different amounts to see.

    Phase 1 iterates on defects visible in the polygon alone -- crossings, blobs,
    merged blocks, duplicates -- and never extrudes. Phase 2 extrudes and checks
    the two defects that need a built prism (runaway roofs, floating bases),
    repairing and re-extruding at most a couple of times.

    Returns {"verts","faces","binfo","footprints","report"}.
    """
    shape = dsm.shape
    fps = [np.asarray(p, np.float32).copy() for p in footprints]
    flat_roof: set = set()
    history: List[Dict] = []

    # ---- Phase 1: footprints only -----------------------------------------
    attempts: Dict[int, int] = {}
    prev: Optional[int] = None
    for it in range(MAX_ITERATIONS):
        defects = find_defects(fps, np.zeros((0, 3), np.float32),
                               np.zeros((0, 3), np.int64), {}, gsd, shape)
        defects = {k: [n for n in v if n in FOOTPRINT_DEFECTS]
                   for k, v in defects.items()}
        defects = {k: v for k, v in defects.items() if v}
        n = len(defects)
        history.append({"phase": 1, "iteration": it,
                        "footprints": len(fps), "defective": n,
                        "kinds": _kind_counts(defects)})
        if verbose:
            detail = "  ".join(f"{k} {v}" for k, v in
                               sorted(_kind_counts(defects).items()))
            print(f"      repair p1.{it}: {len(fps)} footprints, {n} defective"
                  + (f"   [{detail}]" if detail else ""))
        if n == 0:
            break
        stalled = prev is not None and n >= prev
        prev = n
        fps, attempts, dropped = _repair_pass(fps, defects, attempts, stalled,
                                              shape, flat_roof)
        if verbose and dropped:
            print(f"      dropped {dropped} unrepairable footprint(s)")
        if not fps:
            break

    # ---- Phase 2: extrude, then fix what only the prism shows -------------
    verts = faces = None
    binfo: Dict = {}
    attempts = {}
    prev = None
    MAX_EXTRUDE_PASSES = 3
    for it in range(MAX_EXTRUDE_PASSES):
        verts, faces, binfo = city_model.build_prisms(
            fps, dsm, ground, gsd, gsd, min_height_m=min_height_m,
            image_np=image_np, roof_percentile=70.0,
            flat_roof_indices=flat_roof)
        defects = find_defects(fps, verts, faces, binfo, gsd, shape,
                               ground_small=ground_small, cell_m=cell_m)
        defects = {k: [n for n in v if n not in FOOTPRINT_DEFECTS]
                   for k, v in defects.items()}
        defects = {k: v for k, v in defects.items() if v}
        n = len(defects)
        history.append({"phase": 2, "iteration": it,
                        "built": len(binfo["buildings"]), "defective": n,
                        "kinds": _kind_counts(defects)})
        if verbose:
            detail = "  ".join(f"{k} {v}" for k, v in
                               sorted(_kind_counts(defects).items()))
            print(f"      repair p2.{it}: {len(binfo['buildings'])} built, "
                  f"{n} defective" + (f"   [{detail}]" if detail else ""))
        if n == 0 or it == MAX_EXTRUDE_PASSES - 1:
            break
        stalled = prev is not None and n >= prev
        prev = n
        fps, attempts, dropped = _repair_pass(fps, defects, attempts, stalled,
                                              shape, flat_roof)
        if verbose and dropped:
            print(f"      dropped {dropped} unrepairable footprint(s)")
        if not fps:
            break

    remaining = len(defects) if defects else 0
    return {
        "verts": verts, "faces": faces, "binfo": binfo, "footprints": fps,
        "report": {
            "iterations": len(history),
            "history": history,
            "final_built": len(binfo.get("buildings", [])),
            "unrepaired": remaining,
            "unrepaired_detail": {int(k): v for k, v in
                                  list(defects.items())[:20]} if defects else {},
            "converged": remaining == 0,
        },
    }
