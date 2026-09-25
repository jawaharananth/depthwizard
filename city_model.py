"""
Clean prism city model: flat ground, buildings as extruded volumes.

WHY THIS REPLACES THE HEIGHTFIELD APPROACH

Until now the whole scene was one heightfield: a grid of vertices lifted by
the DSM, with the satellite image projected straight down onto it. Buildings
were therefore *bumps in the ground*, not objects. That has three consequences
no amount of texturing or shading can fix:

  * a wall is a steep part of the terrain, so it is a ramp, not a plane -- the
    depth model smooths across every roof edge and the ground slopes up into
    each building;
  * planar UVs give a near-vertical face almost no texture area, so a handful
    of roof texels get stretched down the entire wall -- the "melting wax"
    look;
  * the roof is whatever the depth model produced, which is a lumpy surface
    rather than a flat roof.

Modelling buildings as separate prisms removes all three at once, because a
prism has genuine vertical wall faces and a genuinely flat roof by
construction. The ground underneath becomes bare earth, which is smooth, so
the satellite texture sits on it correctly.

The trade is explicit: a prism is a simplification. A building with a pitched
or stepped roof becomes a flat-topped block. That is a deliberate abstraction
-- the same one made by every vector city model -- and it is honest in a way
the lumpy version was not, because it does not pretend to roof detail the
input cannot support.
"""
import math

import numpy as np
import cv2

import segmentation as seg

# Largest height difference accepted across a single fitted roof, in metres.
# Real pitched roofs on buildings of this scale rise a few metres; anything
# beyond that means the footprint covers more than one structure, so the roof
# is kept flat rather than ramped between them.
MAX_ROOF_RELIEF_M = 5.0

# Pixels eroded from a footprint before its roof height is sampled. Boundary
# pixels straddle a height discontinuity and read high; see build_prisms.
ROOF_SAMPLE_ERODE_PX = 4


def _self_intersects(poly: np.ndarray) -> bool:
    """
    Does this ring cross itself?

    approxPolyDP can turn a ragged watershed contour into a ring whose edges
    cross, and ear clipping has no defined answer for such a polygon -- it
    emits overlapping triangles. Measured: 18 of 2605 footprints. Rare, but
    each one renders as a visible shard, so they are detected and replaced by
    their convex hull rather than triangulated blindly.
    """
    n = len(poly)
    if n < 4:
        return False

    def crosses(a, b, c, d):
        def side(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
        d1, d2 = side(c, d, a), side(c, d, b)
        d3, d4 = side(a, b, c), side(a, b, d)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue          # adjacent edges share a vertex
            if crosses(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return True
    return False


def _ear_clip(poly: np.ndarray) -> list:
    """
    Triangulate a simple polygon by ear clipping.

    Needed because real building footprints are concave (L-shapes, courtyards,
    notches) and a fan triangulation from vertex 0 -- what this project used
    before -- produces self-intersecting shards on anything non-convex. The
    earlier workaround was to replace every footprint with its minimum-area
    rectangle, which is always convex but throws the actual building shape
    away and turns a city block into a domino.

    Returns a list of index triples into `poly`.
    """
    n = len(poly)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # Work counter-clockwise so an "ear" is a convex corner with positive area.
    idx = list(range(n))
    signed = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                 for i in range(n))
    if signed < 0:
        idx.reverse()

    def point_in_tri(p, a, b, c):
        d1, d2, d3 = area2(p, a, b), area2(p, b, c), area2(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if area2(a, b, c) <= 0:
                continue                     # reflex corner, not an ear
            # An ear must contain no other vertex of the polygon.
            if any(point_in_tri(poly[j], a, b, c)
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break                            # degenerate ring; keep what we have
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def _regularise_rectilinear(poly: np.ndarray, max_area_change: float = 0.45):
    """
    Snap a footprint onto its own right-angle grid.

    approxPolyDP runs on a contour traced from a RASTER mask, so every
    staircase artefact along a diagonal wall survives as a short arbitrary-angle
    edge. Extruded, those read as jagged, spiky roofs -- the single most visible
    defect in the rendered city, and one no amount of segmentation quality
    fixes, because the raggedness is introduced after the mask.

    Real buildings are overwhelmingly rectilinear: walls meet at right angles,
    even when the building as a whole sits at an angle to the image grid. So the
    footprint is rotated onto the dominant axis of its own minimum-area
    rectangle, each edge is classified as running along X or along Y, runs of
    like edges collapse to a single straight line at their length-weighted mean,
    and the corners are rebuilt as the intersections of those lines. The result
    is an L, T, U or plain rectangle -- the shapes buildings actually are.

    The rotation is what makes this safe on a city that is not aligned to north:
    "rectilinear" is enforced in the building's own frame, not the image's.

    Returns the original polygon unchanged if the snap would distort its area by
    more than max_area_change, which is the signal that the ring was not a
    building outline in the first place (a merged blob, a courtyard, a sliver).
    """
    if len(poly) < 4:
        return poly

    poly = poly.astype(np.float32)
    orig_area = abs(cv2.contourArea(poly))
    if orig_area < 1.0:
        return poly

    # Dominant axis from the minimum-area rectangle, so a building at 30 deg to
    # the image grid is regularised against ITS axes rather than the image's.
    angle = cv2.minAreaRect(poly)[2]
    theta = np.deg2rad(angle)
    c, s = float(np.cos(-theta)), float(np.sin(-theta))
    fwd = np.array([[c, -s], [s, c]], np.float32)
    inv = np.array([[c, s], [-s, c]], np.float32)

    centre = poly.mean(axis=0)
    local = (poly - centre) @ fwd.T

    n = len(local)
    # 0 = edge runs along X (a horizontal wall), 1 = along Y (a vertical wall).
    cls = []
    for i in range(n):
        d = local[(i + 1) % n] - local[i]
        cls.append(0 if abs(d[0]) >= abs(d[1]) else 1)

    # Collapse consecutive like-classified edges into one wall each.
    runs = []          # (class, [edge indices])
    for i in range(n):
        if runs and runs[-1][0] == cls[i]:
            runs[-1][1].append(i)
        else:
            runs.append((cls[i], [i]))
    # The ring wraps: if the last run matches the first, merge them.
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1].extend(runs[-1][1])
        runs.pop()

    # A rectilinear ring alternates X-walls and Y-walls, so it needs an even
    # number of at least four. Anything else is not a shape this can snap.
    if len(runs) < 4 or len(runs) % 2 != 0:
        return poly

    # Each wall becomes a single line: y = const for an X-wall, x = const for a
    # Y-wall, placed at the mean of its edges weighted by their length so a long
    # true wall outvotes the short staircase steps hanging off it.
    lines = []
    for klass, idxs in runs:
        num = 0.0
        den = 0.0
        for i in idxs:
            a, b = local[i], local[(i + 1) % n]
            if klass == 0:                      # X-wall: fix its y
                w = abs(b[0] - a[0]) + 1e-6
                num += w * 0.5 * (a[1] + b[1])
            else:                               # Y-wall: fix its x
                w = abs(b[1] - a[1]) + 1e-6
                num += w * 0.5 * (a[0] + b[0])
            den += w
        lines.append((klass, num / den))

    # Corners are the intersections of consecutive walls. Because the runs
    # alternate, each consecutive pair is one X-wall and one Y-wall, which meet
    # at exactly one point -- a true right angle by construction.
    snapped = []
    m = len(lines)
    for i in range(m):
        k0, v0 = lines[i]
        k1, v1 = lines[(i + 1) % m]
        if k0 == k1:
            return poly
        x = v0 if k0 == 1 else v1
        y = v0 if k0 == 0 else v1
        snapped.append((x, y))

    out = (np.asarray(snapped, np.float32) @ inv.T) + centre

    new_area = abs(cv2.contourArea(out))
    if new_area < 1.0 or abs(new_area - orig_area) / orig_area > max_area_change:
        return poly
    return out


def extract_footprints(seg_labels: np.ndarray, ndsm: np.ndarray = None,
                       min_area_px: int = 300, simplify_frac: float = 0.012,
                       band_m: float = 3.0) -> list:
    """
    Building footprints as simplified polygons, split by roof height.

    Connected components of the building mask alone do not give buildings.
    Once segmentation covers a realistic ~40% of a dense tile, neighbouring
    structures touch and the whole city collapses into a handful of giant
    blobs -- measured on JAX_068, 99 components for the entire scene, of which
    only 20 survived as prisms. Worse, a percentile roof height over a blob
    that spans several buildings and the street between them is not any of
    their heights.

    Adjacent buildings are almost always different heights, so quantising the
    normalised surface into bands (one band ~ a couple of storeys) and taking
    components *within* a band separates them, and gives each resulting region
    a roof at a consistent height by construction.

    approxPolyDP tolerance scales with each contour's own perimeter, so a
    large block and a small shed simplify proportionally.
    """
    mask = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if ndsm is None:
        regions = [mask]
    else:
        h = np.where(mask > 0, ndsm, np.nan)
        finite = np.isfinite(h)
        if not finite.any():
            return []
        top = float(np.nanpercentile(h, 99.5))
        n_bands = max(1, int(np.ceil(top / max(band_m, 0.5))))
        regions = []
        for b in range(n_bands):
            lo, hi = b * band_m, (b + 1) * band_m
            band = ((h >= lo) & (h < hi)).astype(np.uint8)
            if band.sum() < min_area_px:
                continue
            # Open to drop the one-pixel fringe where a band boundary cuts
            # across a sloped roof, which would otherwise spawn slivers.
            band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k)
            regions.append(band)

    polys = []
    for band in regions:
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area_px:
                continue
            eps = simplify_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
            if len(approx) < 3:
                continue
            polys.append(_regularise_rectilinear(approx))
    return polys


# A repaired footprint must stay the shape it was. The convex hull does not:
# it is the SMALLEST shape containing the ring, which for a C-shape, an L-shape
# or a figure-of-eight is dramatically larger than the building. Measured on one
# uploaded tile, 17 of 200 rings self-intersected and the worst spanned 9403 m2
# at solidity 0.325 -- hulling those produced the enormous flat plates that made
# the render unusable.
MIN_SOLIDITY = 0.35        # below this the ring is a merged blob, not a building
MIN_REPAIR_AREA_PX = 20


def _repair_ring(poly, shape):
    """Return a valid simple ring for a self-intersecting footprint, or None.

    Rasterise the ring and re-trace its outer contour. Filling uses the even-odd
    rule, so the crossing lobes that made the ring invalid resolve into a single
    filled region, and the contour of that region is simple by construction --
    while still following the building's actual outline rather than jumping to
    its convex hull.
    """
    H, W = shape
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    pad = 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    sub_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(sub_mask, [poly.astype(np.int32) - [x0, y0]], 1)
    if sub_mask.sum() < MIN_REPAIR_AREA_PX:
        return None

    cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        return None
    approx = approx + [x0, y0]

    # A ring that is still not simple, or that is a sprawling blob rather than a
    # building, is dropped. Losing a footprint costs one building; keeping a bad
    # one costs a plate across the whole scene.
    if _self_intersects(approx):
        return None
    area = cv2.contourArea(approx.astype(np.float32))
    if area < MIN_REPAIR_AREA_PX:
        return None
    hull = cv2.convexHull(approx.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and (area / hull_area) < MIN_SOLIDITY:
        return None
    return approx


def build_prisms(footprints: list, dsm: np.ndarray, dtm: np.ndarray,
                 gsd_x_m: float, gsd_y_m: float, min_height_m: float = 2.5,
                 roof_percentile: float = 70.0, image_np: np.ndarray = None,
                 flat_roof_indices: set = None, roof_dsm: np.ndarray = None,
                 lean_px_per_m: tuple = None):
    """
    One flat-roofed prism per footprint: vertical walls plus a flat roof.

    Height comes from the DSM inside the footprint against the terrain beneath
    it. The roof uses a percentile rather than the maximum so a single noisy
    pixel, or an aerial on the roof, does not set the whole building's height.

    The base is deliberately sunk to the LOW end of the terrain under the
    footprint. A building based at the average terrain leaves a visible gap on
    the downhill side; sinking it means the wall simply continues below ground
    where nothing can see it.
    """
    verts, faces, index, colors, uvs = [], [], [], [], []
    skipped = 0

    # roof_dsm, when given, is sampled for roof height/plane-fit ONLY -- base
    # height still comes from dtm against the real (possibly squared) dsm's
    # terrain, and nothing else here changes. See build_city_image.py's
    # organic_roofs option for why a caller would pass a second surface
    # instead of just using dsm directly.
    _roof_src = roof_dsm if roof_dsm is not None else dsm
    H, W = dsm.shape
    for _poly_i, poly in enumerate(footprints):
        if _self_intersects(poly):
            repaired = _repair_ring(poly, (H, W))
            if repaired is None or len(repaired) < 3:
                skipped += 1
                continue
            poly = repaired
        xs = np.clip(poly[:, 0].astype(int), 0, W - 1)
        ys = np.clip(poly[:, 1].astype(int), 0, H - 1)

        x0, y0, w, h = cv2.boundingRect(poly.astype(np.int32))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x0 + w, W), min(y0 + h, H)
        if x1 <= x0 or y1 <= y0:
            skipped += 1
            continue
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, [poly.astype(np.int32) - [x0, y0]], 1)
        inside = sub.astype(bool)
        if inside.sum() < 20:
            skipped += 1
            continue

        # Sample the roof from the footprint's INTERIOR, not its full extent.
        #
        # A footprint boundary sits on a real height discontinuity, and both MVS
        # matching and the guided upsample smear across it, so edge pixels carry
        # a mixture of roof and whatever is beyond. Measured against LiDAR on
        # JAX_165, eroding the sampling mask by 4 px cut the roof-height bias
        # from +1.30 m to +1.10 m at the 80th percentile.
        #
        # The erosion is only for SAMPLING. The polygon itself is unchanged, so
        # the building keeps its true extent -- this affects what height is read,
        # never where the walls stand.
        _er = cv2.erode(sub, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ROOF_SAMPLE_ERODE_PX * 2 + 1,) * 2)).astype(bool)
        # Fall back to the full footprint when erosion leaves too little to
        # sample -- a small shed can erode away entirely.
        inside_sample = _er if _er.sum() >= 20 else inside
        # roof_h, the min-height gate and the roof_lo/roof_hi clamp band all
        # come from the ROBUST dsm, unconditionally -- these decide whether a
        # building exists and roughly how tall it is, and that decision must
        # not move just because a second, noisier surface is also available.
        # Verified directly (DFC2019 JAX_004): sourcing this gate from the
        # noisier roof_dsm instead dropped surviving buildings 58->27 purely
        # from more of them failing min_height_m, an effect that had nothing
        # to do with roof shape and everything to do with which array a
        # threshold check happened to read.
        roof_vals = dsm[y0:y1, x0:x1][inside_sample]
        roof_h = float(np.percentile(roof_vals, roof_percentile))
        # The organic sample, used ONLY below to inform the plane's SHAPE.
        roof_vals_organic = (_roof_src[y0:y1, x0:x1][inside_sample]
                             if roof_dsm is not None else roof_vals)

        # Base on the LOW end of the terrain under the footprint: a building
        # based at average terrain leaves a visible gap on the downhill side,
        # whereas sinking it means the wall continues below ground where
        # nothing can see it.
        base_h = float(np.percentile(dtm[y0:y1, x0:x1][inside], 5))
        if roof_h - base_h < min_height_m:
            skipped += 1
            continue

        # ROOF PLANE, fitted to the measured surface.
        #
        # Every roof used to be one flat value -- the 80th percentile of the DSM
        # inside the footprint -- so a pitched, shed or stepped roof came out as
        # a flat lid at an arbitrary height, and a whole town of them reads as
        # packing crates. The DSM does carry the roof's slope; it was simply
        # being collapsed to a scalar before anything could use it.
        #
        # A least-squares plane z = ax + by + c over the footprint's own pixels
        # recovers that slope. It is a measurement, not a style: a flat roof
        # fits a plane with a ~ b ~ 0 and is unchanged, while a shed or a tilted
        # roof gets its real pitch. It also needs no extra topology -- each roof
        # vertex simply moves to its own height -- so it costs one small solve
        # per building and nothing downstream changes.
        #
        # Fitted on the middle of the height distribution so a rooftop plant
        # room, an aerial or a parapet does not tilt the whole roof.
        # Index the SAME pixels roof_vals came from. These diverged when roof
        # sampling moved to the eroded mask: the plane was fitted against
        # coordinates from the full footprint while the heights came from its
        # interior, so the two arrays no longer matched in length.
        ys_i, xs_i = np.nonzero(inside_sample)
        # The clamp band's WIDTH must come from whichever sample the plane is
        # actually fitted against. When roof_dsm is given, dsm has already
        # been squared to one constant per building by prismify_buildings
        # (see build_city_image.py) -- robust roof_vals therefore has zero
        # spread, and a band built from it would clamp every organic vertex
        # straight back to a single flat value regardless of any real tilt
        # the plane found, silently discarding the whole point of roof_dsm.
        # Confirmed directly: this was exactly why an earlier version of this
        # fix restored building count/audit correlation but produced zero
        # roof variation in the final mesh. The organic sample's own
        # percentile band is used instead, shifted by the same amount the
        # plane itself is recentred by below, so both stay anchored to the
        # same trusted roof_h.
        _organic_shift = 0.0
        if roof_dsm is not None:
            _organic_shift = roof_h - float(np.mean(roof_vals_organic))
            roof_lo = float(np.percentile(roof_vals_organic, 10) + _organic_shift)
            roof_hi = float(np.percentile(roof_vals_organic, 90) + _organic_shift)
        else:
            roof_lo = float(np.percentile(roof_vals, 10))
            roof_hi = float(np.percentile(roof_vals, 90))
        plane = None
        # Footprints the repair loop has flagged get a flat roof at the robust
        # percentile instead of a fitted plane. A runaway plane and a height
        # that disagrees with every neighbour are both symptoms of the roof
        # SAMPLE being wrong -- a crane, a tree crown, a shadow edge -- and a
        # plane fitted to a wrong sample tilts to follow it. Refusing to fit is
        # the repair; the percentile is robust to the same contamination.
        _force_flat = bool(flat_roof_indices) and _poly_i in flat_roof_indices
        if ys_i.size >= 12:
            lo, hi = np.percentile(roof_vals, [15, 90])
            keep = (roof_vals >= lo) & (roof_vals <= hi)
            if keep.sum() >= 8:
                A = np.column_stack([
                    (xs_i[keep] + x0).astype(np.float32),
                    (ys_i[keep] + y0).astype(np.float32),
                    np.ones(int(keep.sum()), np.float32)])
                # Fit z-values come from the organic sample (raw signal when
                # roof_dsm is given, otherwise identical to roof_vals) so the
                # plane's SHAPE reflects real per-pixel variation where it
                # exists. The keep mask above is still chosen from the robust
                # sample -- which points to trust is a different question
                # from what height those points report.
                _fit_z = roof_vals_organic[keep].astype(np.float32)
                try:
                    coef, *_ = np.linalg.lstsq(A, _fit_z, rcond=None)
                    fitted = A @ coef
                    resid = float(np.std(_fit_z - fitted))
                    # Reject a fit that explains less than the flat value does --
                    # a noisy or occluded roof should stay flat rather than be
                    # given a slope the data does not support. Compared
                    # against the SAME sample the fit was done against.
                    if resid < float(np.std(_fit_z)) * 0.98:
                        # Reject a plane whose predicted tilt across THIS
                        # footprint exceeds what the roof's own height spread
                        # supports. Fitting well in a least-squares sense does
                        # not stop a plane from extrapolating absurdly once it
                        # leaves the pixels it was fitted on.
                        corner_z = (coef[0] * poly[:, 0] + coef[1] * poly[:, 1]
                                    + coef[2])
                        tilt = float(corner_z.max() - corner_z.min())
                        band = roof_hi - roof_lo
                        # A second, height-relative cap alongside the absolute
                        # one: a roof cannot plausibly rise more than about
                        # half the building's OWN wall height, or it reads as
                        # a blade rather than a pitch -- a 3 m single-storey
                        # house does not have a 5 m ridge. This is not the
                        # "scale the allowance with the band" mistake the
                        # comment above already rejected (that scaled with a
                        # NOISE-DERIVED quantity, granting the most freedom
                        # exactly where data was least trustworthy); roof_h
                        # and base_h both come from the robust dsm, completely
                        # independent of the organic sample the plane itself
                        # was fitted from, so this cap does not loosen with
                        # bad data the way that one did. Found necessary by
                        # direct inspection: Karnataka's real per-pixel signal
                        # (confirmed genuine, std 0.9-1.8 m per building, not
                        # noise-floor) was passing every existing gate yet
                        # rendering as visibly jagged, implausible spikes on
                        # this scene's mostly single-storey building stock.
                        _height_cap = max(1.0, 0.5 * (roof_h - base_h))
                        _tilt_cap = min(MAX_ROOF_RELIEF_M, _height_cap)
                        # A plane is only fitted to a roof that is actually ONE
                        # roof. A wide height band inside a footprint means the
                        # polygon spans several structures of different heights,
                        # and the best-fitting plane through them is a ramp
                        # between two buildings, not anyone's roof.
                        #
                        # Scaling the allowance WITH the band was the mistake in
                        # the first attempt: it granted the most freedom exactly
                        # where the data was least trustworthy, leaving tilt at
                        # 43 m for the 99th percentile. The allowance is now an
                        # absolute cap, so a mixed footprint gets a flat roof at
                        # a robust height instead of a blade.
                        if (not _force_flat
                                and band <= _tilt_cap
                                and tilt <= _tilt_cap):
                            plane = coef
                            if roof_dsm is not None:
                                # Recentre: shift the plane's constant term so
                                # its own mean over the fitted sample equals
                                # the ROBUST roof_h, not whatever absolute
                                # level the noisier organic sample happened to
                                # average to. This keeps the real, informative
                                # part (the fitted SHAPE/tilt) while anchoring
                                # its absolute height to the same trusted
                                # value the flat fallback and every gate above
                                # already agreed on -- a building's average
                                # roof height must not silently change
                                # depending on whether its roof turned out
                                # pitched or flat.
                                mean_fitted = float(np.mean(A @ plane))
                                plane = plane.copy()
                                plane[2] += (roof_h - mean_fitted)
                        else:
                            plane = None
                except np.linalg.LinAlgError:
                    plane = None

        # Roof colour, measured from the image inside this footprint.
        #
        # A uniform grey city is legible but wrong: real building stock is
        # brick, white render, dark membrane roofing, painted metal. The median
        # is used rather than the mean so a bright HVAC unit or a skylight does
        # not shift the whole roof, and it is lifted slightly toward white
        # because a roof seen in a lit 3D scene should not be as dark as its
        # nadir photograph, where it is partly self-shadowed.
        if image_np is not None:
            # The building's own measured colour, unaltered. Roofs render
            # from the texture itself; this colour is for the walls, so where
            # the image is oblique enough to show this building's facade, the
            # facade's own pixels are sampled -- the swept strip between the
            # roof outline and where its base appears, lean * height away.
            # Otherwise the roof's median is the only real sample there is.
            rgb = None
            if lean_px_per_m is not None:
                _h = float(roof_h - base_h)
                _sx = -lean_px_per_m[0] * _h
                _sy = -lean_px_per_m[1] * _h
                if np.hypot(_sx, _sy) >= 3.0:
                    _pts = np.vstack([poly, poly + [_sx, _sy]]).astype(np.float32)
                    _hull = cv2.convexHull(_pts).reshape(-1, 2)
                    _fx0 = int(max(0, np.floor(_hull[:, 0].min())))
                    _fy0 = int(max(0, np.floor(_hull[:, 1].min())))
                    _fx1 = int(min(W, np.ceil(_hull[:, 0].max()) + 1))
                    _fy1 = int(min(H, np.ceil(_hull[:, 1].max()) + 1))
                    if _fx1 - _fx0 > 1 and _fy1 - _fy0 > 1:
                        _fm = np.zeros((_fy1 - _fy0, _fx1 - _fx0), np.uint8)
                        cv2.fillPoly(_fm, [(_hull - [_fx0, _fy0]).astype(np.int32)], 1)
                        cv2.fillPoly(_fm, [(poly - [_fx0, _fy0]).astype(np.int32)], 0)
                        _fpix = image_np[_fy0:_fy1, _fx0:_fx1][_fm > 0]
                        if len(_fpix) >= 12:
                            rgb = np.median(_fpix, axis=0).astype(np.float32) / 255.0
            if rgb is None:
                patch = image_np[y0:y1, x0:x1][inside]
                rgb = np.median(patch, axis=0).astype(np.float32) / 255.0
        else:
            rgb = np.array([0.82, 0.80, 0.76], np.float32)

        n = len(poly)
        off = len(verts)
        # World coordinates: X east, Y up, -Z south, matching build_ground_mesh.
        for px, py in poly:
            verts.append((px * gsd_x_m, base_h, -py * gsd_y_m))
        for px, py in poly:
            if plane is not None:
                zh = float(plane[0] * px + plane[1] * py + plane[2])
                # Clamp to the roof's own MIDDLE, not its maximum.
                #
                # Clamping at roof_vals.max() is no constraint at all: a single
                # tall pixel anywhere in the footprint -- a neighbouring tower
                # clipped by the polygon, an aerial, an MVS outlier -- lets a
                # plane extrapolate to that height at a far corner. Measured
                # across 2605 buildings, roof tilt reached 49 m at the 99th
                # percentile and 79 m at worst, with 165 buildings tilted more
                # than 20 m across a single roof. Rendered, those are the
                # blade-like spikes standing between buildings.
                #
                # A roof can only be as tilted as the surface inside it actually
                # is, so the 10th-90th percentile band bounds it. That admits a
                # genuine pitch while making a runaway extrapolation impossible.
                zh = min(max(zh, roof_lo), roof_hi)
                zh = max(zh, base_h + min_height_m)
            else:
                zh = roof_h
            verts.append((px * gsd_x_m, zh, -py * gsd_y_m))
        for _ in range(2 * n):
            colors.append(rgb)
        # Planar top-down UV, same convention as build_ground_mesh's
        # (px/W, 1 - py/H), emitted for both rings so the array stays
        # aligned with verts/colors. The roof is the one building surface a
        # nadir image actually photographs, so it can carry the real
        # satellite texture pixel-for-pixel with no distortion; the base
        # ring's copy is never sampled by the viewer, which only maps this
        # UV onto the roof triangles and leaves walls untextured -- stretching
        # a roof-edge texel down a wall is a different, uglier problem.
        for px, py in poly:
            uvs.append((px / max(W - 1, 1), 1.0 - py / max(H - 1, 1)))
        for px, py in poly:
            uvs.append((px / max(W - 1, 1), 1.0 - py / max(H - 1, 1)))

        # Walls: one quad per footprint edge, emitted as two triangles. These
        # are exactly vertical because both rings share X and Z.
        for i in range(n):
            j = (i + 1) % n
            b0, b1 = off + i, off + j
            t0, t1 = off + n + i, off + n + j
            faces.append((b0, t0, t1))
            faces.append((b0, t1, b1))

        for (a, b, c) in _ear_clip(poly):
            faces.append((off + n + a, off + n + b, off + n + c))

        index.append({
            # Which INPUT footprint this prism came from. Callers that pair
            # buildings back to polygons cannot use position, because skipped
            # footprints shift every later index -- a validation script doing
            # zip(buildings, polys) silently compared one building's height
            # against another building's footprint the moment anything was
            # skipped, and reported MAE 12.48 m for a pipeline measuring 2.60 m.
            "poly_index": int(_poly_i),
            "vertex_offset": off, "vertex_count": 2 * n, "n_sides": n,
            "base_h": base_h, "roof_h": roof_h,
            "height_m": roof_h - base_h,
            "centroid_px": (float(poly[:, 0].mean()), float(poly[:, 1].mean())),
        })

    if not verts:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
                {"buildings": [], "skipped": skipped,
                 "colors": np.zeros((0, 3), np.float32),
                 "uvs": np.zeros((0, 2), np.float32)})

    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)

    # Wall winding: a quad wound the wrong way is back-face culled and the
    # building looks hollow from outside. Rather than reason about polygon
    # orientation per footprint, fix it empirically -- point each wall normal
    # away from its own building centre.
    f = _orient_outward(v, f, index)

    # Drop degenerate triangles before they leave the builder. Orthogonalising a
    # footprint can leave two vertices coincident, and the resulting zero-area
    # face has no normal -- it renders as nothing but trips every downstream
    # geometry check, so the warning outlives the triangle that caused it.
    if len(f):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        keep = area2 > 1e-9
        if not keep.all():
            f = f[keep]
    return v, f, {"buildings": index, "skipped": skipped,
                  "colors": np.asarray(colors, dtype=np.float32),
                  "uvs": np.asarray(uvs, dtype=np.float32)}


def _orient_outward(verts: np.ndarray, faces: np.ndarray, index: list) -> np.ndarray:
    """
    Flip any face whose normal points into its own building.

    Contour orientation from findContours depends on whether a region is an
    outer boundary or a hole, so assuming one winding leaves some buildings
    inside-out. Testing the normal against the direction to the building's own
    centre decides it per face and cannot be fooled by that.
    """
    faces = faces.copy()
    for rec in index:
        o, c = rec["vertex_offset"], rec["vertex_count"]
        block = verts[o:o + c]
        centre = block.mean(axis=0)
        sel = np.where((faces[:, 0] >= o) & (faces[:, 0] < o + c))[0]
        for fi in sel:
            a, b, cc = verts[faces[fi, 0]], verts[faces[fi, 1]], verts[faces[fi, 2]]
            n = np.cross(b - a, cc - a)
            if abs(n[1]) > max(abs(n[0]), abs(n[2])):
                # Roof face: must point up.
                if n[1] < 0:
                    faces[fi] = faces[fi][::-1]
            else:
                outward = ((a + b + cc) / 3.0) - centre
                outward[1] = 0.0
                if float(np.dot(n[[0, 2]], outward[[0, 2]])) < 0:
                    faces[fi] = faces[fi][::-1]
    return faces


def flatten_ground(dtm: np.ndarray, seg_labels: np.ndarray,
                   smooth_m: float = 6.0, gsd_m: float = 1.0) -> np.ndarray:
    """
    Bare-earth surface for the ground mesh.

    The DTM still carries the residue of whatever the depth model did around
    each building, and any residual bump there is now pure error: the building
    standing on that spot is a separate prism. Smoothing at a scale wider than
    a building removes the residue while keeping real terrain, which varies
    over much longer distances.
    """
    sigma_px = max(1.0, smooth_m / max(gsd_m, 1e-6))
    return cv2.GaussianBlur(dtm.astype(np.float32), (0, 0), sigmaX=sigma_px)


# A tree crown wider than it is tall is the norm; measured directly on a
# real scene (Karnataka kn_a) the worst RAW crown heights before this cap
# even applies reached 19.2 m on a 15.6 m span and 14-19 m on 5-8 m spans --
# taller than the scene's own assumed tallest structure (14 m) and clearly
# depth-field noise, not real trees, since 2.5x was permissive enough to let
# them through uncapped. 1.2 still allows a genuinely slender tree (palm,
# conifer) to read as taller than wide; anything beyond that is a spike.
CANOPY_MAX_ASPECT = 1.2

# Vegetation classification here is a colour heuristic (Excess Green
# Index), not a learned semantic model -- see segmentation.py's own
# docstring. A real, observed failure mode: a large greenish-toned open
# area (a grass field, an airfield median, a shadowed tarmac patch with a
# green colour cast) reads as "vegetation" over its whole extent, and
# because it is ONE large contiguous blob rather than many small tree
# crowns, cv2.approxPolyDP simplifies it down to a handful of vertices --
# a large, sharp-edged wedge, not anything that reads as foliage. A real
# tree cluster in an urban scene is bounded; an area this large almost
# never is one. 2000 sq. m is generous for a real dense grove and still
# well under what a misclassified field/airfield patch covers.
CANOPY_MAX_AREA_M2 = 2000.0

# Compactness = 4*pi*area/perimeter^2, 1.0 for a perfect circle. Merged-blob
# ground artifacts measured at 0.02-0.12 on a real scene; ordinary vegetation
# contours (including irregular real canopy) median 0.36. 0.22 sits in the
# real gap between the two populations, not a guessed round number.
CANOPY_MIN_COMPACTNESS = 0.22


def build_canopy(seg_labels: np.ndarray, dsm: np.ndarray, ground: np.ndarray,
                 gsd_x_m: float, gsd_y_m: float, min_area_px: int = 250,
                 min_height_m: float = 2.0, band_m: float = 4.0):
    """
    Tree canopy as low rounded volumes.

    A prism city with no vegetation reads as a model of a car park: the
    reference imagery this is aimed at is full of green, and every real tile
    here has substantial tree cover that the pipeline already segments and
    then throws away. Canopy is emitted as its own mesh so it can take a green
    material rather than the buildings' stone.

    Canopy is extruded like a building but with the top ring inset toward the
    patch centre, which gives a tapered crown instead of a green box. Trees are
    not boxes, and at this scale a taper is enough to read as foliage.
    """
    veg = (seg_labels == seg.CLASS_IDX["vegetation"]).astype(np.uint8)
    if not veg.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, k)

    ndsm = np.maximum(dsm - ground, 0.0)
    h = np.where(veg > 0, ndsm, np.nan)
    if not np.isfinite(h).any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    top = float(np.nanpercentile(h, 99.0))
    n_bands = max(1, int(np.ceil(top / max(band_m, 0.5))))
    # A second, absolute ceiling alongside the aspect-ratio one: no single
    # patch's crown may exceed what most of the scene's OWN canopy actually
    # supports. The aspect cap alone was measured insufficient on a real
    # scene (Karnataka kn_a) -- a wide patch (15.6 m span) still capped to
    # 18.7 m, barely different from its uncapped 19.2 m, because a
    # proportional cap grows right along with a patch that is itself an
    # outlier. This is a scene-relative statistic, not a borrowed constant,
    # so it holds on a Tier C scene with no absolute metres just as well as
    # a metric one.
    # The median, not a high percentile: measured directly on a real scene
    # (Karnataka kn_a), even p95 of the vegetation height field was 23.3 m --
    # depth estimation over canopy is noisy (complex, non-planar surfaces
    # that stereo/MVS handles poorly, a known limitation, not scene content)
    # and that noise is a long tail, not a few outliers, so ANY high
    # percentile is itself contaminated. The median (3.6 m on that same
    # scene) sits where most real canopy pixels actually are and is immune
    # to the tail. A real tallest tree can reasonably run a few times the
    # scene's typical canopy height; 3x is generous without re-admitting the
    # 15-40 m spikes a percentile-based ceiling let through.
    _veg_median = float(np.nanpercentile(h, 50.0))
    _veg_ceiling = max(_veg_median * 3.0, min_height_m)

    verts, faces = [], []
    count = 0
    for b in range(n_bands):
        lo, hi = b * band_m, (b + 1) * band_m
        band = ((h >= lo) & (h < hi)).astype(np.uint8)
        if band.sum() < min_area_px:
            continue
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            _cnt_area = cv2.contourArea(cnt)
            if _cnt_area < min_area_px:
                continue

            # Real tree crowns are round-ish; the morphological CLOSE above
            # merges many small, separately-real vegetation patches into one
            # blob whenever they sit close together, and that merged blob is
            # long and jagged, not round. Measured on a real scene: the huge
            # ground-artifact blobs score compactness 0.02-0.12, ordinary
            # contours score a median of 0.36. Reject shapes shaped nothing
            # like foliage before they ever reach the mesh.
            _perim = cv2.arcLength(cnt, True)
            if _perim <= 0:
                continue
            _compactness = 4.0 * math.pi * _cnt_area / (_perim * _perim)
            if _compactness < CANOPY_MIN_COMPACTNESS:
                continue

            eps = 0.02 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
            if len(poly) < 3:
                continue

            x0, y0, w, hh = cv2.boundingRect(poly.astype(np.int32))
            x1, y1 = min(x0 + w, dsm.shape[1]), min(y0 + hh, dsm.shape[0])
            x0, y0 = max(x0, 0), max(y0, 0)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(sub, [poly.astype(np.int32) - [x0, y0]], 1)
            inside = sub.astype(bool)
            if inside.sum() < 20:
                continue

            _area_m2 = float(inside.sum()) * gsd_x_m * gsd_y_m
            if _area_m2 > CANOPY_MAX_AREA_M2:
                continue   # almost certainly a misclassified open area, not a tree cluster

            crown = float(np.percentile(dsm[y0:y1, x0:x1][inside], 75))
            base = float(np.percentile(ground[y0:y1, x0:x1][inside], 20))
            if crown - base < min_height_m:
                continue

            # A crown cannot be far taller than the patch it grows from. The
            # depth field spikes inside small vegetation patches, and without
            # this a ten-pixel patch takes the tallest height in the scene and
            # renders as a needle. The cap is an aspect ratio, not a height in
            # metres, so it holds in a Tier C scene that has no scale at all.
            span_m = math.sqrt(max(float(inside.sum()), 1.0)
                               * gsd_x_m * gsd_y_m)
            tallest = base + min(CANOPY_MAX_ASPECT * span_m, _veg_ceiling)
            if crown > tallest:
                crown = tallest
                if crown - base < min_height_m:
                    continue

            centre = poly.mean(axis=0)
            n = len(poly)
            off = len(verts)
            for px, py in poly:
                verts.append((px * gsd_x_m, base, -py * gsd_y_m))
            for px, py in poly:
                # Inset the crown so the volume tapers upward.
                ix = centre[0] + (px - centre[0]) * 0.62
                iy = centre[1] + (py - centre[1]) * 0.62
                verts.append((ix * gsd_x_m, crown, -iy * gsd_y_m))

            for i in range(n):
                j = (i + 1) % n
                b0, b1 = off + i, off + j
                t0, t1 = off + n + i, off + n + j
                faces.append((b0, t0, t1))
                faces.append((b0, t1, b1))
            for (a, bb, c) in _ear_clip(poly):
                faces.append((off + n + a, off + n + bb, off + n + c))
            count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int64)
    return v, f, count


def _organic_tree(cx_m: float, base_h: float, crown_h: float, cz_m: float,
                  radius_m: float, n_sides: int = 8, n_rings: int = 4,
                  trunk_frac: float = 0.15, seed: int = 0):
    """One tree: a thin trunk plus a rounded, continuous-mesh crown.

    The crown is several stitched latitude rings, not a single cone ring --
    a real continuous surface per tree, the same idea a heightfield terrain
    uses (many small connected faces forming a smooth surface), just scoped
    to one tree's own footprint instead of the whole ground. That scoping is
    the actual fix for what a continuous GROUND mesh got wrong for
    vegetation (tested directly this session): a terrain grid's vertex
    spacing is metres wide, so a real crown compressed into one or two grid
    cells has no choice but to look like a spike between flat neighbours.
    A per-tree mesh sets its own vertex spacing to the crown's own real
    size, so the same continuous-surface idea can actually round off the
    way a crown does instead of aliasing.

    Ring radius follows sin(pi*t) for t in [0,1] from the trunk join to the
    apex -- narrow where it meets the trunk, fullest at mid-height, tapering
    to a point at top: an ellipsoid-ish silhouette, not a boring cone. Small
    per-vertex jitter (+-4%, seeded by the tree's own position so it is
    stable across reloads, not shimmering) breaks up the perfect radial
    symmetry a bare revolve would otherwise show, since no two real crowns
    are that regular. Real forestry crown/height ratio sizes the whole
    shape, same as before this only changes.
    """
    rng = np.random.RandomState(seed)
    verts, faces = [], []
    trunk_top = base_h + (crown_h - base_h) * trunk_frac
    trunk_r = max(radius_m * 0.12, 0.05)
    for px, py in [(cx_m + trunk_r * math.cos(2 * math.pi * i / n_sides),
                    cz_m + trunk_r * math.sin(2 * math.pi * i / n_sides))
                   for i in range(n_sides)]:
        verts.append((px, base_h, py))
    for px, py in [(cx_m + trunk_r * math.cos(2 * math.pi * i / n_sides),
                    cz_m + trunk_r * math.sin(2 * math.pi * i / n_sides))
                   for i in range(n_sides)]:
        verts.append((px, trunk_top, py))
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append((i, n_sides + i, n_sides + j))
        faces.append((i, n_sides + j, j))

    crown_span = crown_h - trunk_top
    ring_start = []
    ring_is_pole = []
    for r in range(n_rings + 1):
        t = r / n_rings
        y = trunk_top + crown_span * t
        rad = radius_m * math.sin(math.pi * t)
        ring_start.append(len(verts))
        if rad < radius_m * 0.02:
            verts.append((cx_m, y, cz_m))
            ring_is_pole.append(True)
        else:
            for i in range(n_sides):
                jitter = 1.0 + 0.04 * (rng.rand() - 0.5) * 2.0
                px = cx_m + rad * jitter * math.cos(2 * math.pi * i / n_sides)
                pz = cz_m + rad * jitter * math.sin(2 * math.pi * i / n_sides)
                verts.append((px, y, pz))
            ring_is_pole.append(False)

    for r in range(n_rings):
        i0, i1 = ring_start[r], ring_start[r + 1]
        if not ring_is_pole[r] and not ring_is_pole[r + 1]:
            for i in range(n_sides):
                j = (i + 1) % n_sides
                faces.append((i0 + i, i1 + i, i1 + j))
                faces.append((i0 + i, i1 + j, i0 + j))
        elif ring_is_pole[r + 1]:          # fan up to the apex point
            for i in range(n_sides):
                j = (i + 1) % n_sides
                faces.append((i0 + i, i1, i0 + j))
        elif ring_is_pole[r]:              # fan down from the base point
            for i in range(n_sides):
                j = (i + 1) % n_sides
                faces.append((i0, i1 + j, i1 + i))
    return verts, faces


def build_trees(seg_labels: np.ndarray, chm: np.ndarray, ground: np.ndarray,
                gsd_x_m: float, gsd_y_m: float, min_height_m: float = 1.5,
                min_spacing_m: float = 2.5, max_trees: int = 6000):
    """Individual tree instances, detected as real local maxima in a real
    per-pixel canopy height raster (Meta HighResCanopyHeight/CHMv2), not a
    merged blob per vegetation patch.

    Each tree's position, height and crown radius all come from that same
    raster at the peak pixel -- nothing here invents a tree where the height
    map doesn't independently support one. A non-max-suppression window
    (min_spacing_m) stops one real crown being detected as several adjacent
    peaks, which monocular height noise would otherwise do freely.

    Capped at max_trees (highest peaks kept) purely for mesh size on a dense
    scene -- a lower real peak is dropped for budget, never faked to fill
    a quota, so a sparse scene legitimately returns fewer trees than the cap.
    """
    import scipy.ndimage as ndi

    veg = (seg_labels == seg.CLASS_IDX["vegetation"])
    if not veg.any() or chm is None:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    h = np.where(veg, chm, -np.inf).astype(np.float32)
    if not np.isfinite(h[veg]).any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    px_spacing = max(1, int(round(min_spacing_m / max(gsd_x_m, 1e-6))))
    footprint = np.ones((2 * px_spacing + 1, 2 * px_spacing + 1), bool)
    local_max = ndi.maximum_filter(h, footprint=footprint, mode="constant",
                                   cval=-np.inf)
    peaks = (h == local_max) & veg & np.isfinite(h) & (h >= min_height_m)
    ys, xs = np.nonzero(peaks)
    if not len(ys):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    heights = h[ys, xs]
    if len(ys) > max_trees:
        keep = np.argsort(-heights)[:max_trees]
        ys, xs, heights = ys[keep], xs[keep], heights[keep]

    verts, faces = [], []
    count = 0
    for y, x, tree_h in zip(ys, xs, heights):
        base_h = float(ground[y, x])
        crown_h = base_h + float(tree_h)
        # Typical broadleaf crown radius is roughly a third of tree height --
        # a real forestry rule of thumb, not fitted to this scene.
        radius_m = max(0.5, float(tree_h) * 0.32)
        cx_m = float(x) * gsd_x_m
        cz_m = -float(y) * gsd_y_m
        # Seeded by the tree's own pixel position, not a running counter, so
        # a given tree's crown jitter is stable across rebuilds rather than
        # depending on iteration order.
        _seed = (int(x) * 73856093) ^ (int(y) * 19349663)
        tv, tf = _organic_tree(cx_m, base_h, crown_h, cz_m, radius_m,
                               seed=_seed & 0x7fffffff)
        off = len(verts)
        verts.extend(tv)
        faces.extend([(a + off, b + off, c + off) for a, b, c in tf])
        count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int64)
    return v, f, count


def build_water(seg_labels: np.ndarray, ground: np.ndarray,
                gsd_x_m: float, gsd_y_m: float, min_area_px: int = 4000):
    """
    Water as a single flat surface per body, slightly below its banks.

    Standing water is flat by definition, so it is the one part of the scene
    that should NOT follow the estimated terrain. Sinking it a little under
    the surrounding ground stops the banks z-fighting with the water plane.
    """
    water = (seg_labels == seg.CLASS_IDX["water"]).astype(np.uint8)
    if not water.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    verts, faces = [], []
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            continue
        xs = np.clip(poly[:, 0].astype(int), 0, ground.shape[1] - 1)
        ys = np.clip(poly[:, 1].astype(int), 0, ground.shape[0] - 1)
        # Median of the bank ring, not its 25th percentile. Water is untextured,
        # so plane-sweep NCC is weak over it and the ground estimate around a
        # body scatters by roughly 2 m; a quartile of that scatter puts the
        # surface a couple of metres UNDER its own banks, which hides the body
        # completely. The median tracks the banks, and the 0.3 m is the only
        # deliberate sink -- enough to stop z-fighting, not enough to submerge.
        level = float(np.median(ground[ys, xs])) - 0.3

        off = len(verts)
        for px, py in poly:
            verts.append((px * gsd_x_m, level, -py * gsd_y_m))
        for (a, b, c) in _ear_clip(poly):
            faces.append((off + a, off + b, off + c))
        count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64), count


def detect_vehicles(image_np: np.ndarray, seg_labels: np.ndarray,
                    ground: np.ndarray, gsd_m: float,
                    min_area_m2: float = 4.0, max_area_m2: float = 24.0,
                    height_m: float = 1.5):
    """
    Cars on open ground, as small boxes.

    A car is roughly 4.5 x 1.8 m. At this ortho's 0.25 m/px that is about 130
    pixels -- large enough to find, small enough that the only usable cues are
    size, shape and local contrast. There is no vehicle class in the
    segmentation, so this is a heuristic detector, and it is gated hard:

      * only on open ground -- never on a roof, in vegetation or on water,
        which removes the large majority of look-alike blobs;
      * area within a real vehicle's footprint, in SQUARE METRES rather than
        pixels, so the gate does not silently change with resolution;
      * elongation between 1.5 and 4.5, since cars are oblong and most
        confusers (manhole covers, patches, bins) are not;
      * fill ratio against the oriented box, so ragged shadow slivers fail.

    Expect false positives on roof vents, skips and shadow edges, and misses on
    cars whose colour matches the tarmac. This is decoration, not a measured
    inventory, and the count is returned so it can be reported as such rather
    than implied to be exact.
    """
    open_ground = ((seg_labels == seg.CLASS_IDX["road"]) |
                   (seg_labels == seg.CLASS_IDX["bare_earth"]))
    if not open_ground.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Local contrast: a vehicle is brighter or darker than the surface it sits
    # on, whichever way round, so both tails are taken.
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, 2.5 / gsd_m))
    diff = gray - local
    sigma = float(np.std(diff[open_ground])) or 1.0
    blobs = ((np.abs(diff) > 1.6 * sigma) & open_ground).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, k)
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, k)

    px_area = gsd_m * gsd_m
    min_px = max(6, int(min_area_m2 / px_area))
    max_px = int(max_area_m2 / px_area)

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    verts, faces = [], []
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_px or area > max_px:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        if w < 1e-3 or h < 1e-3:
            continue
        long_side, short_side = max(w, h), min(w, h)
        elong = long_side / short_side
        if elong < 1.5 or elong > 4.5:
            continue
        if area / (w * h) < 0.55:
            continue
        # Reject anything longer than a bus or narrower than a bicycle.
        if long_side * gsd_m > 14.0 or short_side * gsd_m < 1.2:
            continue

        box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
        ys = np.clip(box[:, 1].astype(int), 0, ground.shape[0] - 1)
        xs = np.clip(box[:, 0].astype(int), 0, ground.shape[1] - 1)
        base = float(np.median(ground[ys, xs]))

        off = len(verts)
        for px, py in box:
            verts.append((px * gsd_m, base, -py * gsd_m))
        for px, py in box:
            verts.append((px * gsd_m, base + height_m, -py * gsd_m))
        for i in range(4):
            j = (i + 1) % 4
            b0, b1, t0, t1 = off + i, off + j, off + 4 + i, off + 4 + j
            faces.append((b0, t0, t1)); faces.append((b0, t1, b1))
        faces.append((off + 4, off + 5, off + 6))
        faces.append((off + 4, off + 6, off + 7))
        count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64), count
