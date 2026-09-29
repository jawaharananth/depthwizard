"""
Tall-building height from single-image physics, fused with a monocular model.

WHY

Every monocular height model under-calls tall buildings: ours compresses the
top end ~1.7x, and a competitor reports -15.6 m bias above 20 m. A global
rescale cannot fix it (the least-squares optimal global scale on validation is
0.98 -- the error is a compression of the TAIL, not a wrong unit). The same
single image carries two independent, geometric height measurements that do
not compress:

  shadow      a building of height h casts a ground shadow h * cot(sun_el)
              long, along the anti-sun direction (sun_az + 180).
  relief      in a terrain-only ortho the roof sits h * tan(zenith_sat) away
              from its base, along sat_az + 180 (away from the satellite). The
              wall facing the satellite is the visible band in between.

Both come from the view's own IMD metadata (meanSunEl/Az, meanSatEl/Az) -- no
ground truth, no second image.

CONVENTIONS (checked on overlays, see scripts/benchmark_tall.py --overlays)

  image: row y DOWN, column x RIGHT, north-up grid (the ortho is a UTM raster).
  azimuth: degrees clockwise from north. Unit step for azimuth a in pixels is
           (dx, dy) = (sin a, -cos a).

MEASUREMENT GEOMETRY

Shadow: rays are cast from just inside the roof along s = sun_az + 180. The
run starts at the roof's anti-sun edge (the roof hides the shadow's near
part) and ends at the shadow tip, which is the ground projection of the top
edge. With the roof displaced by the lean vector, the visible run is

    D = h * (cot(sun_el) - tan(z) * cos(sat_az - sun_az))

so h = D / that factor. (If the satellite looks from the shadow side the
visible shadowed wall extends the dark run by exactly the lean, which the
same formula carries through the sign of the cosine.)

Relief: the satellite-facing roof outline is translated toward the satellite
(sat_az) by t pixels; the base edge is the strongest oriented-gradient peak
beyond the roof edge. h = t * gsd / tan(z). At the most-nadir DFC2019 view
(z = 5.4 deg) this is 0.38 px per metre of height at 0.25 m, so it only
resolves genuinely tall buildings; it is reported with a confidence that says
so.

FUSION

A building fires only when physics is confident AND disagrees with the model
by more than 25 %. Then the building height becomes the inverse-variance
blend of model and physics, applied to the building's own roof plateau as a
multiplicative ratio weighted by how "roof-like" each pixel is (pixels below
30 % of the roof height -- ground, the ramp at the edge -- are left alone).
Candidates are only compact, non-vegetated components the model already
thinks are > 8 m, so ground and trees are never touched.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MIN_MODEL_H = 8.0          # only buildings the model already calls > 8 m
DISAGREE_FRAC = 0.25       # physics must differ from the model by more than this
MIN_AREA_M2 = 60.0
MAX_AREA_M2 = 30000.0
MIN_SOLIDITY = 0.6
MAX_GREEN = 0.04           # mean (2G-R-B)/(R+G+B) above this = vegetation


# --------------------------------------------------------------------------
# metadata / geometry
# --------------------------------------------------------------------------

def read_imd_geometry(imd_path: str) -> dict:
    """sun_el / sun_az / sat_el / sat_az / off_nadir (deg) from a WorldView IMD."""
    keys = {"meanSunEl": "sun_el", "meanSunAz": "sun_az", "meanSatEl": "sat_el",
            "meanSatAz": "sat_az", "meanOffNadirViewAngle": "off_nadir"}
    out = {}
    with open(imd_path) as f:
        for line in f:
            k, _, v = line.strip().rstrip(";").partition("=")
            k = k.strip()
            if k in keys:
                try:
                    out[keys[k]] = float(v.strip().strip('"'))
                except ValueError:
                    pass
    return out


def az_vec(az_deg: float) -> np.ndarray:
    """Unit pixel step (dx, dy) for a compass azimuth on a north-up, y-down grid."""
    a = math.radians(az_deg)
    return np.array([math.sin(a), -math.cos(a)])


def view_geometry(sun_el_deg, sun_az_deg, sat_el_deg=None, sat_az_deg=None,
                  off_nadir_deg=None) -> dict:
    s = az_vec(sun_az_deg + 180.0)                      # shadow direction
    if sat_el_deg is not None:
        zen = 90.0 - sat_el_deg                          # incidence at the ground
    elif off_nadir_deg is not None:
        zen = off_nadir_deg                              # slightly under-states it
    else:
        zen = None
    k_lean = math.tan(math.radians(zen)) if zen is not None else 0.0
    lean = az_vec(sat_az_deg + 180.0) if sat_az_deg is not None else np.zeros(2)
    k_shadow = 1.0 / math.tan(math.radians(max(sun_el_deg, 1.0)))
    cosd = float(lean @ s) if sat_az_deg is not None else 0.0
    return {"s": s, "lean": lean, "k_shadow": k_shadow, "k_lean": k_lean,
            "zenith_deg": zen, "shadow_factor": k_shadow - k_lean * cosd,
            "cos_lean_shadow": cosd}


# --------------------------------------------------------------------------
# image cues
# --------------------------------------------------------------------------

def shadow_mask(image: np.ndarray, gsd_m: float) -> np.ndarray:
    """Cast shadow = dark AND sky-lit (blue-shifted), after removing cars.

    shadow_correction.detect_shadow_mask (darker than a local mean with
    sigma = side/32) was measured on these orthos to mark dark cars and tree
    crowns but MISS the large building shadows: inside a 30 m shadow the
    "local" reference is itself shadow. Sampled on JAX_166, sunlit asphalt /
    roof / grass have blue fraction B/(R+G+B) 0.32-0.35 and brightness
    390-540; building shadows 0.38-0.48 and 160-310; tree crowns 0.36 at
    ~270. Shadows are lit only by the blue sky, so blue-shift plus darkness
    separates them from dark-but-sunlit material. A 1.25 m median removes
    cars and roof clutter first.
    """
    k = max(3, int(round(1.25 / gsd_m)) | 1)
    f = cv2.medianBlur(image, k).astype(np.float32)
    L = f.sum(axis=2)
    bfrac = f[..., 2] / (L + 1.0)
    ref = float(np.percentile(L, 70))           # sunlit level of this scene
    m = ((L < 0.65 * ref) & (bfrac > 0.365)) | (L < 0.40 * ref)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kk)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kk)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] * gsd_m * gsd_m < 6.0
    small[0] = False
    m[small[lab]] = 0
    return m.astype(bool)


def greenness(image: np.ndarray) -> np.ndarray:
    f = image.astype(np.float32)
    return (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1.0)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------

def find_candidates(ndsm: np.ndarray, image: np.ndarray, gsd_m: float,
                    min_h: float = MIN_MODEL_H) -> list:
    """Compact, non-green components the model already calls > min_h.

    Components of (ndsm > 4 m) are split to their roof plateau: the largest
    connected part above 60 % of the component's 95th percentile. Merged
    neighbours of different height separate there; a low annex attached to a
    tower is left out of the tower's plateau.
    """
    px_area = gsd_m * gsd_m
    lo = (ndsm > 4.0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lo, connectivity=8)
    green = greenness(image)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] * px_area < MIN_AREA_M2:
            continue
        x0, y0, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                        st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        sub = lab[y0:y0 + h, x0:x0 + w] == i
        hv = ndsm[y0:y0 + h, x0:x0 + w]
        p95 = float(np.percentile(hv[sub], 95))
        if p95 < min_h:
            continue
        core = (sub & (hv > 0.6 * p95)).astype(np.uint8)
        # a plateau can contain several separate towers: take each part
        cn, clab, cst, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        for j in range(1, cn):
            area = cst[j, cv2.CC_STAT_AREA] * px_area
            if not (MIN_AREA_M2 <= area <= MAX_AREA_M2):
                continue
            cm = clab == j
            h_model = float(np.median(hv[cm]))
            if h_model < min_h:
                continue
            cnts, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
            solidity = cm.sum() / max(cv2.contourArea(hull), 1.0)
            g = float(np.mean(green[y0:y0 + h, x0:x0 + w][cm]))
            reason = None
            if solidity < MIN_SOLIDITY:
                reason = f"not compact (solidity {solidity:.2f})"
            elif g > MAX_GREEN:
                reason = f"vegetated (greenness {g:.3f})"
            mask = np.zeros(ndsm.shape, bool)
            mask[y0:y0 + h, x0:x0 + w] = cm
            out.append({"id": len(out), "_mask": mask, "bbox": [int(x0), int(y0), int(w), int(h)],
                        "area_m2": round(float(area), 1), "h_model": round(h_model, 2),
                        "solidity": round(float(solidity), 2), "greenness": round(g, 3),
                        "rejected": reason})
    return out


def _edge_points(mask: np.ndarray, direction: np.ndarray, step: float = 2.0,
                 max_pts: int = 120) -> np.ndarray:
    """Mask pixels whose neighbour `step` px along `direction` is outside."""
    ys, xs = np.nonzero(mask)
    qx = np.round(xs + direction[0] * step).astype(int)
    qy = np.round(ys + direction[1] * step).astype(int)
    H, W = mask.shape
    ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
    out = np.zeros(ys.shape, bool)
    out[ok] = ~mask[qy[ok], qx[ok]]
    pts = np.c_[xs[out], ys[out]].astype(np.float32)
    if len(pts) > max_pts:
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    return pts


# --------------------------------------------------------------------------
# shadow height
# --------------------------------------------------------------------------

def measure_shadow(cand: dict, shadow: np.ndarray, ndsm: np.ndarray, geo: dict,
                   gsd_m: float, h_max_m: float = 150.0) -> dict:
    """Per-ray shadow runs from the candidate's anti-sun roof edge.

    Each ray starts 1.5 m inside the roof mask, walks along s, and must meet
    shadow within 3 m of the mask edge (the model's mask edge is soft). The
    run continues with a 2 m gap tolerance (cars, kerbs) until it ends. D is
    from the first shadow pixel (the photographed roof edge) to the tip.

    A tip that lands on a raised surface (a lower roof, a tree crown) is at
    that surface's height h2, not at ground. With the same lean algebra as
    the module docstring, D = (h - h2) * shadow_factor, so h = D/f + h2; h2 is
    read from the model nDSM at the tip (the model is not compressed at the
    low heights typical of h2). A ray whose shadow ends against something
    RISING above the tip (a taller wall that stops the shadow), or that leaves
    the image, is truncated and discarded, not counted short.
    """
    s = geo["s"]
    f = geo["shadow_factor"]
    mask = cand["_mask"]
    pts = _edge_points(mask, s)
    H, W = shadow.shape
    back = 1.5 / gsd_m
    start_tol = 3.0 / gsd_m
    gap_tol = max(2, int(round(2.0 / gsd_m)))
    max_len = h_max_m * geo["k_shadow"] / gsd_m
    look = int(round(2.0 / gsd_m))
    runs, h_rays, rays, n_noshadow, n_trunc = [], [], [], 0, 0
    for x0, y0 in pts:
        start = end = None
        gap = 0
        t = -back
        valid = True
        while t < max_len + start_tol:
            x = int(round(x0 + s[0] * t)); y = int(round(y0 + s[1] * t))
            if not (0 <= x < W and 0 <= y < H):
                valid = False
                break
            if start is None:
                if shadow[y, x] and not mask[y, x]:
                    start = end = t
                elif t > start_tol:
                    break
            else:
                if shadow[y, x]:
                    end = t
                    gap = 0
                else:
                    gap += 1
                    if gap > gap_tol:
                        break
            t += 1.0
        if not valid:
            n_trunc += 1
            continue
        if start is None:
            n_noshadow += 1
            continue
        xe, ye = x0 + s[0] * end, y0 + s[1] * end
        xi, yi = int(round(xe)), int(round(ye))
        h2 = float(np.median(ndsm[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]))
        # what lies just beyond the tip
        xa = int(round(xe + s[0] * look)); ya = int(round(ye + s[1] * look))
        if not (0 <= xa < W and 0 <= ya < H):
            n_trunc += 1
            continue
        if ndsm[ya, xa] > h2 + 3.0 or mask[ya, xa]:
            n_trunc += 1                         # shadow stopped by a rise
            continue
        L = end - start + 1.0
        runs.append(L)
        h_rays.append(L * gsd_m / f + max(h2, 0.0))
        rays.append((float(x0 + s[0] * start), float(y0 + s[1] * start), float(xe), float(ye)))
    res = {"h_shadow": None, "sigma_shadow": None, "conf_shadow": 0.0,
           "n_rays": len(runs), "n_rays_noshadow": n_noshadow, "n_rays_truncated": n_trunc,
           "_rays": rays}
    if len(runs) < 6:
        res["shadow_note"] = f"only {len(runs)} usable rays"
        return res
    r = np.array(h_rays)
    # Rays grazing the silhouette are short; interior rays carry the full
    # length. The 65th percentile sits on the interior plateau.
    h = float(np.percentile(r, 65))
    q25, q75 = np.percentile(r, [25, 75])
    rel_iqr = float((q75 - q25) / max(h, 1.0))
    frac = len(runs) / max(1, len(runs) + n_noshadow)
    mad = float(np.median(np.abs(r - np.median(r))))
    sig_stat = 1.4826 * mad / math.sqrt(len(runs)) * 3.0
    sigma = math.sqrt(sig_stat ** 2 + (0.10 * h) ** 2 + (2 * gsd_m / f) ** 2)
    conf = float(np.clip(frac, 0, 1) * np.clip(1.0 - rel_iqr, 0, 1) * min(1.0, len(runs) / 15.0))
    res.update({"h_shadow": round(h, 2), "sigma_shadow": round(sigma / max(conf, 0.2), 2),
                "conf_shadow": round(conf, 2),
                "shadow_rel_iqr": round(rel_iqr, 2), "shadow_ray_frac": round(frac, 2)})
    return res


# --------------------------------------------------------------------------
# relief displacement height
# --------------------------------------------------------------------------

def _local_maxima(p: np.ndarray) -> np.ndarray:
    return np.nonzero((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]))[0] + 1


def measure_displacement(cand: dict, gray: np.ndarray, geo: dict, gsd_m: float,
                         h_max_m: float = 150.0) -> dict:
    """Roof edge -> base edge distance on the satellite-facing side.

    The satellite-facing boundary of the roof mask is sampled; for each point
    the image derivative along the direction toward the satellite is
    profiled and the profiles are averaged (the lean is one translation for
    the whole building, so the base edge adds up while facade texture and
    clutter average out). The roof edge is the strongest local maximum
    within +-3 m of the mask edge; the base edge the strongest local maximum
    at least 2 px beyond it.

    Confidence needs all of: the base peak stands > 4 robust sigma above the
    profile's median, is >= 35 % of the roof-edge peak, and beats the next
    candidate peak by 20 %. At the most-nadir DFC view (0.38 px per metre of
    height) this is rarely met below ~20 m, which is the honest answer.
    """
    res = {"h_disp": None, "sigma_disp": None, "conf_disp": 0.0}
    if geo["k_lean"] <= 1e-3:
        res["disp_note"] = "no satellite geometry"
        return res
    u = -geo["lean"]                                   # toward the satellite
    pts = _edge_points(cand["_mask"], u)
    if len(pts) < 10:
        res["disp_note"] = "too few satellite-side edge points"
        return res
    px_per_m = geo["k_lean"] / gsd_m
    t0 = -int(round(3.0 / gsd_m))
    tmax = int(h_max_m * px_per_m) - t0 + 3
    ts = np.arange(t0, tmax + 1, dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gd = np.abs(gx * u[0] + gy * u[1])
    H, W = gray.shape
    X = pts[:, 0:1] + u[0] * ts[None, :]
    Y = pts[:, 1:2] + u[1] * ts[None, :]
    keep = ((X >= 0) & (X < W - 1) & (Y >= 0) & (Y < H - 1)).all(axis=1)
    if keep.sum() < 10:
        res["disp_note"] = "profiles leave the image"
        return res
    prof = cv2.remap(gd, X[keep].astype(np.float32), Y[keep].astype(np.float32),
                     cv2.INTER_LINEAR).mean(axis=0)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    mx = _local_maxima(prof)
    near = mx[(ts[mx] >= t0) & (ts[mx] <= -t0)]
    if near.size == 0:
        res["disp_note"] = "no roof edge"
        return res
    i_roof = int(near[np.argmax(prof[near])])
    far = mx[ts[mx] >= ts[i_roof] + 2]
    if far.size == 0:
        res["disp_note"] = "no base edge"
        return res
    order = far[np.argsort(prof[far])[::-1]]
    i_base = int(order[0])
    med = float(np.median(prof))
    rsd = 1.4826 * float(np.median(np.abs(prof - med))) + 1e-6
    z = (prof[i_base] - med) / rsd
    rel_roof = (prof[i_base] - med) / max(prof[i_roof] - med, 1e-6)
    runner = (prof[order[1]] - med) / max(prof[i_base] - med, 1e-6) if order.size > 1 else 0.0
    d_px = float(ts[i_base] - ts[i_roof])
    h = d_px / px_per_m
    ok = z > 4.0 and rel_roof >= 0.35 and runner < 0.8
    conf = min(1.0, float(np.clip((z - 4.0) / 4.0, 0, 1) * np.clip(1.0 - runner, 0, 1) * 2)) if ok else 0.0
    sigma = math.sqrt(2.0) / px_per_m + 0.1 * h        # 1 px per edge + 10 %
    res.update({"h_disp": round(h, 2), "sigma_disp": round(sigma / max(conf, 0.2), 2),
                "conf_disp": round(conf, 2), "disp_px": d_px, "disp_z": round(float(z), 1),
                "disp_runner_up": round(float(runner), 2),
                "_disp_roof_t": float(ts[i_roof])})
    return res


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def model_sigma(h_model: float) -> float:
    """Model's per-building 1-sigma. Deliberately wide at the top end, where
    the model is known to compress (~1.7x), so confident physics dominates."""
    return 1.0 + 0.30 * h_model


def refine(image: np.ndarray, ndsm: np.ndarray, gsd_m: float, sun_el_deg: float,
           sun_az_deg: float, sat_el_deg: float = None, sat_az_deg: float = None,
           off_nadir_deg: float = None, min_conf: float = 0.45,
           max_rel_sigma: float = 0.25) -> tuple[np.ndarray, dict]:
    """Fused nDSM + per-building report. Returns a NEW array; `ndsm` is untouched."""
    geo = view_geometry(sun_el_deg, sun_az_deg, sat_el_deg, sat_az_deg, off_nadir_deg)
    cands = find_candidates(ndsm, image, gsd_m)
    sh = shadow_mask(image, gsd_m)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 0.8)
    out = ndsm.astype(np.float32).copy()
    for c in cands:
        c["fired"] = False
        if c["rejected"]:
            continue
        c.update(measure_shadow(c, sh, ndsm, geo, gsd_m))
        c.update(measure_displacement(c, gray, geo, gsd_m))
        ests = []
        for k in ("shadow", "disp"):
            h, sg, cf = c.get(f"h_{k}"), c.get(f"sigma_{k}"), c.get(f"conf_{k}", 0)
            if h is None or cf < min_conf or h <= 2.0 or sg / h > max_rel_sigma:
                continue
            # Plausibility window around the model: the known failure is a
            # ~1.7x compression, not a 5x one; a physics reading far outside
            # [0.5, 3] x model has locked onto something else (a neighbour's
            # shadow, a kerb edge) and is not used.
            if not (0.5 * c["h_model"] <= h <= 3.0 * c["h_model"]):
                c[f"{k}_note"] = "outside [0.5, 3] x model; not used"
                continue
            ests.append((h, sg, k))
        if len(ests) == 2 and abs(ests[0][0] - ests[1][0]) > 3 * math.hypot(ests[0][1], ests[1][1]):
            c["fuse_note"] = "shadow and relief disagree; neither used"
            ests = []
        if not ests:
            continue
        w = np.array([1 / e[1] ** 2 for e in ests])
        h_p = float(np.sum(w * [e[0] for e in ests]) / w.sum())
        s_p = float(1 / math.sqrt(w.sum()))
        c["h_physics"], c["sigma_physics"] = round(h_p, 2), round(s_p, 2)
        c["physics_from"] = "+".join(e[2] for e in ests)
        hm = c["h_model"]
        if abs(h_p - hm) / hm <= DISAGREE_FRAC:
            c["fuse_note"] = "physics agrees with model within 25%"
            continue
        sm = model_sigma(hm)
        wm, wp = 1 / sm ** 2, 1 / s_p ** 2
        h_f = (wm * hm + wp * h_p) / (wm + wp)
        ratio = float(np.clip(h_f / hm, 0.5, 3.0))
        region = cv2.dilate(c["_mask"].astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        wt = np.clip((ndsm - 0.3 * hm) / (0.4 * hm), 0, 1) * region
        out = out * (1 + (ratio - 1) * wt)
        c["fired"], c["h_fused"], c["ratio"] = True, round(h_f, 2), round(ratio, 3)
    rep = {"geometry": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in geo.items()},
           "candidates": cands,
           "n_candidates": sum(1 for c in cands if not c["rejected"]),
           "n_fired": sum(1 for c in cands if c["fired"])}
    return out.astype(np.float32), rep


def fmt_candidate(c: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.1f}"
    if c["rejected"]:
        return f"#{c['id']} model {c['h_model']:.1f} m REJECTED {c['rejected']}"
    return (f"#{c['id']} {c['area_m2']:.0f} m2 model {c['h_model']:.1f} | shadow {f(c.get('h_shadow'))}"
            f" (c{c.get('conf_shadow', 0):.2f}, {c.get('n_rays', 0)} rays) | relief {f(c.get('h_disp'))}"
            f" (c{c.get('conf_disp', 0):.2f}) | {'FIRED -> ' + f(c.get('h_fused')) if c['fired'] else c.get('fuse_note', 'no confident physics')}")


# --------------------------------------------------------------------------
# overlay (for eyeballing geometry / sign conventions)
# --------------------------------------------------------------------------

def draw_overlay(image: np.ndarray, rep: dict, path: str) -> None:
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    geo = rep["geometry"]
    H, W = image.shape[:2]
    # compass arrows: yellow = shadow direction, magenta = roof lean direction
    c0 = np.array([60, 60])
    for vec, col, lab in ((geo["s"], (0, 255, 255), "shadow"), (geo["lean"], (255, 0, 255), "lean")):
        v = np.array(vec)
        if np.linalg.norm(v) > 0:
            p1 = tuple(int(a) for a in c0 + 45 * v)
            cv2.arrowedLine(vis, tuple(int(a) for a in c0), p1, col, 2, tipLength=0.3)
            cv2.putText(vis, lab, (p1[0] + 3, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.putText(vis, "N^", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for c in rep["candidates"]:
        m = c["_mask"].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = (80, 80, 80) if c["rejected"] else ((0, 0, 255) if c["fired"] else (0, 255, 0))
        cv2.drawContours(vis, cnts, -1, col, 1)
        for (x0, y0, x1, y1) in c.get("_rays", [])[::2]:
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
        if c.get("h_disp") is not None and c.get("conf_disp", 0) > 0:
            # base outline = roof outline moved back toward the satellite
            t = c["disp_px"] + c["_disp_roof_t"]
            sh = -np.array(geo["lean"]) * t
            moved = [(cc + sh[None, None, :]).astype(np.int32) for cc in cnts]
            cv2.drawContours(vis, moved, -1, (255, 0, 255), 1)
        x, y = c["bbox"][:2]
        f = lambda v: "-" if v is None else f"{v:.0f}"
        txt = f"m{c['h_model']:.0f} s{f(c.get('h_shadow'))} r{f(c.get('h_disp'))}"
        if c["fired"]:
            txt += f">{c['h_fused']:.0f}"
        cv2.putText(vis, txt, (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    cv2.imwrite(path, vis)
