"""Recalibrate a DSM GeoTIFF from user Ground Control Points (absolute elevations).

The SIH problem statement allows "a lower-resolution DEM ... or a limited set of
Ground Control Points" to map scale-agnostic depth to absolute metric elevation.
calibration/gcp.py solves a height *ratio* from building heights; this module
works on the DSM raster itself, with GCPs that are ABSOLUTE elevations (e.g. a
surveyed roof at 612.4 m above the geoid), and writes a new GeoTIFF.

THE CORRECTNESS POINT

A GCP's z is an absolute elevation, but the pipeline's height *error* lives in
the above-ground part: terrain comes from a DEM anchor (datum shift, maybe a
tilt), buildings from a depth model (scale error). Fitting ``z = a*DSM + b``
directly is wrong: with DSM values around 600 m, a 1.2x factor would add 120 m
to the ground. So the two candidate corrections are:

  offset   DSM' = DSM + c                       c = median(z_i - DSM(x_i))
  affine   DSM' = DSM + (s-1)*nDSM + c          nDSM = max(DSM - ground, 0)

i.e. affine scales only the above-ground part and shifts the datum by c.
It is fitted as a (Huber-robust) line through r_i = z_i - DSM(x_i) against
nDSM(x_i): r = (s-1)*nDSM + c.

SAMPLING

A GCP is placed by a person on a roof or a road, not on an exact pixel, and the
DSM is noisy per pixel. Each GCP is sampled as the median over the connected
region around it whose DSM is within max(30% of the local above-ground height,
1 m) of the seed value (capped to the 20k pixels nearest the seed): a whole flat
roof, or a patch of road.

MODEL SELECTION

Leave-one-out cross-validation scores none / offset / affine: each point is
predicted by a correction fitted without it. "auto" picks the lowest LOO RMSE
and never one worse than offset. Affine is only eligible with enough leverage
(>=5 points, nDSM span >=5 m, >=2 points in the upper 40% of the span) and a
plausible scale (0.8 <= s <= 3); otherwise it falls back to offset and says why.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

AFFINE_MIN_POINTS = 5
AFFINE_MIN_SPAN_M = 5.0
AFFINE_UPPER_FRAC = 0.4       # "upper 40% of the span"
AFFINE_MIN_UPPER = 2
SCALE_MIN, SCALE_MAX = 0.8, 3.0
REGION_CAP_PX = 20000
REGION_TOL_FRAC = 0.30
REGION_TOL_MIN_M = 1.0
SPREAD_WARN_M = 3.0
METHODS = ("none", "offset", "affine")


# ---------------------------------------------------------------------------
# Ground layer
# ---------------------------------------------------------------------------

def estimate_ground(dsm: np.ndarray, valid: np.ndarray, gsd_m: float,
                    block_m: float = 20.0, window_blocks: int = 5) -> np.ndarray:
    """Bare-ground estimate from the DSM alone (used when the scene has no
    ground/DTM layer): per-block 10th percentile, then a grey-scale opening
    (min then max filter over ``window_blocks`` blocks, ~100 m -- wider than
    typical buildings) to remove structures while following terrain relief,
    then a light smoothing and a bilinear upsample back to full resolution.
    """
    from scipy import ndimage as ndi

    h, w = dsm.shape
    b = int(max(4, round(block_m / max(gsd_m, 1e-6))))
    b = min(b, max(4, min(h, w) // 4))
    nby, nbx = math.ceil(h / b), math.ceil(w / b)
    pad = np.full((nby * b, nbx * b), np.nan, np.float32)
    pad[:h, :w] = np.where(valid, dsm, np.nan)
    blocks = pad.reshape(nby, b, nbx, b).transpose(0, 2, 1, 3).reshape(nby, nbx, b * b)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = np.nanpercentile(blocks, 10, axis=2)
    if np.all(np.isnan(p)):
        return np.zeros_like(dsm, np.float32)
    # Fill empty blocks with the nearest valid block so filters stay defined.
    nan = np.isnan(p)
    if nan.any():
        idx = ndi.distance_transform_edt(nan, return_distances=False, return_indices=True)
        p = p[tuple(idx)]
    k = max(1, min(window_blocks, nby, nbx))
    g = ndi.maximum_filter(ndi.minimum_filter(p, size=k, mode="nearest"), size=k, mode="nearest")
    g = ndi.uniform_filter(g, size=min(3, nby, nbx), mode="nearest")
    # Bilinear upsample: block i's value sits at pixel centre (i + 0.5) * b - 0.5.
    rr = (np.arange(h, dtype=np.float32) + 0.5) / b - 0.5
    cc = (np.arange(w, dtype=np.float32) + 0.5) / b - 0.5
    rows = np.clip(rr, 0, nby - 1)
    cols = np.clip(cc, 0, nbx - 1)
    r0 = np.floor(rows).astype(int); r1 = np.minimum(r0 + 1, nby - 1); fr = (rows - r0)[:, None]
    c0 = np.floor(cols).astype(int); c1 = np.minimum(c0 + 1, nbx - 1); fc = (cols - c0)[None, :]
    top = g[r0][:, c0] * (1 - fc) + g[r0][:, c1] * fc
    bot = g[r1][:, c0] * (1 - fc) + g[r1][:, c1] * fc
    ground = (top * (1 - fr) + bot * fr).astype(np.float32)
    return ground


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_region(dsm: np.ndarray, ground: np.ndarray, valid: np.ndarray,
                  col: float, row: float, cap: int = REGION_CAP_PX,
                  radius_px: int = 150) -> Optional[Dict]:
    """Median DSM / ground / nDSM over the connected region around (col,row).

    Region: pixels 8-connected to the seed whose DSM is within
    max(30% of the seed's above-ground height, 1 m) of the seed value,
    limited to a (2*radius+1)^2 window and the ``cap`` pixels nearest the seed.
    Returns None if the seed is off-raster or on nodata.
    """
    from scipy import ndimage as ndi

    h, w = dsm.shape
    ci, ri = int(math.floor(col)), int(math.floor(row))
    if not (0 <= ri < h and 0 <= ci < w):
        return None
    y0, y1 = max(0, ri - 1), min(h, ri + 2)
    x0, x1 = max(0, ci - 1), min(w, ci + 2)
    nb = dsm[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    if not valid[ri, ci] and nb.size == 0:
        return None
    v0 = float(np.median(nb)) if nb.size else float(dsm[ri, ci])
    g0 = float(ground[ri, ci])
    tol = max(REGION_TOL_FRAC * max(v0 - g0, 0.0), REGION_TOL_MIN_M)

    wy0, wy1 = max(0, ri - radius_px), min(h, ri + radius_px + 1)
    wx0, wx1 = max(0, ci - radius_px), min(w, ci + radius_px + 1)
    win = dsm[wy0:wy1, wx0:wx1]
    m = valid[wy0:wy1, wx0:wx1] & (np.abs(win - v0) <= tol)
    sr, sc = ri - wy0, ci - wx0
    if not m[sr, sc]:
        # Seed pixel itself is an outlier vs its 3x3 median: reseed on the
        # closest-valued neighbour that is inside the band.
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = sr + dy, sc + dx
                if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx]:
                    d = abs(float(win[yy, xx]) - v0)
                    if best is None or d < best[0]:
                        best = (d, yy, xx)
        if best is None:
            return None
        sr, sc = best[1], best[2]
    lab, _ = ndi.label(m, structure=np.ones((3, 3), bool))
    comp = lab == lab[sr, sc]
    ys, xs = np.nonzero(comp)
    if ys.size > cap:
        d2 = (ys - sr) ** 2 + (xs - sc) ** 2
        keep = np.argpartition(d2, cap - 1)[:cap]
        ys, xs = ys[keep], xs[keep]
    dv = win[ys, xs].astype(np.float64)
    gv = ground[wy0:wy1, wx0:wx1][ys, xs].astype(np.float64)
    nd = np.maximum(dv - gv, 0.0)
    return {"dsm": float(np.median(dv)), "ground": float(np.median(gv)),
            "ndsm": float(np.median(nd)), "n_px": int(ys.size),
            "pixel_value": float(dsm[ri, ci]) if valid[ri, ci] else None,
            "band_m": round(tol, 2)}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _huber_line(x: np.ndarray, y: np.ndarray, iters: int = 20) -> tuple:
    """Robust (Huber IRLS) fit y = k*x + c."""
    A = np.column_stack([x, np.ones_like(x)])
    wts = np.ones_like(y)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(iters):
        res = y - A @ coef
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
        scale = max(mad, 0.25)
        k = 1.345 * scale
        a = np.abs(res)
        wts = np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))
        sw = np.sqrt(wts)
        new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(new, coef, atol=1e-6):
            coef = new
            break
        coef = new
    return float(coef[0]), float(coef[1])


def affine_guard(ndsm: np.ndarray, min_points: int = AFFINE_MIN_POINTS) -> Optional[str]:
    """None if the points carry enough leverage to fit a scale, else why not."""
    n = ndsm.size
    if n < min_points:
        return f"affine needs >= {min_points} points, have {n}"
    span = float(ndsm.max() - ndsm.min())
    if span < AFFINE_MIN_SPAN_M:
        return (f"points span only {span:.1f} m of above-ground height "
                f"(need >= {AFFINE_MIN_SPAN_M:.0f} m) -- scale is not observable")
    thr = float(ndsm.min()) + (1 - AFFINE_UPPER_FRAC) * span
    upper = int((ndsm >= thr).sum())
    if upper < AFFINE_MIN_UPPER:
        return (f"only {upper} point(s) in the upper {AFFINE_UPPER_FRAC*100:.0f}% of the "
                f"height span (need >= {AFFINE_MIN_UPPER}) -- one tall point would set the scale alone")
    return None


def fit_offset(r: np.ndarray) -> Dict:
    return {"offset_m": float(np.median(r)), "scale": 1.0}


def fit_affine(r: np.ndarray, ndsm: np.ndarray, min_points: int = AFFINE_MIN_POINTS) -> Dict:
    """Returns {"scale","offset_m"} or {"refused": reason}."""
    why = affine_guard(ndsm, min_points)
    if why:
        return {"refused": why}
    k, c = _huber_line(ndsm, r)
    s = 1.0 + k
    if not (SCALE_MIN <= s <= SCALE_MAX):
        return {"refused": f"fitted above-ground scale {s:.3f} is outside the plausible "
                           f"[{SCALE_MIN}, {SCALE_MAX}] range -- points are inconsistent "
                           f"or the DSM's relief is wrong in shape, not just scale"}
    return {"scale": s, "offset_m": c}


def _predict_r(params: Dict, ndsm: np.ndarray) -> np.ndarray:
    """Predicted residual z - DSM at points with above-ground height ndsm."""
    return (params["scale"] - 1.0) * ndsm + params["offset_m"]


def loo(r: np.ndarray, ndsm: np.ndarray) -> Dict:
    """Leave-one-out residuals for each method. Affine folds that fail the
    guards fall back to offset (the same thing the full procedure would do)."""
    n = r.size
    out: Dict = {"none": r.copy()}
    if n < 2:
        return out
    off, aff = np.empty(n), np.empty(n)
    fallbacks = 0
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        o = fit_offset(r[m])
        off[i] = r[i] - _predict_r(o, ndsm[i:i + 1])[0]
        a = fit_affine(r[m], ndsm[m], min_points=AFFINE_MIN_POINTS - 1)
        if "refused" in a:
            fallbacks += 1
            a = o
        aff[i] = r[i] - _predict_r(a, ndsm[i:i + 1])[0]
    out["offset"] = off
    out["affine"] = aff
    out["affine_fold_fallbacks"] = fallbacks
    return out


def _rmse(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def solve(z: np.ndarray, dsm_s: np.ndarray, ndsm_s: np.ndarray,
          mode: str = "auto", xy: Optional[np.ndarray] = None) -> Dict:
    """Pure fitting step on sampled values. z: absolute GCP elevations,
    dsm_s: DSM sampled at the points, ndsm_s: above-ground height there."""
    z = np.asarray(z, float); dsm_s = np.asarray(dsm_s, float)
    ndsm_s = np.asarray(ndsm_s, float)
    if mode not in ("auto", "offset", "affine"):
        raise ValueError("mode must be auto, offset or affine")
    n = z.size
    if n < 1:
        raise ValueError("no usable GCPs")
    r = z - dsm_s
    warnings: List[str] = []
    fits = {"none": {"scale": 1.0, "offset_m": 0.0}, "offset": fit_offset(r)}
    aff = fit_affine(r, ndsm_s)
    affine_ok = "refused" not in aff
    if affine_ok:
        fits["affine"] = aff
    lo = loo(r, ndsm_s)
    loo_rmse = {k: (round(_rmse(lo[k]), 3) if k in lo else None) for k in METHODS}
    if not affine_ok:
        loo_rmse["affine"] = None
    if n < 2:
        warnings.append("only 1 GCP: offset applied without any cross-validation -- "
                        "add points to check it")

    if mode == "offset":
        chosen = "offset"
    elif mode == "affine":
        if affine_ok:
            chosen = "affine"
        else:
            chosen = "offset"
            warnings.append(f"affine refused, fell back to offset: {aff['refused']}")
    else:
        chosen = "offset"
        if n >= 2:
            best = loo_rmse["offset"]
            if affine_ok and loo_rmse["affine"] is not None and loo_rmse["affine"] < best:
                chosen, best = "affine", loo_rmse["affine"]
            if loo_rmse["none"] is not None and loo_rmse["none"] < best:
                chosen = "none"
                warnings.append("the uncorrected DSM already predicts the GCPs better than "
                                "any fitted correction (LOO) -- file written unchanged")
        if not affine_ok:
            warnings.append(f"affine not considered: {aff['refused']}")

    p = fits[chosen]
    fitted_r = r - _predict_r(p, ndsm_s)
    # Diagnostics on the offset-only residuals: what a single shift cannot explain.
    off_res = r - fits["offset"]["offset_m"]
    spread = float(np.std(off_res)) if n >= 2 else 0.0
    if n >= 3 and spread > SPREAD_WARN_M:
        msg = (f"offset residual spread {spread:.1f} m > {SPREAD_WARN_M:.0f} m: the points "
               f"do not agree on one datum shift")
        hints = []
        if n >= 4 and xy is not None and np.ptp(xy[:, 0]) + np.ptp(xy[:, 1]) > 0:
            A = np.column_stack([np.ones(n), xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()])
            coef = np.linalg.lstsq(A, off_res, rcond=None)[0]
            rem = off_res - A @ coef
            r2 = 1 - float(np.sum(rem ** 2)) / max(float(np.sum((off_res - off_res.mean()) ** 2)), 1e-12)
            if r2 > 0.5:
                grad = math.hypot(coef[1], coef[2]) * 1000.0
                hints.append(f"residuals trend with position (R^2 {r2:.2f}, {grad:.1f} m/km) -- "
                             f"terrain slope / datum tilt the DEM anchor did not capture")
        if n >= 4 and np.std(ndsm_s) > 0:
            cr = float(np.corrcoef(off_res, ndsm_s)[0, 1])
            if abs(cr) > 0.7:
                hints.append(f"residuals correlate with above-ground height (r={cr:.2f}) -- "
                             f"a height-scale error; affine mode addresses this")
        if not hints:
            hints.append("likely terrain slope, a height-scale error, or a mis-entered point")
        warnings.append(msg + "; " + "; ".join(hints))
    if n >= 4 and n >= 2 and chosen != "none":
        lr = lo.get(chosen, lo["none"])
        scale = max(1.4826 * float(np.median(np.abs(lr - np.median(lr)))), 0.5)
        bad = [int(i) for i in np.nonzero(np.abs(lr) > 3 * scale)[0]]
        if bad:
            warnings.append(f"suspect point(s) {bad}: leave-one-out error > 3x the robust "
                            f"spread -- check their coordinates/elevations")

    return {
        "method": chosen, "chosen": chosen, "mode": mode, "n": int(n),
        "offset_m": round(p["offset_m"], 4), "scale": round(p["scale"], 5),
        "fits": {k: {"offset_m": round(v["offset_m"], 4), "scale": round(v["scale"], 5)}
                 for k, v in fits.items()},
        "affine_refused": None if affine_ok else aff["refused"],
        "residuals_before_m": [round(float(v), 3) for v in r],
        "residuals_m": [round(float(v), 3) for v in fitted_r],
        "rmse_in_sample_m": round(_rmse(fitted_r), 3),
        "loo_rmse": loo_rmse,
        "loo_residuals_m": {k: [round(float(v), 3) for v in lo[k]] for k in METHODS if k in lo},
        "affine_loo_fold_fallbacks": lo.get("affine_fold_fallbacks"),
        "offset_residual_spread_m": round(spread, 3),
        "warnings": warnings,
    }


def apply(dsm: np.ndarray, ground: np.ndarray, valid: np.ndarray,
          scale: float, offset_m: float) -> np.ndarray:
    nd = np.maximum(dsm - ground, 0.0)
    out = dsm + (scale - 1.0) * nd + offset_m
    return np.where(valid, out, dsm).astype(np.float32)


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------

def find_ground_layer(dsm_path: str) -> Optional[str]:
    base = dsm_path[:-4]
    for stem in (base.rsplit("_DSM", 1)[0] if "_DSM" in base else base,
                 base.rsplit("_rDSM", 1)[0] if "_rDSM" in base else base):
        for suf in ("_DTM.tif", "_ground.tif", "_dtm.tif"):
            p = stem + suf
            if os.path.isfile(p):
                return p
    return None


def gcp_output_path(dsm_path: str) -> str:
    return dsm_path[:-4] + "_gcp.tif"


def _gsd_m(ds) -> float:
    a = abs(ds.transform.a)
    if ds.crs is not None and ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2
        return a * 111320.0 * max(math.cos(math.radians(lat)), 0.01)
    if ds.transform.is_identity:
        return 0.5
    return a


def points_to_pixels(points: Sequence[Dict], ds) -> tuple:
    """Returns (cols, rows, z, xy_map, dropped). Points may give col/row
    (pixels) or x/y in ``crs`` (default: the DSM's own CRS)."""
    from pyproj import CRS, Transformer
    cols, rows, zs, xys, dropped, ids = [], [], [], [], [], []
    inv = ~ds.transform
    tcache: Dict = {}
    hagl = []
    for i, p in enumerate(points):
        # z is an absolute elevation. A point may instead give h_agl, a known
        # height above ground (e.g. "this building is 45 m tall"); it is
        # turned into an elevation once the ground under it has been sampled.
        h = float("nan")
        try:
            if p.get("z") is None and p.get("h_agl") is not None:
                h = float(p["h_agl"])
                z = float("nan")
                if not math.isfinite(h) or h < 0:
                    raise ValueError
            else:
                z = float(p["z"])
                if not math.isfinite(z):
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            dropped.append({"index": i, "reason": "missing/invalid z (or h_agl)"}); continue
        if p.get("col") is not None and p.get("row") is not None:
            c, r = float(p["col"]), float(p["row"])
            x, y = ds.transform * (c, r)
        else:
            try:
                x, y = float(p["x"]), float(p["y"])
            except (KeyError, TypeError, ValueError):
                dropped.append({"index": i, "reason": "need x/y or col/row"}); continue
            crs = p.get("crs")
            if crs:
                if ds.crs is None:
                    dropped.append({"index": i, "reason": "DSM has no CRS; give col/row"}); continue
                if crs not in tcache:
                    tcache[crs] = Transformer.from_crs(CRS.from_user_input(crs),
                                                       CRS.from_user_input(ds.crs.to_wkt()),
                                                       always_xy=True)
                x, y = tcache[crs].transform(x, y)
            c, r = inv * (x, y)
        cols.append(c); rows.append(r); zs.append(z); xys.append((x, y)); ids.append(i)
        hagl.append(h)
    return (np.array(cols), np.array(rows), np.array(zs), np.array(xys).reshape(-1, 2),
            dropped, ids, np.array(hagl))


def calibrate_file(dsm_path: str, points: Sequence[Dict], mode: str = "auto",
                   out_path: Optional[str] = None, ground_path: Optional[str] = None) -> Dict:
    """Sample, fit, and write ``<dsm>_gcp.tif`` (+ ``_gcp.json``). Returns the
    report. Raises ValueError with a user-facing message on bad input."""
    import rasterio

    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1).astype(np.float32)
        nd = ds.nodata
        profile = ds.profile.copy()
        tags = ds.tags()
        gsd = _gsd_m(ds)
        cols, rows, z, xy, dropped, ids, hagl = points_to_pixels(points, ds)
    valid = np.isfinite(dsm)
    if nd is not None:
        valid &= dsm != nd

    ground_src = "estimated from the DSM (block 10th percentile + ~100 m grey opening)"
    ground = None
    gp = ground_path or find_ground_layer(dsm_path)
    if gp:
        with rasterio.open(gp) as g:
            if g.shape == dsm.shape:
                ground = g.read(1).astype(np.float32)
                ground_src = f"scene ground layer {os.path.basename(gp)}"
    if ground is None:
        ground = estimate_ground(dsm, valid, gsd)

    samples, keep = [], []
    for k in range(z.size):
        s = sample_region(dsm, ground, valid, cols[k], rows[k])
        if s is None:
            dropped.append({"index": ids[k], "reason": "outside the DSM or on nodata"})
            continue
        if math.isfinite(hagl[k]):
            z[k] = s["ground"] + hagl[k]
        samples.append(s); keep.append(k)
    if not keep:
        raise ValueError("none of the GCPs fall on valid DSM pixels" +
                         (f" ({dropped[0]['reason']})" if dropped else ""))
    keep = np.array(keep)
    rep = solve(z[keep], np.array([s["dsm"] for s in samples]),
                np.array([s["ndsm"] for s in samples]), mode=mode, xy=xy[keep])
    if dropped:
        rep["warnings"].insert(0, f"{len(dropped)} point(s) ignored: " +
                               "; ".join(f"#{d['index']} {d['reason']}" for d in dropped))
    product = tags.get("PRODUCT") or ("rDSM" if dsm_path.endswith("_rDSM.tif") else "DSM")
    if product == "rDSM":
        rep["warnings"].append("source is a relative rDSM: absolute heights now rest "
                               "entirely on these GCPs")

    new = apply(dsm, ground, valid, rep["scale"], rep["offset_m"])
    if nd is not None:
        new = np.where(valid, new, nd).astype(np.float32)
    out_path = out_path or gcp_output_path(dsm_path)
    profile.update(dtype="float32", count=1, compress="deflate")
    old_datum = tags.get("VERTICAL_DATUM", "unknown")
    new_tags = dict(tags)
    new_tags.update({
        "VERTICAL_DATUM": f"user GCPs ({rep['n']} points, {rep['method']}); "
                          f"previously: {old_datum}",
        "GCP_METHOD": rep["method"], "GCP_MODE": mode, "GCP_N": str(rep["n"]),
        "GCP_SCALE_ABOVE_GROUND": str(rep["scale"]), "GCP_OFFSET_M": str(rep["offset_m"]),
        "GCP_RESIDUALS_M": json.dumps(rep["residuals_m"]),
        "GCP_RMSE_IN_SAMPLE_M": str(rep["rmse_in_sample_m"]),
        "GCP_LOO_RMSE_M": json.dumps(rep["loo_rmse"]),
        "GCP_GROUND_LAYER": ground_src,
        "GCP_FORMULA": "DSM' = DSM + (scale-1)*max(DSM-ground,0) + offset",
        "GCP_SOURCE_DSM": os.path.basename(dsm_path),
    })
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(new, 1)
        dst.update_tags(**{k: str(v) for k, v in new_tags.items()})

    rep.update({
        "points": [dict(index=ids[k], col=round(float(cols[k]), 2), row=round(float(rows[k]), 2),
                        z=float(z[k]), dsm_sampled=round(s["dsm"], 3),
                        ground_sampled=round(s["ground"], 3), ndsm_sampled=round(s["ndsm"], 3),
                        region_px=s["n_px"], band_m=s["band_m"],
                        residual_m=rep["residuals_m"][j])
                   for j, (k, s) in enumerate(zip(keep, samples))],
        "dropped": dropped, "ground_layer": ground_src, "gsd_m": round(gsd, 4),
        "source_dsm": os.path.basename(dsm_path), "output_file": os.path.basename(out_path),
        "vertical_datum": new_tags["VERTICAL_DATUM"], "product": product,
    })
    with open(out_path[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return rep
