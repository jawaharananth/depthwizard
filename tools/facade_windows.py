"""
Window-grid detection from the REAL facade pixels of an oblique satellite texture.

    venv\\Scripts\\python.exe tools\\facade_windows.py --scene scenes/hitec_2x2 [--viz DIR]

What it does
------------
1. Measures the image's relief displacement (the "lean": where a point h metres
   above its base appears in the image) directly from the texture, instead of
   trusting scene.json lean_m_per_m:
     * direction = the dominant orientation of long straight line segments
       (building vertical corners and mullions all project along it);
     * magnitude = upper quartile of (corner-segment length / DSM roof height)
       for segments that start on a >25 m roof and end on the ground.
   scene.json's value is reported alongside for comparison.
2. For every building footprint edge that faces the sensor, samples the
   parallelogram of texture between the roof edge and where the base appears
   (lean * height away) into a rectified strip: s along the wall (ground
   metres), r down the wall (image texels along the lean).
3. Floor pitch: periodicity of the row profile (mean over s) -> period along
   the lean in texels / lean magnitude = metres of height per floor.
   Window pitch: periodicity of the column profile (mean over r) -> metres
   along the wall.  Periodicity statistic = best explained-variance fraction
   of a sinusoid(+2nd harmonic) over the allowed period range.
4. Confidence = empirical CDF of that statistic against a NULL bank: the same
   strip geometry translated onto real non-facade texture in the same image
   (open ground / flat roof interiors, chosen with the DSM).  So a confidence
   of 0.95 means the pattern is stronger than 95 % of equally-sized patches of
   ground and roof in this very image -- no synthetic noise model.
5. Writes
     facade_windows.json   per building, geojson feature order
     facade_window_mask.png  RGB, SAME pixel grid and bounds as the texture
                             (scene.json texture_file), i.e. aligned with the
                             facade UVs the viewer samples (it samples facade
                             texels out of that texture by planar UV):
                               R = window-cell (glass) likelihood 0..255
                               G = 255 detected lattice / 96 analysed, none / 0 not facade
                               B = confidence * 255
World frame (same as viewer/index.html realCoordsAt and build_city_image.py
context placement): x = easting - affine[2], z = northing - affine[5]
(source-CRS units; divided by crs_units_per_m when scene.json has it, which
HITEC does not -> 1).  x east, z north (z <= 0 inside the tile).
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter1d, grey_opening

Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------------------------------- helpers
def load_scene(scene_dir):
    meta = json.load(open(os.path.join(scene_dir, "scene.json")))
    a = meta["affine_transform"]
    k = meta.get("crs_units_per_m") or 1.0
    return meta, a, k


def merc_scale(a, n_px):
    """Web-Mercator map units per ground metre at the tile centre."""
    y = a[5] + a[4] * n_px / 2
    lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
    return 1.0 / math.cos(math.radians(lat)), lat


def ground_from_dsm(dsm):
    small = cv2.resize(dsm, (1024, 1024), interpolation=cv2.INTER_AREA)
    g = grey_opening(small, size=(51, 51))           # ~150 m structuring element
    g = cv2.GaussianBlur(g, (0, 0), 8)
    return cv2.resize(g, dsm.shape[::-1], interpolation=cv2.INTER_LINEAR)


def measure_lean(gray_u8, ndsm, dsm_per_tex):
    """Lean in texture texels per metre of height, from the image itself."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray_u8)[0].reshape(-1, 4)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    L = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(-dy, dx)) % 180.0
    sel = L > 25
    # ignore the 0/90 deg pixel-grid / JPEG-block artefacts
    grid = (np.minimum(ang % 90, 90 - ang % 90) < 1.5)
    h, e = np.histogram(ang[sel & ~grid], bins=360, range=(0, 180), weights=L[sel & ~grid])
    h = np.convolve(h, np.ones(5) / 5, mode="same")
    peak = float(e[np.argmax(h)] + 0.25)
    # refine: length-weighted mean within +-2.5 deg of the peak
    w = sel & ~grid & (np.abs(ang - peak) < 2.5)
    peak = float(np.average(ang[w], weights=L[w]))
    a = math.radians(peak)
    d = np.array([math.cos(a), -math.sin(a)])       # image (col,row) unit, roof side
    # magnitude: corner segments from a tall roof down to the ground
    s = (np.abs(ang - peak) < 3) & (L > 15)
    l, Ls = lines[s], L[s]
    up = (l[:, 1] < l[:, 3])[:, None]
    top = np.where(up, l[:, :2], l[:, 2:])
    bot = np.where(up, l[:, 2:], l[:, :2])
    pr = (top + d * 4) / dsm_per_tex
    pb = (bot - d * 6) / dsm_per_tex
    Ht = map_coordinates(ndsm, [pr[:, 1], pr[:, 0]], order=1)
    Hb = map_coordinates(ndsm, [pb[:, 1], pb[:, 0]], order=1)
    k = (Ht > 25) & (Hb < 6)
    ratio = Ls[k] / Ht[k]
    # LSD breaks long edges and DSM roof heights are percentiles, so a full
    # corner edge sits in the upper part of the distribution: use the 75th pct.
    mag = float(np.percentile(ratio, 75))
    return {"direction_deg_from_east_ccw": round(peak, 2),
            "texels_per_m": round(mag, 3),
            "n_segments": int(k.sum()),
            "ratio_pct25_50_75_90": [round(float(x), 3) for x in np.percentile(ratio, [25, 50, 75, 90])],
            "unit": d}


def best_periodicity(y, dt, pmin, pmax):
    """Scan periods (in the same units as dt*index) and return (R2, period, fit, phase_profile)."""
    n = len(y)
    if n < 6:
        return 0.0, None, None
    t = np.arange(n) * dt
    var = float((y ** 2).sum()) + 1e-9
    Ps = pmin * 1.025 ** np.arange(int(np.log(pmax / pmin) / np.log(1.025)) + 1)
    Ps = Ps[Ps * 2.5 <= t[-1]]         # need >= ~2.5 periods in the profile
    if not len(Ps):
        return 0.0, None, None
    best = (0.0, None, None)
    for harm, sel in ((1, Ps / 2 < 2.4), (2, Ps / 2 >= 2.4)):   # 2nd harmonic only if resolvable
        P = Ps[sel]
        if not len(P):
            continue
        w = 2 * np.pi * t[None, :] / P[:, None]
        cols = [np.cos(w), np.sin(w)]
        if harm == 2:
            cols += [np.cos(2 * w), np.sin(2 * w)]
        X = np.stack(cols, 2)                                  # (nP, n, c)
        XtX = np.einsum("pnc,pnd->pcd", X, X) + 1e-6 * np.eye(X.shape[2])
        Xty = np.einsum("pnc,n->pc", X, y)
        coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        r2 = np.einsum("pc,pc->p", coef, Xty) / var            # explained variance fraction
        r2 = r2 - (X.shape[2] / n) * 0.5                       # small-sample penalty
        i = int(np.argmax(r2))
        if r2[i] > best[0]:
            best = (float(r2[i]), float(P[i]), None)
    return best


def detrend(y, sigma):
    return y - gaussian_filter1d(y, sigma, mode="nearest")


class StripSampler:
    def __init__(self, gray, d_unit, lean_tpm, step=0.5):
        self.g = gray
        self.d = d_unit
        self.v = lean_tpm
        self.step = step

    def geometry(self, p0, p1, H):
        e = p1 - p0
        L = float(np.hypot(*e))
        w = e / (L + 1e-9)
        R = self.v * H                       # facade extent along -d, texels
        s = np.arange(0.0, L, self.step)
        r = np.arange(-1.0, R + 1.0, self.step)
        return w, s, r, L, R

    def sample(self, p0, w, s, r):
        S, Rr = np.meshgrid(s, r)            # rows = r, cols = s
        X = p0[0] + S * w[0] - Rr * self.d[0]
        Y = p0[1] + S * w[1] - Rr * self.d[1]
        ok = (X > 1) & (Y > 1) & (X < self.g.shape[1] - 2) & (Y < self.g.shape[0] - 2)
        vals = map_coordinates(self.g, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
        return vals, X, Y, ok


def analyse_strip(vals, s, r, R, tex_m, lean_tpm, step):
    """Floor & window periodicity statistics of one rectified strip."""
    rin = (r >= 1.0) & (r <= R - 1.0)
    L = s[-1] + step if len(s) else 0
    sin_ = (s >= 0.08 * L) & (s <= 0.92 * L)
    out = {"floor": (0.0, None, None), "window": (0.0, None, None)}
    if rin.sum() >= 12 and sin_.sum() >= 4:
        prof = vals[rin][:, sin_].mean(1)
        # floor pitch 2.4..5.5 m -> texels along the lean
        pmin, pmax = 2.4 * lean_tpm, 5.5 * lean_tpm
        pf = detrend(prof, sigma=1.2 * pmax / step)
        out["floor"] = best_periodicity(pf, step, pmin, pmax)
        out["floor_prof"] = pf
    if sin_.sum() >= 12 and rin.sum() >= 4:
        prof = vals[rin][:, :].mean(0)
        # window pitch 1.0..8.0 m along the wall -> texels (ground m / tex_m)
        pmin, pmax = 1.0 / tex_m, 8.0 / tex_m
        pw = detrend(prof, sigma=1.2 * pmax / step)
        out["window"] = best_periodicity(pw, step, pmin, pmax)
        out["window_prof"] = pw
    return out


def fold(prof, step, P, nb=8):
    t = np.arange(len(prof)) * step
    ph = ((t / P) % 1.0 * nb).astype(int)
    f = np.array([prof[ph == b].mean() if (ph == b).any() else 0.0 for b in range(nb)])
    return f


def duty_and_cells(prof, step, P):
    """Majority phase (> mean or < mean) = cell/glass, minority = frame/slab line."""
    f = fold(prof, step, P)
    hi = f > f.mean()
    frac_hi = hi.mean()
    glass_is_high = frac_hi >= 0.5
    duty = frac_hi if glass_is_high else 1 - frac_hi
    return float(duty), bool(glass_is_high)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scenes/hitec_2x2")
    ap.add_argument("--viz", default=None)
    ap.add_argument("--null-per-strip", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    sd = args.scene
    meta, a, kcrs = load_scene(sd)
    tex_path = os.path.join(sd, meta.get("texture_file") or "terrain_texture.png")
    rgb = np.array(Image.open(tex_path).convert("RGB"))
    NT = rgb.shape[0]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    del rgb
    n_src = meta["resolution"][0]
    tex_map = a[0] * n_src / NT                     # map units per texel
    ms, lat = merc_scale(a, n_src) if "3857" in str(meta.get("crs")) else (1.0, None)
    tex_m = tex_map / ms                            # ground metres per texel

    # DSM (image frame) -> nDSM, for lean magnitude and null selection
    import rasterio
    dsm_path = [f for f in os.listdir(sd) if f.endswith("_DSM.tif")][0]
    with rasterio.open(os.path.join(sd, dsm_path)) as src:
        dsm = src.read(1).astype(np.float32)
        nod = src.nodata
    if nod is not None:
        dsm[dsm == nod] = np.nan
    dsm = np.nan_to_num(dsm, nan=float(np.nanmedian(dsm)))
    ndsm = np.maximum(dsm - ground_from_dsm(dsm), 0).astype(np.float32)
    dsm_per_tex = NT / dsm.shape[0]
    del dsm

    lean = measure_lean(gray, ndsm, dsm_per_tex)
    d = lean.pop("unit")
    v = lean["texels_per_m"]
    print(f"lean measured: {lean}")
    sl = meta.get("lean_m_per_m")
    scene_lean = None
    if sl:
        scene_lean = {"lean_m_per_m": sl,
                      "direction_deg_from_east_ccw": round(math.degrees(math.atan2(sl[1], sl[0])), 1),
                      "magnitude_m_per_m": round(math.hypot(*sl), 3)}
    lean_m_per_m_map = v * tex_map                  # map units of image offset per metre height
    lean["magnitude_map_units_per_m"] = round(lean_m_per_m_map, 3)
    lean["magnitude_ground_m_per_m"] = round(lean_m_per_m_map / ms, 3)
    lean["world_vector_x_z"] = [round(d[0] * lean_m_per_m_map / kcrs, 4),
                                round(-d[1] * lean_m_per_m_map / kcrs, 4)]

    grayf = gray.astype(np.float32)
    del gray
    samp = StripSampler(grayf, d, v)

    feats = json.load(open(os.path.join(sd, "buildings.geojson")))["features"]
    ndsm_small = ndsm  # 4096 grid

    def to_tex(P):
        return np.c_[(P[:, 0] - a[2]) / tex_map, (a[5] - P[:, 1]) / tex_map]

    # ------------------------------------------------ pass 1: real strips
    strips = []
    for bi, f in enumerate(feats):
        H = float(f["properties"].get("height_m") or 0)
        ring = np.array(f["geometry"]["coordinates"][0][:-1], float)
        c = to_tex(ring)
        if len(c) < 3:
            continue
        # outward-normal orientation: probe just off the longest edge's midpoint
        kk = int(np.argmax(np.hypot(*(np.roll(c, -1, 0) - c).T)))
        e0 = c[(kk + 1) % len(c)] - c[kk]
        n0 = np.array([e0[1], -e0[0]]) / (np.hypot(*e0) + 1e-9)
        probe = c[kk] + e0 / 2 + n0 * 0.3
        inside = cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2),
                                      (float(probe[0]), float(probe[1])), False)
        orient = -1.0 if inside > 0 else 1.0
        for k in range(len(c)):
            p0, p1 = c[k], c[(k + 1) % len(c)]
            e = p1 - p0
            Le = float(np.hypot(*e))
            if Le * tex_m < 3.0:
                continue
            # outward normal in image coords (y down)
            n = np.array([e[1], -e[0]]) / Le * orient
            facing = float(-(n @ d))
            R = v * H
            if facing < 0.35 or R < 6.0:
                continue
            strips.append((bi, k, p0, p1, H, facing))
    print(f"{len(strips)} sensor-facing wall strips >= 6 texels tall")

    results = []
    for (bi, k, p0, p1, H, facing) in strips:
        w, s, r, L, R = samp.geometry(p0, p1, H)
        vals, X, Y, ok = samp.sample(p0, w, s, r)
        if ok.mean() < 0.9:
            continue
        an = analyse_strip(vals, s, r, R, tex_m, v, samp.step)
        results.append(dict(bi=bi, k=k, p0=p0, w=w, s=s, r=r, L=L, R=R, H=H, facing=facing,
                            f=an["floor"], wnd=an["window"],
                            fprof=an.get("floor_prof"), wprof=an.get("window_prof")))
    print(f"{len(results)} strips analysed  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------ pass 2: null bank
    # same strip geometry translated onto non-facade texture of this image:
    # open ground (nDSM < 1.5 m everywhere under the strip) or a flat roof
    # interior (nDSM > 4 m, spread < 2 m) -- far from any wall edge.
    nulls = []   # (nr, ns, floorR2, windowR2)
    NS = ndsm_small.shape[0]
    for res in results:
        got, tries = 0, 0
        while got < args.null_per_strip and tries < 60:
            tries += 1
            q = rng.uniform(40, NT - 40 - res["L"], 2)
            corners = np.array([q, q + res["w"] * res["L"], q - d * res["R"],
                                q + res["w"] * res["L"] - d * res["R"],
                                q + res["w"] * res["L"] / 2 - d * res["R"] / 2])
            if corners.min() < 2 or corners.max() > NT - 3:
                continue
            hv = map_coordinates(ndsm_small, [corners[:, 1] / dsm_per_tex, corners[:, 0] / dsm_per_tex], order=0)
            is_ground = hv.max() < 1.5
            is_roof = hv.min() > 4 and np.ptp(hv) < 2.0
            if not (is_ground or is_roof):
                continue
            vals, X, Y, ok = samp.sample(q, res["w"], res["s"], res["r"])
            if ok.mean() < 0.9:
                continue
            an = analyse_strip(vals, res["s"], res["r"], res["R"], tex_m, v, samp.step)
            nulls.append((len(res["r"]), len(res["s"]), an["floor"][0], an["window"][0],
                          1 if is_ground else 0))
            got += 1
    nulls = np.array(nulls, float)
    print(f"{len(nulls)} null strips ({int(nulls[:,4].sum())} ground, {int((1-nulls[:,4]).sum())} roof) "
          f"({time.time()-t0:.0f}s)")

    def conf(stat, nr, ns, col):
        # null strips of similar size (profile length drives the chance R2)
        m = (np.abs(np.log(nulls[:, 0] / nr)) < 0.35) & (np.abs(np.log(nulls[:, 1] / ns)) < 0.6)
        if m.sum() < 40:
            m = np.abs(np.log(nulls[:, 0] / nr)) + 0.5 * np.abs(np.log(nulls[:, 1] / ns))
            m = m <= np.sort(m)[min(80, len(m) - 1)]
        ref = nulls[m, col]
        return float((ref < stat).mean()), int(m.sum())

    for res in results:
        nr, ns = len(res["r"]), len(res["s"])
        res["cf"], res["nf"] = conf(res["f"][0], nr, ns, 2) if res["f"][1] else (0.0, 0)
        res["cw"], res["nw"] = conf(res["wnd"][0], nr, ns, 3) if res["wnd"][1] else (0.0, 0)

    # ------------------------------------------------ per building
    by_b = {}
    for res in results:
        by_b.setdefault(res["bi"], []).append(res)
    per = []
    mask = np.zeros((NT, NT, 3), np.uint8)
    n_det = 0
    for bi, f in enumerate(feats):
        H = float(f["properties"].get("height_m") or 0)
        ring = np.array(f["geometry"]["coordinates"][0][:-1], float)
        cxy = ring.mean(0)
        rec = {"id": f["properties"].get("id"), "index": bi,
               "centroid_world_xz": [round((cxy[0] - a[2]) / kcrs, 2), round((cxy[1] - a[5]) / kcrs, 2)],
               "height_m": H}
        rs = by_b.get(bi, [])
        if not rs:
            rec.update(status="no_visible_facade" if v * H >= 6 else "facade_below_6px",
                       facade_px=round(v * H, 1),
                       floor_pitch_m=None, window_pitch_m=None, confidence=0.0,
                       glass_ratio_estimate=None)
            per.append(rec)
            continue
        m = len(rs)
        # Bonferroni across this building's walls: p_min * m
        bf = max(rs, key=lambda x: x["cf"])
        bw = max(rs, key=lambda x: x["cw"])
        cf = max(0.0, 1 - (1 - bf["cf"]) * m)
        cw = max(0.0, 1 - (1 - bw["cw"]) * m)
        fp = bf["f"][1] / v if bf["f"][1] else None       # texels along lean / (texels per m)
        wp = bw["wnd"][1] * tex_m if bw["wnd"][1] else None
        conf_all = 1 - (1 - cf) * (1 - cw) if (cf > 0.5 and cw > 0.5) else max(cf, cw) * 0.5
        glass = None
        if cf >= 0.9 and bf["fprof"] is not None:
            df, _ = duty_and_cells(bf["fprof"], samp.step, bf["f"][1])
            glass = df
        if cw >= 0.9 and bw["wprof"] is not None:
            dw, _ = duty_and_cells(bw["wprof"], samp.step, bw["wnd"][1])
            glass = dw if glass is None else glass * dw
        rec.update(status="analysed", walls_analysed=m, facade_px=round(v * H, 1),
                   floor_pitch_m=round(fp, 2) if fp and cf >= 0.5 else None,
                   floor_confidence=round(cf, 3),
                   # the raw measurement: period along the lean in texture texels
                   # (floor_pitch_m = this / lean texels_per_m, so it carries the
                   # lean-magnitude uncertainty; this number does not)
                   floor_period_texels=round(bf["f"][1], 2) if bf["f"][1] else None,
                   window_period_texels=round(bw["wnd"][1], 2) if bw["wnd"][1] else None,
                   floors_estimate=(int(round(H / fp)) if fp and cf >= 0.9 else None),
                   window_pitch_m=round(wp, 2) if wp and cw >= 0.5 else None,
                   window_confidence=round(cw, 3),
                   confidence=round(conf_all, 3),
                   glass_ratio_estimate=(round(glass, 2) if glass is not None else None),
                   glass_ratio_method=("INFERRED: duty cycle of the detected periodic "
                                       "profile(s); majority phase taken as glass" if glass is not None else None),
                   best_wall_facing=round(max(r_["facing"] for r_ in rs), 2))
        per.append(rec)

        # ---- UV-space (texture pixel grid) mask for each wall of this building
        for res in rs:
            fok = res["cf"] >= 0.9 and res["f"][1]
            wok = res["cw"] >= 0.9 and res["wnd"][1]
            s, r = res["s"], res["r"]
            S_, R_ = np.meshgrid(s, r)
            X = np.round(res["p0"][0] + S_ * res["w"][0] - R_ * d[0]).astype(int)
            Y = np.round(res["p0"][1] + S_ * res["w"][1] - R_ * d[1]).astype(int)
            inside = (R_ >= 0) & (R_ <= res["R"]) & (X >= 0) & (Y >= 0) & (X < NT) & (Y < NT)
            X, Y = X[inside], Y[inside]
            if not (fok or wok):
                g = mask[Y, X, 1]
                mask[Y, X, 1] = np.maximum(g, 96)
                continue
            n_det += 1
            cell = np.ones(S_.shape, np.float32)
            # separable lattice model fitted to THIS wall's pixels
            for ok_, prof, P, coord in ((fok, res["fprof"], res["f"][1], R_),
                                        (wok, res["wprof"], res["wnd"][1], S_)):
                if not ok_ or prof is None:
                    continue
                fcurve = fold(prof, samp.step, P)
                hi = fcurve > fcurve.mean()
                glass_hi = hi.mean() >= 0.5
                ph = ((coord / P) % 1.0 * 8).astype(int)
                cellax = hi[ph] if glass_hi else ~hi[ph]
                cell *= cellax.astype(np.float32)
            c_ = max(res["cf"] if fok else 0, res["cw"] if wok else 0)
            mask[Y, X, 0] = np.maximum(mask[Y, X, 0], (cell[inside] * 255).astype(np.uint8))
            mask[Y, X, 1] = 255
            mask[Y, X, 2] = np.maximum(mask[Y, X, 2], np.uint8(round(c_ * 255)))

    # ------------------------------------------------ summary + write
    confs = np.array([p["confidence"] for p in per if p.get("status") == "analysed"])
    cfs = np.array([p["floor_confidence"] for p in per if p.get("status") == "analysed"])
    cws = np.array([p["window_confidence"] for p in per if p.get("status") == "analysed"])
    fps = np.array([p["floor_pitch_m"] for p in per if p.get("floor_pitch_m") and p["floor_confidence"] >= 0.9])
    wps = np.array([p["window_pitch_m"] for p in per if p.get("window_pitch_m") and p["window_confidence"] >= 0.9])
    # floor pitch fallback for buildings with no detected floors: INFERRED
    for p in per:
        if p.get("floor_pitch_m") is None or p.get("floor_confidence", 0) < 0.9:
            if p["height_m"] >= 2.5 and len(fps) >= 20:
                med = float(np.median(fps))
                n_fl = max(1, int(round(p["height_m"] / med)))
                p["floor_pitch_inferred_m"] = round(p["height_m"] / n_fl, 2)
                p["floor_pitch_inferred_method"] = (
                    f"INFERRED: height_m / round(height_m / {med:.2f}); {med:.2f} m = median of "
                    f"this scene's detected floor pitches (conf>=0.9)")

    def hist(x, edges=(0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001)):
        h, _ = np.histogram(x, bins=edges)
        return {f"{edges[i]:.2f}-{min(edges[i+1],1):.2f}": int(h[i]) for i in range(len(h))}

    summary = {
        "buildings_total": len(feats),
        "buildings_with_sensor_facing_facade_ge_6px": int(len(by_b)),
        "walls_analysed": len(results),
        "walls_with_detected_lattice_in_mask": n_det,
        "buildings_floor_conf_ge_0.9": int((cfs >= 0.9).sum()),
        "buildings_window_conf_ge_0.9": int((cws >= 0.9).sum()),
        "buildings_both_ge_0.9": int(((cfs >= 0.9) & (cws >= 0.9)).sum()),
        "confidence_histogram": hist(confs),
        "floor_confidence_histogram": hist(cfs),
        "window_confidence_histogram": hist(cws),
        "detected_floor_pitch_m_pct10_50_90": ([round(float(x), 2) for x in np.percentile(fps, [10, 50, 90])]
                                                if len(fps) else None),
        "detected_window_pitch_m_pct10_50_90": ([round(float(x), 2) for x in np.percentile(wps, [10, 50, 90])]
                                                 if len(wps) else None),
        "null_strips": int(len(nulls)),
        "null_false_alarm_note": "confidence c means the statistic beats a fraction c of same-size "
                                 "ground/roof strips from this image; Bonferroni-corrected over a "
                                 "building's walls. Expect ~10% of pure-noise buildings above 0.9 "
                                 "before correction.",
    }
    out = {
        "source": f"real texture pixels: {os.path.basename(tex_path)} ({NT}x{NT}, "
                  f"{tex_m:.3f} ground m/texel = {tex_map:.4f} {meta.get('crs')} units)",
        "texture_file": os.path.basename(tex_path),
        "capture_time_utc": meta.get("capture_time_utc"),
        "method": "rectified facade strips (roof edge -> base along measured lean); floor pitch "
                  "from row-profile periodicity, window pitch from column-profile periodicity; "
                  "confidence vs an in-image null bank of ground/roof strips",
        "lean_measured": lean,
        "lean_scene_json": scene_lean,
        "lean_note": ("MEASURED here from the texture. It disagrees with scene.json lean_m_per_m "
                      "(which the viewer uses to find facade texels) in both direction and size; "
                      "floor_pitch_m scales as 1/lean magnitude, whose uncertainty is ~+-15 %"),
        "world_frame": "x = easting - affine[2], z = northing - affine[5] (source CRS units / "
                       "crs_units_per_m, which is 1 here); x east, z north -- same as viewer "
                       "realCoordsAt and build_city_image context placement",
        "footprint_caveat": "buildings.geojson polygons are often fragments of a building and do "
                            "not always sit on the roof outline in the texture; walls whose strip "
                            "lands on roof/ground simply fail the null test",
        "mask_file": "facade_window_mask.png",
        "mask_encoding": {"grid": f"{NT}x{NT}, same bounds as texture_file (texture UV space)",
                          "R": "window-cell (glass) likelihood from the lattice fitted to that wall's "
                               "own pixels (0/255); INFERRED which phase is glass (majority phase)",
                          "G": "255 = lattice detected (conf>=0.9 on floor or window axis); "
                               "96 = facade strip analysed, no lattice; 0 = not a facade strip",
                          "B": "confidence * 255"},
        "summary": summary,
        "per_building": per,
    }
    json.dump(out, open(os.path.join(sd, "facade_windows.json"), "w"), indent=1)
    Image.fromarray(mask).save(os.path.join(sd, "facade_window_mask.png"), optimize=True)
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time()-t0:.0f}s")

    # ------------------------------------------------ debug images
    if args.viz:
        os.makedirs(args.viz, exist_ok=True)
        tex = np.array(Image.open(tex_path).convert("RGB"))
        good = sorted([r_ for r_ in results if r_["cf"] >= 0.9 or r_["cw"] >= 0.9],
                      key=lambda r_: -(r_["cf"] + r_["cw"]) * r_["R"])
        tiles = []
        for res in good[:24]:
            vals, *_ = samp.sample(res["p0"], res["w"], res["s"], res["r"])
            im = np.clip(vals, 0, 255).astype(np.uint8)
            im = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            if res["cf"] >= 0.9:
                P = res["f"][1] / samp.step * 3
                y = 3 * (1.0 / samp.step)
                while y < im.shape[0]:
                    im[int(y), :: 4] = (255, 60, 60)
                    y += P
            if res["cw"] >= 0.9:
                P = res["wnd"][1] / samp.step * 3
                x = 0.0
                while x < im.shape[1]:
                    im[:: 4, int(x)] = (60, 255, 255)
                    x += P
            im = cv2.copyMakeBorder(im[:360, :360], 16, 2, 2, 2, cv2.BORDER_CONSTANT)
            cv2.putText(im, f"b{res['bi']} cf{res['cf']:.2f} cw{res['cw']:.2f}", (2, 12), 0, 0.4,
                        (255, 255, 0), 1)
            tiles.append(cv2.resize(im, (240, 240)) if max(im.shape[:2]) > 240 else
                         cv2.copyMakeBorder(im, 0, 240 - im.shape[0], 0, 240 - im.shape[1],
                                            cv2.BORDER_CONSTANT))
        if tiles:
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)]
            Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.viz, "facade_strips_top.png"))
        # texture crops with mask overlay around the strongest detections
        crops = []
        for res in good[:6]:
            cx, cy = (res["p0"] + res["w"] * res["L"] / 2 - d * res["R"] / 2).astype(int)
            R0 = 150
            x0, y0 = max(0, cx - R0), max(0, cy - R0)
            t_ = tex[y0:y0 + 2 * R0, x0:x0 + 2 * R0].copy()
            mk = mask[y0:y0 + 2 * R0, x0:x0 + 2 * R0]
            ov = t_.copy()
            ov[mk[..., 1] == 255] = (0.5 * ov[mk[..., 1] == 255] + [0, 60, 0]).astype(np.uint8)
            ov[mk[..., 0] > 128] = (0.4 * ov[mk[..., 0] > 128] + [0, 0, 150]).astype(np.uint8)
            ov[(mk[..., 1] == 96)] = (0.6 * ov[mk[..., 1] == 96] + [90, 0, 0]).astype(np.uint8)
            crops.append(np.concatenate([t_, ov], 1))
        if crops:
            Image.fromarray(np.concatenate(crops, 0)).save(os.path.join(args.viz, "facade_mask_overlay.png"))
        # confidence histograms
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        ax[0].hist(cfs, bins=20, range=(0, 1), color="#4a78b5"); ax[0].set_title("floor confidence")
        ax[1].hist(cws, bins=20, range=(0, 1), color="#4a78b5"); ax[1].set_title("window confidence")
        if len(fps):
            ax[2].hist(fps, bins=25, color="#b5784a"); ax[2].set_title("floor pitch m (conf>=0.9)")
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.viz, "facade_confidence.png"), dpi=90)


if __name__ == "__main__":
    main()
