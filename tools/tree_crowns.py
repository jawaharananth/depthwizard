"""
Individual tree crowns for a DepthWizard scene, from REAL canopy height.

    venv\\Scripts\\python.exe tools\\tree_crowns.py hitec_2x2
    venv\\Scripts\\python.exe tools\\tree_crowns.py periyar_forest

Height sources (both real, both recorded per crown):
  h    Meta/WRI Global Canopy Height v2 (DINOv3, 2025; AWS open data
       s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3,
       EPSG:3857 1.194 m, whole metres). Fetched once over plain HTTP and
       cached in cache/chmv2_<scene>_native.npy; registered to the scene texture by
       phase correlation (greenness vs canopy) because v2 was made from other
       imagery. Falls back to v1 (the CHM the build used) if v2 is unavailable.
  hd   DSM - ground at the crown's highest pixel: what the BUILT scene's
       surface holds there. Ground is the pipeline's own ground where it can be
       recovered exactly (periyar: DSM - CHMv1 on canopy pixels, verified to
       0.05 m p99 smoothness), else a DTM from the DSM with buildings and
       vegetation masked (low percentile per 9.6 m block, filled, smoothed).
Crowns: height-adaptive h-maxima markers + marker watershed on the smoothed
canopy height, restricted to canopy pixels (h >= 1.5 m, outside building
footprints and their lean-displaced roofs) and to >= 40% of the crown's own
peak. Crown SHAPE is inferred (segmentation of a 1.2 m product), not traced.
Colours: mean / dark-quartile / light-quartile sRGB of the scene texture
(the same image the viewer drapes) inside each crown.

Output (see FIELDS): scenes/<scene>/tree_crowns.json (compact arrays) and,
when --binary or when the count exceeds 60k, tree_crowns.bin (float32
records) with the JSON as header. Coordinates are viewer world metres, same
frame as terrain.glb -- see tools/scene_frame.py.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_frame import ROOT, SceneFrame  # noqa: E402

CHMV2_URL = ("http://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/"
             "dinov3_global_chm_v2_ml3/chm/{qk}.tif")
SCENES = {
    "hitec_2x2": {"chm_v1": "archive/hitec_city_2x2/source/hitec2x2_chm.npy",
                  "cache": "cache/chmv2_hitec_2x2_native.npy"},
    "periyar_forest": {"chm_v1": "periyar_meta_chm_native.npy",
                       "cache": "cache/chmv2_periyar_forest_native.npy",
                       "v1_blur_native": 1.0},
}
# Showcase places built by build_showcase.py keep the canopy map they used.
_sc = os.path.join(ROOT, "work", "showcase")
for _id in (sorted(os.listdir(_sc)) if os.path.isdir(_sc) else []):
    if os.path.exists(os.path.join(_sc, _id, "chm.npy")) and _id not in SCENES:
        SCENES[_id] = {"chm_v1": f"work/showcase/{_id}/chm.npy", "cache": f"cache/chmv2_{_id}_native.npy"}
FIELDS = [
    ("x", "crown centroid, world x (m, east)"),
    ("z", "crown centroid, world z (m, north; = -row*gsd)"),
    ("y", "ground elevation under the centroid, world y (m) -- trunk base"),
    ("h", "tree height above ground (m), canopy-height product peak in crown"),
    ("hd", "DSM - ground at the crown's highest pixel (m): the built scene's surface"),
    ("yt", "rendered terrain.glb surface y at the centroid (m; includes its canopy bump)"),
    ("r", "area-equivalent crown radius (m) = sqrt(area/pi)"),
    ("ra", "ellipse semi-major axis (m), 2*sqrt(lambda1) of crown pixel covariance"),
    ("rb", "ellipse semi-minor axis (m)"),
    ("rot", "three.js rotation.y (rad) turning local +X onto the major axis"),
    ("R", "mean crown colour, sRGB 0-255"), ("G", ""), ("B", ""),
    ("Rd", "dark quartile (lowest-luminance 25% of crown pixels) mean, sRGB"), ("Gd", ""), ("Bd", ""),
    ("Rl", "light quartile (highest-luminance 25%) mean, sRGB"), ("Gl", ""), ("Bl", ""),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- inputs ----
def quadkey(x, y, z):
    s = ""
    for i in range(z, 0, -1):
        m = 1 << (i - 1)
        s += str((1 if x & m else 0) + (2 if y & m else 0))
    return s


def fetch_chmv2(sf, cache_path):
    """CHMv2 window covering the scene at native 1.194 m (uint8 metres)."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    import rasterio
    from rasterio.windows import from_bounds
    o = 20037508.342789244
    t = 2 * o / 1024
    cx = sf.left + sf.W * sf.gsd / 2
    cy = sf.top - sf.H * sf.gsd / 2
    qk = quadkey(int((cx + o) // t), int((o - cy) // t), 10)
    with rasterio.open("/vsicurl/" + CHMV2_URL.format(qk=qk)) as r:
        w = from_bounds(sf.left, sf.top - sf.H * sf.gsd, sf.left + sf.W * sf.gsd, sf.top,
                        r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, a)
    log(f"CHMv2 tile {qk} window {a.shape} cached -> {cache_path}")
    return a


def register_to_texture(chm, tex):
    """Sub-pixel shift (dx, dy) in working px that aligns chm with the texture's
    greenness; (0, 0) when the correlation peak is too weak to trust."""
    s = 2048
    t = cv2.resize(tex, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(2 * t[:, :, 1] - t[:, :, 0] - t[:, :, 2], (0, 0), 1.5)
    c = cv2.GaussianBlur(cv2.resize(chm, (s, s), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    a = (g - g.mean()) / (g.std() + 1e-6)
    b = (c - c.mean()) / (c.std() + 1e-6)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((s, s), cv2.CV_32F))
    k = chm.shape[0] / s
    if resp < 0.08:
        return 0.0, 0.0, float(resp)
    return float(dx * k), float(dy * k), float(resp)


def building_hull_mask(sf, lean):
    """Footprints plus the swept hull to their lean-displaced roof image."""
    m = np.zeros((sf.H, sf.W), np.uint8)
    p = os.path.join(sf.dir, "buildings.geojson")
    if not os.path.exists(p):
        return m.astype(bool)
    for f in json.load(open(p)).get("features", []):
        g = f.get("geometry") or {}
        rings = g.get("coordinates", [])
        if g.get("type") == "MultiPolygon":
            rings = [r for poly in rings for r in poly[:1]]
        else:
            rings = rings[:1]
        h = float(f.get("properties", {}).get("height_m") or 0.0)
        for ring in rings:
            xy = np.asarray(ring, np.float64)
            c, r = sf.px_from_crs(xy[:, 0], xy[:, 1])
            pts = np.stack([c, r], 1)
            if lean is not None and h > 0:
                sh = -np.asarray(lean) * h / sf.gsd   # roof appears NW of base (measured)
                pts = cv2.convexHull(np.vstack([pts, pts + sh]).astype(np.float32)).reshape(-1, 2)
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)


def dtm_from_dsm(dsm, exclude, block=16):
    """Ground from the DSM: low percentile per block over non-excluded pixels,
    holes filled by nearest, smoothed, bilinear back to full resolution."""
    H, W = dsm.shape
    nb = H // block
    d = dsm.reshape(nb, block, nb, block).transpose(0, 2, 1, 3).reshape(nb, nb, -1)
    ok = (~exclude).reshape(nb, block, nb, block).transpose(0, 2, 1, 3).reshape(nb, nb, -1)
    ok &= np.isfinite(d)
    v = np.where(ok, d, np.nan)
    frac = ok.mean(-1)
    with np.errstate(all="ignore"):
        g = np.nanpercentile(v, 10, axis=-1)
    g[frac < 0.15] = np.nan
    bad = ~np.isfinite(g)
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    g = g[tuple(idx)]
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------ crowns -------
def detect_tile(hs, veg, lum, gsd, peak_dyn=0.4, lum_weight=0.6):
    """Markers + watershed on one (tile+halo) block. Returns labels, tops, heights.

    A 1.2 m canopy-height product is smooth over closed canopy (one blob for a
    whole grove), so crown tops come from a combined surface: canopy height
    plus the texture's sunlit-crown brightness (z-scored over canopy, weighted
    as `lum_weight` metres). Gaps between crowns are both lower and darker,
    which is exactly what the watershed follows.
    """
    E = hs.copy()
    if veg.any():
        ls = cv2.GaussianBlur(lum, (0, 0), 1.5)
        v = ls[veg]
        E = hs + lum_weight * (ls - np.median(v)) / (v.std() + 1e-6)
    E = np.where(veg, E, E[veg].min() - 1 if veg.any() else 0).astype(np.float32)
    q = np.round((E - E.min()) * 10).astype(np.int32)   # decimetres for h_maxima
    hm = h_maxima(q, int(peak_dyn * 10)).astype(bool) & veg & (hs >= 2.0)
    lab, n = ndi.label(hm, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(hs.shape, np.int32), np.zeros((0, 2)), np.zeros(0)
    cy, cx = np.array(ndi.center_of_mass(hm, lab, np.arange(1, n + 1))).T
    ph = ndi.maximum(hs, lab, np.arange(1, n + 1))
    pe = ndi.maximum(E, lab, np.arange(1, n + 1))
    # height-adaptive suppression: a crown top needs >= max(1.5 m, 0.25 h) of
    # clearance from any stronger accepted top (crown radius ~0.2-0.35 h).
    order = np.argsort(-pe)
    cell = max(2.0, 0.25 * float(ph.max()) / gsd)
    grid = {}
    keep = []
    for i in order:
        rmin = max(1.5, 0.25 * ph[i]) / gsd
        gx, gy = int(cx[i] // cell), int(cy[i] // cell)
        ok = True
        for ax in (gx - 1, gx, gx + 1):
            for ay in (gy - 1, gy, gy + 1):
                for j in grid.get((ax, ay), ()):
                    if (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2 < rmin * rmin:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep.append(i)
            grid.setdefault((gx, gy), []).append(i)
    keep = np.array(keep)
    mk = np.zeros(hs.shape, np.int32)
    mk[np.round(cy[keep]).astype(int), np.round(cx[keep]).astype(int)] = np.arange(1, len(keep) + 1)
    labels = watershed(-E, mk, mask=veg)
    # crown = pixels at >= 40% of its own peak, within 0.6 h + 2 m of the top
    peak = np.zeros(len(keep) + 1, np.float32)
    peak[1:] = ph[keep]
    top_x = np.zeros(len(keep) + 1, np.float32)
    top_y = np.zeros(len(keep) + 1, np.float32)
    top_x[1:], top_y[1:] = cx[keep], cy[keep]
    yy, xx = np.indices(hs.shape, dtype=np.float32)
    pl = peak[labels]
    dist2 = (xx - top_x[labels]) ** 2 + (yy - top_y[labels]) ** 2
    rmax = (0.6 * pl + 2.0) / gsd
    labels[(hs < 0.4 * pl) | (dist2 > rmax * rmax)] = 0
    return labels, np.stack([cx[keep], cy[keep]], 1), ph[keep]


def crown_stats(labels, n, hs, hd, tex, gsd):
    """Per-label geometry + colour, vectorised. Label 0 is background."""
    m = labels > 0
    L = labels[m]
    yy, xx = np.nonzero(m)
    xx = xx.astype(np.float64) + 0.5
    yy = yy.astype(np.float64) + 0.5
    cnt = np.bincount(L, minlength=n + 1).astype(np.float64)
    c = np.maximum(cnt, 1)
    sx = np.bincount(L, xx, n + 1) / c
    sy = np.bincount(L, yy, n + 1) / c
    cxx = np.bincount(L, xx * xx, n + 1) / c - sx ** 2
    cyy = np.bincount(L, yy * yy, n + 1) / c - sy ** 2
    cxy = np.bincount(L, xx * yy, n + 1) / c - sx * sy
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    ang_img = 0.5 * np.arctan2(2 * cxy, cxx - cyy)   # major axis angle, image (col,row)
    # image dir (cos a, sin a) -> world (dx, dz) = (cos a, -sin a); three.js
    # rotation.y maps +X to (cos t, 0, -sin t), so t = a.
    hmax = ndi.maximum(hs, labels, np.arange(n + 1))
    # hd at each crown's highest pixel
    order = np.lexsort((hs[m], L))
    last = np.r_[np.nonzero(np.diff(L[order]))[0], len(L) - 1]
    hd_top = np.zeros(n + 1)
    hd_top[L[order][last]] = hd[m][order][last]
    # colours: mean + luminance quartiles
    rgb = tex[m].astype(np.float64)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    mean = np.stack([np.bincount(L, rgb[:, k], n + 1) / c for k in range(3)], 1)
    o = np.lexsort((lum, L))
    Ls = L[o]
    start = np.searchsorted(Ls, np.arange(n + 1))
    rank = np.arange(len(Ls)) - start[Ls]
    q = np.maximum(cnt[Ls] // 4, 1)
    dark = rank < q
    light = rank >= cnt[Ls] - q
    rs = rgb[o]
    cd = np.maximum(np.bincount(Ls[dark], minlength=n + 1), 1)
    cl = np.maximum(np.bincount(Ls[light], minlength=n + 1), 1)
    dk = np.stack([np.bincount(Ls[dark], rs[dark, k], n + 1) / cd for k in range(3)], 1)
    lt = np.stack([np.bincount(Ls[light], rs[light, k], n + 1) / cl for k in range(3)], 1)
    return dict(area_px=cnt, cx=sx, cy=sy, ra=2 * np.sqrt(l1) * gsd, rb=2 * np.sqrt(l2) * gsd,
                rot=ang_img, h=hmax, hd=hd_top, rgb=mean, dark=dk, light=lt)


def run(scene, tile=1024, halo=128, binary=None, overlay_dir=None):
    t0 = time.time()
    cfg = SCENES[scene]
    sf = SceneFrame(os.path.join("scenes", scene))
    gsd, H, W = sf.gsd, sf.H, sf.W
    log(scene, "grid", W, H, "gsd", round(gsd, 4))
    tex = sf.read_texture(W)
    dsm = sf.read_dsm()

    # --- canopy height: v2 (registered) with v1 fallback ------------------
    v1n = np.load(os.path.join(ROOT, cfg["chm_v1"])).astype(np.float32)
    if "v1_blur_native" in cfg:   # exactly how build_periyar.py prepared v1
        v1n = np.where(v1n > 0, cv2.GaussianBlur(v1n, (0, 0), cfg["v1_blur_native"]), 0).astype(np.float32)
    v1 = cv2.resize(v1n, (W, H), interpolation=cv2.INTER_LINEAR) if v1n.shape != (H, W) else v1n
    del v1n
    src = {}
    try:
        v2n = fetch_chmv2(sf, os.path.join(ROOT, cfg["cache"])).astype(np.float32)
        chm = cv2.resize(v2n, (W, H), interpolation=cv2.INTER_LINEAR)
        dx, dy, resp = register_to_texture(chm, tex)
        if dx or dy:
            chm = cv2.warpAffine(chm, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        src["h"] = ("Meta/WRI Global Canopy Height v2 (dinov3_global_chm_v2_ml3, 1.194 m, EPSG:3857), "
                    f"registered to scene texture by shift ({-dx * gsd:+.2f}, {-dy * gsd:+.2f}) m "
                    f"(x, image-down), phase-corr response {resp:.3f}")
        log("CHMv2 registration shift px", round(dx, 2), round(dy, 2), "resp", round(resp, 3))
    except Exception as e:   # offline: the CHM the build itself used
        log("CHMv2 unavailable (", e, ") -> v1")
        chm = v1.copy()
        src["h"] = "Meta HighResCanopyHeight v1 (the build's own CHM)"

    # --- ground ------------------------------------------------------------
    lean = sf.meta.get("lean_m_per_m")
    bld = building_hull_mask(sf, lean)
    if scene == "periyar_forest":
        # build_city_image: dsm[veg] = ground + chm_v1, veg = chm_v1 > 1
        ground = dsm - np.where(v1 > 1.0, v1, 0.0)
        src["ground"] = "pipeline ground recovered exactly: DSM - CHMv1 on the build's canopy mask (v1 > 1 m)"
    else:
        t = tex.astype(np.float32)
        green = (2 * t[:, :, 1] - t[:, :, 0] - t[:, :, 2]) > 10
        excl = bld | (v1 >= 1.0) | (chm >= 1.0) | green
        ground = dtm_from_dsm(dsm, excl)
        del t, green, excl
        src["ground"] = ("DTM from the DSM: p10 per 9.6 m block of pixels outside buildings, canopy (v1/v2 >= 1 m) "
                         "and green texture; nearest-filled, smoothed")
    datum, datum_note = sf.datum()
    hd_full = np.nan_to_num(dsm - ground).astype(np.float32)
    del dsm

    # --- canopy mask ----------------------------------------------------------
    t = tex.astype(np.float32)
    lumf = (t @ np.float32([0.2126, 0.7152, 0.0722])).astype(np.float32)
    grey_bright = (lumf > 165) & ((t.max(2) - t.min(2)) < 28)   # concrete/roof, never canopy
    del t
    # canopy: height >= 1.5 m (grown by 2 px to absorb the 1.2 m product's
    # soft edges), never building, never bright paving/soil in the texture
    veg = cv2.dilate((chm >= 1.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    veg &= ~bld & ~grey_bright & (lumf < 150)
    hs = cv2.GaussianBlur(np.where(veg, chm, 0).astype(np.float32), (0, 0), 1.2)
    hs = np.maximum(hs, 0)
    src["mask"] = "canopy height >= 1.5 m (+2 px), texture luminance < 150, outside building footprints + lean-displaced roofs, not bright-grey"

    # --- tiles ---------------------------------------------------------------
    out = {k: [] for k in ("cx", "cy", "area_px", "ra", "rb", "rot", "h", "hd", "rgb", "dark", "light")}
    full_labels = np.zeros((H, W), np.int32) if overlay_dir else None
    next_id = 1
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            y0, x0 = max(0, ty - halo), max(0, tx - halo)
            y1, x1 = min(H, ty + tile + halo), min(W, tx + tile + halo)
            lab, tops, ph = detect_tile(hs[y0:y1, x0:x1], veg[y0:y1, x0:x1], lumf[y0:y1, x0:x1], gsd)
            if len(tops) == 0:
                continue
            gx, gy = tops[:, 0] + x0, tops[:, 1] + y0
            own = (gx >= tx) & (gx < tx + tile) & (gy >= ty) & (gy < ty + tile)
            remap = np.zeros(len(tops) + 1, np.int32)
            remap[1:][own] = np.arange(1, own.sum() + 1)
            lab = remap[lab]
            n = int(own.sum())
            if n == 0 or not (lab > 0).any():
                continue                    # a tile with no crowns at all (desert, water)
            st = crown_stats(lab, n, hs[y0:y1, x0:x1], hd_full[y0:y1, x0:x1],
                             tex[y0:y1, x0:x1], gsd)
            # a crown whose mean colour is bright soil/concrete is not a tree
            # (canopy in these images is dark: mean luminance ~40-110)
            lumc = st["rgb"][1:] @ np.array([0.2126, 0.7152, 0.0722])
            mc = st["rgb"][1:]
            chroma = mc.max(1) - mc.min(1)
            roofish = ((lumc > 90) & (chroma < 22)) | (mc[:, 2] > mc[:, 1] + 6)   # grey/white/blue
            ok = (st["area_px"][1:] >= 3) & (st["h"][1:] >= 2.0) & (lumc < 140) & ~roofish
            for k in out:
                v = st[k][1:][ok]
                if k == "cx":
                    v = v + x0
                if k == "cy":
                    v = v + y0
                out[k].append(v)
            if full_labels is not None:
                keep_ids = np.zeros(n + 1, np.int32)
                keep_ids[1:][ok] = np.arange(next_id, next_id + ok.sum())
                sub = keep_ids[lab]
                blk = full_labels[y0:y1, x0:x1]
                blk[:] = np.where(blk > 0, blk, sub)
            next_id += int(ok.sum())
        log(f"row {ty}: {next_id - 1} crowns so far")
    for k in out:
        out[k] = np.concatenate(out[k]) if out[k] else np.zeros(0)
    N = len(out["cx"])
    x, z = sf.world_xz(out["cx"], out["cy"])
    gi = np.clip(out["cy"].astype(int), 0, H - 1), np.clip(out["cx"].astype(int), 0, W - 1)
    yg = ground[gi] - datum
    yt = sf.terrain_y(x, z)
    r = np.sqrt(out["area_px"] / np.pi) * gsd
    rec = np.column_stack([x, z, yg, out["h"], out["hd"], yt, r, out["ra"], out["rb"], out["rot"],
                           out["rgb"], out["dark"], out["light"]]).astype(np.float32)

    # --- stats + write -------------------------------------------------------
    h = rec[:, 3]
    area_m2 = out["area_px"] * gsd * gsd
    canopy_px = int(veg.sum())
    stats = {
        "count": int(N),
        "height_m": {"min": float(h.min()), "p10": float(np.percentile(h, 10)), "p50": float(np.median(h)),
                     "p90": float(np.percentile(h, 90)), "max": float(h.max())},
        "height_hist_m": {"bins": [2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 60],
                          "counts": np.histogram(h, [2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 60])[0].tolist()},
        "crown_radius_m": {"p10": float(np.percentile(r, 10)), "p50": float(np.median(r)),
                           "p90": float(np.percentile(r, 90))},
        "hd_minus_h_m": {"p10": float(np.percentile(rec[:, 4] - h, 10)),
                         "p50": float(np.median(rec[:, 4] - h)), "p90": float(np.percentile(rec[:, 4] - h, 90))},
        "canopy_area_m2": round(canopy_px * gsd * gsd, 1),
        "crown_area_m2": round(float(area_m2.sum()), 1),
        "crowns_per_ha_canopy": round(N / max(canopy_px * gsd * gsd / 1e4, 1e-9), 1),
        "seconds": round(time.time() - t0, 1),
    }
    names = [f for f, _ in FIELDS]
    header = {
        "schema": "depthwizard.tree_crowns/1",
        "scene": scene,
        "frame": {
            "units": "metres; viewer world = terrain.glb frame (three.js, Y up)",
            "x": f"col*gsd = X_{sf.meta['crs'].replace(':', '')} - {sf.left}",
            "z": f"-row*gsd = Y_{sf.meta['crs'].replace(':', '')} - {sf.top}",
            "y": f"absolute elevation - {datum:.3f}",
            "gsd_m": gsd, "grid": [W, H], "datum_m": datum, "datum_source": datum_note,
            "uv": "u = x/(gsd*(W-1)), v = 1 + z/(gsd*(H-1)) into scene texture_file",
        },
        "sources": src,
        "crown_shape": "INFERRED: marker watershed of a 1.2 m canopy-height product; ellipse = 2-sigma of crown pixels",
        "colour_source": sf.meta.get("texture_file") or "terrain_texture.png",
        "fields": {f: d for f, d in FIELDS},
        "count": int(N),
        "stats": stats,
    }
    use_bin = binary if binary is not None else N > 60000
    jp = os.path.join(sf.dir, "tree_crowns.json")
    if use_bin:
        bp = os.path.join(sf.dir, "tree_crowns.bin")
        rec.astype("<f4").tofile(bp)
        header["binary"] = {"file": "tree_crowns.bin", "dtype": "float32", "byte_order": "little",
                            "records": int(N), "stride_floats": len(names), "fields": names,
                            "layout": "record-major: record i = floats [i*stride, (i+1)*stride)"}
    else:
        dec = {"x": 2, "z": 2, "y": 2, "h": 1, "hd": 1, "yt": 2, "r": 2, "ra": 2, "rb": 2, "rot": 3}
        for i, f in enumerate(names):
            col = rec[:, i]
            header[f] = (np.round(col).astype(int).tolist() if f[0] in "RGB"
                         else np.round(col, dec[f]).tolist())
    with open(jp, "w") as fh:
        json.dump(header, fh, separators=(",", ":"))
    # The viewer's elevation readouts need the same datum (absolute = y + datum).
    try:
        sjp = os.path.join(sf.dir, "scene.json")
        sj = json.load(open(sjp, encoding="utf-8"))
        sj["elevation_datum_m"] = round(float(datum), 3)
        sj["elevation_datum_source"] = datum_note
        json.dump(sj, open(sjp, "w", encoding="utf-8"), indent=2)
    except Exception as e:
        log(f"elevation datum not written to scene.json: {e}")
    log(f"wrote {jp}" + (" + tree_crowns.bin" if use_bin else ""), f"{N} crowns",
        f"{os.path.getsize(jp) / 1e6:.1f} MB json")
    print(json.dumps(stats, indent=1))

    if overlay_dir:
        write_overlay(scene, tex, full_labels, out, overlay_dir)
    return header


def write_overlay(scene, tex, labels, out, odir, crops=None):
    os.makedirs(odir, exist_ok=True)
    b = np.zeros(labels.shape, np.uint8)
    b[(labels != np.roll(labels, 1, 0)) | (labels != np.roll(labels, 1, 1))] = 1
    b &= (labels > 0) | np.roll(labels > 0, 1, 0) | np.roll(labels > 0, 1, 1)
    ov = tex.copy()
    ov[b > 0] = (255, 230, 0)
    for cx, cy, h in zip(out["cx"], out["cy"], out["h"]):
        col = (255, 0, 0) if h >= 8 else (255, 0, 255)
        cv2.circle(ov, (int(cx), int(cy)), 2, col, -1)
    from PIL import Image
    Image.fromarray(ov).resize((2048, 2048), Image.BOX).save(os.path.join(odir, f"{scene}_crowns_full.jpg"), quality=88)
    H = labels.shape[0]
    crops = crops or [(H // 2 - 300, H // 2 - 300), (H // 4, H // 4), (3 * H // 4 - 300, H // 4 + 200)]
    for i, (y, x) in enumerate(crops):
        Image.fromarray(np.concatenate([tex[y:y + 600, x:x + 600], ov[y:y + 600, x:x + 600]], 1)).save(
            os.path.join(odir, f"{scene}_crowns_crop{i}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--binary", action="store_true", default=None)
    ap.add_argument("--overlay-dir", default=None)
    a = ap.parse_args()
    run(a.scene, binary=a.binary, overlay_dir=a.overlay_dir)
