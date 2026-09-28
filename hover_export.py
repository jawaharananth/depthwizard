"""Exact per-pixel height and uncertainty for the viewer's cursor readout.

The viewer shows two numbers under the cursor: the height the model
estimated there and its 1-sigma uncertainty. They are read from rasters, not
from the rendered mesh, so the number is the product's own value (the one the
DSM file carries and the benchmarks score), not a property of how buildings
were drawn.

Format: little-endian uint16 in centimetres (0 .. 655.34 m), 65535 = no
data, row-major from the top-left of the image -- the same orientation as the
scene's texture, so a texture UV addresses it directly. Longest side <= 1024.

`scene_facts` gathers the numbers the viewer's information panel shows, all
derived from the same rasters.
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

MAX_SIDE = 1024
NODATA = 65535


def _fit(a: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = a.shape
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return a.astype(np.float32)
    m = np.isfinite(a).astype(np.float32)
    v = np.where(m > 0, a, 0).astype(np.float32)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    num = cv2.resize(v, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


def _encode(a: np.ndarray) -> bytes:
    q = np.where(np.isfinite(a), np.clip(np.round(a * 100.0), 0, NODATA - 1), NODATA)
    return q.astype("<u2").tobytes()


def _decode(path: str, h: int, w: int) -> np.ndarray:
    """Inverse of _encode: metres, NaN where the file says no data. Reading the
    code as a number turned a masked image border into 655.35 m walls."""
    q = np.fromfile(path, dtype="<u2").reshape(h, w)
    return np.where(q == NODATA, np.nan, q.astype(np.float32) / 100).astype(np.float32)


def write(out_dir: str, agl: np.ndarray, sigma: np.ndarray | None,
          ground: np.ndarray | None = None) -> dict:
    """Write hover_agl.u16 (+ hover_sigma.u16, + hover_ground.f32); returns the scene.json entry.

    ground, when given, is the terrain in the 3D mesh's own vertical frame
    (metres, the values the ground mesh was built from). The viewer uses it
    to stand a LiDAR surface on exactly the same ground as the estimate, so
    a comparison shows height differences only.
    """
    a = _fit(np.maximum(agl, 0.0) if agl is not None else agl)
    with open(os.path.join(out_dir, "hover_agl.u16"), "wb") as f:
        f.write(_encode(a))
    entry = {"width": int(a.shape[1]), "height": int(a.shape[0]), "agl": "hover_agl.u16",
             "encoding": "uint16 little-endian, centimetres, 65535 = no data, rows from image top"}
    if sigma is not None:
        s = _fit(sigma)
        if s.shape != a.shape:
            s = cv2.resize(np.nan_to_num(s, nan=0.0), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        with open(os.path.join(out_dir, "hover_sigma.u16"), "wb") as f:
            f.write(_encode(s))
        entry["sigma"] = "hover_sigma.u16"
    else:
        for stale in ("hover_sigma.u16",):
            p = os.path.join(out_dir, stale)
            if os.path.exists(p):
                os.remove(p)
    if ground is not None:
        g = _fit(np.asarray(ground, np.float32))
        if g.shape != a.shape:
            g = cv2.resize(np.nan_to_num(g, nan=float(np.nanmedian(g))), (a.shape[1], a.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        with open(os.path.join(out_dir, "hover_ground.f32"), "wb") as f:
            f.write(np.nan_to_num(g, nan=float(np.nanmedian(g))).astype("<f4").tobytes())
        entry["ground"] = "hover_ground.f32"
    return entry


def scene_facts(agl: np.ndarray, sigma: np.ndarray | None, gsd_m: float | None,
                extra: dict | None = None) -> dict:
    """What the information panel shows. Every number from the rasters."""
    a = np.asarray(agl, np.float32)
    fin = np.isfinite(a)
    facts = {}
    if gsd_m:
        facts["ground_m"] = [round(a.shape[1] * gsd_m), round(a.shape[0] * gsd_m)]
        facts["pixel_m"] = round(float(gsd_m), 3)
    if fin.any():
        facts["height_p99_m"] = round(float(np.percentile(a[fin], 99)), 1)
        facts["height_max_m"] = round(float(a[fin].max()), 1)
        built = a[fin] > 2.5
        facts["above_2p5m_share"] = round(float(built.mean()), 3)
    if sigma is not None:
        s = np.asarray(sigma, np.float32)
        sf = np.isfinite(s)
        # Typical +/- on what stands above the ground (> 1 m): over every
        # pixel, a bare desert scene reports +/-0.02 m -- true of flat sand,
        # but not the figure anyone hovering a roof or tree will meet.
        if s.shape == a.shape:
            raised = sf & fin & (a > 1.0)
            if raised.mean() >= 0.005:
                sf = raised
        if sf.any():
            facts["sigma_median_m"] = round(float(np.median(s[sf])), 2)
    if extra:
        facts.update(extra)
    return facts


def lonlat_centre(transform, crs, width, height):
    """(lon, lat) of the image centre, or None."""
    try:
        from rasterio.warp import transform as _tr
        x = transform.c + transform.a * width / 2 + transform.b * height / 2
        y = transform.f + transform.d * width / 2 + transform.e * height / 2
        lon, lat = _tr(crs, "EPSG:4326", [x], [y])
        if math.isfinite(lon[0]) and math.isfinite(lat[0]):
            return [round(lon[0], 5), round(lat[0], 5)]
    except Exception:
        pass
    return None


def update_scene_json(scene_dir: str, hover: dict, facts: dict) -> None:
    p = os.path.join(scene_dir, "scene.json")
    meta = json.load(open(p)) if os.path.exists(p) else {}
    meta["hover"] = hover
    meta["facts"] = {**meta.get("facts", {}), **facts}
    json.dump(meta, open(p, "w"), indent=2)


def burn_buildings(out_dir: str, geojson_path: str, affine6, image_wh) -> int:
    """Write each building's own height (as drawn in 3D) into the readout rasters.

    The 3D buildings may carry heights fused from real sources (Open Buildings
    2.5D) or a per-building median, which differ from the raw per-pixel field.
    The cursor must read what the eye sees, so inside every footprint the
    readout takes that building's height_m, and its uncertainty is the
    building's own interval (90 % half-width / 1.645 = 1 sigma).
    """
    import rasterio.features
    from rasterio.transform import Affine
    meta_p = os.path.join(out_dir, "scene.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    hv = meta.get("hover")
    if not hv or not os.path.exists(geojson_path):
        return 0
    w, h = hv["width"], hv["height"]
    W, H = image_wh
    tf = Affine(*affine6[:6]) * Affine.scale(W / w, H / h)
    g = json.load(open(geojson_path))
    hs, ss = [], []
    for f in g.get("features", []):
        p = f.get("properties", {})
        if isinstance(p.get("height_m"), (int, float)) and f.get("geometry"):
            hs.append((f["geometry"], float(p["height_m"])))
            if isinstance(p.get("interval_half_width_m"), (int, float)):
                ss.append((f["geometry"], float(p["interval_half_width_m"]) / 1.645))
    if not hs:
        return 0
    bh = rasterio.features.rasterize(hs, out_shape=(h, w), transform=tf, fill=np.nan, dtype="float32")
    agl = _decode(os.path.join(out_dir, hv["agl"]), h, w)
    # a building only where there is image under it: no-data stays no-data
    agl = np.where(np.isfinite(bh) & np.isfinite(agl), bh, agl)
    with open(os.path.join(out_dir, hv["agl"]), "wb") as f:
        f.write(_encode(agl))
    if ss and hv.get("sigma"):
        bs = rasterio.features.rasterize(ss, out_shape=(h, w), transform=tf, fill=np.nan, dtype="float32")
        sg = _decode(os.path.join(out_dir, hv["sigma"]), h, w)
        sg = np.where(np.isfinite(bs) & np.isfinite(sg), bs, sg)
        with open(os.path.join(out_dir, hv["sigma"]), "wb") as f:
            f.write(_encode(sg))
    else:
        sg = None
    # The sheet's numbers follow what the readout now holds.
    gsd = (meta.get("facts") or {}).get("pixel_m")
    fresh = scene_facts(np.where(agl >= 0, agl, np.nan), sg, gsd)
    fresh.pop("ground_m", None); fresh.pop("pixel_m", None)
    fresh["buildings_drawn"] = len(hs)
    update_scene_json(out_dir, hv, fresh)
    return len(hs)
