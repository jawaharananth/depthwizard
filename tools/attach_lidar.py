"""Attach airborne LiDAR to a scene: the viewer's LiDAR wipe and the accuracy facts.

    python tools/attach_lidar.py scenes/zurich LIDAR_DSM.tif LIDAR_DTM.tif [--source "swisstopo ..."]

LiDAR height above ground (DSM - DTM) is resampled onto the scene's hover
grid -- the image extent, top-left origin -- by reprojecting through the
scene's own georeference (scene.json affine_transform + crs), so the two
rasters describe the same ground pixel for pixel. Written as lidar_agl.u16
(same encoding as hover_agl.u16).

The scene's facts gain the comparison of the estimate against it, every
pixel with LiDAR: RMSE, MAE, bias and correlation, plus the tallest height in
each. Nothing is fitted to the LiDAR.
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hover_export as hx  # noqa: E402


def read_u16(path, w, h):
    a = np.fromfile(path, dtype="<u2").reshape(h, w).astype(np.float32)
    a[a == hx.NODATA] = np.nan
    return a / 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("dsm")
    ap.add_argument("dtm")
    ap.add_argument("--source", default="airborne LiDAR")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.scene_dir, "scene.json")))
    hv = meta["hover"]
    w, h = hv["width"], hv["height"]
    W, H = meta["resolution"]
    t = meta.get("affine_transform")
    crs = (meta.get("facts") or {}).get("crs") or meta.get("crs")
    if not t or not crs:
        raise SystemExit("scene has no georeference; LiDAR cannot be placed on it")
    tf = Affine(*t[:6]) * Affine.scale(W / w, H / h)          # hover grid
    out = {}
    for name, path in (("dsm", a.dsm), ("dtm", a.dtm)):
        dst = np.full((h, w), np.nan, np.float32)
        with rasterio.open(path) as s:
            reproject(rasterio.band(s, 1), dst, dst_transform=tf, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.average)
        out[name] = dst
    lidar = np.clip(out["dsm"] - out["dtm"], 0, None)
    with open(os.path.join(a.scene_dir, "lidar_agl.u16"), "wb") as f:
        f.write(hx._encode(lidar))
    est = read_u16(os.path.join(a.scene_dir, hv["agl"]), w, h)
    m = np.isfinite(lidar) & np.isfinite(est)
    e = est[m] - lidar[m]
    facts = {"lidar": {
        "source": a.source, "pixels": int(m.sum()),
        "rmse_m": round(float(np.sqrt(np.mean(e ** 2))), 2), "mae_m": round(float(np.mean(np.abs(e))), 2),
        "bias_m": round(float(e.mean()), 2), "r": round(float(np.corrcoef(est[m], lidar[m])[0, 1]), 3),
        "tallest_estimate_m": round(float(np.percentile(est[m], 99.9)), 1),
        "tallest_lidar_m": round(float(np.percentile(lidar[m], 99.9)), 1)}}
    hv["lidar"] = "lidar_agl.u16"
    hx.update_scene_json(a.scene_dir, hv, facts)
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
