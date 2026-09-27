"""Give an existing scene the viewer's exact cursor readout (height + uncertainty).

    python tools/attach_hover.py scenes/hitec_2x2 hitec2x2.tif

For scenes built before hover rasters were staged. Runs the production model
on the scene's source image exactly as the pipeline does (the trained 0.3 m
scale, block-wise on large scenes), computes the calibrated 1-sigma
uncertainty (checkpoints/height_best/sigma_calibration.json), and writes
hover_agl.u16 / hover_sigma.u16 plus the information panel's facts into the
scene. The source image must cover the same extent as the scene's texture
(true for every scene this project builds: the texture IS the source image).
"""
import json
import os
import sys

import cv2
import numpy as np
import rasterio
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import depth_model as dm  # noqa: E402
import hover_export as hx  # noqa: E402

TRAIN_GSD = 0.3


def ground_gsd(ds):
    import math
    if ds.crs is None:
        return None
    a = abs(ds.transform.a)
    if ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2
        return a * 111320.0 * math.cos(math.radians(lat))
    if ds.crs.to_epsg() == 3857:
        from rasterio.warp import transform as tr
        lat = tr(ds.crs, "EPSG:4326", [ds.bounds.left], [(ds.bounds.top + ds.bounds.bottom) / 2])[1][0]
        return a * math.cos(math.radians(lat))
    return a


def main():
    scene_dir, src = sys.argv[1], sys.argv[2]
    with rasterio.open(src) as ds:
        rgb = ds.read([1, 2, 3]).transpose(1, 2, 0)
        gsd = ground_gsd(ds)
        centre = hx.lonlat_centre(ds.transform, ds.crs, ds.width, ds.height) if ds.crs else None
        crs = str(ds.crs) if ds.crs else None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / max(np.percentile(rgb, 99.5), 1) * 255, 0, 255).astype(np.uint8)
    H, W = rgb.shape[:2]
    bb = dm.DepthBackbone(metric=True)
    z = max(1.0, min(3.0, (gsd or TRAIN_GSD) / TRAIN_GSD))
    z = min(z, (80e6 / (W * H)) ** 0.5) if z > 1 else 1.0
    im = Image.fromarray(rgb)
    work = im if z < 1.05 else im.resize((round(W * z), round(H * z)), Image.BICUBIC)
    print(f"model pass on {work.size[0]}x{work.size[1]} (x{z:.2f})", flush=True)
    agl = np.maximum(bb.predict_tiled(work, raw_metric=True), 0).astype(np.float32)
    agl = cv2.resize(agl, (W, H), interpolation=cv2.INTER_AREA)
    cal = json.load(open(os.path.join(dm.DepthBackbone.METRIC_CKPT, "sigma_calibration.json")))
    s = min(1.0, 1024 / max(W, H))
    small = im.resize((max(64, int(W * s)), max(64, int(H * s))), Image.BILINEAR)
    _, std = bb.predict_tta(small, raw_metric=True, views=4)
    std = cv2.resize(std.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    sigma = np.sqrt((cal["k"] * std) ** 2 + cal["s0"] ** 2 + (cal.get("c_height", 0.0) * agl) ** 2)
    hover = hx.write(scene_dir, agl, sigma)
    marker = json.load(open(os.path.join(dm.DepthBackbone.METRIC_CKPT, "metric_ok.json")))
    facts = hx.scene_facts(agl, sigma, gsd, {"centre_lonlat": centre, "crs": crs,
                                             "model": marker.get("checkpoint", "metric")})
    hx.update_scene_json(scene_dir, hover, facts)
    print(json.dumps({"hover": hover, "facts": facts}, indent=2))


if __name__ == "__main__":
    main()
