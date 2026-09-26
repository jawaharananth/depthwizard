"""Projection accuracy: how far the modelled roofs sit from where the image shows them.

    python scripts/projection_accuracy.py scenes/hitec_2x2 archive/hitec_city_2x2/source/hitec2x2.tif

The SIH rubric scores "projection accuracy". The ground drape is exact by
construction (the texture is the input image on its own geotransform), so the
measurable part is the buildings: every roof outline is placed where the
viewer draws its texture -- footprint + roof_offset_m + roof_lean_m_per_m *
height -- and its distance to the nearest strong edge in the image is read
from a distance transform. Reported as median / p75 / p90 in metres, overall
and by height band, plus the same numbers WITHOUT the lean correction so the
correction's effect is visible.
"""
import json
import os
import sys

import cv2
import numpy as np
import rasterio


def outline_points(px):
    pts = []
    for a, b in zip(px[:-1], px[1:]):
        n = max(2, int(np.hypot(*(b - a))))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + (b - a) * t)
    return np.vstack(pts)


def main(scene_dir, tif):
    meta = json.load(open(os.path.join(scene_dir, "scene.json")))
    g = json.load(open(os.path.join(scene_dir, "buildings.geojson")))
    with rasterio.open(tif) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        T = src.transform
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    edges = cv2.Canny(gray, 40, 110)
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    H, W = dist.shape
    gsd = abs(T.a)
    inv = ~T
    lean = meta.get("roof_lean_m_per_m") or [0.0, 0.0]
    off = meta.get("roof_offset_m") or [0.0, 0.0]
    # world (x east, z north) metres -> pixel (col right, row down)
    lpx = np.array([lean[0] / gsd, -lean[1] / gsd]); opx = np.array([off[0] / gsd, -off[1] / gsd])

    rows = []
    for f in g["features"]:
        h = f["properties"].get("height_m") or 0
        if h < 3 or f["geometry"]["type"] != "Polygon":
            continue
        px = np.array([inv * (x, y) for x, y in f["geometry"]["coordinates"][0]], np.float32)
        pts = outline_points(px)
        for tag, shift in (("corrected", opx + lpx * h), ("uncorrected", np.zeros(2))):
            q = np.round(pts + shift).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            if ok.mean() < 0.9:
                break
            rows.append((tag, h, float(np.median(dist[q[ok, 1], q[ok, 0]])) * gsd))
    out = {}
    for tag in ("corrected", "uncorrected"):
        d = np.array([r[2] for r in rows if r[0] == tag]); hh = np.array([r[1] for r in rows if r[0] == tag])
        res = {"n": int(d.size), "median_m": round(float(np.median(d)), 2),
               "p75_m": round(float(np.percentile(d, 75)), 2), "p90_m": round(float(np.percentile(d, 90)), 2)}
        for lo, hi in ((3, 10), (10, 25), (25, 1e9)):
            s = (hh >= lo) & (hh < hi)
            if s.sum() >= 10:
                res[f"p75_m_{lo}-{'' if hi > 1e8 else int(hi)}m"] = round(float(np.percentile(d[s], 75)), 2)
        out[tag] = res
    out["ground"] = "exact by construction: the texture is the input image on its own geotransform"
    out["method"] = ("median distance from each roof outline (as drawn by the viewer) to the nearest "
                     "Canny edge of the image, per building; buildings >= 3 m")
    print(json.dumps(out, indent=2))
    meta["projection_accuracy"] = out
    json.dump(meta, open(os.path.join(scene_dir, "scene.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
