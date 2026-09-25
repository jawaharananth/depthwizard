"""
Sun azimuth + elevation from an image's own shadows, using KNOWN building
heights (a finished build's per-building GeoJSON) and the measured lean.

A building of height h whose roof appears at polygon P has its base at
P - lean*h (oblique image). Its ground shadow is the sweep of that base
along the anti-sun direction by h*cot(elevation). The sun direction that
makes these predicted shadow regions darkest (roofs and bases excluded) is
the capture's sun. Tall buildings only: their shadows are long enough to
constrain elevation.

    python scripts/measure_sun.py hitec_city.tif final_out/city_X_buildings.geojson LEANX LEANY
"""
import json
import sys

import cv2
import numpy as np
import rasterio
from PIL import Image


def main(tif, gj, lean_x, lean_y, min_h=15.0):
    with rasterio.open(tif) as ds:
        inv = ~ds.transform
        gsd = abs(ds.transform.a)
    img = np.array(Image.open(tif if tif.endswith(".png") else tif).convert("RGB")).astype(np.float32)
    H, W = img.shape[:2]
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = (lum < np.percentile(lum, 18)).astype(np.float32)
    feats = json.load(open(gj))["features"]
    lean = np.array([lean_x, lean_y])

    occupied = np.zeros((H, W), np.uint8)
    bld = []
    for f in feats:
        ring = np.array([inv * tuple(p) for p in f["geometry"]["coordinates"][0]], np.float32)
        h = float(f["properties"]["height_m"])
        base = ring - lean * h
        cv2.fillPoly(occupied, [ring.astype(np.int32)], 1)
        cv2.fillPoly(occupied, [base.astype(np.int32)], 1)
        if h >= min_h:
            bld.append((base, h))
    free = occupied == 0
    print(f"{len(bld)} buildings >= {min_h} m used; dark threshold = 18th percentile")

    def score(az, el):
        # shadow points AWAY from the sun; image x = east, y = south
        sa = np.radians(az + 180.0)
        d = np.array([np.sin(sa), -np.cos(sa)]) / gsd  # px per metre of ground
        cot = 1.0 / np.tan(np.radians(el))
        num = den = 0.0
        for base, h in bld:
            tip = base + d * h * cot
            hull = cv2.convexHull(np.vstack([base, tip]).astype(np.float32)).reshape(-1, 2)
            x0, y0 = np.floor(hull.min(0)).astype(int)
            x1, y1 = np.ceil(hull.max(0)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [(hull - [x0, y0]).astype(np.int32)], 1)
            sel = (m > 0) & free[y0:y1, x0:x1]
            n = sel.sum()
            if n < 20:
                continue
            num += dark[y0:y1, x0:x1][sel].sum()
            den += n
        return num / den if den else 0.0

    best = max(((score(a, e), a, e) for a in range(60, 301, 6) for e in range(20, 81, 4)))
    _, a0, e0 = best
    best = max(((score(a, e), a, e) for a in range(a0 - 6, a0 + 7, 1)
                for e in range(max(10, e0 - 4), min(85, e0 + 5), 1)))
    base = score(a0 + 90, e0)
    print(f"sun azimuth {best[1]} deg, elevation {best[2]} deg "
          f"(dark fraction in predicted shadows {best[0]:.2f}; same elevation, "
          f"azimuth rotated 90 deg: {base:.2f})")
    return best[1], best[2]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
