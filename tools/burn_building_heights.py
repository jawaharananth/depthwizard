"""Make the cursor readout agree with the drawn buildings.

    python tools/burn_building_heights.py scenes/hitec_core [--provenance-from]

The hover raster (hover_agl.u16) is the model's height field, while each
building solid takes the height in buildings.geojson -- which, for a scene
built with supplied heights (a close-up inheriting its whole-area scene's
calibrated heights), is a different and better number. Pointing at a roof
must read the height of the roof that is drawn, so inside every outline
the raster is set to that building's height. Ground, trees and everything
outside the outlines keep the model's values. The panel facts (tall end,
share above 2.5 m) are recomputed from the result.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import rasterio
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
NODATA = 65535


def main(scene_dir):
    sj = os.path.join(scene_dir, "scene.json")
    meta = json.load(open(sj, encoding="utf-8"))
    hv = meta["hover"]
    H, W = hv["height"], hv["width"]
    p = os.path.join(scene_dir, hv["agl"])
    q = np.fromfile(p, dtype="<u2").reshape(H, W)
    agl = np.where(q == NODATA, np.nan, q / 100.0).astype(np.float32)
    with rasterio.open(sorted(glob.glob(os.path.join(scene_dir, "*_DSM.tif")))[0]) as s:
        tf = s.transform * s.transform.scale(s.width / W, s.height / H)
        crs = s.crs
    g = json.load(open(os.path.join(scene_dir, "buildings.geojson"), encoding="utf-8"))
    gcrs = ((g.get("crs") or {}).get("properties") or {}).get("name") or str(crs)
    to_r = Transformer.from_crs(gcrs, crs, always_xy=True)
    inv = ~tf
    n = 0
    order = sorted(g["features"], key=lambda f: (f.get("properties") or {}).get("height_m") or 0)
    for f in order:                       # tall last: a tower wins over a podium it overlaps
        h = (f.get("properties") or {}).get("height_m")
        geom = f.get("geometry") or {}
        if h is None or h < 2.5 or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        rings = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in rings:
            ext = poly[0]
            pts = [inv * to_r.transform(x, y) for x, y in ext]
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1)
            # keep a one-pixel margin off the outline: edge pixels mix roof and street
            m = cv2.erode(m, np.ones((3, 3), np.uint8)) if m.sum() > 20 else m
            sel = (m > 0) & np.isfinite(agl)
            if sel.any():
                agl[sel] = float(h)
                n += 1
    out = np.where(np.isfinite(agl), np.clip(np.round(agl * 100), 0, NODATA - 1), NODATA).astype("<u2")
    out.tofile(p)
    fin = np.isfinite(agl)
    facts = meta.setdefault("facts", {})
    facts["height_p99_m"] = round(float(np.percentile(agl[fin], 99)), 1)
    facts["height_max_m"] = round(float(agl[fin].max()), 1)
    facts["above_2p5m_share"] = round(float((agl[fin] > 2.5).mean()), 3)
    hv["buildings_burned"] = n
    json.dump(meta, open(sj, "w", encoding="utf-8"), indent=2)
    print(f"{scene_dir}: {n} building outlines written into {hv['agl']}; "
          f"tall end {facts['height_p99_m']} m, max {facts['height_max_m']} m")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
