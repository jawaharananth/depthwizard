"""
Build HITEC City, Hyderabad using real Google Open Buildings V3 polygons
(S2 cell 3bd, confidence >= 0.65) as ground-truth footprints -- 1260 real
polygons landing inside this exact scene, vs the 281 Microsoft Global ML
Building Footprints supplied for the same area. Heights come from the
existing Open Buildings 2.5D Temporal height raster fusion (separate real
source, same as the MS-footprints build), via the gt_building_polygons
code path fixed earlier this session.

    python build_hitec_obv3.py
"""
import json

import numpy as np
from PIL import Image

MOSAIC = "hitec_city.tif"
POLY_FILE = "hitec_obv3_polygons_px.json"

if __name__ == "__main__":
    data = json.load(open(POLY_FILE))
    polygons = data["polygons"]

    mosaic = np.array(Image.open(MOSAIC).convert("RGB"))
    gt_mask = np.zeros(mosaic.shape[:2], np.uint8)
    import cv2
    for p in polygons:
        cv2.fillPoly(gt_mask, [np.array(p, np.int32)], 1)

    import build_city_image as bci
    bci.build(MOSAIC, "hitec_obv3", max_px=2048, use_dem=True,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons)
