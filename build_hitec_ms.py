"""
Rebuild HITEC City, Hyderabad using real Microsoft Global ML Building
Footprints (India, 2026-08-13 snapshot) as ground-truth polygons, instead
of the automated watershed discovery that scored VERDICT POOR (silhouette
IoU 0.088, recall 0.089) earlier this session on the exact same mosaic.

No real height attribute exists for India in the MS data (height=-1.0
sentinel for every record) -- heights still come from the existing
Open Buildings height fusion / depth-model estimate, unchanged. This
tests footprint geometry only.

    python build_hitec_ms.py
"""
import json

import numpy as np
from PIL import Image

MOSAIC = "hitec_city.tif"
POLY_FILE = "hitec_ms_polygons_px.json"

if __name__ == "__main__":
    data = json.load(open(POLY_FILE))
    polygons = data["polygons"]

    mosaic = np.array(Image.open(MOSAIC).convert("RGB"))
    gt_mask = np.zeros(mosaic.shape[:2], np.uint8)
    import cv2
    for p in polygons:
        cv2.fillPoly(gt_mask, [np.array(p, np.int32)], 1)

    import build_city_image as bci
    bci.build(MOSAIC, "hitec_ms", max_px=2048, use_dem=True,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons)
