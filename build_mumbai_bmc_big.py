"""
Build the Colaba, Mumbai demo scene from real BMC (Bruhanmumbai Municipal
Corporation) property-assessment records -- individually surveyed building
footprints, 655 of them, 580 carrying a REAL floor count from the municipal
record (Esri World Imagery for the base texture, BMC's own public ArcGIS
FeatureServer for the footprints and floor counts).

This is the one scene in the whole project where height is a MEASUREMENT,
not an estimate: floor count x a standard storey height (3.3 m) is a real
number from a real civic record, not a shadow calibration or a depth-field
guess. Where the record has no floor count (blank or "Not Yet Surveyed"),
the pipeline falls back to its usual depth-field estimate for that one
building -- honestly labelled per-building in the export, not blended
together into one number for the whole scene.

    python build_mumbai_bmc.py
"""
import json

import numpy as np
from PIL import Image

MOSAIC = "mumbai_bmc_big/mosaic.png"
POLY_FILE = "mumbai_bmc_big/polygons_px.json"

if __name__ == "__main__":
    data = json.load(open(POLY_FILE))
    polygons = data["polygons"]
    heights = data["heights"]

    mosaic = np.array(Image.open(MOSAIC).convert("RGB"))
    gt_mask = np.zeros(mosaic.shape[:2], np.uint8)
    import cv2
    for p in polygons:
        cv2.fillPoly(gt_mask, [np.array(p, np.int32)], 1)

    import build_city_image as bci
    bci.build(MOSAIC, "mumbai_colaba_big", max_px=2048, use_dem=False,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons,
              gt_building_heights=heights)
