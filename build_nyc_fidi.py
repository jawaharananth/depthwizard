"""
Build the Lower Manhattan Financial District demo scene from real NYC Open
Data Building Footprints (DoITT/City Planning) -- govt-surveyed polygons,
394 of them, every one carrying a REAL height_roof (photogrammetric /
LiDAR-derived, feet, converted to metres here) from NYC's own public
Socrata dataset (5zhs-2jue). Esri World Imagery for the base texture.

Same class of source as Mumbai BMC and Karnataka RAMP: height here is a
real civic MEASUREMENT, not a depth-field estimate or shadow calibration.

    python build_nyc_fidi.py
"""
import json

import numpy as np
from PIL import Image

MOSAIC = "nyc_fidi/mosaic.png"
POLY_FILE = "nyc_fidi/polygons_px.json"

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
    bci.build(MOSAIC, "nyc_fidi", max_px=2048, use_dem=False,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons,
              gt_building_heights=heights)
