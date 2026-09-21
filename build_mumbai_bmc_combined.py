"""
Colaba, Mumbai, with BOTH real municipal building records (BMC, with real
floor-count heights) AND Google Open Buildings filling the gap where BMC's
formal property-assessment registry has no coverage -- confirmed to be
exactly the informal settlement visible in the source image, which municipal
tax records structurally exclude.

1,023 buildings added this way, on top of BMC's 655. Combined footprints are
honestly split by provenance per building: BMC buildings keep their real
floor-count height; Google Open Buildings footprints have no height record
of their own and fall back to the depth-field estimate, same as any
ground-truth-only scene tonight.
"""
import json

import numpy as np
from PIL import Image
import cv2

MOSAIC = "mumbai_bmc/mosaic.png"
POLY_FILE = "mumbai_bmc/polygons_combined.json"

if __name__ == "__main__":
    data = json.load(open(POLY_FILE))
    polygons = data["polygons"]
    heights = data["heights"]

    mosaic = np.array(Image.open(MOSAIC).convert("RGB"))
    gt_mask = np.zeros(mosaic.shape[:2], np.uint8)
    for p in polygons:
        cv2.fillPoly(gt_mask, [np.array(p, np.int32)], 1)

    import build_city_image as bci
    bci.build(MOSAIC, "mumbai_colaba_full", max_px=1536, use_dem=False,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons,
              gt_building_heights=heights)
