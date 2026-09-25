"""
HITEC City, Hyderabad -- full demo build.

  footprints  real Google Open Buildings V3 polygons (1260 in scene)
  heights     real Open Buildings 2.5D Temporal raster, fused per building
  vegetation  real per-pixel canopy height (Meta HighResCanopyHeight),
              folded into the ground as one continuous displaced mesh
  shading     normal / AO / roughness baked at full texture resolution
              from that same surface

    python build_hitec_final.py
"""
import json

import numpy as np
from PIL import Image

MOSAIC = "hitec_city.tif"
POLY_FILE = "hitec_obv3_polygons_px.json"
CHM_FILE = "hitec_chm.npy"

if __name__ == "__main__":
    polygons = json.load(open(POLY_FILE))["polygons"]

    mosaic = np.array(Image.open(MOSAIC).convert("RGB"))
    gt_mask = np.zeros(mosaic.shape[:2], np.uint8)
    import cv2
    for p in polygons:
        cv2.fillPoly(gt_mask, [np.array(p, np.int32)], 1)

    chm = np.load(CHM_FILE)

    import build_city_image as bci
    bci.build(MOSAIC, "hitec_final", max_px=2048, use_dem=True,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons,
              canopy_height_map=chm,
              continuous_vegetation=True,
              ground_grid=1024,
              pbr_maps=True,
              crown_radius_ratio=0.18,
              display_texture_path="hitec_city_z19.png",
              context_image="hitec_context")
