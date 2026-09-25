"""
HITEC City, Hyderabad -- 2.45 km x 2.45 km (2x2 of the original tile).

  footprints  real Google Open Buildings V3 polygons (7390 in area) +
              264 Microsoft-detected structures Open Buildings missed
  structures  flyovers / metro viaduct / skywalks from OpenStreetMap
  context     + LOD1 context city (Microsoft footprints, OB heights)
  heights     real Open Buildings 2.5D raster, lean- and offset-corrected
  vegetation  real per-pixel canopy height, one continuous displaced mesh
  display     real zoom-19 (~0.3 m) imagery of the same bounds
  context     real ~10 km surrounding imagery, flat, labelled as context

    python build_hitec2x2.py
"""
import json

import numpy as np
from PIL import Image

MOSAIC = "hitec2x2.tif"

if __name__ == "__main__":
    import os
    # Regularised copy (tools/regularise_footprints.py: buildingregulariser,
    # guarded -- original kept where area moved >10% or IoU < 0.85) when it
    # exists; same order and count as the raw file.
    _fp_file = ("hitec2x2_polygons_px_reg.json"
                if os.path.exists("hitec2x2_polygons_px_reg.json")
                and os.environ.get("DW_NO_REGULARISE") != "1"
                else "hitec2x2_polygons_px.json")
    print(f"footprints: {_fp_file}")
    polygons = json.load(open(_fp_file))["polygons"]
    # + structures Microsoft detected that Open Buildings did not (sheds,
    # small blocks): a second, independent real detector, <10% overlap.
    polygons += json.load(open("hitec2x2_ms_extra_px.json"))["polygons"]
    mosaic = np.array(Image.open("hitec2x2.png").convert("RGB"))
    gt_mask = np.zeros(mosaic.shape[:2], np.uint8)
    import cv2
    for p in polygons:
        cv2.fillPoly(gt_mask, [np.array(p, np.int32)], 1)

    import build_city_image as bci
    bci.build(MOSAIC, "hitec2x2", max_px=4096, use_dem=True, auto_zoom=False,
              gt_building_mask=gt_mask.astype(bool),
              gt_building_polygons=polygons,
              canopy_height_map=np.load("hitec2x2_chm.npy"),
              continuous_vegetation=True,
              ground_grid=1400,
              pbr_maps=True,
              crown_radius_ratio=0.18,
              display_texture_path="hitec2x2_z19.png",
              depth_image_path="hitec2x2_z19.png",
              context_image="hitec2x2_context",
              context_far_image="hitec2x2_far",
              require_open_buildings=True,
              capture_date="2026-01-27",
              osm_structures=True,
              context_city_footprints=None)
