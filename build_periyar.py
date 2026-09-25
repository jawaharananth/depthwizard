"""
Periyar Tiger Reserve (Thekkady, Kerala) -- 2.45 km x 2.45 km forest + lake.

  imagery    Esri World Imagery z18 (Vantor Vivid, captured 2023-02-08)
  canopy     Meta/WRI Global Canopy Height Map (1.2 m, EPSG:3857 COG),
             one continuous displaced canopy surface
  terrain    Copernicus GLO-30 DEM, corrected for X-band canopy penetration
  sun        solar ephemeris at the capture date, assumed ~10:30 LMT pass
  context    real z16 (11.6 km) + z14 (46.5 km) imagery, flat

    python build_periyar.py
"""
import numpy as np

import build_city_image as bci

if __name__ == "__main__":
    import cv2
    chm = np.load("periyar_meta_chm_native.npy").astype(np.float32)
    # The product stores whole metres (uint8); unsmoothed, a gentle canopy
    # slope renders as 1 m terraces. Sub-crown smoothing removes only that.
    chm = np.where(chm > 0, cv2.GaussianBlur(chm, (0, 0), 1.0), 0.0).astype(np.float32)
    bci.build("periyar.tif", "periyar", max_px=4096, use_dem=True, auto_zoom=False,
              gt_building_mask=np.zeros((4096, 4096), bool),
              gt_building_polygons=[],
              canopy_height_map=chm,
              canopy_mask_from_chm=True,
              continuous_vegetation=True,
              canopy_blur_m=1.2,
              terrain_from_dem=True,
              ground_grid=1400,
              pbr_maps=True,
              context_image="periyar_context",
              context_far_image="periyar_far",
              capture_utc="2023-02-08T05:21")
