import numpy as np

from calibration.georeferenced import calibrate_georeferenced, calibrate_georeferenced_per_terrain
from calibration.object_scale import calibrate_object_based
from calibration.shadow_hybrid import calibrate_shadow_hybrid


def calibrate(relative_depth: np.ndarray, image_np: np.ndarray,
              geo_info: dict = None, dem_path: str = None, seg_labels: np.ndarray = None,
              gsd_x_m: float = None, gsd_y_m: float = None,
              sun_elevation_deg: float = None, sun_azimuth_deg: float = None):
    if geo_info is not None and dem_path is not None:
        if seg_labels is not None:
            try:
                dsm, meta = calibrate_georeferenced_per_terrain(
                    relative_depth, dem_path, geo_info["bounds_wgs84"], seg_labels)
                return dsm, meta
            except Exception as e:
                print(f"[calibrate] Tier A per-terrain failed ({e}), trying Tier A global fit")
        try:
            dsm, meta = calibrate_georeferenced(
                relative_depth, dem_path, geo_info["bounds_wgs84"])
            return dsm, meta
        except Exception as e:
            print(f"[calibrate] Tier A failed ({e}), falling back to Tier B")

    # Tier B: shadow-hybrid vertical calibration -- verified against real DFC2019
    # LiDAR ground truth. Buildings get direct shadow-measured height (proven
    # better: 5.3m vs 6.7m RMSE); non-building terrain uses a depth-model
    # regression fit from the same shadow references (weak correlation, but
    # still better than assuming flat terrain -- see shadow_hybrid.py). Needs
    # seg_labels (building footprints) + real sun angle + real-world scale
    # (gsd). If gsd isn't supplied directly, fall back to object-detection-based
    # GSD estimation (imperfect, but only used for horizontal scale here --
    # the old object_scale height *formula*, a blind "10% of image height"
    # guess, is retired entirely: benchmarked against real DFC2019 ground
    # truth it gave ~20-29m RMSE, worse than no calibration at all).
    if seg_labels is not None and sun_elevation_deg is not None and sun_azimuth_deg is not None:
        gx, gy = gsd_x_m, gsd_y_m
        if gx is None or gy is None:
            _, obj_meta = calibrate_object_based(relative_depth, image_np)
            gx = gy = obj_meta.get("m_per_px_estimate")

        if gx is not None and gy is not None:
            dsm, meta = calibrate_shadow_hybrid(
                relative_depth, image_np, seg_labels, gx, gy, sun_elevation_deg, sun_azimuth_deg)
            if dsm is not None:
                return dsm, meta
            print(f"[calibrate] Tier B shadow-hybrid insufficient references ({meta}), falling back to Tier C")

    return relative_depth, {"tier": "C_relative_only",
                             "note": "No metric reference available; output is relative height, not absolute elevation."}
