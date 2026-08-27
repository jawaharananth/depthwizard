"""
Tier B''': the actual final design -- shadow-direct heights for buildings
(proven better: 5.57m vs 6.74m RMSE on real DFC2019 urban terrain), blended
with the depth-model-based regression for everything else.

Why not flat-zero for non-building terrain (shadow_primary.py's approach):
real hills/ground genuinely aren't flat, and killing all relief signal there
made hilly/forest/sparse RMSE worse despite the depth model's weak (r~0.08)
correlation -- some real shape signal beats none for continuous terrain,
even an imperfect one. Why not depth-model regression for buildings (the
OLD shadow_based.py design): proven worse there specifically, since the
regression's global slope/intercept is only as good as that same weak
correlation, while shadow gives a direct per-building measurement with no
model in the loop at all.

So: two different signals, two different jobs, each used where it's
actually shown to win.
"""
import numpy as np

from calibration.shadow_based import calibrate_shadow_based
from calibration.shadow_primary import calibrate_shadow_primary
import segmentation as seg


def calibrate_shadow_hybrid(relative_depth: np.ndarray, image_np: np.ndarray, seg_labels: np.ndarray,
                             gsd_x_m: float, gsd_y_m: float,
                             sun_elevation_deg: float, sun_azimuth_deg: float):
    base_dsm, base_meta = calibrate_shadow_based(
        relative_depth, image_np, seg_labels, gsd_x_m, gsd_y_m, sun_elevation_deg, sun_azimuth_deg)
    building_dsm, building_meta = calibrate_shadow_primary(
        image_np, seg_labels, gsd_x_m, gsd_y_m, sun_elevation_deg, sun_azimuth_deg)

    if base_dsm is None and building_dsm is None:
        return None, {"tier": "B_shadow_hybrid", "status": "both_sub_tiers_failed",
                       "base_meta": base_meta, "building_meta": building_meta}

    if base_dsm is None:
        # no usable regression fit but buildings resolved directly: use 0 baseline
        # for non-building terrain rather than fail the whole tier.
        dsm = np.zeros_like(relative_depth)
    else:
        dsm = base_dsm.copy()

    building_mask = seg_labels == seg.CLASS_IDX["building"]
    if building_dsm is not None:
        dsm[building_mask] = building_dsm[building_mask]

    return dsm, {
        "tier": "B_shadow_hybrid",
        "base_regression": base_meta if base_dsm is not None else None,
        "building_direct": building_meta if building_dsm is not None else None,
    }
