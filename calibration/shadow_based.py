"""
Tier B': vertical scale calibration from shadow geometry, replacing the old
object_scale.py height formula (`approx_max_height_m = m_per_px * shape[0] *
0.1`), which had no physical basis -- verified against real DFC2019 LiDAR
ground truth, it produced ~20-29m RMSE, worse than useless.

Shadow length + sun elevation gives an independent real-meters height
estimate per building (shadow_correction.py, already validated against
synthetic ground truth). Using those as reference points, exactly like Tier
A's DEM-grid reference points, lets us fit a real relative-depth -> meters
scale/shift -- the only physically legitimate way to get absolute height
without a DEM or GCPs.

Requires: real-world scale (gsd_x_m/gsd_y_m) and sun position. Without both,
this tier is skipped -- no substitute heuristic, see object_scale.py's
docstring for why guessing further is worse than admitting Tier C (relative-only).
"""
import numpy as np
from scipy import stats

import shadow_correction


def calibrate_shadow_based(relative_depth: np.ndarray, image_np: np.ndarray, seg_labels: np.ndarray,
                            gsd_x_m: float, gsd_y_m: float,
                            sun_elevation_deg: float, sun_azimuth_deg: float,
                            min_confidence: str = "medium", min_references: int = 3):
    """
    Returns (metric_dsm, meta) or (None, meta) if too few shadow references
    were found to fit a stable regression.
    """
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = confidence_rank[min_confidence]

    # dsm=relative_depth (uncalibrated) here on purpose: cross_validate_heights'
    # "ai_height_m" becomes percentile95(relative_depth[footprint]) - percentile10(relative_depth) --
    # a RELATIVE unit, not meters yet. Paired with the shadow's independently-measured
    # real meters, that's exactly the (x, y) correspondence a scale/shift fit needs.
    shadow_results = shadow_correction.cross_validate_heights(
        image_np, seg_labels, relative_depth, sun_elevation_deg, sun_azimuth_deg, gsd_x_m, gsd_y_m)

    refs = [r for r in shadow_results
            if r["shadow_height_m"] is not None and confidence_rank[r["confidence"]] >= min_rank]

    if len(refs) < min_references:
        return None, {"tier": "B_shadow_based", "status": "insufficient_shadow_references",
                       "n_refs": len(refs), "n_buildings_detected": len(shadow_results)}

    rel_vals = np.array([r["ai_height_m"] for r in refs])   # relative-depth units (mislabeled key, see above)
    true_vals = np.array([r["shadow_height_m"] for r in refs])  # real meters, from shadow geometry

    a, b, r_value, _, _ = stats.linregress(rel_vals, true_vals)
    metric_dsm = a * relative_depth + b

    return metric_dsm, {
        "tier": "B_shadow_based", "a": float(a), "b": float(b), "r_value": float(r_value),
        "n_refs": len(refs), "n_buildings_detected": len(shadow_results),
        "sun_elevation_deg": sun_elevation_deg, "sun_azimuth_deg": sun_azimuth_deg,
    }
