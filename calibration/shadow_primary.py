"""
Tier B'': shadow height DIRECTLY as the building-height signal, not as a
reference point for fitting a scale to the neural depth model.

Both Depth Anything V2 and UniDepth v2 were benchmarked against real DFC2019
LiDAR and found to have near-zero correlation (r~0.08) with true building
height on nadir satellite imagery -- neither model has ever seen this imaging
geometry in training, so their relative-depth VALUES aren't trustworthy for
height, even though calibration/shadow_based.py's regression against them
technically "worked" (it just inherited that near-zero correlation).

This tier skips the neural depth model for building height entirely: each
building's shadow gives a direct, physically-measured height (no model in
the loop). Buildings without a measurable shadow get the group median height
imputed (honest fallback, not a guess dressed as precision). Non-building
ground is left at 0 (height ABOVE LOCAL GROUND convention, since there's no
absolute elevation reference here -- see dfc2019_benchmark.py's AGL note).
"""
import numpy as np
import cv2

import segmentation as seg
import shadow_correction


def calibrate_shadow_primary(image_np: np.ndarray, seg_labels: np.ndarray,
                              gsd_x_m: float, gsd_y_m: float,
                              sun_elevation_deg: float, sun_azimuth_deg: float,
                              min_confidence: str = "medium"):
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = confidence_rank[min_confidence]

    building_mask = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    n, cc_labels, stats, _ = cv2.connectedComponentsWithStats(building_mask, connectivity=8)

    # dsm=zeros: cross_validate_heights' "ai_height_m" isn't used by this tier at
    # all (that's the whole point), but the function still needs a same-shape
    # array for its internal shadow-mask/component bookkeeping.
    placeholder = np.zeros(seg_labels.shape, dtype=np.float32)
    shadow_results = shadow_correction.cross_validate_heights(
        image_np, seg_labels, placeholder, sun_elevation_deg, sun_azimuth_deg, gsd_x_m, gsd_y_m)

    resolved = {r["building_id"]: r["shadow_height_m"] for r in shadow_results
                if r["shadow_height_m"] is not None and confidence_rank[r["confidence"]] >= min_rank}

    dsm = np.zeros(seg_labels.shape, dtype=np.float32)  # ground = 0 (AGL convention)

    if not resolved:
        return None, {"tier": "B_shadow_primary", "status": "no_buildings_with_measurable_shadow",
                       "n_buildings_detected": len(shadow_results)}

    fallback_height = float(np.median(list(resolved.values())))
    n_direct_px, n_imputed_px = 0, 0

    for building_id in range(1, n):
        comp = cc_labels == building_id
        area = int(comp.sum())
        if area < 30:
            continue
        if building_id in resolved:
            dsm[comp] = resolved[building_id]
            n_direct_px += area
        else:
            dsm[comp] = fallback_height
            n_imputed_px += area

    total_building_px = n_direct_px + n_imputed_px
    return dsm, {
        "tier": "B_shadow_primary",
        "n_buildings_detected": len(shadow_results),
        "n_buildings_resolved_directly": len(resolved),
        "fallback_height_m": fallback_height,
        "direct_shadow_coverage_pct": round(100 * n_direct_px / max(total_building_px, 1), 1),
        "sun_elevation_deg": sun_elevation_deg, "sun_azimuth_deg": sun_azimuth_deg,
    }
