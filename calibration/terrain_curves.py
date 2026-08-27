import numpy as np
from scipy import stats

TERRAIN_CLASSES = ["urban", "hilly", "forest", "sparse"]


def fit_per_terrain_curves(relative_depth: np.ndarray, metric_reference: np.ndarray,
                            terrain_mask: np.ndarray):
    curves = {}
    calibrated = np.zeros_like(relative_depth)

    for idx, name in enumerate(TERRAIN_CLASSES):
        class_pixels = terrain_mask == idx
        valid = class_pixels & ~np.isnan(metric_reference)

        if valid.sum() < 10:
            all_valid = ~np.isnan(metric_reference)
            if all_valid.sum() < 3:
                curves[name] = (1.0, 0.0)
                calibrated[class_pixels] = relative_depth[class_pixels]
                continue
            a, b, _, _, _ = stats.linregress(
                relative_depth[all_valid], metric_reference[all_valid])
        else:
            a, b, _, _, _ = stats.linregress(
                relative_depth[valid], metric_reference[valid])

        curves[name] = (a, b)
        calibrated[class_pixels] = a * relative_depth[class_pixels] + b

    return calibrated, curves
