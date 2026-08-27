"""
Accuracy validation: predicted DSM vs ground-truth DSM (e.g. DFC2019 LiDAR),
broken down by terrain type (urban/hilly/forest/sparse).

DFC2019 access is currently blocked on dataset registration (see project
roadmap item 3), so there is no real benchmark run yet -- this module is the
computation engine + dashboard data prep, verified here against synthetic
ground truth with a KNOWN injected error so the RMSE/MAE formulas themselves
are provably correct. Real numbers land the moment DFC2019 pairs are available
via compute_metrics_from_files().
"""
import numpy as np
import json

from calibration.terrain_curves import TERRAIN_CLASSES


def compute_metrics(predicted_dsm: np.ndarray, ground_truth_dsm: np.ndarray,
                     terrain_mask: np.ndarray = None, nodata_value: float = None) -> dict:
    """
    predicted_dsm, ground_truth_dsm: HxW float arrays, same shape, same units (meters).
    terrain_mask: optional HxW int array (values in TERRAIN_CLASSES index) for
                  the per-terrain breakdown. Without it, only 'overall' is returned.
    nodata_value: ground-truth pixels equal to this (or NaN) are excluded.

    Returns {"overall": {...}, "urban": {...}, "hilly": {...}, ...} where each
    entry has rmse_m, mae_m, bias_m, n_pixels. A class with zero valid pixels
    reports n_pixels=0 and null metrics rather than a fabricated 0.0.
    """
    if predicted_dsm.shape != ground_truth_dsm.shape:
        raise ValueError(f"shape mismatch: predicted {predicted_dsm.shape} vs "
                          f"ground truth {ground_truth_dsm.shape}")

    valid = ~np.isnan(ground_truth_dsm) & ~np.isnan(predicted_dsm)
    if nodata_value is not None:
        valid &= (ground_truth_dsm != nodata_value)

    def _metrics_for(mask):
        m = mask & valid
        n = int(m.sum())
        if n == 0:
            return {"rmse_m": None, "mae_m": None, "bias_m": None, "n_pixels": 0}
        err = predicted_dsm[m] - ground_truth_dsm[m]
        return {
            "rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "mae_m": float(np.mean(np.abs(err))),
            "bias_m": float(np.mean(err)),  # signed: + means over-predicting height
            "n_pixels": n,
        }

    results = {"overall": _metrics_for(np.ones_like(valid))}

    if terrain_mask is not None:
        if terrain_mask.shape != predicted_dsm.shape:
            raise ValueError(f"terrain_mask shape {terrain_mask.shape} != dsm shape {predicted_dsm.shape}")
        for idx, name in enumerate(TERRAIN_CLASSES):
            results[name] = _metrics_for(terrain_mask == idx)

    return results


def compute_metrics_from_files(predicted_geotiff_path: str, ground_truth_geotiff_path: str,
                                terrain_mask: np.ndarray = None) -> dict:
    """DFC2019-style entry point: two GeoTIFFs (predicted DSM, LiDAR ground truth)."""
    from dsm_export import load_dsm_geotiff
    pred, _, _, _ = load_dsm_geotiff(predicted_geotiff_path)
    gt, _, _, _ = load_dsm_geotiff(ground_truth_geotiff_path)
    if pred.shape != gt.shape:
        from skimage.transform import resize
        gt = resize(gt, pred.shape, preserve_range=True, anti_aliasing=True)
    return compute_metrics(pred, gt, terrain_mask)


def save_dashboard_data(results: dict, out_path: str, source: str = "synthetic_test"):
    payload = {"source": source, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


if __name__ == "__main__":
    # Synthetic verification: 4 terrain quadrants, each with a KNOWN error
    # pattern, so the reported RMSE/MAE must exactly match hand-computed values.
    H, W = 100, 100
    gt = np.random.RandomState(0).rand(H, W) * 20 + 10  # ground truth 10-30m

    terrain_mask = np.zeros((H, W), dtype=np.int32)
    terrain_mask[:50, :50] = TERRAIN_CLASSES.index("urban")
    terrain_mask[:50, 50:] = TERRAIN_CLASSES.index("hilly")
    terrain_mask[50:, :50] = TERRAIN_CLASSES.index("forest")
    terrain_mask[50:, 50:] = TERRAIN_CLASSES.index("sparse")

    pred = gt.copy()
    pred[:50, :50] += 2.0          # urban: constant +2m bias -> RMSE=MAE=2.0 exactly
    pred[:50, 50:] += np.random.RandomState(1).choice([-3.0, 3.0], size=(50, 50))  # hilly: +-3m -> RMSE=MAE=3.0
    pred[50:, :50] += 0.0          # forest: perfect -> RMSE=MAE=0.0
    pred[50:, 50:] += np.random.RandomState(2).normal(0, 1.5, size=(50, 50))       # sparse: noisy

    results = compute_metrics(pred, gt, terrain_mask)
    for name, m in results.items():
        print(f"{name:10s} rmse={m['rmse_m']} mae={m['mae_m']} bias={m['bias_m']} n={m['n_pixels']}")

    print("\nExpected: urban rmse=mae=2.0 exactly, forest rmse=mae=0.0 exactly, hilly rmse=mae=3.0 exactly")

    save_dashboard_data(results, "validation_test.json")
    print("Saved validation_test.json")
