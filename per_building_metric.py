"""Per-building height RMSE, on the same 18 DFC2019 benchmark tiles
rescore_baseline.py scores at the whole-tile-pixel level.

WHY THIS EXISTS

rescore_baseline.py's RMSE is a per-PIXEL average across an entire tile --
roads, bare ground, hillside, everything. That number can look good largely
because most of a scene is flat ground a model gets right almost by
default (measured: the "hilly" terrain class alone is 1.82m RMSE, dragging
the whole-tile average down). It is a different, generally easier number
than "how far off is this building's height," the per-building figure
commonly reported against airborne LiDAR.

This script computes the closest honest equivalent from what this project
already has: reuses rescore_baseline.py's exact orthorectification,
prediction, and calibration pipeline unmodified, then aggregates by
individual building INSTANCE (a connected component of LAS class 6 in the
ground truth) instead of by pixel. One number per building: its predicted
height is the MEDIAN prediction over its own pixels; its true height is the
median of ground_truth_ndsm over the same pixels. That is what "per-building
RMSE" means in the literature this compares against, and it is a
meaningfully harder, more honest number than a whole-tile pixel average.

Buildings are NOT this project's own extracted footprints (region_footprints.py)
here -- those depend on the very depth field being scored, which would let a
bad prediction hide by simply failing to detect the building it got wrong.
Ground-truth-labelled building pixels only, so the count of buildings scored
is independent of what the model itself predicts.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

from rescore_baseline import TRUTH, RGB, ground_truth_ndsm, predict_ndsm, MIN_CORRELATION
from depth_model import DepthBackbone

MIN_BUILDING_PX = 20   # smaller than this, a "building" is likely LiDAR noise on an edge


def building_instances(cls: np.ndarray) -> np.ndarray:
    """Connected components of LAS class 6 (building) pixels. Returns an int
    label image, 0 = not a building, 1..N = one building each.
    """
    mask = (cls == 6).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BUILDING_PX:
            labels[labels == i] = 0
    return labels


def main(max_tiles: int = 18) -> None:
    model = DepthBackbone()
    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(TRUTH, "*_DSM.tif"))})[:max_tiles]

    pred_heights, true_heights, per_tile = [], [], []

    for tile in tiles:
        if not glob.glob(os.path.join(RGB, f"{tile}_*_RGB.tif")):
            continue
        try:
            pred_full, o, metric, cal = predict_ndsm(tile, model, tiled=True)
        except Exception as exc:
            print(f"{tile}: SKIPPED -- {type(exc).__name__}: {exc}")
            continue

        gt_data = o["truth"]
        gt_dsm = gt_data["dsm"].copy()
        gt_dsm[~gt_data["valid_mask"]] = np.nan
        gt_ndsm = ground_truth_ndsm(gt_dsm, gt_data["cls"])

        size = gt_ndsm.shape[0]
        pred = cv2.resize(pred_full.astype(np.float32), (size, size),
                          interpolation=cv2.INTER_AREA)

        fin = np.isfinite(gt_ndsm) & np.isfinite(pred)
        corr = (float(np.corrcoef(pred[fin], gt_ndsm[fin])[0, 1])
                if fin.sum() > 100 else 0.0)
        if not np.isfinite(corr) or corr < MIN_CORRELATION:
            print(f"{tile}: REJECTED -- correlation {corr:.3f} < {MIN_CORRELATION}")
            continue

        if not metric:
            s = np.nanpercentile(gt_ndsm, 99) / max(np.percentile(pred, 99), 1e-6)
            pred = pred * s

        inst = building_instances(gt_data["cls"])
        n_buildings = int(inst.max())
        tile_pred, tile_true = [], []
        for b in range(1, n_buildings + 1):
            m = (inst == b) & np.isfinite(gt_ndsm) & np.isfinite(pred)
            if m.sum() < MIN_BUILDING_PX:
                continue
            tile_true.append(float(np.median(gt_ndsm[m])))
            tile_pred.append(float(np.median(pred[m])))

        if not tile_true:
            print(f"{tile}: 0 scoreable buildings")
            continue

        tt, tp = np.array(tile_true), np.array(tile_pred)
        tile_rmse = float(np.sqrt(np.mean((tp - tt) ** 2)))
        pred_heights.extend(tile_pred)
        true_heights.extend(tile_true)
        per_tile.append({"tile": tile, "n_buildings": len(tile_true),
                         "rmse_m": round(tile_rmse, 2)})
        print(f"{tile}: {len(tile_true)} buildings, per-building RMSE {tile_rmse:.2f} m")

    if not pred_heights:
        print("no tile produced a scoreable building")
        return

    pred_arr, true_arr = np.array(pred_heights), np.array(true_heights)
    overall_rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    overall_mae = float(np.mean(np.abs(pred_arr - true_arr)))
    bias = float(np.mean(pred_arr - true_arr))

    print()
    print("=" * 64)
    print(f"Per-building height RMSE, {len(pred_heights)} buildings, "
          f"{len(per_tile)} of {len(tiles)} tiles scored:")
    print(f"  RMSE {overall_rmse:.3f} m   MAE {overall_mae:.3f} m   "
          f"bias {bias:+.3f} m")
    print("=" * 64)

    payload = {
        "source": f"DFC2019_orthorectified_{len(per_tile)}tiles_per_building",
        "metric": "per-building median height, ground-truth LAS-class-6 instances",
        "n_buildings": len(pred_heights),
        "overall": {"rmse_m": overall_rmse, "mae_m": overall_mae, "bias_m": bias},
        "per_tile": per_tile,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/per_building_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote results/per_building_results.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 18)
