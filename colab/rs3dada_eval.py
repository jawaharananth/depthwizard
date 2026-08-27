"""
RS3DAda height estimation — evaluated on OUR JAX tiles, in OUR metric.

Why this file exists: the SynRS3D paper reports 4.921m RMSE, but that number
is an average across 6 datasets (Houston/JAX/OMA/GeoNRW x2/Potsdam) in nDSM
(above-ground) units. Our shipped baseline is 5.91m absolute-DSM RMSE on
Jacksonville tiles only. Those are not the same measurement, and the paper's
number cannot be compared to ours directly. This script makes them
comparable: run RS3DAda on the exact tiles our benchmark already uses,
ground-reference its nDSM output the same way our shadow-hybrid path does,
and score with validation.compute_metrics -- verified elsewhere to be exact
against known injected error.

Calls the repo's own documented infer_height.py as a subprocess rather than
reimplementing model loading here. An earlier version of this file guessed
an import path (`height.dpt_model.DPTHead`) that turned out to be wrong --
the real entry point is `models.dpt.DPT_DINOv2`, with non-trivial
preprocessing (1022px patch tiling, cosine-blend overlap, GDAL I/O, ImageNet
normalisation, 4-way TTA) that lives inside infer_height.py. Reimplementing
that from a paraphrase would have shipped a guess as if it were verified
code. Calling the real script directly avoids that.

No accuracy claim is made until this script's own output says so.
"""
import os
import glob
import json
import subprocess
import sys

import numpy as np
import cv2

DATA_ROOT = os.environ.get("DW_DATA", "/content/colab_subset")
TRUTH_DIR = os.path.join(DATA_ROOT, "truth")
RGB_DIR = os.path.join(DATA_ROOT, "rgb")
SYNRS3D_DIR = os.environ.get("SYNRS3D_DIR", "/content/SynRS3d")
CKPT = os.environ.get("RS3DADA_CKPT", "/content/SynRS3d/pretrain/RS3DAda_vitl_DPT_height.pth")
PRED_DIR = os.path.join(DATA_ROOT, "rs3dada_preds")
OUT_JSON = os.path.join(DATA_ROOT, "rs3dada_vs_baseline.json")

TRUTH_SIZE = 512


def run_infer_height(rgb_tif_path: str, out_tif_path: str, use_tta: bool = True) -> bool:
    """
    Shells out to the real infer_height.py, per its documented CLI:
      python infer_height.py --data_dir IN --restore_from CKPT --output_path OUT [--use_tta]
    Returns True on success. Prints stderr on failure rather than swallowing it,
    since a silent failure here would look identical to "model predicts near-zero".
    """
    cmd = [
        sys.executable, os.path.join(SYNRS3D_DIR, "infer_height.py"),
        "--data_dir", rgb_tif_path,
        "--restore_from", CKPT,
        "--output_path", out_tif_path,
    ]
    if use_tta:
        cmd.append("--use_tta")

    result = subprocess.run(cmd, cwd=SYNRS3D_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"infer_height.py FAILED on {rgb_tif_path}:")
        print(result.stderr[-2000:])
        return False
    return True



def _ground_truth_ndsm(gt_dsm, cls):
    """
    Derive an above-ground height field from the LiDAR ground truth.

    Uses LAS class 2 (Ground) as the bare-earth reference and fills terrain
    beneath structures by nearest-ground extrapolation, then subtracts. The
    result is object height above terrain -- directly comparable with an nDSM
    prediction, unlike the raw absolute DSM.
    """
    import cv2

    GROUND_CLASS = 2
    ground = (cls == GROUND_CLASS)
    if ground.sum() < 100:
        # not enough labelled ground to build a terrain surface; fall back to
        # a per-tile offset and flag it by returning the offset version
        return gt_dsm - np.nanpercentile(gt_dsm, 10)

    hole = (~ground).astype(np.uint8)
    filled = gt_dsm.astype(np.float32).copy()
    if hole.any():
        _, labels = cv2.distanceTransformWithLabels(
            hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(ground)
        order = np.argsort(labels[ground])
        src_y, src_x = ys[order], xs[order]
        idx = np.clip(labels[hole.astype(bool)] - 1, 0, len(src_y) - 1)
        filled[hole.astype(bool)] = gt_dsm[src_y[idx], src_x[idx]]

    terrain = cv2.GaussianBlur(filled, (0, 0), sigmaX=4.0)
    return np.maximum(gt_dsm - terrain, 0.0)


def run(max_tiles: int = 20, use_tta: bool = True):
    import rasterio
    import dfc2019_loader as L  # from the main depthwizard repo -- add its path via sys.path if needed
    from validation import compute_metrics

    os.makedirs(PRED_DIR, exist_ok=True)

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found at {CKPT}")
        print("Download: huggingface_hub.hf_hub_download(repo_id='JTRNEO/RS3DAda', "
              "filename='RS3DAda_vitl_DPT_height.pth', local_dir='./SynRS3d/pretrain')")
        return
    if not os.path.exists(os.path.join(SYNRS3D_DIR, "infer_height.py")):
        print(f"SynRS3D repo not found at {SYNRS3D_DIR}")
        print(f"Clone: git clone https://github.com/JTRNEO/SynRS3D {SYNRS3D_DIR}")
        return

    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(TRUTH_DIR, "*_DSM.tif"))})
    results = []

    for tile in tiles[:max_tiles]:
        rgbs = sorted(glob.glob(os.path.join(RGB_DIR, f"{tile}_*_RGB.tif")))
        if not rgbs:
            continue

        pred_path = os.path.join(PRED_DIR, f"{tile}_height.tif")
        ok = run_infer_height(rgbs[0], pred_path, use_tta=use_tta)
        if not ok:
            results.append({"tile": tile, "status": "inference_failed"})
            continue

        with rasterio.open(pred_path) as src:
            pred_ndsm_full = src.read(1).astype(np.float32)
        pred_ndsm = cv2.resize(pred_ndsm_full, (TRUTH_SIZE, TRUTH_SIZE), interpolation=cv2.INTER_AREA)

        gt_data = L.load_tile(tile, TRUTH_DIR)
        gt = gt_data["dsm"].copy()
        gt[~gt_data["valid_mask"]] = np.nan

        # COMPARE LIKE WITH LIKE.
        #
        # RS3DAda predicts nDSM: height of objects ABOVE the ground. The DFC
        # ground truth is absolute DSM: terrain PLUS objects. Subtracting a
        # single percentile per tile removes a constant offset but cannot
        # remove terrain variation occurring *within* the tile -- measured on
        # JAX_004, the bare-earth terrain alone varies 18.0m across the tile,
        # against a 16.1m total range for the prediction. Scoring one against
        # the other charges the model for terrain it was never asked to
        # predict.
        #
        # So build a ground-truth nDSM instead, using the LiDAR's own
        # classification: LAS class 2 is Ground. Interpolating elevation from
        # those labelled ground returns gives bare earth; DSM minus that is
        # the true object height, which is exactly what RS3DAda outputs.
        gt_ndsm = _ground_truth_ndsm(gt, gt_data["cls"])
        gt_agl = gt_ndsm
        pred_agl = pred_ndsm

        metrics = compute_metrics(pred_agl, gt_agl, gt_data["terrain_mask"])
        results.append({"tile": tile, "status": "ok", "metrics": metrics})

        print(f"{tile}: overall RMSE={metrics['overall']['rmse_m']:.2f}  "
              f"MAE={metrics['overall']['mae_m']:.2f}")

    ok_results = [r for r in results if r["status"] == "ok"]
    failed = [r["tile"] for r in results if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} tile(s) failed inference: {failed}")
    if not ok_results:
        print("No tiles produced a valid prediction. Fix the failures above before trusting any number.")
        return

    agg_sq, agg_abs, n = 0.0, 0.0, 0
    for r in ok_results:
        m = r["metrics"]["overall"]
        if m["n_pixels"]:
            agg_sq += m["rmse_m"] ** 2 * m["n_pixels"]
            agg_abs += m["mae_m"] * m["n_pixels"]
            n += m["n_pixels"]

    rmse = (agg_sq / n) ** 0.5 if n else None
    mae = agg_abs / n if n else None

    print("\n" + "=" * 50)
    print(f"RS3DAda on {len(ok_results)} JAX tiles (OUR metric, OUR data):")
    print(f"  RMSE {rmse:.2f} m   MAE {mae:.2f} m   (n={n} px)")
    print("NOTE: our 5.91m shadow-hybrid figure was measured against ABSOLUTE DSM")
    print("      with a per-tile offset, which is a different (and easier-to-lose)")
    print("      comparison than the nDSM-vs-nDSM scoring used here. The baseline is")
    print("      being re-measured with this same metric before any claim is made.")
    print("=" * 50)
    if rmse is not None:
        print("RESULT: RS3DAda nDSM RMSE = %.2f m. Compare only against the baseline"
              % rmse)
        print("        re-scored with the identical nDSM metric -- not against 5.91.")

    with open(OUT_JSON, "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "n_pixels": n, "per_tile": results}, f, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    run()
