"""How accuracy holds up when the input is coarser than the training imagery.

    python scripts/resolution_test.py

ISRO's evaluation imagery is likely nearer 0.6-1.0 m than the 0.3 m the
model was trained at. Each blind DFC2019 tile (the orthos benchmark_blind.py
writes, 0.25 m) is degraded to a coarser ground sample by area-averaging --
what a coarser sensor records -- then resampled back up to the working grid,
which is what the pipeline's auto-zoom does for a coarse upload. Heights are
scored against the LiDAR nDSM exactly as in the blind benchmark, with no
fitting to truth.
"""
import glob
import os
import sys

import cv2
import numpy as np
import rasterio
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import depth_model as dm                                 # noqa: E402
import benchmark_blind as B                              # noqa: E402
from rescore_baseline import ground_truth_ndsm           # noqa: E402

GSDS = (0.25, 0.5, 0.75, 1.0)


def main():
    bb = dm.DepthBackbone(metric=True)
    tiles = sorted(glob.glob(os.path.join(ROOT, "final_out", "blind_input", "JAX_*.tif")))
    res = {g: {"sq": 0.0, "ab": 0.0, "n": 0, "P": [], "T": []} for g in GSDS}
    for tp in tiles:
        tile = os.path.basename(tp)[:7]
        with rasterio.open(tp) as s:
            img = s.read([1, 2, 3]).transpose(1, 2, 0)
            native = abs(s.transform.a)
        gt_tile = B.L.load_tile(tile, B.TRUTH)
        gt = gt_tile["dsm"].copy(); gt[~gt_tile["valid_mask"]] = np.nan
        g_ndsm = ground_truth_ndsm(gt, gt_tile["cls"])
        for g in GSDS:
            f = native / g
            if f < 1:
                small = cv2.resize(img, (max(8, int(img.shape[1] * f)), max(8, int(img.shape[0] * f))),
                                   interpolation=cv2.INTER_AREA)
                work = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                work = img
            pred = bb.predict_tiled(Image.fromarray(work), raw_metric=True)
            pred = cv2.resize(pred.astype(np.float32), (g_ndsm.shape[1], g_ndsm.shape[0]),
                              interpolation=cv2.INTER_AREA)
            m = np.isfinite(g_ndsm) & np.isfinite(pred)
            e = pred[m] - g_ndsm[m]
            r = res[g]; r["sq"] += float((e ** 2).sum()); r["ab"] += float(np.abs(e).sum()); r["n"] += int(m.sum())
            r["P"].append(pred[m][::37]); r["T"].append(g_ndsm[m][::37])
        print(f"{tile} done", flush=True)
    lines = ["| input GSD | RMSE | MAE | pooled corr |", "|---|---|---|---|"]
    for g in GSDS:
        r = res[g]
        corr = np.corrcoef(np.concatenate(r["P"]), np.concatenate(r["T"]))[0, 1]
        lines.append(f"| {g:.2f} m | {np.sqrt(r['sq'] / r['n']):.2f} m | {r['ab'] / r['n']:.2f} m | {corr:.3f} |")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(ROOT, "BENCHMARK_RESOLUTION.md"), "w", encoding="utf-8") as fh:
        fh.write("# Accuracy vs input resolution (blind, DFC2019, "
                 f"{len(tiles)} held-out tiles)\n\n"
                 "Each tile's 0.25 m ortho is area-averaged to the coarser ground sample (what a "
                 "coarser sensor records), resampled back to the working grid (the pipeline's "
                 "auto-zoom), and scored against the LiDAR nDSM with no fitting to truth.\n\n"
                 + txt + "\n\nRegenerate: `python scripts/resolution_test.py`.\n")


if __name__ == "__main__":
    main()
