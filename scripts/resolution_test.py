"""How accuracy holds up when the input is coarser than the training imagery.

    python scripts/resolution_test.py

ISRO's FAQ (22 Sep 2026) fixes final evaluation on Cartosat-2S at 0.6 m. Each
held-out DFC2019 tile (the 0.25 m orthos benchmark_blind.py writes) is degraded
to a coarser ground sample by area-averaging -- what a coarser sensor records --
and then either

  * resampled back up to the working grid ("upsampled": the pipeline's
    auto-zoom), or
  * fed at its own coarse pixel count ("native pixels": what happens when a
    large scene is too big to upsample).

Heights are scored against the LiDAR nDSM exactly as in the blind benchmark,
with no fitting to truth: per pixel (RMSE, MAE, pooled correlation) and per
building (LiDAR building instances, median height on both sides -- the metric
the literature reports).

Only dfc2019_height.benchmark_tiles() are scored: the tiles the checkpoint
never saw. Simulated coarsening measures the resolution gap, not the sensor
gap (Cartosat colour and look angle differ from WorldView-3).
"""
import glob
import json
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
import dfc2019_height as DH                              # noqa: E402
from rescore_baseline import ground_truth_ndsm           # noqa: E402

# (label, ground sample in metres, how the coarse image is fed)
#   "full"     resampled back to the 0.25 m working grid
#   "native"   fed at its own coarse pixel count
#   "pipeline" upsampled the way build_city_image's auto-zoom does it:
#              x min(3, gsd / 0.3), never more -- beyond 3x a pixel is invented
# ISRO's FAQ asks for 0.35-10 m without overfitting to 0.6 m.
CONDITIONS = (
    ("0.25 m native", 0.25, "full"),
    ("0.35 m, upsampled", 0.35, "full"),
    ("0.50 m, upsampled", 0.50, "full"),
    ("0.60 m, upsampled", 0.60, "full"),
    ("0.60 m, native pixels", 0.60, "native"),
    ("0.75 m, upsampled", 0.75, "full"),
    ("1.00 m, upsampled", 1.00, "full"),
    ("2.5 m, auto-zoom x3", 2.5, "pipeline"),
    ("5 m, auto-zoom x3", 5.0, "pipeline"),
    ("10 m, auto-zoom x3", 10.0, "pipeline"),
)
NATIVE_TRAIN_GSD_M = 0.3        # as in build_city_image
MIN_BUILDING_PX = 40            # >= 10 m2 on the 0.5 m truth grid


def main(conditions=None, write=True):
    global CONDITIONS
    if conditions is not None:
        CONDITIONS = conditions
    bb = dm.DepthBackbone(metric=True)
    held = set(DH.benchmark_tiles())
    tiles = [t for t in sorted(glob.glob(os.path.join(ROOT, "final_out", "blind_input", "JAX_*.tif")))
             if os.path.basename(t)[:7] in held]
    res = {c[0]: {"sq": 0.0, "ab": 0.0, "n": 0, "P": [], "T": [], "B": []} for c in CONDITIONS}
    for tp in tiles:
        tile = os.path.basename(tp)[:7]
        with rasterio.open(tp) as s:
            img = s.read([1, 2, 3]).transpose(1, 2, 0)
            native = abs(s.transform.a)
        gt_tile = B.L.load_tile(tile, B.TRUTH)
        gt = gt_tile["dsm"].copy(); gt[~gt_tile["valid_mask"]] = np.nan
        g_ndsm = ground_truth_ndsm(gt, gt_tile["cls"])
        fin = np.isfinite(g_ndsm)
        bm = ((gt_tile["cls"] == 6) & fin).astype(np.uint8)
        nl, lab, st, _ = cv2.connectedComponentsWithStats(bm, connectivity=4)
        for label, g, mode in CONDITIONS:
            f = native / g
            if f < 1:
                small = cv2.resize(img, (max(8, int(img.shape[1] * f)), max(8, int(img.shape[0] * f))),
                                   interpolation=cv2.INTER_AREA)
                if mode == "full":
                    work = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
                elif mode == "pipeline" or isinstance(mode, tuple):
                    # ("to", m): upsample to m metres per pixel, uncapped
                    z = g / mode[1] if isinstance(mode, tuple) else min(3.0, g / NATIVE_TRAIN_GSD_M)
                    work = cv2.resize(small, (round(small.shape[1] * z), round(small.shape[0] * z)),
                                      interpolation=cv2.INTER_CUBIC)
                else:
                    work = small
            else:
                work = img
            pred = bb.predict_tiled(Image.fromarray(work), raw_metric=True)
            pred = cv2.resize(pred.astype(np.float32), (g_ndsm.shape[1], g_ndsm.shape[0]),
                              interpolation=cv2.INTER_AREA)
            m = fin & np.isfinite(pred)
            e = pred[m] - g_ndsm[m]
            r = res[label]; r["sq"] += float((e ** 2).sum()); r["ab"] += float(np.abs(e).sum()); r["n"] += int(m.sum())
            r["P"].append(pred[m][::37]); r["T"].append(g_ndsm[m][::37])
            for i in range(1, nl):
                if st[i, cv2.CC_STAT_AREA] < MIN_BUILDING_PX:
                    continue
                bi = lab == i
                r["B"].append((float(np.median(g_ndsm[bi])), float(np.median(pred[bi]))))
        print(f"{tile} done", flush=True)

    base = None
    lines = ["| input | per-pixel RMSE | MAE | pooled corr | per-building RMSE | bias | buildings <= 20 m | vs native |",
             "|---|---|---|---|---|---|---|---|"]
    out = {}
    for label, _, _ in CONDITIONS:
        r = res[label]
        corr = float(np.corrcoef(np.concatenate(r["P"]), np.concatenate(r["T"]))[0, 1])
        b = np.array(r["B"])
        be = b[:, 1] - b[:, 0]
        pb = float(np.sqrt(np.mean(be ** 2)))
        low = b[:, 0] <= 20
        pb20 = float(np.sqrt(np.mean(be[low] ** 2)))
        if base is None:
            base = pb
        out[label] = {"rmse": np.sqrt(r["sq"] / r["n"]), "mae": r["ab"] / r["n"], "corr": corr,
                      "bldg_rmse": pb, "bldg_bias": float(be.mean()), "bldg_rmse_le20": pb20,
                      "n_buildings": int(len(b))}
        lines.append(f"| {label} | {out[label]['rmse']:.2f} m | {out[label]['mae']:.2f} m | {corr:.3f} | "
                     f"{pb:.2f} m | {be.mean():+.2f} m | {pb20:.2f} m | {100 * (pb / base - 1):+.1f} % |")
    txt = "\n".join(lines)
    print(txt)
    if not write:
        return out
    json.dump(out, open(os.path.join(ROOT, "final_out", "resolution_test.json"), "w"), indent=2)
    with open(os.path.join(ROOT, "docs", "BENCHMARK_RESOLUTION.md"), "w", encoding="utf-8") as fh:
        fh.write("# Accuracy vs input resolution, 0.25-10 m (blind, DFC2019, "
                 f"{len(tiles)} held-out tiles, {out[CONDITIONS[0][0]]['n_buildings']} buildings)\n\n"
                 "ISRO evaluates on Cartosat-2S at 0.6 m and asks for models that hold from 0.35 m "
                 "to 10 m. Each held-out tile's 0.25 m ortho is area-averaged to the coarser ground "
                 "sample (what a coarser sensor records), then resampled back to the 0.25 m working "
                 "grid, fed at its own coarse pixel count, or (2.5 m and coarser) upsampled at most "
                 "3x exactly as the pipeline's auto-zoom does. Scored against the LiDAR nDSM with "
                 "no fitting to truth; per building = median height of each LiDAR building instance "
                 "on both sides. Model: the raw height field (depth_model.predict_tiled).\n\n"
                 + txt +
                 "\n\nAt 2.5 m and coarser a typical building is a few pixels or less, so building "
                 "heights blur toward their surroundings and are under-called; the absolute DSM "
                 "then rests mostly on the terrain model it is anchored to. At 5 m and 10 m the "
                 "model returns close to zero everywhere: those rows equal the predict-zero floor "
                 "on these tiles. Upsampling does not rescue it (measured 2026-10-01, same tiles, "
                 "per-building RMSE at 2.5 m: native pixels 3.55 m, x3 3.73 m, to 0.6 m 3.73 m, "
                 "to 0.3 m 3.71 m, to 0.25 m 3.74 m): the checkpoint was trained on 0.3-1.0 m "
                 "ground samples, so coarse input needs multi-resolution training, not resampling. "
                 "Products from inputs coarser than 2 m carry a RESOLUTION_NOTE tag saying so. "
                 "Simulated by downsampling WorldView-3, so this measures "
                 "the resolution gap, not the sensor gap.\n\n"
                 "Regenerate: `python scripts/resolution_test.py`.\n")


if __name__ == "__main__":
    main()
