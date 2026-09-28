"""How far must a 0.6 m image be upsampled before the model runs?

    python scripts/zoom_test.py

The 18 held-out DFC2019 tiles, area-averaged to 0.6 m (ISRO's resolution),
then upsampled by each zoom before inference: x2.0 is the trained 0.3 m.
Per-building and per-pixel error against the LiDAR, as resolution_test.py.
A smaller zoom means fewer pixels through the model on a large scene.
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
import depth_model as dm  # noqa: E402
import benchmark_blind as B  # noqa: E402
import dfc2019_height as DH  # noqa: E402
from rescore_baseline import ground_truth_ndsm  # noqa: E402

ZOOMS = (2.0, 1.75, 1.5, 1.25)
MIN_BUILDING_PX = 40


def main():
    bb = dm.DepthBackbone(metric=True)
    held = set(DH.benchmark_tiles())
    tiles = [t for t in sorted(glob.glob(os.path.join(ROOT, "final_out", "blind_input", "JAX_*.tif")))
             if os.path.basename(t)[:7] in held]
    res = {z: {"sq": 0.0, "n": 0, "B": []} for z in ZOOMS}
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
        f = native / 0.6
        small = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        for z in ZOOMS:
            work = cv2.resize(small, (round(small.shape[1] * z), round(small.shape[0] * z)),
                              interpolation=cv2.INTER_CUBIC)
            pred = bb.predict_tiled(Image.fromarray(work), raw_metric=True)
            pred = cv2.resize(pred.astype(np.float32), (g_ndsm.shape[1], g_ndsm.shape[0]),
                              interpolation=cv2.INTER_AREA)
            m = fin & np.isfinite(pred)
            e = pred[m] - g_ndsm[m]
            r = res[z]; r["sq"] += float((e ** 2).sum()); r["n"] += int(m.sum())
            for i in range(1, nl):
                if st[i, cv2.CC_STAT_AREA] < MIN_BUILDING_PX:
                    continue
                bi = lab == i
                r["B"].append((float(np.median(g_ndsm[bi])), float(np.median(pred[bi]))))
    out = {}
    for z in ZOOMS:
        b = np.array(res[z]["B"]); be = b[:, 1] - b[:, 0]
        out[f"x{z}"] = {"effective_m": round(0.6 / z, 3), "rmse": float(np.sqrt(res[z]["sq"] / res[z]["n"])),
                        "bldg_rmse": float(np.sqrt(np.mean(be ** 2))), "bldg_bias": float(be.mean()),
                        "n_buildings": int(len(b))}
        print(f"x{z} ({0.6 / z:.2f} m): pixel RMSE {out[f'x{z}']['rmse']:.3f}  per-building "
              f"{out[f'x{z}']['bldg_rmse']:.3f} m  bias {be.mean():+.3f}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "final_out", "zoom_test.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
