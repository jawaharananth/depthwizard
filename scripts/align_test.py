"""Does fitting each crop to a whole-image prediction help a metric model?

    python scripts/align_test.py

depth_model.predict_tiled can affine-align every crop to a ~518 px whole-image
prediction (align=True, the old behaviour) or trust the metric crops as they
are (align=False). Scored on the 18 held-out DFC2019 tiles, native and at
ISRO's 0.6 m (upsampled, as the pipeline runs it), per pixel and per building,
exactly as scripts/resolution_test.py scores.
"""
import glob, json, os, sys
import cv2
import numpy as np
import rasterio
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import depth_model as dm
import benchmark_blind as B
import dfc2019_height as DH
from rescore_baseline import ground_truth_ndsm

bb = dm.DepthBackbone(metric=True)
held = set(DH.benchmark_tiles())
tiles = [t for t in sorted(glob.glob(os.path.join(ROOT, "final_out", "blind_input", "JAX_*.tif")))
         if os.path.basename(t)[:7] in held]
conds = [(g, al) for g in (0.25, 0.6) for al in (True, False)]
R = {c: {"sq": 0.0, "n": 0, "ab": 0.0, "B": []} for c in conds}
for tp in tiles:
    tile = os.path.basename(tp)[:7]
    with rasterio.open(tp) as s:
        img = s.read([1, 2, 3]).transpose(1, 2, 0); native = abs(s.transform.a)
    gt = B.L.load_tile(tile, B.TRUTH); d = gt["dsm"].copy(); d[~gt["valid_mask"]] = np.nan
    t = ground_truth_ndsm(d, gt["cls"]); fin = np.isfinite(t)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(((gt["cls"] == 6) & fin).astype(np.uint8), connectivity=4)
    for g, al in conds:
        f = native / g
        work = img if f >= 1 else cv2.resize(cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)),
                                                         interpolation=cv2.INTER_AREA),
                                              (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        p = bb.predict_tiled(Image.fromarray(work), raw_metric=True, align=al).astype(np.float32)
        p = cv2.resize(p, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
        m = fin & np.isfinite(p); e = p[m] - t[m]
        r = R[(g, al)]; r["sq"] += float((e ** 2).sum()); r["ab"] += float(np.abs(e).sum()); r["n"] += int(m.sum())
        for i in range(1, nl):
            if st[i, cv2.CC_STAT_AREA] >= 40:
                bi = lab == i; r["B"].append((float(np.median(t[bi])), float(np.median(p[bi]))))
    print(tile, "done", flush=True)
out = {}
for (g, al), r in R.items():
    b = np.array(r["B"]); be = b[:, 1] - b[:, 0]
    out[f"{g} align={al}"] = {"rmse": (r["sq"] / r["n"]) ** 0.5, "mae": r["ab"] / r["n"],
                              "bldg_rmse": float(np.sqrt(np.mean(be ** 2))), "bldg_bias": float(be.mean())}
    print(f"{g} m align={al}: pixel RMSE {out[f'{g} align={al}']['rmse']:.3f}  MAE {out[f'{g} align={al}']['mae']:.3f}  "
          f"building RMSE {out[f'{g} align={al}']['bldg_rmse']:.3f}  bias {be.mean():+.3f}")
json.dump(out, open(os.path.join(ROOT, "final_out", "align_test.json"), "w"), indent=2)
