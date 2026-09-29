"""Score a candidate checkpoint on every held-out test that decides promotion.

    python scripts/compare_ckpt.py checkpoints/height_run03 [checkpoints/height_best ...]

For each checkpoint directory:
  * DFC2019, the 18 held-out tiles, at ISRO's 0.6 m (area-averaged, upsampled
    back as the pipeline runs it): per-pixel RMSE / MAE / correlation and
    per-building RMSE / bias -- as scripts/resolution_test.py;
  * swisstopo LiDAR, the four landscapes of scripts/lidar_landscapes.py at
    0.6 m: height-above-ground RMSE / bias, and the absolute surface
    (anchoring.py, GLO-30, measured k) against the LiDAR DSM.

None of these tiles is in any training split (DFC benchmark_tiles(); Swiss
benchmark tiles plus a 1 km buffer are excluded by training/prepare_swiss.py).
The checkpoint is loaded by pointing DepthBackbone.METRIC_CKPT at it, so the
production checkpoint is never touched.
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
import depth_model as dm                                   # noqa: E402
import anchoring                                           # noqa: E402
import dem_source                                          # noqa: E402
import benchmark_blind as B                                # noqa: E402
import dfc2019_height as DH                                # noqa: E402
import lidar_landscapes as LL                              # noqa: E402
from rescore_baseline import ground_truth_ndsm             # noqa: E402


def dfc(bb):
    held = set(DH.benchmark_tiles())
    tiles = [t for t in sorted(glob.glob(os.path.join(ROOT, "final_out", "blind_input", "JAX_*.tif")))
             if os.path.basename(t)[:7] in held]
    sq = ab = n = 0.0
    P, T, Bl, Bt = [], [], [], []
    for tp in tiles:
        tile = os.path.basename(tp)[:7]
        with rasterio.open(tp) as s:
            img = s.read([1, 2, 3]).transpose(1, 2, 0)
            f = abs(s.transform.a) / 0.6
        gt = B.L.load_tile(tile, B.TRUTH)
        d = gt["dsm"].copy(); d[~gt["valid_mask"]] = np.nan
        t = ground_truth_ndsm(d, gt["cls"]); fin = np.isfinite(t)
        small = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        work = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        p = bb.predict_tiled(Image.fromarray(work), raw_metric=True).astype(np.float32)
        p = cv2.resize(p, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
        m = fin & np.isfinite(p); e = p[m] - t[m]
        sq += float((e ** 2).sum()); ab += float(np.abs(e).sum()); n += int(m.sum())
        P.append(p[m][::37]); T.append(t[m][::37])
        nl, lab, st, _ = cv2.connectedComponentsWithStats(((gt["cls"] == 6) & fin).astype(np.uint8), connectivity=4)
        for i in range(1, nl):
            if st[i, cv2.CC_STAT_AREA] >= 40:
                bi = lab == i
                Bl.append((float(np.median(t[bi])), float(np.median(p[bi]))))
                Bt.append(tile)
        print(f"  dfc {tile}", flush=True)
    b = np.array(Bl); be = b[:, 1] - b[:, 0]
    return {"rmse": (sq / n) ** 0.5, "mae": ab / n,
            "corr": float(np.corrcoef(np.concatenate(P), np.concatenate(T))[0, 1]),
            "bldg_rmse": float(np.sqrt(np.mean(be ** 2))), "bldg_bias": float(be.mean()),
            "n_buildings": int(len(b)),
            # per building: tile, LiDAR median, predicted median (for calibration)
            "buildings": [[tl, round(x, 3), round(y, 3)] for tl, (x, y) in zip(Bt, Bl)]}


def swiss(bb, tag):
    out = {}
    for land, place, lat, lon in LL.SITES:
        tile, fd, ft, fi = LL.pick(lat, lon)
        dsm_p, dtm_p, img_p = LL.get(LL.asset(fd, 0.5)), LL.get(LL.asset(ft, 0.5)), LL.get(LL.asset(fi, 0.1))
        with rasterio.open(dsm_p) as s:
            dsm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
            x0, ytop = s.transform.c, s.transform.f
        with rasterio.open(dtm_p) as s:
            dtm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        truth = LL.block(np.clip(dsm - dtm, 0, None), 2)
        cache = os.path.join(LL.WORK, f"pred_{tile}_{tag}_blocks.npy")
        if os.path.exists(cache):
            pred = np.load(cache)
        else:
            with rasterio.open(img_p) as s:
                rgb = s.read([1, 2, 3]).transpose(1, 2, 0)
                src_gsd = abs(s.transform.a)
            n06 = int(round(rgb.shape[1] * src_gsd / 0.6))
            img06 = cv2.resize(rgb, (n06, n06), interpolation=cv2.INTER_AREA)
            del rgb
            work = cv2.resize(img06, (n06 * 2, n06 * 2), interpolation=cv2.INTER_CUBIC)
            pred = bb.predict_tiled(Image.fromarray(work), raw_metric=True).astype(np.float32)
            pred = cv2.resize(np.maximum(pred, 0), truth.shape[::-1], interpolation=cv2.INTER_AREA)
            np.save(cache, pred)
        dd = dem_source.sample_grid(x0, ytop, truth.shape[0], 1.0, "EPSG:2056", dem_source="glo30",
                                    width_px=truth.shape[1])
        surf, _ = anchoring.absolute_surface(np.asarray(dd["dem"], np.float32), pred, 1.0,
                                             anchoring.DEFAULT_K["glo30"])
        out[land] = {"agl": LL.stats(pred, truth), "absolute": LL.stats(surf, LL.block(dsm, 2))}
        print(f"  swiss {land}", flush=True)
    return out


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("DW_THREADS", "6")))
    res = {}
    for ck in sys.argv[1:]:
        ck = os.path.abspath(ck)
        dm.DepthBackbone.METRIC_CKPT = ck
        # the backbone is a flyweight keyed on its constructor args, not on
        # METRIC_CKPT: without this every checkpoint after the first would be
        # scored with the first one's weights
        dm.DepthBackbone._instances.clear()
        bb = dm.DepthBackbone(metric=True)
        tag = os.path.basename(ck)
        print(f"== {tag}", flush=True)
        res[tag] = {"swiss": swiss(bb, tag), "dfc_0.6m": dfc(bb)}
        json.dump(res, open(os.path.join(ROOT, "final_out", "compare_ckpt.json"), "w"), indent=2)
    for tag, r in res.items():
        d = r["dfc_0.6m"]
        print(f"\n{tag}: DFC 0.6 m pixel RMSE {d['rmse']:.3f} MAE {d['mae']:.3f} corr {d['corr']:.3f} | "
              f"building RMSE {d['bldg_rmse']:.3f} bias {d['bldg_bias']:+.3f}")
        for land, s in r["swiss"].items():
            print(f"   {land:9s} AGL RMSE {s['agl']['rmse']:.2f} bias {s['agl']['bias']:+.2f} | "
                  f"absolute RMSE {s['absolute']['rmse']:.2f} bias {s['absolute']['bias']:+.2f}")


if __name__ == "__main__":
    main()
