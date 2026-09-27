"""Our model on the competing entry's own benchmark, by their own rules.

    python scripts/head_to_head.py            # native 0.3 m and ISRO's 0.6 m

zaidnansari2011/sih2026-depthwizard publishes, for DFC2019 Track 1, 80 whole
validation tiles: per-building RMSE 3.464 m, whole-tile RMSE 6.008 m / MAE
1.681 m / r 0.843, buildings <= 20 m 1.667 m, bias above 20 m -15.64 m.
This reproduces their tile list and their metrics exactly:

  * split  (tools/prepare_data.py): regions = sorted({tile.rsplit('_',1)[0]})
           over tiles with RGB and AGL; np.random.default_rng(1337).shuffle;
           first round(0.15 n) test, next round(0.15 n) val;
  * tiles  (tools/evaluate.py): val-region tiles in sorted RGB-file order,
           np.random.default_rng(1337).choice(80, replace=False);
  * whole tile: every pixel where truth and prediction are finite, pooled;
  * per building (depthwizard/metrics.py building_instances): class 6,
           8-connected, >= 25 m2 at 0.3 m, median height on both sides.

Contamination is reported, not hidden: Track 1 and Track 3 cover the same
Jacksonville areas, and our model trained on 35 of them (training/dfc_tiles).
Tiles in those areas are scored and flagged; the clean subset excludes them.
Omaha was never seen by our model.

Tiles are read straight from the IEEE DataPort zips through /vsizip/.
"""
import json
import os
import sys
import zipfile

import cv2
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import depth_model as dm  # noqa: E402

DL = os.path.join(os.path.expanduser("~"), "Downloads")
RGB_ZIP = os.path.join(DL, "Train-Track1-RGB.zip")
TRUTH_ZIP = os.path.join(DL, "Train-Track1-Truth.zip")
SEED, N_TILES, GSD = 1337, 80, 0.3
THEIRS = {"bldg_rmse": 3.464, "rmse": 6.008, "mae": 1.681, "corr": 0.843,
          "bldg_le20_rmse": 1.667, "bias_gt20": -15.64, "at_0.6m_bldg_rmse": 3.999}


def trained_areas():
    s = set()
    for sub in ("train", "val"):
        d = os.path.join(ROOT, "training", "dfc_tiles", sub)
        if os.path.isdir(d):
            s |= {"_".join(f.split("_")[1:3]) for f in os.listdir(d) if f.startswith("DFC_")}
    return s


def their_tiles():
    rgb = {os.path.basename(n)[:-8]: n for n in zipfile.ZipFile(RGB_ZIP).namelist() if n.endswith("_RGB.tif")}
    tz = zipfile.ZipFile(TRUTH_ZIP).namelist()
    agl = {os.path.basename(n)[:-8]: n for n in tz if n.endswith("_AGL.tif")}
    cls = {os.path.basename(n)[:-8]: n for n in tz if n.endswith("_CLS.tif")}
    paired = sorted(t for t in rgb if t in agl)
    regions = sorted({t.rsplit("_", 1)[0] for t in paired})
    order = list(regions)
    np.random.default_rng(SEED).shuffle(order)
    n = len(order)
    n_test, n_val = max(1, round(0.15 * n)), max(1, round(0.15 * n))
    val_regions = set(order[n_test:n_test + n_val])
    cand = [t for t in sorted(rgb) if t.rsplit("_", 1)[0] in val_regions and t in agl]
    pick = list(np.random.default_rng(SEED).choice(cand, min(N_TILES, len(cand)), replace=False))
    return pick, rgb, agl, cls, len(val_regions)


def read(zp, inner):
    with rasterio.open(f"/vsizip/{zp}/{inner}") as s:
        return s.read()


def building_instances(pred, target, cls):
    mask = (cls == 6) & np.isfinite(pred) & np.isfinite(target)
    if not mask.any():
        return np.array([]), np.array([])
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.array([]), np.array([])
    idx = np.arange(1, n + 1)
    counts = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = counts * GSD * GSD >= 25.0
    t_med = np.asarray(ndimage.median(target, lab, idx))
    p_med = np.asarray(ndimage.median(pred, lab, idx))
    good = keep & np.isfinite(t_med) & np.isfinite(p_med)
    return p_med[good], t_med[good]


def to_rgb8(a):
    a = np.transpose(a[:3], (1, 2, 0))
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a[np.isfinite(a)], [2, 98])
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return a


def main():
    tiles, rgb, agl, cls, n_val_regions = their_tiles()
    seen = trained_areas()
    print(f"{len(tiles)} tiles from {n_val_regions} val regions; "
          f"{sum(t.rsplit('_', 1)[0] in seen for t in tiles)} in areas our model trained on", flush=True)
    bb = dm.DepthBackbone(metric=True)
    acc = {c: {"all": {"P": [], "T": [], "BP": [], "BT": []}, "clean": {"P": [], "T": [], "BP": [], "BT": []}}
           for c in ("0.3m", "0.6m")}
    for i, t in enumerate(tiles):
        img = to_rgb8(read(RGB_ZIP, rgb[t]))
        truth = read(TRUTH_ZIP, agl[t])[0].astype(np.float32)
        cl = read(TRUTH_ZIP, cls[t])[0] if t in cls else None
        clean = t.rsplit("_", 1)[0] not in seen
        for cond in ("0.3m", "0.6m"):
            work = img
            if cond == "0.6m":
                small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
                work = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            pred = bb.predict_tiled(Image.fromarray(work), raw_metric=True).astype(np.float32)
            if pred.shape != truth.shape:
                pred = cv2.resize(pred, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_LINEAR)
            m = np.isfinite(truth) & np.isfinite(pred)
            bp, bt = building_instances(pred, truth, cl) if cl is not None else (np.array([]), np.array([]))
            for key in (("all", "clean") if clean else ("all",)):
                a = acc[cond][key]
                a["P"].append(pred[m][::7]); a["T"].append(truth[m][::7])
                a["BP"].append(bp); a["BT"].append(bt)
        print(f"  {i + 1}/{len(tiles)} {t}{'' if clean else '  (area seen in training)'}", flush=True)

    out = {"theirs": THEIRS, "n_tiles": len(tiles),
           "n_clean": sum(t.rsplit("_", 1)[0] not in seen for t in tiles), "ours": {}}
    for cond, d in acc.items():
        out["ours"][cond] = {}
        for key, a in d.items():
            if not a["P"]:
                continue
            p, t = np.concatenate(a["P"]), np.concatenate(a["T"])
            e = p - t
            bp, bt = np.concatenate(a["BP"]), np.concatenate(a["BT"])
            be = bp - bt
            r = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                 "corr": float(np.corrcoef(p, t)[0, 1]), "n_buildings": int(bp.size),
                 "bldg_rmse": float(np.sqrt((be ** 2).mean())), "bldg_bias": float(be.mean())}
            lo, hi = bt <= 20, bt > 20
            if lo.any():
                r["bldg_le20_rmse"] = float(np.sqrt((be[lo] ** 2).mean()))
            if hi.any():
                r["bias_gt20"] = float(be[hi].mean()); r["n_gt20"] = int(hi.sum())
            out["ours"][cond][key] = r
    json.dump(out, open(os.path.join(ROOT, "final_out", "head_to_head.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
