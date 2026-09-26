"""DFC2019 Track 3 -> GAMUS-format training tiles (satellite domain).

    python training/prepare_dfc.py --out training/dfc_tiles --views 4 --px 768

GAMUS is aerial imagery; the evaluation imagery ISRO will use is satellite.
A GAMUS-only model scored no better than the old checkpoint on blind DFC2019
tiles, so the second training set is DFC2019 itself: each tile is
orthorectified from several of its satellite views (different dates, sun and
look angles -- real augmentation, not synthetic), paired with the LiDAR
height above ground, and written as <stem>_v<k>_RGB.h5 / _AGL.h5 with an
"image" dataset -- the exact layout gamus.GamusHeight already reads.

Split: dfc2019_height.split_tiles(), which removes the 18 benchmark tiles
(and with them every tile scripts/benchmark_blind.py scores) from both train
and val. 768 px over the 256 m tile gives 0.33 m, GAMUS's own ground sample.
"""
import argparse
import glob
import os
import sys

import cv2
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dfc2019_height as D          # noqa: E402
import dfc2019_loader as L          # noqa: E402
import ortho                        # noqa: E402
import rescore_baseline             # noqa: E402

META_DIR = os.path.join(ROOT, "dfc2019_data", "metadata", "Track3-Metadata")


def pick_views(tile, rgb_dir, meta_dir, k, max_off_nadir=30.0):
    """Up to k views under max_off_nadir, preferring distinct dates."""
    views = []
    for p in sorted(glob.glob(os.path.join(rgb_dir, f"{tile}_*_RGB.tif"))):
        try:
            imd = L.parse_imd(L.imd_path_for_rgb(p, meta_dir))
        except Exception:
            continue
        a = imd.get("meanOffNadirViewAngle")
        if a is not None and a <= max_off_nadir:
            views.append((a, p, str(imd.get("firstLineTime", ""))[:10]))
    views.sort()
    out, dates = [], set()
    for a, p, d in views:                      # distinct dates first
        if d not in dates:
            out.append(p); dates.add(d)
        if len(out) == k:
            return out
    for a, p, d in views:                      # then fill
        if p not in out:
            out.append(p)
        if len(out) == k:
            break
    return out


def truth_on_frame(r):
    """LiDAR nDSM placed on the ortho frame at its real inset; NaN elsewhere."""
    ndsm = rescore_baseline.ground_truth_ndsm(r["truth"]["dsm"], r["truth"]["cls"])
    H, W = r["image"].shape[:2]
    inset = int(r.get("truth_inset_px") or 0)
    inner = (H - 2 * inset, W - 2 * inset)
    frame = np.full((H, W), np.nan, np.float32)
    frame[inset:inset + inner[0], inset:inset + inner[1]] = cv2.resize(
        ndsm.astype(np.float32), (inner[1], inner[0]), interpolation=cv2.INTER_LINEAR)
    return frame


def write(path, arr):
    with h5py.File(path, "w") as h:
        h.create_dataset("image", data=arr, compression="gzip", compression_opts=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "dfc_tiles"))
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--px", type=int, default=768)
    a = ap.parse_args()

    train, val = D.split_tiles(D.available_tiles())
    for split, tiles in (("train", train), ("val", val)):
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        n = 0
        for t in tiles:
            for k, p in enumerate(pick_views(t, D.RGB_DIR, META_DIR, a.views)):
                stem = f"DFC_{t}_v{k}"
                if os.path.exists(os.path.join(d, stem + "_AGL.h5")):
                    n += 1
                    continue
                try:
                    r = ortho.orthorectify(t, D.TRUTH_DIR, D.RGB_DIR, META_DIR,
                                           out_px=a.px, rgb_path=p)
                except Exception as e:
                    print(f"  {t} v{k}: skipped ({type(e).__name__}: {e})")
                    continue
                agl = truth_on_frame(r)
                if np.isfinite(agl).mean() < 0.5:
                    print(f"  {t} v{k}: skipped (truth covers {np.isfinite(agl).mean():.0%})")
                    continue
                write(os.path.join(d, stem + "_RGB.h5"), r["image"].astype(np.uint8))
                write(os.path.join(d, stem + "_AGL.h5"), agl.astype(np.float16))
                n += 1
            print(f"{split} {t}: {n} views so far", flush=True)
        print(f"{split}: {n} tile-views from {len(tiles)} tiles")


if __name__ == "__main__":
    main()
