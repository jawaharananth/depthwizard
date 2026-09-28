"""DFC2019 Track 1 -> training tiles for run 05 (tall buildings, Omaha + Jacksonville).

    python training/prepare_dfc_track1.py --out training/dfc_track1 --views 6

Track 1 is 108 areas (Jacksonville, Omaha), each photographed 19-42 times by
WorldView-3, 1024 px at 0.3 m, with airborne-LiDAR height above ground per view.
Held out, never written:
  * the competing entry's validation and test areas, reproduced exactly from
    their split (scripts/head_to_head.py: seed 1337, 15 % test, 15 % val), so
    the 80-tile head-to-head stays a clean comparison after run 05;
  * our own 18 benchmark areas (dfc2019_height.benchmark_tiles()) and every
    area in our Track 3 validation split (the run's model selection).
Of each remaining area, up to --views dates, evenly spread through the list.
Written as <name>_RGB.h5 / _AGL.h5 ("image" dataset, uint8 / float16, gzip) --
the layout gamus.GamusHeight reads. Missing heights stay NaN (masked in the loss).
"""
import argparse
import json
import os
import sys
import zipfile

import h5py
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dfc2019_height as DH  # noqa: E402

DL = os.path.join(os.path.expanduser("~"), "Downloads")
RGB_ZIP = os.path.join(DL, "Train-Track1-RGB.zip")
TRUTH_ZIP = os.path.join(DL, "Train-Track1-Truth.zip")
SEED = 1337


def held_out_regions():
    rgb = {os.path.basename(n)[:-8]: n for n in zipfile.ZipFile(RGB_ZIP).namelist() if n.endswith("_RGB.tif")}
    agl = {os.path.basename(n)[:-8] for n in zipfile.ZipFile(TRUTH_ZIP).namelist() if n.endswith("_AGL.tif")}
    paired = sorted(t for t in rgb if t in agl)
    regions = sorted({t.rsplit("_", 1)[0] for t in paired})
    order = list(regions)
    np.random.default_rng(SEED).shuffle(order)
    n = len(order)
    n_test, n_val = max(1, round(0.15 * n)), max(1, round(0.15 * n))
    theirs = set(order[:n_test + n_val])
    ours = set(DH.benchmark_tiles())
    vdir = os.path.join(ROOT, "training", "dfc_tiles", "val")
    ours |= {"_".join(f.split("_")[1:3]) for f in os.listdir(vdir) if f.startswith("DFC_")} if os.path.isdir(vdir) else set()
    return rgb, paired, regions, theirs, ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "dfc_track1"))
    ap.add_argument("--views", type=int, default=6)
    a = ap.parse_args()
    rgb, paired, regions, theirs, ours = held_out_regions()
    train_regions = [r for r in regions if r not in theirs and r not in ours]
    out = os.path.join(a.out, "train")
    os.makedirs(out, exist_ok=True)
    written = []
    for reg in train_regions:
        views = sorted(t for t in paired if t.rsplit("_", 1)[0] == reg)
        if len(views) > a.views:
            idx = np.linspace(0, len(views) - 1, a.views).round().astype(int)
            views = [views[i] for i in sorted(set(idx))]
        for t in views:
            name = f"DFC1_{t}"
            rp, hp = os.path.join(out, name + "_RGB.h5"), os.path.join(out, name + "_AGL.h5")
            if os.path.exists(rp) and os.path.exists(hp):
                written.append(name); continue
            with rasterio.open(f"/vsizip/{RGB_ZIP}/{rgb[t]}") as s:
                img = s.read([1, 2, 3]).transpose(1, 2, 0)
            with rasterio.open(f"/vsizip/{TRUTH_ZIP}/Track1-Truth/{t}_AGL.tif") as s:
                h = s.read(1).astype(np.float32)
                if s.nodata is not None:
                    h[h == s.nodata] = np.nan
            h[(h < -5) | (h > 400)] = np.nan
            if img.dtype != np.uint8 or img.shape[:2] != h.shape:
                continue
            with h5py.File(rp, "w") as f:
                f.create_dataset("image", data=img, compression="gzip", compression_opts=4)
            with h5py.File(hp, "w") as f:
                f.create_dataset("image", data=h.astype(np.float16), compression="gzip", compression_opts=4)
            written.append(name)
        print(f"  {reg}: {len(views)} views", flush=True)
    rep = {"areas_total": len(regions), "held_out_theirs": sorted(theirs), "held_out_ours": sorted(ours & set(regions)),
           "train_areas": train_regions, "tiles": len(written), "views_per_area": a.views}
    json.dump(rep, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(f"{len(written)} tiles from {len(train_regions)} areas "
          f"({len(theirs)} competitor val/test and {len(ours & set(regions))} of ours held out)")


if __name__ == "__main__":
    main()
