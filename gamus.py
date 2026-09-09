"""GAMUS dataset: download a subset, and serve RGB/height pairs for training.

GAMUS is the dataset the SAC reference repository recommends for this problem
statement, specifically for "addressing domain gaps between natural and top-down
imagery". Measured on the shipped backbone it supplies exactly that: height
correlation 0.299 against true nDSM on unseen cities, with one tile NEGATIVE
(-0.248). The gap is real and this is the data that closes it.

WHAT IT GIVES THAT DFC2019 DOES NOT

  * nDSM in METRES (AGL, above ground level), paired 1:1 with RGB. Training on
    it produces a model that predicts absolute height directly, which removes
    the Tier C scale assumption ("the tallest structure is about 40 m") and the
    per-tile affine alignment the benchmark currently relies on.
  * Six-class semantic labels including building and road.
  * Three cities (PHL, NYC, DC on the HuggingFace copy) and no Jacksonville, so
    it does not overlap the tiles this project was tuned on.

MEASURED PROPERTIES OF THE LABELS, which drive the choices below:

    p50    1.67 m        51.3% of pixels below 2 m
    p90   24.46 m
    p99   36.91 m
    max  146.91 m        heavy tail: a few pixels are 4x the 99th percentile
    min   -5.00 m        AGL IS NOT NON-NEGATIVE

That last one matters. AGL is DSM minus DTM and both carry error, so ground
pixels scatter either side of zero. Clamping to >= 0 would teach the model that
the noise floor is one-sided; the targets are left signed and only the extreme
tail is clipped.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import List, Optional, Tuple

import numpy as np

REPO = "earthflow/GAMUS"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
API = f"https://huggingface.co/api/datasets/{REPO}"
DATA_DIR = "gamus_data"

# Clip the target's upper tail. 146 m is a real building, but a handful of such
# pixels in a batch dominate an L1 loss and stall everything else. 120 m keeps
# every genuine high-rise in these cities while bounding the gradient.
HEIGHT_CLIP_M = 120.0

# Below this the model is being asked to resolve DSM/DTM noise rather than
# structure; used only for reporting, never to mask the loss.
GROUND_BAND_M = 2.0


def list_tiles(split: str = "train") -> List[str]:
    """Tile stems available for a split, e.g. 'DC_02_26'."""
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)
    names = [s["rfilename"] for s in meta.get("siblings", [])]
    pat = re.compile(rf"^images/{split}/(.+)_RGB\.h5$")
    return sorted(m.group(1) for m in (pat.match(n) for n in names) if m)


def download(split: str = "train", limit: Optional[int] = None,
             cities: Optional[List[str]] = None,
             out_dir: str = DATA_DIR, verbose: bool = True) -> List[str]:
    """Fetch RGB + AGL (+ CLS) for a subset of tiles.

    The full dataset is 80 GB. Every tile is a separate file, so a subset can be
    pulled without it -- which is what makes this trainable on one machine.
    """
    stems = list_tiles(split)
    if cities:
        want = tuple(c.upper() + "_" for c in cities)
        stems = [s for s in stems if s.upper().startswith(want)]
    if limit:
        # Spread across the list rather than taking a prefix: tiles are named
        # by grid position, so the first N are one corner of one city.
        step = max(1, len(stems) // limit)
        stems = stems[::step][:limit]

    dest = os.path.join(out_dir, split)
    os.makedirs(dest, exist_ok=True)
    got = []
    for i, stem in enumerate(stems, 1):
        ok = True
        for kind, folder in (("RGB", "images"), ("AGL", "heights"), ("CLS", "classes")):
            path = os.path.join(dest, f"{stem}_{kind}.h5")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            url = f"{BASE}{folder}/{split}/{stem}_{kind}.h5"
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                if kind == "CLS":
                    continue          # labels are optional for height training
                print(f"  {stem}: {kind} failed ({e})")
                ok = False
                break
        if ok:
            got.append(stem)
        if verbose and (i % 25 == 0 or i == len(stems)):
            mb = sum(os.path.getsize(os.path.join(dest, f))
                     for f in os.listdir(dest)) / 1e6
            print(f"  {i}/{len(stems)} tiles, {mb:.0f} MB")
    return got


def _read(path: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as h:
        return h["image"][:]


class GamusHeight:
    """RGB/height crops for training. A torch Dataset when torch is present.

    Crops are taken at native resolution rather than resizing the 1024 tile down
    to the network's 518. Absolute height regression depends on apparent scale:
    the model learns what a shadow of a given length, or a roof of a given pixel
    size, means in metres. Resizing changes the ground sample distance and
    therefore changes the answer, so it would teach a relationship that does not
    hold at inference.
    """

    def __init__(self, root: str = DATA_DIR, split: str = "train",
                 crop: int = 518, augment: bool = True,
                 crops_per_tile: int = 4):
        self.dir = os.path.join(root, split)
        self.crop = crop
        self.augment = augment
        self.crops_per_tile = max(1, crops_per_tile)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(
                f"{self.dir} not found -- run: python gamus.py download --split {split}")
        self.stems = sorted(
            f[:-len("_RGB.h5")] for f in os.listdir(self.dir) if f.endswith("_RGB.h5"))
        if not self.stems:
            raise RuntimeError(f"no tiles in {self.dir}")
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_tile

    def _load(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        rgb = _read(os.path.join(self.dir, f"{stem}_RGB.h5"))
        agl = _read(os.path.join(self.dir, f"{stem}_AGL.h5")).astype(np.float32)
        return rgb, agl

    def __getitem__(self, idx: int):
        stem = self.stems[idx // self.crops_per_tile]
        rgb, agl = self._load(stem)
        H, W = agl.shape
        c = min(self.crop, H, W)

        if self.augment:
            y = np.random.randint(0, H - c + 1)
            x = np.random.randint(0, W - c + 1)
        else:
            y, x = (H - c) // 2, (W - c) // 2
        rgb = rgb[y:y + c, x:x + c]
        agl = agl[y:y + c, x:x + c]

        if self.augment:
            # Flips and 90-degree rotations only. A nadir image has no up, so
            # these are label-preserving; a vertical flip of a ground-level
            # photo would not be.
            k = np.random.randint(4)
            if k:
                rgb, agl = np.rot90(rgb, k, (0, 1)), np.rot90(agl, k, (0, 1))
            if np.random.rand() < 0.5:
                rgb, agl = rgb[:, ::-1], agl[:, ::-1]
            # Mild photometric jitter. Deliberate: final evaluation is on ISRO
            # RGB-band imagery with different radiometry from these US cities,
            # and a model that has only ever seen one sensor's response will
            # transfer badly.
            if np.random.rand() < 0.8:
                gain = np.float32(np.random.uniform(0.88, 1.12))
                bias = np.float32(np.random.uniform(-12, 12))
                rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255)

        rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        rgb = np.transpose(rgb, (2, 0, 1))

        agl = np.ascontiguousarray(agl, dtype=np.float32)
        agl = np.clip(agl, -HEIGHT_CLIP_M, HEIGHT_CLIP_M)
        valid = np.isfinite(agl).astype(np.float32)
        agl = np.nan_to_num(agl, nan=0.0)

        try:
            import torch
            return (torch.from_numpy(rgb), torch.from_numpy(agl),
                    torch.from_numpy(valid))
        except ImportError:
            return rgb, agl, valid


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="GAMUS subset downloader")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--split", default="train", choices=["train", "val", "test"])
    d.add_argument("--limit", type=int, default=400)
    d.add_argument("--cities", nargs="*", default=None,
                   help="e.g. --cities DC PHL")
    sub.add_parser("list").add_argument("--split", default="train")
    a = ap.parse_args()

    if a.cmd == "list":
        t = list_tiles(a.split)
        print(f"{len(t)} tiles in {a.split}")
        import collections
        c = collections.Counter(s.split("_")[0] for s in t)
        for k, v in c.most_common():
            print(f"  {k:<6} {v}")
        return

    print(f"downloading up to {a.limit} {a.split} tiles"
          + (f" from {a.cities}" if a.cities else ""))
    got = download(a.split, a.limit, a.cities)
    print(f"{len(got)} tiles ready in {os.path.join(DATA_DIR, a.split)}")


if __name__ == "__main__":
    main()
