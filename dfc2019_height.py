"""DFC2019 RGB/height crops for training, in the same shape GamusHeight yields.

WHY THIS EXISTS

htc_bins.py / train_height.py --htc trains on GAMUS alone. GAMUS is three US
cities (PHL, NYC, DC); DFC2019 Track 3 (Jacksonville, Omaha) is a second,
independently-collected dataset with a different sensor (WorldView-3) and a
different ground-truth pipeline (airborne LiDAR DSM, not the DTM-subtracted
AGL GAMUS ships). Training on both, and holding out whole tiles of each for
validation, is a real cross-dataset check rather than a second sample from
the same distribution -- the same reason train_height.py already treats
DFC2019 as this project's independent scoring set elsewhere (rescore_baseline.py).

This module does NOT invent a new ground-truth pipeline. It reuses exactly
the two functions already validated throughout this project:
ortho.orthorectify() (RPC-based RGB/truth alignment, used for every DFC2019
benchmark scene built this session) and rescore_baseline.ground_truth_ndsm()
(DSM minus ground-class DTM, the same nDSM every rescoring in this project is
measured against). A tile's ortho+nDSM pair is computed once and cached in
memory -- orthorectify's cubic RGB resampling is the expensive part, and a
training epoch revisits every tile several times for different crops.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

import dfc2019_loader as L
import ortho
import rescore_baseline

TRUTH_DIR = "dfc2019_data/truth/Track3-Truth"
RGB_DIR = "dfc2019_data/rgb/Track3-RGB-1"

# Tiles with the most sun-glare / off-nadir distortion in the RGB, measured
# during this session's ortho work; excluded rather than silently degrading
# every batch that happens to sample them.
KNOWN_BAD_TILES: Tuple[str, ...] = ()


def benchmark_tiles(truth_dir: str = TRUTH_DIR, n: int = 18) -> List[str]:
    """The exact tile set rescore_baseline.py scores as this project's
    independent DFC2019 check -- same selection, same order (sorted stems,
    first `n`), so this always matches whatever that script is actually
    scoring, rather than a copy of a list that can drift out of sync with it.
    """
    import glob
    stems = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(truth_dir, "*_DSM.tif"))})
    return stems[:n]


def available_tiles(truth_dir: str = TRUTH_DIR) -> List[str]:
    return L.list_available_tiles(truth_dir)


def split_tiles(tiles: List[str], val_every: int = 5,
                exclude_benchmark: bool = True,
                truth_dir: str = TRUTH_DIR) -> Tuple[List[str], List[str]]:
    """Deterministic tile-level split: every `val_every`-th tile (sorted) is
    held out. Splitting by whole tile, not by pixel, is what makes this a
    real generalisation test -- a random pixel-level split would let crops
    from the same tile leak between train and val.

    exclude_benchmark=True (the default, and the only safe choice for
    anything that might get licensed into production) removes
    rescore_baseline.py's 18 benchmark tiles from BOTH splits entirely.
    Measured on the full available set: 14 of those 18 tiles land in what
    would otherwise be the train split and 4 in val -- so without this
    exclusion, training on DFC2019 trains directly on the tiles this
    project's headline RMSE/MAE numbers are scored against. See
    train_height.py's module docstring for why that number would then be
    meaningless.
    """
    tiles = sorted(tiles)
    if exclude_benchmark:
        bench = set(benchmark_tiles(truth_dir))
        tiles = [t for t in tiles if t not in bench]
    val = tiles[::val_every]
    train = [t for t in tiles if t not in set(val)]
    return train, val


class DFC2019Height:
    """Same (rgb, agl, valid) contract as gamus.GamusHeight."""

    def __init__(self, stems: List[str], truth_dir: str = TRUTH_DIR,
                rgb_dir: str = RGB_DIR, crop: int = 518, augment: bool = True,
                crops_per_tile: int = 4, out_px: int = 1024):
        self.stems = [s for s in stems if s not in KNOWN_BAD_TILES]
        if not self.stems:
            raise RuntimeError("no DFC2019 tiles given")
        self.truth_dir = truth_dir
        self.rgb_dir = rgb_dir
        self.crop = crop
        self.augment = augment
        self.crops_per_tile = max(1, crops_per_tile)
        self.out_px = out_px
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_tile

    def _load(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        if stem in self._cache:
            return self._cache[stem]

        r = ortho.orthorectify(stem, self.truth_dir, self.rgb_dir, self.rgb_dir,
                               out_px=self.out_px)
        rgb = r["image"]                                   # (out_px, out_px, 3) uint8
        dsm, cls = r["truth"]["dsm"], r["truth"]["cls"]
        ndsm = rescore_baseline.ground_truth_ndsm(dsm, cls)

        if ndsm.shape != rgb.shape[:2]:
            import cv2
            ndsm = cv2.resize(ndsm, (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        self._cache[stem] = (rgb, ndsm.astype(np.float32))
        return self._cache[stem]

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
            k = np.random.randint(4)
            if k:
                rgb, agl = np.rot90(rgb, k, (0, 1)), np.rot90(agl, k, (0, 1))
            if np.random.rand() < 0.5:
                rgb, agl = rgb[:, ::-1], agl[:, ::-1]
            if np.random.rand() < 0.8:
                gain = np.float32(np.random.uniform(0.88, 1.12))
                bias = np.float32(np.random.uniform(-12, 12))
                rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255)

        rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        rgb = np.transpose(rgb, (2, 0, 1))

        agl = np.ascontiguousarray(agl, dtype=np.float32)
        # rescore_baseline.ground_truth_ndsm already zeroes non-ground-truth
        # pixels via NaN; the orthorectified frame beyond the truth tile's
        # own extent (see ortho.orthorectify's `extent_m` padding) has no
        # truth at all and must not be scored either.
        valid = np.isfinite(agl).astype(np.float32)
        agl = np.clip(np.nan_to_num(agl, nan=0.0), -20.0, 120.0)

        import torch
        return torch.from_numpy(rgb), torch.from_numpy(agl), torch.from_numpy(valid)


if __name__ == "__main__":
    tiles = available_tiles()
    train, val = split_tiles(tiles)
    print(f"{len(tiles)} DFC2019 tiles available: {len(train)} train / {len(val)} val")
    ds = DFC2019Height(train[:2], crop=256, crops_per_tile=1, augment=False)
    rgb, agl, valid = ds[0]
    print("sample:", rgb.shape, agl.shape, valid.shape,
          "agl range", float(agl.min()), float(agl.max()),
          "valid frac", float(valid.mean()))
