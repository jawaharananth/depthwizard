"""
Benchmark across every usable tile.

WHY THIS MATTERS MORE THAN ANY SINGLE NUMBER

Every accuracy figure this project has quoted came from one tile. One tile
cannot distinguish a method that works from a scene that happened to suit it.
Running the whole set is what turns "2.63 m MAE" into a claim with a spread
behind it -- and the spread is the honest part, because it shows where the
method fails as well as where it succeeds.

This uses the monocular path deliberately: MVS needs a ~15 minute plane sweep
per tile, so a full-set MVS benchmark is hours. The monocular path is what runs
on any tile today, and its numbers are directly comparable across all of them.
"""
import glob
import json
import os
import sys
import time

import numpy as np

import rescore_baseline as rb


def main(limit=None):
    truth = rb.TRUTH
    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(truth, "*_DSM.tif"))})
    tiles = [t for t in tiles if glob.glob(os.path.join(rb.RGB, f"{t}_*_RGB.tif"))]
    if limit:
        tiles = tiles[:limit]
    print(f"benchmarking {len(tiles)} tiles\n")
    rb.main(len(tiles), tiled=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
