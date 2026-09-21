"""
Measure building-class over-recall against DFC2019 LiDAR labels.

The comment in segmentation.py already documents the failure mode: canopy
elevated like a roof gets swept into the building mask by the height cue, and
a treed campus reads ~40% building against ~8% actually built. This script
measures that on the same 10-tile set the blind benchmark uses, so the size
of the problem (and of any fix) is a number, not a guess.

Ground truth CLS codes (ASPRS, confirmed against the actual files): 2=ground,
5=high vegetation, 6=building. Runs the SAME orthorectification path as
benchmark_blind.py's blind (no-ground-truth) prediction, and only reads CLS
afterward to score -- CLS never touches the prediction.

    python scripts/eval_segmentation.py [n_tiles]
"""
import glob
import os
import sys

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import segmentation as seg
from depth_model import DepthBackbone
import benchmark_blind as bb

TRUTH = bb.TRUTH


def cls_for_tile(tile: str) -> np.ndarray:
    path = os.path.join(TRUTH, f"{tile}_CLS.tif")
    with rasterio.open(path) as d:
        return d.read(1)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(bb.TRUTH, "*_DSM.tif"))})
    tiles = [t for t in tiles if glob.glob(os.path.join(bb.RGB, f"{t}_*_RGB.tif"))][:n]

    model = DepthBackbone()
    rows = []
    for tile in tiles:
        try:
            o = bb._orthorectify_blind(tile, 1024)
            pil = Image.fromarray(o["image"])
            height = model.predict_tiled(pil)
            labels, stats = seg.segment(o["image"], height=height)
        except Exception as exc:
            print(f"{tile}: SKIP -- {type(exc).__name__}: {exc}")
            continue

        cls = cls_for_tile(tile)
        cls = np.array(Image.fromarray(cls).resize(
            (labels.shape[1], labels.shape[0]), Image.NEAREST))

        gt_building = cls == 6
        gt_veg = cls == 5
        pred_building = labels == seg.CLASS_IDX["building"]

        tp = (pred_building & gt_building).sum()
        fp_veg = (pred_building & gt_veg).sum()
        fp_other = (pred_building & ~gt_building & ~gt_veg).sum()
        fn = (~pred_building & gt_building).sum()

        precision = tp / max(pred_building.sum(), 1)
        recall = tp / max(gt_building.sum(), 1)
        canopy_frac_of_pred = fp_veg / max(pred_building.sum(), 1)

        row = {
            "tile": tile,
            "pred_building_pct": round(100 * pred_building.sum() / labels.size, 1),
            "true_building_pct": round(100 * gt_building.sum() / labels.size, 1),
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "canopy_frac_of_pred_building": round(float(canopy_frac_of_pred), 3),
        }
        rows.append(row)
        print(f"{tile}: pred_building={row['pred_building_pct']}%  "
              f"true_building={row['true_building_pct']}%  "
              f"precision={row['precision']}  recall={row['recall']}  "
              f"canopy_leak={row['canopy_frac_of_pred_building']}")

    if rows:
        print()
        print(f"mean precision       {np.mean([r['precision'] for r in rows]):.3f}")
        print(f"mean recall          {np.mean([r['recall'] for r in rows]):.3f}")
        print(f"mean canopy leakage  {np.mean([r['canopy_frac_of_pred_building'] for r in rows]):.3f}  "
              f"of predicted building area is actually high-vegetation by LiDAR")


if __name__ == "__main__":
    main()
