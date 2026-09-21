"""
Build a Mumbai demo scene using the Dabra & Upadhyay (2023) human-verified
segmentation mask instead of our own learned classifier.

Our segmentation measures mean precision 0.203 on the building class (see
scripts/eval_segmentation.py) -- roads and bare ground get misread as
buildings often enough that footprints pack edge to edge with no street gaps.
That is a classifier problem, not a pipeline problem: everything downstream
(depth, prisms, mesh repair, texture) is unaffected by which mask feeds it.
This substitutes a CVAT-annotated ground-truth mask for one tile at a time
from the Mendeley dataset (CC BY-NC-SA 4.0, Dabra & Upadhyay, IISER Bhopal),
honestly labelled as using externally-sourced ground truth rather than our
own segmentation for these scenes.

    python build_mumbai_gt.py <tile_name> <scene_name>
"""
import sys

import numpy as np
from PIL import Image

import segmentation as seg

DATASET = "mumbai_dataset_full/Dataset/Main_Dataset"

# Dabra & Upadhyay RGB class codes -> our CLASS_IDX names.
CLASS_RGB = {
    (200, 200, 200): "building",       # built-up
    (250, 235, 185): "building",       # informal settlements -- still structures
    (100, 100, 150): "road",           # impervious surfaces (roads, lots)
    (40, 120, 240): "water",
    (80, 140, 50): "vegetation",
    (200, 160, 40): "bare_earth",      # barren
    (0, 0, 0): "bare_earth",           # unlabelled, 0.08% of the dataset
}


def ground_truth_labels(mask_path, size) -> np.ndarray:
    m = np.array(Image.open(mask_path).convert("RGB").resize(size, Image.NEAREST))
    labels = np.full(m.shape[:2], seg.CLASS_IDX["bare_earth"], dtype=np.int32)
    for rgb, name in CLASS_RGB.items():
        match = np.all(np.abs(m.astype(int) - np.array(rgb)) < 12, axis=-1)
        labels[match] = seg.CLASS_IDX[name]
    return labels


def make_patched_segment(mask_path):
    def patched_segment(image_np, height=None):
        h, w = image_np.shape[:2]
        labels = ground_truth_labels(mask_path, (w, h))
        total = labels.size
        stats = {name: float((labels == idx).sum()) / total
                  for name, idx in seg.CLASS_IDX.items()}
        return labels, stats
    return patched_segment


if __name__ == "__main__":
    tile = sys.argv[1]
    scene_name = sys.argv[2] if len(sys.argv) > 2 else tile.replace(".", "_")

    img_path = f"{DATASET}/images/{tile}.tif"
    mask_path = f"{DATASET}/masks/{tile}.png"
    upscaled_path = f"india/{scene_name}.png"

    Image.open(img_path).convert("RGB").resize((2400, 2400), Image.LANCZOS).save(upscaled_path)

    seg.segment = make_patched_segment(mask_path)
    import build_city_image as bci
    bci.build(upscaled_path, scene_name, max_px=1024, use_dem=False)
