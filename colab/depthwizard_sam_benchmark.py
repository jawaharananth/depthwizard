"""
DepthWizard — SAM building segmentation benchmark (Colab / GPU)
==============================================================

Purpose: decide, on measured evidence, whether SAM should replace the current
hand-tuned CV segmentation in `segmentation.py`.

Why this exists
---------------
The local CV segmentation was found to mark roof EDGES rather than roof
INTERIORS -- its primary building signal is edge density, so it detects
boundaries, not regions. Every downstream stage inherits fragments, and no
amount of morphological post-processing reconstructs regions that were never
detected. SAM produces genuine instance masks, which is the right foundation.

This is a BENCHMARK, not an assumption. It measures both approaches against
the same real ground truth and prints a comparison table. If SAM does not win,
that result stands and the swap does not happen.

Ground truth
------------
DFC2019 Track 3 CLS rasters carry LAS-spec labels; class 6 is Buildings. This
is real labelled data, so precision/recall/IoU are measured, not eyeballed.

Scale note: RGB tiles are 2048x2048, while CLS/DSM are 512x512 (0.5 m GSD).
Everything is compared in the 512x512 truth grid; RGB is downsampled to match.

Run this as a Colab cell sequence, or `python depthwizard_sam_benchmark.py`
after setting DATA_ROOT.
"""
import os
import glob
import json
import time

import numpy as np
import cv2

DATA_ROOT = os.environ.get("DW_DATA", "/content/colab_subset")
TRUTH_DIR = os.path.join(DATA_ROOT, "truth")
RGB_DIR = os.path.join(DATA_ROOT, "rgb")
OUT_JSON = os.path.join(DATA_ROOT, "sam_benchmark_results.json")

BUILDING_CLASS = 6      # LAS spec, per DFC2019 data/README
UNLABELED_CLASS = 65    # excluded from scoring per contest rules
TRUTH_SIZE = 512


# ---------------------------------------------------------------------------
# Setup (Colab)
# ---------------------------------------------------------------------------
SETUP = r"""
!pip -q install segment-anything rasterio
!wget -q -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# 375 MB ViT-B. ViT-H (2.4 GB) is more accurate but slower; start with ViT-B.
"""


def load_sam(checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b"):
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM device: {device}")
    if device == "cpu":
        print("WARNING: CPU inference will be very slow. This benchmark assumes a GPU runtime.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

    # Tuned for overhead imagery: denser sampling than the default so small
    # rooftops are proposed, and a low area floor so nothing is dropped on
    # size before evidence is considered.
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=48,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        crop_n_layers=1,               # extra pass on crops recovers small objects
        crop_n_points_downscale_factor=2,
        min_mask_region_area=16,
    )


# ---------------------------------------------------------------------------
# Building classification of SAM masks
# ---------------------------------------------------------------------------
def classify_masks_as_buildings(masks, rgb, dsm=None):
    """
    SAM is class-agnostic: it segments everything, including roads, trees, and
    parking lots. Each mask is scored on the same physical evidence the local
    pipeline uses, so the comparison is like-for-like and the decision stays
    explainable rather than a black box.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scene_spread = float(np.percentile(gray, 95) - np.percentile(gray, 5)) or 1.0

    g = gray
    mean = cv2.blur(g, (5, 5))
    var = np.clip(cv2.blur(g * g, (5, 5)) - mean * mean, 0, None)

    kept = []
    for m in masks:
        seg_mask = m["segmentation"]
        area = int(seg_mask.sum())
        if area < 4:
            continue

        ys, xs = np.where(seg_mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0

        # boundary contrast: a roof differs materially from the ground beside it
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mu8 = seg_mask.astype(np.uint8)
        outer = (cv2.dilate(mu8, k, iterations=2).astype(bool)) & ~seg_mask
        interior = float(gray[seg_mask].mean())
        exterior = float(gray[outer].mean()) if outer.any() else interior
        edge = float(np.clip(abs(interior - exterior) / (0.35 * scene_spread), 0, 1))

        # rectilinearity: buildings fill their minimum rotated rectangle
        cnts, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape = 0.0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            ra = max(rect[1][0] * rect[1][1], 1e-6)
            shape = float(np.clip(cv2.contourArea(c) / ra, 0, 1))

        # interior smoothness: roofs are more uniform than canopy
        v = float(var[seg_mask].mean())
        texture = float(np.clip(1.0 - abs(np.log10(max(v, 1e-3)) - 1.6) / 2.2, 0, 1))

        # elevation above a surrounding ring, when a DSM is available
        height = 0.0
        if dsm is not None:
            pad = max(3, int(0.4 * max(w, h)))
            ry0, ry1 = max(0, y0 - pad), min(dsm.shape[0], y1 + pad)
            rx0, rx1 = max(0, x0 - pad), min(dsm.shape[1], x1 + pad)
            ring = dsm[ry0:ry1, rx0:rx1]
            inner = dsm[seg_mask]
            if ring.size and inner.size:
                rise = float(np.percentile(inner, 75) - np.percentile(ring, 25))
                spread = float(np.percentile(dsm, 95) - np.percentile(dsm, 5)) or 1.0
                height = float(np.clip(rise / (0.08 * spread), 0, 1))

        score = max(edge, shape, texture, height) * 0.5 + \
            (edge + shape + texture + height) / 4.0 * 0.5

        kept.append({
            "segmentation": seg_mask, "area": area, "score": score,
            "edge": edge, "shape": shape, "texture": texture, "height": height,
        })

    return kept


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mask_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p = pred & valid
    g = gt & valid
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou,
            "tp": tp, "fp": fp, "fn": fn}


def instance_recall(pred: np.ndarray, gt: np.ndarray, min_overlap=0.3) -> dict:
    """
    Per-building recall by size bucket -- the number that matters for the
    'no small building left behind' requirement. A GT instance counts as found
    when at least min_overlap of its pixels are predicted.
    """
    n, cc, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    buckets = {"tiny": [0, 0], "small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for i in range(1, n):
        comp = cc == i
        a = int(comp.sum())
        key = ("tiny" if a < 15 else "small" if a < 50 else "medium" if a < 200 else "large")
        buckets[key][1] += 1
        if (pred & comp).sum() / max(a, 1) >= min_overlap:
            buckets[key][0] += 1
    return {k: {"found": v[0], "total": v[1],
                "recall": (v[0] / v[1]) if v[1] else None}
            for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Baseline: the current local CV segmentation
# ---------------------------------------------------------------------------
def baseline_building_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Mirrors segmentation.py's building rule so the comparison is honest.
    Reproduced here rather than imported so this file runs standalone on Colab.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2].astype(np.float32)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    dens = cv2.blur(edges, (9, 9))
    return (dens > np.percentile(dens, 85)) & (val > 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(max_tiles: int = 10, score_threshold: float = 0.35):
    import rasterio

    generator = load_sam()

    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(TRUTH_DIR, "*_CLS.tif"))})
    results = []

    for tile in tiles[:max_tiles]:
        rgbs = sorted(glob.glob(os.path.join(RGB_DIR, f"{tile}_*_RGB.tif")))
        if not rgbs:
            continue

        with rasterio.open(rgbs[0]) as src:
            rgb_full = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        with rasterio.open(os.path.join(TRUTH_DIR, f"{tile}_CLS.tif")) as src:
            cls = src.read(1)
        with rasterio.open(os.path.join(TRUTH_DIR, f"{tile}_DSM.tif")) as src:
            dsm = src.read(1).astype(np.float32)

        # compare in the truth grid
        rgb = cv2.resize(rgb_full, (TRUTH_SIZE, TRUTH_SIZE), interpolation=cv2.INTER_AREA)
        gt = (cls == BUILDING_CLASS)
        valid = (cls != UNLABELED_CLASS)

        t0 = time.time()
        masks = generator.generate(rgb)
        scored = classify_masks_as_buildings(masks, rgb, dsm)
        elapsed = time.time() - t0

        sam_pred = np.zeros((TRUTH_SIZE, TRUTH_SIZE), bool)
        for m in scored:
            if m["score"] >= score_threshold:
                sam_pred |= m["segmentation"]

        base_pred = baseline_building_mask(rgb)

        row = {
            "tile": tile,
            "sam_masks_total": len(masks),
            "sam_masks_kept": int(sum(1 for m in scored if m["score"] >= score_threshold)),
            "seconds": round(elapsed, 1),
            "sam": mask_metrics(sam_pred, gt, valid),
            "baseline": mask_metrics(base_pred, gt, valid),
            "sam_instance_recall": instance_recall(sam_pred, gt),
            "baseline_instance_recall": instance_recall(base_pred, gt),
        }
        results.append(row)

        print(f"\n{tile}  ({elapsed:.1f}s, {len(masks)} SAM masks -> {row['sam_masks_kept']} kept)")
        print(f"  SAM       IoU {row['sam']['iou']:.3f}  P {row['sam']['precision']:.3f}  "
              f"R {row['sam']['recall']:.3f}  F1 {row['sam']['f1']:.3f}")
        print(f"  baseline  IoU {row['baseline']['iou']:.3f}  P {row['baseline']['precision']:.3f}  "
              f"R {row['baseline']['recall']:.3f}  F1 {row['baseline']['f1']:.3f}")

    if not results:
        print("No tiles processed -- check DW_DATA path.")
        return

    def avg(key, sub):
        return float(np.mean([r[key][sub] for r in results]))

    print("\n" + "=" * 62)
    print(f"AGGREGATE over {len(results)} tiles")
    print("=" * 62)
    print(f"{'metric':<12}{'baseline (CV)':>16}{'SAM':>12}{'delta':>12}")
    for sub in ("iou", "precision", "recall", "f1"):
        b, s = avg("baseline", sub), avg("sam", sub)
        print(f"{sub:<12}{b:>16.3f}{s:>12.3f}{s-b:>+12.3f}")

    print("\nper-building recall (GT instances found):")
    for bucket in ("tiny", "small", "medium", "large"):
        bt = sum(r["baseline_instance_recall"][bucket]["found"] for r in results)
        st = sum(r["sam_instance_recall"][bucket]["found"] for r in results)
        tot = sum(r["sam_instance_recall"][bucket]["total"] for r in results)
        if tot:
            print(f"  {bucket:<7} baseline {bt:4d}/{tot:<4d} ({bt/tot:5.1%})   "
                  f"SAM {st:4d}/{tot:<4d} ({st/tot:5.1%})")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    sam_f1, base_f1 = avg("sam", "f1"), avg("baseline", "f1")
    print("\nVERDICT:", "SAM wins -- proceed with the swap"
          if sam_f1 > base_f1 else
          "SAM does NOT beat the baseline -- do not swap, investigate why")


if __name__ == "__main__":
    run()
