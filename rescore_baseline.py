"""
Score the pipeline against DFC2019 airborne LiDAR, on registered rasters.

This replaces an earlier version of this script whose numbers were not
meaningful. Three defects, all of which produced plausible-looking output:

  1. It read the raw Track 3 RGB straight off disk. Those are un-orthorectified
     satellite frames carrying an RPC model, so resizing the north-up truth
     raster onto them compares two different pieces of ground. Verified by eye
     on JAX_033: the storage tanks sit bottom-left in the image and top-right
     in the truth. Every RMSE this project has published came from that
     comparison.

  2. It took the first RGB view in the folder. Views of one tile range from
     4.8 to 29 degrees off-nadir, and off-nadir angle displaces a roof from
     its footprint by height * tan(angle) -- 55 m for a tall building at 19
     degrees, which is larger than most of the errors being measured.

  3. It passed one hardcoded sun angle (33.5 deg elevation, 158.6 az) and one
     hardcoded GSD for every tile and every view, when both are per-view
     metadata and vary widely across the set.

All three are fixed here: the image is orthorectified onto the truth grid
through its own RPC model, the straightest view is selected, and sun angle and
GSD come from that view's IMD. Registration is then checked per tile via
correlation rather than assumed, and a tile that fails is reported rather than
folded into the aggregate.
"""
import glob
import json
import os
import sys

import numpy as np
import cv2
from PIL import Image

import ortho
import segmentation as seg
import shadow_correction
import dsm_refine
import dtm as dtm_mod
from validation import compute_metrics
from depth_model import DepthBackbone

TRUTH = "dfc2019_data/truth/Track3-Truth"
RGB = "dfc2019_data/rgb/Track3-RGB-1"
METADATA = "dfc2019_data/metadata/Track3-Metadata"

MIN_CORRELATION = 0.25  # below this the pair is not registered; the RMSE is noise


def ground_truth_ndsm(gt_dsm, cls):
    """Object height above bare earth, from the LiDAR's own ground returns."""
    ground = (cls == 2)          # LAS class 2 = Ground
    if ground.sum() < 100:
        return gt_dsm - np.nanpercentile(gt_dsm, 10)
    hole = (~ground).astype(np.uint8)
    filled = gt_dsm.astype(np.float32).copy()
    if hole.any():
        _, labels = cv2.distanceTransformWithLabels(
            hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(ground)
        order = np.argsort(labels[ground])
        sy, sx = ys[order], xs[order]
        idx = np.clip(labels[hole.astype(bool)] - 1, 0, len(sy) - 1)
        filled[hole.astype(bool)] = gt_dsm[sy[idx], sx[idx]]
    terrain = cv2.GaussianBlur(filled, (0, 0), sigmaX=4.0)
    return np.maximum(gt_dsm - terrain, 0.0)


def predict_ndsm(tile, model, tiled=True, out_px=1024):
    """Predicted object height above terrain, on the truth's own grid."""
    o = ortho.orthorectify(tile, TRUTH, RGB, METADATA, out_px=out_px)
    img = o["image"]
    pil = Image.fromarray(img)

    height = model.predict_tiled(pil) if tiled else model.predict(pil)
    labels, _ = seg.segment(img, height=height)
    refined = dsm_refine.refine_dsm(height, img)

    cal = shadow_correction.calibrate_scale(
        img, labels, refined, o["sun_elev_deg"], o["sun_azimuth_deg"],
        o["gsd_m"], o["gsd_m"])
    scale = cal["scale_m_per_unit"]
    if scale is None or cal["n"] < 10:
        # No metric scale from this scene. Rather than invent one, hand the
        # unitless field back and let the caller scale-align it, which is what
        # a Tier C result honestly is.
        scale, metric = 1.0, False
    else:
        metric = True

    dsm = refined * scale
    terrain = dtm_mod.estimate_dtm(dsm, labels)
    return np.maximum(dsm - terrain, 0.0), o, metric, cal


def main(max_tiles=5, tiled=True):
    model = DepthBackbone()
    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(TRUTH, "*_DSM.tif"))})[:max_tiles]

    agg_sq = agg_abs = n = 0
    per_tile, rejected = [], []

    for tile in tiles:
        if not glob.glob(os.path.join(RGB, f"{tile}_*_RGB.tif")):
            continue
        try:
            pred_full, o, metric, cal = predict_ndsm(tile, model, tiled=tiled)
        except Exception as exc:
            rejected.append({"tile": tile, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"{tile}: SKIPPED -- {exc}")
            continue

        gt_data = o["truth"]
        gt = gt_data["dsm"].copy()
        gt[~gt_data["valid_mask"]] = np.nan
        gt_ndsm = ground_truth_ndsm(gt, gt_data["cls"])

        size = gt_ndsm.shape[0]
        pred = cv2.resize(pred_full.astype(np.float32), (size, size),
                          interpolation=cv2.INTER_AREA)

        fin = np.isfinite(gt_ndsm) & np.isfinite(pred)
        corr = (float(np.corrcoef(pred[fin], gt_ndsm[fin])[0, 1])
                if fin.sum() > 100 else 0.0)
        if not np.isfinite(corr) or corr < MIN_CORRELATION:
            rejected.append({"tile": tile, "reason": f"unregistered (corr={corr:.3f})"})
            print(f"{tile}: REJECTED -- correlation {corr:.3f} < {MIN_CORRELATION}")
            continue

        if not metric:
            # Unitless field: align its scale before scoring, and say so.
            s = np.nanpercentile(gt_ndsm, 99) / max(np.percentile(pred, 99), 1e-6)
            pred = pred * s

        full = compute_metrics(pred, gt_ndsm, gt_data["terrain_mask"])
        m = full["overall"]
        per_tile.append({
            "tile": tile, "metrics": full, "correlation": round(corr, 3),
            "view": os.path.basename(o["rgb_path"]),
            "off_nadir_deg": o["off_nadir_deg"],
            "scale_source": "shadow" if metric else "aligned (Tier C)",
            "shadow_samples": cal.get("n", 0),
        })
        print(f"{tile}: RMSE={m['rmse_m']:.2f}  MAE={m['mae_m']:.2f}  "
              f"corr={corr:.2f}  view={os.path.basename(o['rgb_path'])} "
              f"({o['off_nadir_deg']}deg)  scale={'shadow' if metric else 'aligned'}")
        if m["n_pixels"]:
            agg_sq += m["rmse_m"] ** 2 * m["n_pixels"]
            agg_abs += m["mae_m"] * m["n_pixels"]
            n += m["n_pixels"]

    if not n:
        print("no tile produced a registered, scoreable result")
        return

    print()
    print("=" * 64)
    print(f"DepthWizard, nDSM vs nDSM on RPC-orthorectified input "
          f"({len(per_tile)} of {len(tiles)} tiles scored):")
    print(f"  RMSE {np.sqrt(agg_sq/n):.2f} m   MAE {agg_abs/n:.2f} m")
    if rejected:
        print(f"  {len(rejected)} tile(s) excluded: "
              + ", ".join(f"{r['tile']} ({r['reason']})" for r in rejected))
    print("=" * 64)

    from calibration.terrain_curves import TERRAIN_CLASSES
    agg = {k: {"sq": 0.0, "abs": 0.0, "n": 0} for k in TERRAIN_CLASSES + ["overall"]}
    for r in per_tile:
        for k, mm in r["metrics"].items():
            if not mm["n_pixels"]:
                continue
            agg[k]["sq"] += mm["rmse_m"] ** 2 * mm["n_pixels"]
            agg[k]["abs"] += mm["mae_m"] * mm["n_pixels"]
            agg[k]["n"] += mm["n_pixels"]
    results = {k: ({"rmse_m": None, "mae_m": None, "n_pixels": 0} if not a["n"] else
                   {"rmse_m": float(np.sqrt(a["sq"] / a["n"])),
                    "mae_m": float(a["abs"] / a["n"]), "n_pixels": a["n"]})
               for k, a in agg.items()}

    payload = {
        "source": f"DFC2019_orthorectified_{len(per_tile)}tiles",
        "metric": "nDSM vs nDSM (object height above terrain)",
        "registration": ("RPC-orthorectified onto the truth grid; each tile's "
                        "correlation with the ground-truth nDSM checked against "
                        f"a {MIN_CORRELATION} floor before it is scored"),
        "note": ("Supersedes all earlier figures from this project. Those were "
                 "computed by resizing the north-up truth raster onto raw, "
                 "un-orthorectified satellite frames, which compares different "
                 "ground. They are withdrawn, not adjusted."),
        "results": results,
        "per_tile": per_tile,
        "rejected": rejected,
    }
    with open("dashboard/validation_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote dashboard/validation_results.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5,
         tiled="--global-depth" not in sys.argv)
