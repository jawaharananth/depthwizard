"""
Runs the actual DepthWizard pipeline against real DFC2019 tiles and validates
predicted DSM against real LiDAR ground truth. No shortcuts: no CRS is
embedded in these RGB tiles (RPC metadata is separate, DFC2019 format), so
Tier A calibration is NOT available here -- this honestly reports whichever
tier the pipeline actually lands on (Tier B object-scale, or Tier C
relative-only) rather than faking a georeferenced run.
"""
import os
import glob
import numpy as np
from PIL import Image

from depth_model import DepthBackbone
from calibration.tiered import calibrate
import segmentation
import dfc2019_loader
import validation


def find_rgb_for_tile(tile_prefix: str, rgb_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(rgb_dir, f"{tile_prefix}_*_RGB.tif")))
    return matches[0] if matches else None


def run_benchmark(truth_dir: str, rgb_dir: str, metadata_dir: str = None,
                   n_tiles: int = 8, out_json: str = "dfc2019_validation_results.json"):
    tiles = dfc2019_loader.list_available_tiles(truth_dir)
    model = DepthBackbone()

    per_tile_results = []
    tier_counts = {}

    for tile_prefix in tiles[:n_tiles]:
        rgb_path = find_rgb_for_tile(tile_prefix, rgb_dir)
        if rgb_path is None:
            print(f"  {tile_prefix}: no RGB image found, skipping")
            continue

        gt_data = dfc2019_loader.load_tile(tile_prefix, truth_dir)
        pil_img = Image.open(rgb_path).convert("RGB")
        image_np = np.array(pil_img)

        relative_depth = model.predict(pil_img)
        seg_labels, _ = segmentation.segment(image_np)

        gsd_x_m = gsd_y_m = sun_el = sun_az = None
        if metadata_dir is not None:
            try:
                imd_path = dfc2019_loader.imd_path_for_rgb(rgb_path, metadata_dir)
                imd = dfc2019_loader.parse_imd(imd_path)
                gsd_x_m = gsd_y_m = imd.get("meanProductGSD")
                sun_el, sun_az = imd.get("meanSunEl"), imd.get("meanSunAz")
            except Exception as e:
                print(f"  {tile_prefix}: metadata parse failed ({e}), no sun-angle calibration available")

        dsm, meta = calibrate(relative_depth, image_np, geo_info=None, dem_path=None, seg_labels=seg_labels,
                               gsd_x_m=gsd_x_m, gsd_y_m=gsd_y_m,
                               sun_elevation_deg=sun_el, sun_azimuth_deg=sun_az)

        tier = meta["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        extra = ""
        if tier == "B_shadow_hybrid" and meta.get("building_direct"):
            bd = meta["building_direct"]
            extra = f" (direct_shadow_coverage={bd['direct_shadow_coverage_pct']}%, n_buildings={bd['n_buildings_detected']})"
        print(f"  {tile_prefix}: tier={tier}, image={os.path.basename(rgb_path)}{extra}")

        if tier != "C_relative_only" and dsm is not None:
            # only a metric prediction can be meaningfully compared in meters
            gt = gt_data["dsm"].copy()
            gt[~gt_data["valid_mask"]] = np.nan
            # DFC images are 2048x2048, DSM/terrain_mask are 512x512 (different crop/scale) --
            # resize prediction down to ground-truth grid for a fair pixel comparison.
            from skimage.transform import resize
            dsm_resized = resize(dsm, gt.shape, preserve_range=True, anti_aliasing=True)

            # DFC2019 ground truth is ABSOLUTE WGS84 elevation (includes ~-20m
            # geoid-offset baseline in Jacksonville). Tier B/shadow-based has no
            # DEM/GCP, so it can only ever recover height ABOVE LOCAL GROUND, not
            # true absolute Z -- comparing raw values bakes in a huge, meaningless
            # constant offset. Ground-reference both to their own local 10th
            # percentile so the comparison measures actual shape/height accuracy,
            # not a datum mismatch. (Only Tier A, anchored to a real DEM, could
            # fairly be compared on absolute elevation.)
            gt_ref = np.nanpercentile(gt, 10)
            pred_ref = np.percentile(dsm_resized, 10)
            gt_agl = gt - gt_ref
            pred_agl = dsm_resized - pred_ref

            metrics = validation.compute_metrics(pred_agl, gt_agl, gt_data["terrain_mask"])
            r_value = meta.get("base_regression", {}).get("r_value") if meta.get("base_regression") else None
            per_tile_results.append({"tile": tile_prefix, "tier": tier, "metrics": metrics, "r_value": r_value})
        else:
            r_value = meta.get("base_regression", {}).get("r_value") if meta.get("base_regression") else None
            per_tile_results.append({"tile": tile_prefix, "tier": tier, "metrics": None, "r_value": r_value})

    print("\nTier distribution across tiles:", tier_counts)

    metric_results = [r for r in per_tile_results if r["metrics"] is not None]
    if not metric_results:
        print("\nNo tiles reached a metric-scale tier (all fell to Tier C, relative-only).")
        print("This is an honest, expected result without external elevation reference (no SRTM/Cartosat DEM "
              "supplied and no reliably-detected reference objects) -- RMSE/MAE in meters cannot be computed "
              "from unitless relative depth. Real numbers require either: (a) a real DEM covering these UTM "
              "coordinates passed via --dem, or (b) Tier B succeeding via detected reference objects.")
        payload = {"source": f"DFC2019_real_{n_tiles}tiles_no_metric_tier",
                   "tier_distribution": tier_counts, "results": {}}
        import json
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    # aggregate per-terrain metrics across tiles (pixel-weighted)
    from calibration.terrain_curves import TERRAIN_CLASSES
    agg = {name: {"sq_err_sum": 0.0, "abs_err_sum": 0.0, "n": 0} for name in TERRAIN_CLASSES + ["overall"]}
    for r in metric_results:
        for name, m in r["metrics"].items():
            if m["n_pixels"] == 0:
                continue
            agg[name]["sq_err_sum"] += (m["rmse_m"] ** 2) * m["n_pixels"]
            agg[name]["abs_err_sum"] += m["mae_m"] * m["n_pixels"]
            agg[name]["n"] += m["n_pixels"]

    final_results = {}
    for name, a in agg.items():
        if a["n"] == 0:
            final_results[name] = {"rmse_m": None, "mae_m": None, "n_pixels": 0}
        else:
            final_results[name] = {
                "rmse_m": float(np.sqrt(a["sq_err_sum"] / a["n"])),
                "mae_m": float(a["abs_err_sum"] / a["n"]),
                "n_pixels": a["n"],
            }

    print("\nAggregated real DFC2019 results:")
    for name, m in final_results.items():
        print(f"  {name:10s} rmse={m['rmse_m']} mae={m['mae_m']} n={m['n_pixels']}")

    payload = {"source": f"DFC2019_real_{len(metric_results)}of{n_tiles}tiles",
               "tier_distribution": tier_counts, "results": final_results,
               "per_tile": per_tile_results}
    import json
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_json}")
    return payload


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run_benchmark("dfc2019_data/truth/Track3-Truth", "dfc2019_data/rgb/Track3-RGB-1",
                  metadata_dir="dfc2019_data/metadata/Track3-Metadata", n_tiles=n)
