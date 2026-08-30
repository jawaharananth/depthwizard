"""
Stage-by-stage verification of the reconstruction pipeline.

Every stage is checked against something external -- LiDAR ground truth,
semantic labels, or a geometric invariant -- rather than being assumed
correct because it ran without raising. Each check prints PASS/FAIL with the
measured value, and the script exits non-zero if any hard invariant fails.

This exists because several defects in this project produced perfectly valid
output that was silently wrong: a sign-inverted height field, buildings
extruded below the terrain, a ground mesh whose faces all pointed downward,
and an accuracy metric comparing two different physical quantities. None of
those raised an exception. All of them would have been caught by an assertion
on an invariant.
"""
import sys
import glob
import os

import numpy as np
import cv2
import rasterio
from PIL import Image

import dfc2019_loader as L
import ortho
import segmentation as seg
import height_cache
import dsm_refine
import dtm as dtm_mod
import mesh_generation as mg
import roof_structure
import shadow_correction
import terrain_maps
from depth_model import DepthBackbone, orientation_check
from validation import compute_metrics

TRUTH = "dfc2019_data/truth/Track3-Truth"
RGB = "dfc2019_data/rgb/Track3-RGB-1"
METADATA = "dfc2019_data/metadata/Track3-Metadata"

BUILDING_CLASS = 6      # LAS spec
GROUND_CLASS = 2

_failures = []
_warnings = []


def check(name, condition, detail, hard=True):
    status = "PASS" if condition else ("FAIL" if hard else "WARN")
    print(f"  [{status}] {name}: {detail}")
    if not condition:
        (_failures if hard else _warnings).append(f"{name}: {detail}")
    return condition


def ground_truth_ndsm(gt_dsm, cls):
    """Object height above bare earth, from the LiDAR's own ground returns."""
    ground = (cls == GROUND_CLASS)
    if ground.sum() < 100:
        return gt_dsm - np.nanpercentile(gt_dsm, 10)
    hole = (~ground).astype(np.uint8)
    filled = gt_dsm.astype(np.float32).copy()
    _, labels = cv2.distanceTransformWithLabels(
        hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(ground)
    order = np.argsort(labels[ground])
    sy, sx = ys[order], xs[order]
    idx = np.clip(labels[hole.astype(bool)] - 1, 0, len(sy) - 1)
    filled[hole.astype(bool)] = gt_dsm[sy[idx], sx[idx]]
    terrain = cv2.GaussianBlur(filled, (0, 0), sigmaX=4.0)
    return np.maximum(gt_dsm - terrain, 0.0)


def main(tile="JAX_164", tiled=True):
    print(f"\n{'='*70}\nVERIFYING PIPELINE ON {tile}\n{'='*70}")

    # ---------- STAGE 1: input ----------
    #
    # The RGB comes through the RPC orthorectifier, not straight off disk.
    # Track 3 RGB tiles are raw satellite frames with no geotransform, so
    # resizing the north-up truth raster onto them -- what this script and the
    # whole benchmark used to do -- compares two different pieces of ground.
    # After orthorectification the two share a grid by construction, and the
    # registration itself is checked below rather than assumed.
    print("\n[1] INPUT (RPC-orthorectified)")
    o = ortho.orthorectify(tile, TRUTH, RGB, METADATA, out_px=2560, extent_m=640.0)
    image_np = o["image"]
    h, w = image_np.shape[:2]
    print(f"       view {os.path.basename(o['rgb_path'])}  "
          f"off-nadir {o['off_nadir_deg']}deg  sun elev {o['sun_elev_deg']}deg  "
          f"{o['gsd_m']:.3f} m/px")
    check("resolution >= 2000px", min(h, w) >= 2000, f"{w}x{h}")
    check("3 colour channels", image_np.shape[2] == 3, f"{image_np.shape[2]} channels")
    check("not blank", image_np.std() > 10, f"std={image_np.std():.1f}")
    # The scene is rendered over 640 m while the LiDAR truth covers only the
    # central 256 m, so full coverage is not expected: the extended frame runs
    # past the edge of what the satellite view captured. Measured at 98.6% for
    # this extent, with the shortfall entirely in the far corners. The check
    # exists to catch a badly mis-placed grid -- which shows up as coverage in
    # the 50-70% range -- not to demand a full frame.
    check("ortho coverage sufficient", o["coverage"] > 0.95,
          f"{o['coverage']*100:.1f}% filled "
          f"(truth covers the central {o['truth_extent_m']:.0f} m of "
          f"{o['extent_m']:.0f} m)")
    check("near-nadir view selected", (o["off_nadir_deg"] or 99) < 10,
          f"{o['off_nadir_deg']} deg off-nadir "
          f"(roof lean = height x tan(angle))", hard=False)

    gt_data = o["truth"]
    gt_dsm = gt_data["dsm"].copy()
    gt_dsm[~gt_data["valid_mask"]] = np.nan
    cls = gt_data["cls"]
    check("ground truth loaded", np.isfinite(gt_dsm).any(),
          f"{gt_data['valid_mask'].mean()*100:.1f}% labelled")

    # ---------- STAGE 2: depth ----------
    print("\n[2] HEIGHT ESTIMATION")
    pil = Image.fromarray(image_np)
    # Same cache key build_city.py writes. They had diverged -- the harness
    # asked for "tiled" while the pipeline stores "tiled_e640" -- so every run
    # missed the cache and recomputed ten minutes of depth inference for a field
    # already sitting on disk. A verification harness that is expensive to run
    # is a harness that stops being run.
    mode = ("tiled_e640" if tiled else "global")
    height = height_cache.load(tile, mode, h)
    if height is None:
        backbone = DepthBackbone()
        height = backbone.predict_tiled(pil) if tiled else backbone.predict(pil)
        height_cache.save(tile, mode, h, height)
    else:
        print(f"       height field reused from cache ({mode}, {h}px)")
    check("output shape matches input", height.shape == (h, w), f"{height.shape}")
    check("normalised to [0,1]", 0 <= height.min() and height.max() <= 1.0001,
          f"[{height.min():.3f}, {height.max():.3f}]")
    check("no NaN", not np.isnan(height).any(), "clean")

    seg_labels, seg_stats = seg.segment(image_np, height=height)
    oc = orientation_check(height, seg_labels)
    if oc.get("checked"):
        check("buildings ABOVE ground (sign not inverted)",
              oc["correct_orientation"],
              f"building {oc['building_mean']:.4f} vs ground {oc['ground_mean']:.4f}")
    else:
        check("orientation checkable", False, oc.get("reason", "?"), hard=False)

    # ---------- STAGE 3: segmentation vs truth ----------
    print("\n[3] SEGMENTATION (vs LiDAR semantic labels)")
    pred_b = (seg_labels == seg.CLASS_IDX["building"])
    gt_b = (cls == BUILDING_CLASS)
    gt_b_big = cv2.resize(gt_b.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = cv2.resize((cls != 65).astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    tp = int((pred_b & gt_b_big & valid).sum())
    fp = int((pred_b & ~gt_b_big & valid).sum())
    fn = int((~pred_b & gt_b_big & valid).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"       building precision {prec:.3f}  recall {rec:.3f}  "
          f"(GT coverage {gt_b_big.mean()*100:.1f}%)")
    check("finds some real buildings", rec > 0.15, f"recall={rec:.3f}", hard=False)
    check("classes sum to 100%", abs(sum(seg_stats.get(c, 0) for c in seg.CLASSES) - 1.0) < 0.01,
          f"{sum(seg_stats.get(c,0) for c in seg.CLASSES)*100:.1f}%")

    # ---------- STAGE 4: refinement ----------
    print("\n[4] EDGE REFINEMENT")
    before = dsm_refine.edge_sharpness(height)
    refined = dsm_refine.refine_dsm(height, image_np)
    after = dsm_refine.edge_sharpness(refined)
    check("edges sharpened", after > before, f"{before:.4f} -> {after:.4f} ({after/max(before,1e-9):.2f}x)")
    check("range preserved", abs(refined.mean() - height.mean()) < 0.15,
          f"mean {height.mean():.3f} -> {refined.mean():.3f}")

    # ---------- STAGE 4b: metric scale ----------
    #
    # Verify what actually ships. This used to multiply by a constant 60 and
    # then, at stage 8, rescale the result so its 99th percentile matched the
    # truth's -- which hands the answer's magnitude to the thing being scored
    # and can only ever measure shape. The shipped path calibrates against
    # shadow geometry instead, so that is what gets checked.
    print("\n[4b] METRIC SCALE (shadow geometry)")
    cal = shadow_correction.calibrate_scale(
        image_np, seg_labels, refined, o["sun_elev_deg"], o["sun_azimuth_deg"],
        o["gsd_m"], o["gsd_m"])
    metric = cal["scale_m_per_unit"] is not None and cal["n"] >= 10
    if metric:
        scale = cal["scale_m_per_unit"]
        print(f"       {scale:.1f} m per unit from {cal['n']} shadow measurements, "
              f"IQR {cal['spread_ratio']*100:.0f}% of median")
        check("shadow scale is stable across buildings", cal["spread_ratio"] < 1.5,
              f"IQR/median = {cal['spread_ratio']:.2f}", hard=False)
    else:
        scale = 60.0
        print(f"       not measurable ({cal.get('reason', str(cal['n']) + ' usable shadows')})"
              f" -- Tier C, heights are relative only")
    # THE SHIPPED PIPELINE, not a parallel reimplementation.
    #
    # This harness used to build its own scene with mesh_generation's legacy
    # heightfield path. build_city.py stopped using any of that -- it now builds
    # prisms from image regions on an MVS surface with a morphological ground --
    # so the harness was verifying code that no longer ships. It reported a hard
    # FAIL (correlation 0.138) for a path nothing runs, while the real pipeline
    # was fine. A test that guards nothing is worse than a missing test, because
    # it looks like coverage.
    import city_model
    import region_footprints as rf
    import mvs_height

    gsd_here = 640.0 / image_np.shape[0]
    try:
        mv = mvs_height.compute(tile, TRUTH, RGB, METADATA, extent_m=640.0)
        dsm = mvs_height.to_image_grid(mv["dsm"], image_np)
        height_source = f"MVS, {mv['views']} views"
    except Exception as exc:
        dsm = refined * scale
        height_source = f"monocular fallback ({type(exc).__name__})"
    check("MVS height source available", "MVS" in height_source, height_source,
          hard=False)

    # ---------- STAGE 5: ground ----------
    print("\n[5] GROUND (morphological opening, segmentation-independent)")
    ground = dtm_mod.ground_from_dsm(dsm, gsd_here)
    check("ground never above surface", bool((ground <= dsm + 1e-3).all()),
          f"max excess {float((ground - dsm).max()):.4f} m")
    ndsm = np.maximum(dsm - ground, 0.0)
    check("object heights are positive", bool((ndsm >= 0).all()),
          f"min {float(ndsm.min()):.3f} m")

    # ---------- STAGE 6: footprints ----------
    print("\n[6] FOOTPRINTS (image regions)")
    rres = rf.extract(image_np, ndsm, gsd_here, seg_labels=seg_labels,
                      min_area_m2=8.0, min_height_m=2.0)
    polys = rres["polygons"]
    check("footprints found", len(polys) > 0,
          f"{len(polys)} kept of {rres['report']['regions_examined']} regions")

    # ---------- STAGE 7: prisms ----------
    print("\n[7] BUILDING GEOMETRY (prisms)")
    bv, bf, binfo = city_model.build_prisms(polys, dsm, ground, gsd_here, gsd_here,
                                            min_height_m=2.0, image_np=image_np)
    n_b = len(binfo["buildings"])
    check("prisms extruded", n_b > 0,
          f"{n_b} of {len(polys)} footprints ({binfo['skipped']} below min height)")
    if len(bv):
        check("no NaN in vertices", not bool(np.isnan(bv).any()), "clean")
        check("face indices in bounds", int(bf.max()) < len(bv),
              f"max {int(bf.max())} < {len(bv)}")
        check("no degenerate faces",
              bool((bf[:, 0] != bf[:, 1]).all() and (bf[:, 1] != bf[:, 2]).all()), "none")
        hh = np.array([r["height_m"] for r in binfo["buildings"]])
        check("heights plausible", bool((hh > 0).all() and hh.max() < 400),
              f"median {np.median(hh):.1f} m, max {hh.max():.1f} m")

    # ---------- STAGE 8: accuracy vs LiDAR ----------
    print("\n[8] HEIGHT ACCURACY (vs airborne LiDAR)")
    gt_ndsm = ground_truth_ndsm(gt_dsm, cls)
    # CROP TO THE TRUTH EXTENT FIRST.
    #
    # The scene is now built over 640 m while the LiDAR tile covers only the
    # central 256 m. Resizing the whole 640 m prediction to the truth raster
    # compares two different pieces of ground -- the same class of error as the
    # RPC misregistration this project already had once, reintroduced here by
    # widening the extent without revisiting the comparison. It shows up exactly
    # as it did then: a plausible-looking RMSE with a correlation of -0.018.
    inset = o["truth_inset_px"]
    tw_px = ndsm.shape[0] - 2 * inset
    pred_small = cv2.resize(ndsm[inset:inset + tw_px, inset:inset + tw_px].astype(np.float32),
                            (512, 512), interpolation=cv2.INTER_AREA)
    pred_ndsm = np.maximum(pred_small, 0.0)
    # A shadow-calibrated field is already in metres, so it is scored as-is.
    # Only an uncalibrated Tier C field gets scale-aligned, and that result is
    # a shape comparison, not a metric one -- labelled below so the two are
    # never read as the same claim.
    if metric:
        s = 1.0
    else:
        s = np.nanpercentile(gt_ndsm, 99) / max(np.percentile(pred_ndsm, 99), 1e-6)
    pred_ndsm = pred_ndsm * s
    # Does the prediction describe THIS ground at all?
    #
    # Registration is guaranteed by construction now that the input is
    # orthorectified onto the truth grid, so a low correlation here no longer
    # means misalignment -- it means the height field does not match the scene
    # it was computed from. Both failures look identical in RMSE alone, which
    # is how this project reported numbers for weeks against unaligned
    # rasters, so the correlation is checked separately either way.
    #
    # Measured on JAX_167 (dense downtown): correlation 0.086 with perfect
    # alignment. The backbone cannot separate rooftop from street in a street
    # canyon, and no amount of meshing recovers a signal that was never there.
    fin = np.isfinite(gt_ndsm) & np.isfinite(pred_ndsm)
    corr = float(np.corrcoef(pred_ndsm[fin], gt_ndsm[fin])[0, 1]) if fin.sum() > 100 else 0.0
    check("prediction describes the actual scene", corr > 0.25,
          f"correlation with ground-truth nDSM = {corr:.3f}")

    m = compute_metrics(pred_ndsm, gt_ndsm, gt_data["terrain_mask"])["overall"]
    print(f"       RMSE {m['rmse_m']:.2f} m   MAE {m['mae_m']:.2f} m   "
          f"({m['n_pixels']:,} px)  "
          f"[{'metric, shadow-calibrated' if metric else 'shape only, scale-aligned'}]")
    check("accuracy in expected range", m["rmse_m"] < 12.0,
          f"RMSE {m['rmse_m']:.2f} m", hard=False)

    print(f"\n{'='*70}")
    if _failures:
        print(f"FAILED {len(_failures)} hard check(s):")
        for f in _failures:
            print(f"  - {f}")
    else:
        print("ALL HARD CHECKS PASSED")
    if _warnings:
        print(f"{len(_warnings)} warning(s):")
        for wn in _warnings:
            print(f"  - {wn}")
    print("=" * 70)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "JAX_164"))
