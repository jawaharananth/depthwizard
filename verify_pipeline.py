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
    o = ortho.orthorectify(tile, TRUTH, RGB, METADATA, out_px=2048)
    image_np = o["image"]
    h, w = image_np.shape[:2]
    print(f"       view {os.path.basename(o['rgb_path'])}  "
          f"off-nadir {o['off_nadir_deg']}deg  sun elev {o['sun_elev_deg']}deg  "
          f"{o['gsd_m']:.3f} m/px")
    check("resolution >= 2000px", min(h, w) >= 2000, f"{w}x{h}")
    check("3 colour channels", image_np.shape[2] == 3, f"{image_np.shape[2]} channels")
    check("not blank", image_np.std() > 10, f"std={image_np.std():.1f}")
    check("ortho covers the whole truth extent", o["coverage"] > 0.99,
          f"{o['coverage']*100:.1f}% of output pixels filled")
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
    mode = "tiled" if tiled else "global"
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
    dsm = refined * scale

    # ---------- STAGE 5: terrain separation ----------
    print("\n[5] TERRAIN (DTM) SEPARATION")
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    check("DTM never above DSM", (terrain <= dsm + 1e-3).all(),
          f"max excess {float((terrain-dsm).max()):.4f}")
    st = dtm_mod.structure_height_stats(dsm, terrain, seg_labels)
    check("no building below terrain", st.get("negative_fraction", 1.0) < 0.001,
          f"{st.get('negative_fraction',1)*100:.2f}% negative")
    check("buildings rise above terrain", st.get("mean_agl", 0) > 0,
          f"mean AGL {st.get('mean_agl',0):.2f}")

    # ---------- STAGE 6: water ----------
    print("\n[6] WATER LEVELLING")
    dsm_w, n_lev, n_rej = terrain_maps.flatten_water(dsm, seg_labels)
    changed = float((np.abs(dsm_w - dsm) > 1e-6).mean() * 100)
    check("water edits are bounded", changed < 15.0,
          f"{n_lev} levelled, {n_rej} rejected, {changed:.2f}% of pixels changed")

    # ---------- STAGE 7: buildings ----------
    print("\n[7] BUILDING GEOMETRY")
    bv, buv, bf, binfo = mg.build_building_meshes(
        seg_labels, dsm_w, 1.0, 1.0, None, min_height_m=1.0, dtm=terrain)
    roof_counts = binfo["counts"]
    n_buildings = sum(roof_counts.values())
    check("buildings extruded", n_buildings > 0, f"{n_buildings} ({roof_counts})")
    if len(bv):
        check("no NaN in vertices", not np.isnan(bv).any(), "clean")
        check("UV count matches vertices", len(buv) == len(bv), f"{len(buv)} vs {len(bv)}")
        check("face indices in bounds", bf.max() < len(bv), f"max {bf.max()} < {len(bv)}")
        check("no degenerate faces",
              (bf[:, 0] != bf[:, 1]).all() and (bf[:, 1] != bf[:, 2]).all(), "none")

        # Every base vertex must sit at or above the terrain beneath it.
        # Walk the ORDERED index rather than grouping by roof type -- buildings
        # are emitted in polygon order with types interleaved, so type-grouped
        # slicing reads the wrong vertices and invents failures.
        # Two distinct defects, only one of which "buried" describes:
        #   roof below terrain -> building entirely invisible
        #   base above terrain -> visible gap beneath the building
        # A base BELOW terrain is correct and desirable: the wall continues
        # underground where it cannot be seen, which is what prevents gaps.
        invisible = floating = 0
        for rec in binfo["buildings"]:
            o, n = rec["vertex_offset"], rec["vertex_count"]
            blk = bv[o:o + n]
            if len(blk) < 8:
                continue
            base_y = float(blk[:4, 1].mean())
            # The building's APEX, not the eave ring. Vertices 4:8 are the
            # eaves; a gable or hip carries its ridge at 8:10, so reading the
            # eave reports a pitched building as hidden while its ridge stands
            # clearly above the terrain. Verified on a real case: eave 39.27
            # against terrain 39.41, but apex 42.40 -- plainly visible.
            roof_y = float(blk[:, 1].max())
            xs = np.clip(blk[:4, 0].astype(int), 0, terrain.shape[1] - 1)
            zs = np.clip((-blk[:4, 2]).astype(int), 0, terrain.shape[0] - 1)
            t_local = float(terrain[zs, xs].mean())
            if roof_y < t_local:
                invisible += 1
            elif base_y > t_local + 0.5:
                floating += 1
        total_b = len(binfo["buildings"])
        check("no building hidden below terrain", invisible == 0,
              f"{invisible} of {total_b} invisible")
        check("no building floating above terrain", floating == 0,
              f"{floating} of {total_b} floating")
        unmeas = binfo.get("unmeasurable", 0)
        total_candidates = total_b + unmeas
        check("most buildings have measurable height",
              unmeas < 0.5 * max(total_candidates, 1),
              f"{unmeas} of {total_candidates} omitted as unmeasurable "
              f"({unmeas/max(total_candidates,1)*100:.0f}%)", hard=False)

    # ---------- STAGE 8: accuracy vs LiDAR ----------
    print("\n[8] HEIGHT ACCURACY (vs airborne LiDAR)")
    gt_ndsm = ground_truth_ndsm(gt_dsm, cls)
    pred_small = cv2.resize((dsm_w - terrain).astype(np.float32), (512, 512),
                             interpolation=cv2.INTER_AREA)
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
