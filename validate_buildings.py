"""
Per-building height validation against LiDAR.

WHY THIS EXISTS

Every accuracy figure this project reports is aggregate: an RMSE over pixels, or
a median height over a scene. Those can look excellent while individual
buildings are badly wrong, because errors of opposite sign cancel. The scene
median currently sits within 0.2 m of the LiDAR median, which says the
DISTRIBUTION is right -- it says nothing about whether any particular building
is.

This scores each extruded prism against the LiDAR surface inside its own
footprint, so a building that is 15 m out cannot be hidden by one that is 15 m
out the other way.

Only buildings inside the truth extent are scored. The scene is built over 640 m
while the LiDAR tile covers the central 256 m; scoring the rest would be
comparing against nothing.
"""
import sys
import numpy as np
import cv2

import ortho
import segmentation as seg
import height_cache
import mvs_height
import dtm as dtm_mod
import city_model
import region_footprints as rf

TRUTH = "dfc2019_data/truth/Track3-Truth"
RGB = "dfc2019_data/rgb/Track3-RGB-1"
METADATA = "dfc2019_data/metadata/Track3-Metadata"


def validate(tile="JAX_165", extent_m=640.0, out_px=2560):
    o = ortho.orthorectify(tile, TRUTH, RGB, METADATA, out_px=out_px, extent_m=extent_m)
    img = o["image"]
    W = img.shape[0]
    gsd = o["gsd_m"]
    inset = o["truth_inset_px"]
    tw = W - 2 * inset

    cls = o["truth"]["cls"]
    gt_dsm = o["truth"]["dsm"].astype(np.float32)
    ground_mask = (cls == 2)
    if ground_mask.sum() < 100:
        raise SystemExit("no LiDAR ground returns in this tile")
    gl = float(np.percentile(gt_dsm[ground_mask], 50))
    # LiDAR height above ground, resampled onto the ortho's truth window.
    gt_h = cv2.resize(gt_dsm - gl, (tw, tw), interpolation=cv2.INTER_LINEAR)
    gt_cls = cv2.resize(cls, (tw, tw), interpolation=cv2.INTER_NEAREST)

    height = height_cache.load(tile, f"tiled_e{int(extent_m)}", out_px)
    seg_labels, _ = seg.segment(img, height=height)
    mv = mvs_height.compute(tile, TRUTH, RGB, METADATA, extent_m=extent_m)
    dsm = mvs_height.to_image_grid(mv["dsm"], img)
    ground = dtm_mod.ground_from_dsm(dsm, gsd)
    ndsm = np.maximum(dsm - ground, 0.0)

    polys = rf.extract(img, ndsm, gsd, seg_labels=seg_labels,
                       min_area_m2=8.0, min_height_m=2.0)["polygons"]
    bv, bf, binfo = city_model.build_prisms(polys, dsm, ground, gsd, gsd,
                                            min_height_m=2.0, image_np=img)

    rows = []
    for rec, poly in zip(binfo["buildings"], polys[:len(binfo["buildings"])]):
        x, y, w_, h_ = cv2.boundingRect(poly.astype(np.int32))
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w_, W), min(y + h_, W)
        if x1 <= x0 or y1 <= y0:
            continue
        gy0, gy1, gx0, gx1 = y0 - inset, y1 - inset, x0 - inset, x1 - inset
        if gy0 < 0 or gx0 < 0 or gy1 > tw or gx1 > tw:
            continue                      # outside the truth extent
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32) - [x0, y0]], 1)
        mb = m.astype(bool)
        if mb.sum() < 20:
            continue
        sub = gt_h[gy0:gy1, gx0:gx1]
        sc = gt_cls[gy0:gy1, gx0:gx1]
        if sub.shape != mb.shape:
            continue
        # Only score where the LiDAR agrees something is built. A footprint
        # sitting mostly on ground has no building height to be compared with,
        # and scoring it would measure the footprint's placement, not its height.
        if float((sc[mb] == 6).mean()) < 0.5:
            continue
        true_h = float(np.percentile(sub[mb], 80))
        if true_h < 2.0:
            continue
        rows.append((rec["height_m"], true_h, float(mb.sum()) * gsd * gsd))

    if not rows:
        raise SystemExit("no buildings fell inside the truth extent")

    a = np.array(rows)
    ours, true, area = a[:, 0], a[:, 1], a[:, 2]
    err = ours - true

    print(f"\nPER-BUILDING HEIGHT vs LiDAR -- {tile}, {len(rows)} buildings "
          f"inside the {o['truth_extent_m']:.0f} m truth extent\n")
    print(f"  MAE            {np.abs(err).mean():7.2f} m")
    print(f"  RMSE           {np.sqrt((err**2).mean()):7.2f} m")
    print(f"  bias           {err.mean():+7.2f} m   (positive = we build too tall)")
    print(f"  median abs err {np.median(np.abs(err)):7.2f} m")
    print(f"  p90 abs err    {np.percentile(np.abs(err),90):7.2f} m")
    print(f"  within  2 m    {(np.abs(err)<2).mean()*100:6.1f}%")
    print(f"  within  5 m    {(np.abs(err)<5).mean()*100:6.1f}%")
    print(f"  within 10 m    {(np.abs(err)<10).mean()*100:6.1f}%")
    print(f"\n  our median height {np.median(ours):6.2f} m   "
          f"LiDAR median {np.median(true):6.2f} m")
    print(f"  (the aggregate agreement that per-building error can hide)")

    print("\n  by true height band:")
    for lo, hi, nm in [(2, 10, "low   2-10 m"), (10, 25, "mid  10-25 m"),
                       (25, 50, "tall 25-50 m"), (50, 1e9, "high  >50 m")]:
        s = (true >= lo) & (true < hi)
        if s.sum() < 3:
            continue
        print(f"    {nm}  n={int(s.sum()):4d}  MAE {np.abs(err[s]).mean():6.2f} m"
              f"  bias {err[s].mean():+6.2f} m")
    return a


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "JAX_165")
