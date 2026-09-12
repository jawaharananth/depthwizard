"""
Validate build_city_image.estimate_sun_azimuth() against real IMD sun angles.

WHY: the fix in build_city_image.py (edge-only probing, reject probes that
land back on a building) was made to address a measured defect -- azimuth
error up to 178 deg on some tiles, near-zero on others -- found by comparing
the heuristic's output to each tile's own IMD meanSunAz. This script is that
same comparison, made re-runnable, over every tile that has one, instead of
the 3 tiles checked by hand during diagnosis.

IMD IS DIAGNOSTIC ONLY. It is read here, after estimate_sun_azimuth() has
already returned its answer, purely to grade that answer. It is never passed
into estimate_sun_azimuth(), calibrate_scale(), or anything upstream of it --
real ISRO evaluation imagery carries no IMD sidecar, so a fix that secretly
depended on one would pass here and fail there. Grep this file for "meanSunAz"
and the only place it is used is inside compare(), after the estimate has
already been computed and returned.

This does not touch DFC2019 LiDAR truth (_DSM.tif / _CLS.tif) at all -- IMD
sun-angle metadata is not ground truth for elevation, and this script does
not compute accuracy metrics, only azimuth error.

Usage:
  python scripts/validate_azimuth.py           # all tiles with IMD available
  python scripts/validate_azimuth.py 5         # first 5
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dfc2019_loader as L
import segmentation as seg
import overlay_rejection
import build_city_image as bci
import height_cache
from depth_model import DepthBackbone, backbone_tag

TRUTH = os.path.join(ROOT, "dfc2019_data", "truth", "Track3-Truth")
RGB = os.path.join(ROOT, "dfc2019_data", "rgb", "Track3-RGB-1")
METADATA = os.path.join(ROOT, "dfc2019_data", "metadata", "Track3-Metadata")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("benchmark_blind", os.path.join(ROOT, "scripts", "benchmark_blind.py"))
_bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bb)


def angular_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a-b, in (-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def compute_estimate(tile: str, out_px: int = 1024) -> tuple[float | None, int]:
    """
    The exact sequence build_city_image.build() runs before calling
    estimate_sun_azimuth() -- height field, overlay strip, segmentation --
    reimplemented here only because build() does not expose the intermediate
    azimuth value on its own. No IMD, no ground truth anywhere in this
    function.
    """
    o = _bb._orthorectify_blind(tile, out_px)
    image_np = o["image"]
    pil = Image.fromarray(image_np)

    key = f"azcheck_{tile}_{backbone_tag()}"
    height = height_cache.load(key, "tiled", out_px)
    if height is None:
        height = DepthBackbone().predict_tiled(pil, verbose=False)
        height_cache.save(key, "tiled", out_px, height)

    seg_pre, _ = seg.segment(image_np)
    image_np, _ov = overlay_rejection.clean(
        image_np, veg_mask=(seg_pre == seg.CLASS_IDX["vegetation"]))
    seg_labels, _ = seg.segment(image_np, height=height)

    n_building_px = int((seg_labels == seg.CLASS_IDX["building"]).sum())
    computed = bci.estimate_sun_azimuth(image_np, seg_labels)
    return computed, n_building_px


def compare(tile: str) -> dict:
    imd_path_candidates = glob.glob(os.path.join(RGB, f"{tile}_*_RGB.tif"))
    if not imd_path_candidates:
        return {"tile": tile, "status": "NO_RGB"}

    computed, n_building_px = compute_estimate(tile)

    # IMD read AFTER the estimate above -- diagnostic grading only, see
    # module docstring. most_nadir_view() picks the same view
    # _orthorectify_blind() used, so this grades the view actually probed.
    import ortho
    rgb_path, imd = ortho.most_nadir_view(tile, RGB, METADATA)
    real_az = imd.get("meanSunAz")
    if real_az is None:
        return {"tile": tile, "status": "NO_IMD"}

    if computed is None:
        return {"tile": tile, "status": "REFUSED", "real_sun_az": real_az,
                "n_building_px": n_building_px}

    err = angular_diff(computed, real_az)
    return {"tile": tile, "status": "OK", "real_sun_az": real_az,
           "computed_sun_az": round(computed, 1), "error_deg": round(err, 1),
           "n_building_px": n_building_px}


def main(max_tiles: int = None):
    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(TRUTH, "*_DSM.tif"))})
    tiles = [t for t in tiles if glob.glob(os.path.join(RGB, f"{t}_*_RGB.tif"))]
    if max_tiles:
        tiles = tiles[:max_tiles]

    print(f"validating azimuth on {len(tiles)} tiles (IMD used only to grade, "
         f"never fed into estimate_sun_azimuth)\n")

    results = []
    for i, tile in enumerate(tiles, 1):
        r = compare(tile)
        results.append(r)
        if r["status"] == "OK":
            print(f"[{i}/{len(tiles)}] {tile}: real={r['real_sun_az']:.1f}  "
                  f"computed={r['computed_sun_az']:.1f}  "
                  f"error={r['error_deg']:+.1f}deg  "
                  f"building_px={r['n_building_px']}")
        else:
            print(f"[{i}/{len(tiles)}] {tile}: {r['status']}"
                  + (f"  building_px={r.get('n_building_px')}" if "n_building_px" in r else ""))

    ok = [r for r in results if r["status"] == "OK"]
    refused = [r for r in results if r["status"] == "REFUSED"]
    other = [r for r in results if r["status"] not in ("OK", "REFUSED")]

    print()
    print("=" * 64)
    if ok:
        errs = np.array([abs(r["error_deg"]) for r in ok])
        print(f"{len(ok)}/{len(tiles)} estimated  |  median |error| "
              f"{np.median(errs):.1f} deg  max {errs.max():.1f} deg  "
              f"mean {errs.mean():.1f} deg")
        print(f"  within 10 deg: {(errs <= 10).sum()}/{len(ok)}   "
              f"within 30 deg: {(errs <= 30).sum()}/{len(ok)}   "
              f"within 90 deg: {(errs <= 90).sum()}/{len(ok)}")
    print(f"{len(refused)}/{len(tiles)} refused (not enough off-building edge signal)")
    if other:
        print(f"{len(other)}/{len(tiles)} skipped: "
              + ", ".join(f"{r['tile']} ({r['status']})" for r in other))
    print("=" * 64)

    write_report(results, tiles)


def write_report(results: list, tiles: list, path: str = None):
    path = path or os.path.join(ROOT, "AZIMUTH_VALIDATION.md")
    ok = [r for r in results if r["status"] == "OK"]
    lines = []
    lines.append("# estimate_sun_azimuth() validation against real IMD sun angles")
    lines.append("")
    lines.append("Generated by `scripts/validate_azimuth.py`. IMD `meanSunAz` is used "
                 "ONLY to grade the already-computed estimate, never fed into it -- "
                 "real evaluation imagery carries no IMD sidecar, so this measures "
                 "exactly what a blind single image can do.")
    lines.append("")
    if ok:
        errs = np.array([abs(r["error_deg"]) for r in ok])
        lines.append(f"**{len(ok)}/{len(tiles)} tiles estimated.** "
                     f"Median |error| **{np.median(errs):.1f} deg**, "
                     f"max {errs.max():.1f} deg, mean {errs.mean():.1f} deg.")
        lines.append(f"Within 10 deg: {(errs<=10).sum()}/{len(ok)}. "
                     f"Within 30 deg: {(errs<=30).sum()}/{len(ok)}. "
                     f"Within 90 deg: {(errs<=90).sum()}/{len(ok)}.")
    lines.append("")
    lines.append("| tile | status | real_sun_az | computed | error_deg | building_px |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        if r["status"] == "OK":
            lines.append(f"| {r['tile']} | OK | {r['real_sun_az']:.1f} | "
                         f"{r['computed_sun_az']:.1f} | {r['error_deg']:+.1f} | "
                         f"{r['n_building_px']} |")
        else:
            lines.append(f"| {r['tile']} | {r['status']} | "
                         f"{r.get('real_sun_az', '')} | -- | -- | "
                         f"{r.get('n_building_px', '')} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
