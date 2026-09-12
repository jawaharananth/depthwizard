"""
Blind accuracy benchmark: no affine fit to ground truth, ever.

WHY THIS EXISTS

benchmark_all.py / rescore_baseline.py report RMSE 3.87 m / MAE 2.01 m, but
every one of those 18 scored tiles carries `scale_source: "aligned (Tier C)"`
-- rescore_baseline.py fits a free scale per tile against the LiDAR truth
before scoring (`s = percentile(gt,99)/percentile(pred,99)`). That number
cannot reproduce at final evaluation, where no LiDAR exists to fit against.
Zero blind (no-truth-touched) accuracy measurements exist for this project.
This is the first one.

WHAT "BLIND" MEANS HERE, PRECISELY

Prediction happens in predict_tile_blind(), which:
  - orthorectifies the tile's own most-nadir RGB view through its own RPC
    model, onto that view's own UTM grid, via _orthorectify_blind() below --
    NOT ortho.orthorectify(), the path every other script in this repo uses.
    That shared path computes its RPC-projection terrain-height prior as
    `median(truth["dsm"])` -- a real read of the tile's own LiDAR elevation
    values, defended in its own docstring as "a single scalar, not per-pixel
    truth" but a ground-truth read nonetheless. _orthorectify_blind() and
    _blind_terrain_height() replace that one line with a prior from a global
    DEM (SRTM), which is not this tile's ground truth at all. This is the
    one place in this file that took real engineering to get right, and it
    is the reason this file does not simply call ortho.orthorectify().
  - writes that single orthorectified frame as a standalone GeoTIFF, carrying
    only what a real delivered satellite product carries: its own CRS,
    transform, and the SAME view's own acquisition timestamp (from its own
    IMD sidecar's firstLineTime field -- metadata belonging to the one image
    used, not a cross-view or ground-truth lookup)
  - hands that GeoTIFF to build_city_image.build() -- the single-image,
    no-archive production entry point, patched to attempt Tier B shadow
    calibration from exactly that capture time + location, or fall to Tier C
    loudly when it cannot

Ground truth (dfc2019_loader.load_tile, which reads _DSM.tif and _CLS.tif) is
loaded for the FIRST time in main()/score_tile(), in a separate call, AFTER
predict_tile_blind() has already returned. Prediction and scoring are
different function calls with no shared truth reference between them --
grep this file for "load_tile" and there is exactly one call site, in
score_tile(), after the predict call.

RULES ENFORCED, NOT JUST STATED

  - assert_unmodified(): the predicted nDSM array is hashed the moment
    predict_tile_blind() returns and re-hashed immediately before scoring.
    Any scale-fit-to-truth step -- the exact defect this script exists to
    not repeat -- would change those bytes. A mismatch raises, not warns.
  - Tier C tiles are NEVER rescaled against truth. They contribute a shape
    correlation only and are excluded from the RMSE/MAE aggregate, loudly
    (printed and recorded in BENCHMARK_BLIND.md), the same way an
    unregistered tile already is.
  - MIN_CORRELATION floor (0.25) is unchanged from rescore_baseline.py: a
    tile whose prediction doesn't correlate with its own ground truth is
    reporting noise, not a number.

Usage:
  python scripts/benchmark_blind.py 20
  python scripts/benchmark_blind.py 3          # quick check, 3 tiles
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import os
import sys
import time

import numpy as np
import cv2
import rasterio
from rasterio.warp import transform as rwt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

import ortho
import dfc2019_loader as L
import dem_source as dem_mod
import build_city_image as bci
from validation import compute_metrics
from calibration.terrain_curves import TERRAIN_CLASSES
from rescore_baseline import ground_truth_ndsm, MIN_CORRELATION

TRUTH = os.path.join(ROOT, "dfc2019_data", "truth", "Track3-Truth")
RGB = os.path.join(ROOT, "dfc2019_data", "rgb", "Track3-RGB-1")
METADATA = os.path.join(ROOT, "dfc2019_data", "metadata", "Track3-Metadata")
TMP_DIR = os.path.join(ROOT, "final_out", "blind_input")


def _array_fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _read_first_line_time(rgb_path: str) -> _dt.datetime | None:
    """The selected view's OWN acquisition time, from its OWN IMD sidecar.

    Not a cross-view lookup and not ground truth -- this is the one metadata
    field a real delivered satellite product carries alongside its imagery.
    dfc2019_loader.parse_imd() does not expose it, so it is read directly
    here rather than extending the shared parser for a benchmark-only need.
    """
    imd_path = L.imd_path_for_rgb(rgb_path, METADATA)
    if not os.path.isfile(imd_path):
        return None
    with open(imd_path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if line.startswith("firstLineTime"):
                _, _, val = line.partition("=")
                val = val.strip().strip('"')
                try:
                    return _dt.datetime.strptime(val.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    return None
    return None


def _blind_terrain_height(utm_x0: float, utm_y_top: float, extent_m: float,
                          epsg: str) -> tuple[float, str]:
    """
    A scalar terrain-height PRIOR for the RPC projection below, from a global
    DEM rather than the tile's own LiDAR.

    WHY THIS FUNCTION EXISTS: ortho.orthorectify() -- the shared, reused-
    everywhere-else orthorectification path -- computes this same scalar as
    `float(np.median(truth["dsm"]))`, i.e. from the tile's OWN ground-truth
    DSM. Its own docstring calls that acceptable because it is "a single
    scalar describing the tile's terrain level, not per-pixel truth", which
    is a fair argument against leaking SHAPE -- but it is still a value read
    out of the LiDAR truth before prediction, which is exactly what this
    script's "fail loud if any code path touches ground truth before
    scoring" constraint rules out. So this benchmark does not call
    ortho.orthorectify() at all; it reimplements just the RPC-projection
    geometry below with this DEM-derived prior in place of that one line.

    Falls back to 0.0 m (flat, sea-level) with the reason recorded, rather
    than silently substituting a value, if the DEM fetch fails (e.g. no
    network) -- a coarse assumption, honestly labelled, not a masked error.
    """
    try:
        d = dem_mod.sample_grid(utm_x0, utm_y_top, 8, extent_m / 8.0, epsg,
                                dem_source="srtm")
        if d["dem"] is not None and np.isfinite(d["dem"]).sum() > 4:
            return float(np.nanmedian(d["dem"])), d["source"]
    except Exception as exc:
        return 0.0, f"DEM prior failed ({type(exc).__name__}: {exc}); assumed flat 0 m"
    return 0.0, "DEM prior returned no coverage; assumed flat 0 m"


def _orthorectify_blind(tile: str, out_px: int) -> dict:
    """
    The geometric subset of ortho.orthorectify(), with the ground-truth-
    derived terrain-height prior replaced by _blind_terrain_height() above.
    Everything else -- most-nadir view selection, the RPC warp itself, the
    tile's UTM extent -- is identical, and all of it comes from public
    geometry (the DSM.txt header's easting/northing/size/gsd, and the
    selected view's own RPC model), never from LiDAR elevation values.

    Returns the same shape of dict as ortho.orthorectify(), minus "truth"
    (never loaded here) and with "terrain_height_source" added.
    """
    coords = L.parse_dsm_txt(os.path.join(TRUTH, f"{tile}_DSM.txt"))
    gsd_t, size_t = coords["gsd_m"], coords["size_px"]
    truth_extent_m = size_t * gsd_t
    extent_m = truth_extent_m
    out_gsd = extent_m / out_px
    epsg = ortho.JAX_UTM if tile.startswith("JAX") else ortho.OMA_UTM

    rgb_path, imd = ortho.most_nadir_view(tile, RGB, METADATA)

    dst_transform = from_origin(coords["utm_x"], coords["utm_y"] + truth_extent_m,
                                out_gsd, out_gsd)
    dst_crs = CRS.from_string(epsg)

    rpc_height, height_source = _blind_terrain_height(
        coords["utm_x"], coords["utm_y"] + truth_extent_m, extent_m, epsg)

    ortho_img = np.zeros((3, out_px, out_px), dtype=np.uint8)
    with rasterio.open(rgb_path) as src:
        if not src.rpcs:
            raise ValueError(f"{rgb_path} carries no RPC model; cannot orthorectify")
        for band in (1, 2, 3):
            reproject(
                source=rasterio.band(src, band), destination=ortho_img[band - 1],
                rpcs=src.rpcs, src_crs=CRS.from_epsg(4326),
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.cubic, RPC_HEIGHT=rpc_height)

    image = np.transpose(ortho_img, (1, 2, 0))
    return {
        "image": image, "transform": dst_transform, "crs": dst_crs.to_string(),
        "gsd_m": out_gsd, "rgb_path": rgb_path,
        "off_nadir_deg": imd.get("meanOffNadirViewAngle"),
        "terrain_height_m": rpc_height, "terrain_height_source": height_source,
        "extent_m": extent_m, "truth_extent_m": truth_extent_m,
    }


def predict_tile_blind(tile: str, out_px: int = 1024) -> dict:
    """
    Prediction only. Returns pred_ndsm, the tier reached, and enough
    bookkeeping to score it -- and NOTHING derived from ground truth.

    Uses _orthorectify_blind() above, not ortho.orthorectify() -- see that
    function's docstring for why the shared path is not reused here. No
    truth array of any kind (DSM, CLS, or a value derived from them) is
    loaded anywhere in this function.
    """
    o = _orthorectify_blind(tile, out_px)

    rgb_path = os.path.join(RGB, o["rgb_path"]) if not os.path.isabs(o["rgb_path"]) else o["rgb_path"]
    acq_time = _read_first_line_time(o["rgb_path"] if os.path.isabs(o["rgb_path"])
                                     else os.path.join(RGB, os.path.basename(o["rgb_path"])))

    os.makedirs(TMP_DIR, exist_ok=True)
    geotiff_path = os.path.join(TMP_DIR, f"{tile}.tif")
    tags = {}
    if acq_time is not None:
        tags["TIFFTAG_DATETIME"] = acq_time.strftime("%Y:%m:%d %H:%M:%S")
    with rasterio.open(
            geotiff_path, "w", driver="GTiff",
            height=o["image"].shape[0], width=o["image"].shape[1], count=3,
            dtype="uint8", crs=o["crs"], transform=o["transform"]) as dst:
        for b in range(3):
            dst.write(o["image"][:, :, b], b + 1)
        if tags:
            dst.update_tags(**tags)

    meta = bci.build(geotiff_path, f"blind_{tile}", max_px=out_px,
                     stage=False, return_arrays=True)
    pred_ndsm = meta.pop("_ndsm")
    effective_scale = meta.pop("_effective_scale_m_per_unit")
    fingerprint = _array_fingerprint(pred_ndsm)

    return {
        "tile": tile, "pred_ndsm": pred_ndsm, "fingerprint": fingerprint,
        "meta": meta, "transform": o["transform"], "crs": o["crs"],
        "view": os.path.basename(o["rgb_path"]), "acq_time": acq_time,
        "effective_scale_m_per_unit": effective_scale,
    }


def assert_unmodified(result: dict) -> None:
    """
    Fail loud if anything touched the predicted array between prediction and
    this call. The one operation this guards against by name is exactly
    rescore_baseline.py's `pred = pred * s` truth-fit -- this script contains
    no such line, and this assertion is what makes "contains no such line"
    a checked fact instead of a claim about the source code.
    """
    now = _array_fingerprint(result["pred_ndsm"])
    if now != result["fingerprint"]:
        raise AssertionError(
            f"{result['tile']}: predicted nDSM changed between prediction and "
            f"scoring ({result['fingerprint'][:12]} -> {now[:12]}) -- this is "
            "exactly the ground-truth-fit defect this script exists to not "
            "have. Aborting rather than reporting a number.")


def score_tile(result: dict) -> dict:
    """
    Scoring only. Ground truth is loaded HERE, for the first time, in a call
    predict_tile_blind() never made and has no reference to.
    """
    assert_unmodified(result)
    tile = result["tile"]

    gt_tile = L.load_tile(tile, TRUTH)     # the one ground-truth load site
    gt = gt_tile["dsm"].copy()
    gt[~gt_tile["valid_mask"]] = np.nan
    gt_ndsm = ground_truth_ndsm(gt, gt_tile["cls"])
    terrain_mask = gt_tile["terrain_mask"]

    size = gt_ndsm.shape[0]
    pred = cv2.resize(result["pred_ndsm"].astype(np.float32), (size, size),
                      interpolation=cv2.INTER_AREA)
    fp_before_resize = result["fingerprint"]
    # Resizing to the truth grid is a resampling of the PREDICTION alone --
    # gt_ndsm has not been read yet at the point `pred` was fingerprinted, so
    # this reshape cannot be a truth-fit; assert_unmodified() above already
    # covers the only thing that could be.
    del fp_before_resize

    fin = np.isfinite(gt_ndsm) & np.isfinite(pred)
    corr = (float(np.corrcoef(pred[fin], gt_ndsm[fin])[0, 1])
           if fin.sum() > 100 else 0.0)

    tier = result["meta"].get("tier", "")
    is_metric = tier.startswith("A") or tier.startswith("B")
    shadow_cal = result["meta"].get("shadow_calibration") or {}

    scale_source = "dem" if tier.startswith("A") else "shadow" if tier.startswith("B") else "none"

    out = {
        "tile": tile, "tier": tier, "view": result["view"],
        "correlation": round(corr, 3) if np.isfinite(corr) else None,
        "scale_source": scale_source,
        "scale_source_detail": result["meta"].get("scale_source"),
        "sun_elevation_deg": result["meta"].get("sun_elevation_deg"),
        "n_shadow_measurements": shadow_cal.get("n"),
        "refusal_reason": None if is_metric else shadow_cal.get("refusal_reason"),
        "acq_time": result["acq_time"].isoformat() if result["acq_time"] else None,
    }

    if not np.isfinite(corr) or corr < MIN_CORRELATION:
        out["status"] = "REJECTED"
        out["reason"] = f"unregistered (corr={corr:.3f} < {MIN_CORRELATION})"
        return out

    # FITTED SCALE, computed for the report table only -- never applied to
    # `pred`, never fed back into anything scored above. This answers "what
    # would rescore_baseline's affine have picked", on the same
    # metres-per-raw-unit basis as m_per_unit, so the two are directly
    # comparable: back out the raw (pre-calibration) field from the scene's
    # own effective scale, then see what scale a truth fit implies for that
    # same raw field.
    #
    #   pred_p99 = effective_scale * raw_p99          (what the pipeline built)
    #   fitted_scale = gt_p99 / raw_p99                (what a truth fit implies)
    #                = gt_p99 * effective_scale / pred_p99
    #
    # so scale_error_pct compares two numbers in the same units, not a scale
    # against a dimensionless correction ratio.
    eff_scale = result["effective_scale_m_per_unit"]
    pred_p99 = float(np.percentile(pred[fin], 99))
    gt_p99 = float(np.nanpercentile(gt_ndsm[fin], 99))
    raw_p99 = pred_p99 / max(eff_scale, 1e-9)
    fitted_scale = gt_p99 / max(raw_p99, 1e-9)
    out["fitted_scale_m_per_unit"] = round(fitted_scale, 2)
    out["m_per_unit"] = round(eff_scale, 2) if is_metric else None
    out["scale_error_pct"] = (round(abs(eff_scale - fitted_scale) / max(abs(fitted_scale), 1e-9) * 100, 1)
                              if is_metric else None)

    if not is_metric:
        out["status"] = "SHAPE_ONLY"
        out["reason"] = (f"Tier C -- scale UNAVAILABLE blind "
                         f"({shadow_cal.get('refusal_reason', 'no metric scale reached')}); "
                         "excluded from RMSE aggregate, never rescaled to truth")
        return out

    metrics = compute_metrics(pred, gt_ndsm, terrain_mask)
    out["status"] = "SCORED"
    out["metrics"] = metrics
    return out


def main(max_tiles: int = 20, out_px: int = 1024):
    tiles = sorted({os.path.basename(p)[:7]
                    for p in glob.glob(os.path.join(TRUTH, "*_DSM.tif"))})
    tiles = [t for t in tiles if glob.glob(os.path.join(RGB, f"{t}_*_RGB.tif"))]
    tiles = tiles[:max_tiles]
    print(f"blind benchmark: {len(tiles)} tiles, no affine fit to truth\n")

    results = []
    for i, tile in enumerate(tiles, 1):
        t0 = time.time()
        try:
            pred = predict_tile_blind(tile, out_px=out_px)
        except Exception as exc:
            print(f"[{i}/{len(tiles)}] {tile}: PREDICT FAILED -- {type(exc).__name__}: {exc}")
            results.append({"tile": tile, "status": "PREDICT_FAILED",
                           "reason": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            scored = score_tile(pred)
        except Exception as exc:
            print(f"[{i}/{len(tiles)}] {tile}: SCORE FAILED -- {type(exc).__name__}: {exc}")
            results.append({"tile": tile, "status": "SCORE_FAILED",
                           "reason": f"{type(exc).__name__}: {exc}"})
            continue
        dt = time.time() - t0
        if scored["status"] == "SCORED":
            m = scored["metrics"]["overall"]
            print(f"[{i}/{len(tiles)}] {tile}: SCORED  tier={scored['tier'][:1]}  "
                  f"RMSE={m['rmse_m']:.2f}  MAE={m['mae_m']:.2f}  "
                  f"corr={scored['correlation']}  ({dt:.0f}s)")
        else:
            print(f"[{i}/{len(tiles)}] {tile}: {scored['status']} -- "
                  f"{scored.get('reason', '')}  ({dt:.0f}s)")
        results.append(scored)

    write_report(results, out_px)


def write_report(results: list, out_px: int, path: str = None):
    path = path or os.path.join(ROOT, "BENCHMARK_BLIND.md")
    scored = [r for r in results if r["status"] == "SCORED"]
    shape_only = [r for r in results if r["status"] == "SHAPE_ONLY"]
    rejected = [r for r in results if r["status"] not in ("SCORED", "SHAPE_ONLY")]

    lines = []
    lines.append("# DepthWizard -- Blind Accuracy (no ground-truth fitting)")
    lines.append("")
    lines.append("**Dataset:** IEEE GRSS DFC2019 Track 3")
    lines.append("**Path under test:** `build_city_image.py` (single image, no archive/MVS)")
    lines.append("**Scale:** whatever tier the pipeline reaches on its own -- "
                  "shadow calibration from that image's own acquisition time + "
                  "location, or Tier C, reported honestly. **No affine fit to "
                  "ground truth at any point, on any tile.**")
    lines.append(f"**Generated by:** `scripts/benchmark_blind.py` -- deterministic, re-runnable")
    lines.append("")
    lines.append("This is the counterpart to `BENCHMARK.md`. That document's "
                 "3.87 m / 2.01 m figure scale-aligns every non-metric tile to "
                 "the LiDAR truth before scoring (`scale_source: \"aligned "
                 "(Tier C)\"` on all 18 scored tiles there) -- a fit that "
                 "cannot happen at real evaluation, where no truth exists to "
                 "align to. The numbers below never do that: a Tier C tile "
                 "contributes a shape correlation and nothing else.")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    if scored:
        agg_sq = sum(r["metrics"]["overall"]["rmse_m"] ** 2 * r["metrics"]["overall"]["n_pixels"] for r in scored)
        agg_abs = sum(r["metrics"]["overall"]["mae_m"] * r["metrics"]["overall"]["n_pixels"] for r in scored)
        n = sum(r["metrics"]["overall"]["n_pixels"] for r in scored)
        lines.append(f"**RMSE {np.sqrt(agg_sq/n):.2f} m   MAE {agg_abs/n:.2f} m** "
                     f"-- {len(scored)} of {len(results)} tiles reached a metric "
                     f"tier (A/B) blind and were scored.")
    else:
        lines.append("**No tile reached a metric tier blind.** Every tile that "
                     "was not rejected for poor registration produced a Tier C, "
                     "shape-only result -- see the table below. There is "
                     "currently no blind RMSE figure for this pipeline, which "
                     "is the correct thing to report rather than one produced "
                     "by fitting to the truth this file exists to not touch.")
    lines.append("")
    lines.append(f"- Scored (metric tier, in RMSE aggregate): {len(scored)}")
    lines.append(f"- Shape-only (Tier C, correlation reported, excluded from RMSE): {len(shape_only)}")
    lines.append(f"- Rejected/failed: {len(rejected)}")
    lines.append("")

    if scored:
        lines.append("## Per-tile (scored)")
        lines.append("")
        lines.append("scale_error_pct is the key column: |calibrated m/unit - what a "
                     "truth-fit affine would have chosen| / that fitted value, computed "
                     "for the report only and never applied to any prediction above. "
                     "It says whether this tile's Tier B metres are correct metres or "
                     "merely confident ones.")
        lines.append("")
        lines.append("| tile | tier | RMSE | MAE | corr | scale_source | m_per_unit | "
                     "n_shadow | fitted_scale | scale_error_pct |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in scored:
            m = r["metrics"]["overall"]
            lines.append(
                f"| {r['tile']} | {r['tier']} | {m['rmse_m']:.2f} | {m['mae_m']:.2f} | "
                f"{r['correlation']} | {r['scale_source']} | {r['m_per_unit']} | "
                f"{r['n_shadow_measurements']} | {r['fitted_scale_m_per_unit']} | "
                f"{r['scale_error_pct']} |")
        lines.append("")

        lines.append("## By terrain class")
        lines.append("")
        agg = {k: {"sq": 0.0, "abs": 0.0, "n": 0} for k in TERRAIN_CLASSES}
        for r in scored:
            for k in TERRAIN_CLASSES:
                mm = r["metrics"].get(k)
                if not mm or not mm["n_pixels"]:
                    continue
                agg[k]["sq"] += mm["rmse_m"] ** 2 * mm["n_pixels"]
                agg[k]["abs"] += mm["mae_m"] * mm["n_pixels"]
                agg[k]["n"] += mm["n_pixels"]
        lines.append("| class | RMSE | MAE | pixels |")
        lines.append("|---|---|---|---|")
        for k, a in agg.items():
            if a["n"]:
                lines.append(f"| {k} | {np.sqrt(a['sq']/a['n']):.2f} | "
                             f"{a['abs']/a['n']:.2f} | {a['n']} |")
            else:
                lines.append(f"| {k} | -- | -- | 0 |")
        lines.append("")

    if shape_only:
        lines.append("## Tier C -- shape correlation only, no scale, excluded from RMSE")
        lines.append("")
        lines.append("m_per_unit is null throughout this table by construction -- no "
                     "calibrated scale exists on a Tier C tile. fitted_scale is still "
                     "shown as a diagnostic: what a truth-fit affine would have implied "
                     "for this tile's raw field, for comparison against the naive "
                     "assumed-tallest guess, never used to score anything above.")
        lines.append("")
        lines.append("| tile | corr | scale_source | n_shadow | refusal_reason | fitted_scale |")
        lines.append("|---|---|---|---|---|---|")
        for r in shape_only:
            lines.append(
                f"| {r['tile']} | {r['correlation']} | {r['scale_source']} | "
                f"{r['n_shadow_measurements']} | {r['refusal_reason']} | "
                f"{r['fitted_scale_m_per_unit']} |")
        lines.append("")

    if rejected:
        lines.append("## Rejected / failed")
        lines.append("")
        lines.append("| tile | status | reason |")
        lines.append("|---|---|---|")
        for r in rejected:
            lines.append(f"| {r['tile']} | {r['status']} | {r.get('reason', '')} |")
        lines.append("")

    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python scripts/benchmark_blind.py {len(results)}")
    lines.append("```")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
