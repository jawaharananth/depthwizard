"""Pre/post-event change detection between two DepthWizard DSMs.

Given the DSM GeoTIFFs from two builds of the same ground -- typically imagery
from before and after an earthquake, flood, fire or demolition -- this reports
where the surface lost height, where it gained height, and where it did not
move.

Three things separate this from a plain raster subtraction.

FIRST, the two DSMs are reprojected onto a common grid before differencing.
Two builds of "the same" scene are not automatically pixel-aligned: they may
come from different views, different extents or different output resolutions,
and subtracting misaligned rasters produces a rim of false change around every
building edge -- the single most common way a change map lies.

SECOND, a constant offset between the two surfaces is removed before anything
is classified. Tier B and Tier C calibration each fix the absolute datum from
scene evidence, so two builds of the same place can sit metres apart vertically
while describing identical geometry. Left in, that offset paints the entire
tile as uniform change. The offset is estimated as the median difference over
pixels that are provisionally stable, so real localised change does not drag it.

THIRD, and most important, the detection threshold is derived from the
pipeline's own measured uncertainty rather than chosen to make the output look
decisive. Each DSM carries an error of roughly sigma; their difference carries
sigma * sqrt(2). Change smaller than that is indistinguishable from noise, and
reporting it as damage would be inventing an event. The caller must therefore
supply the accuracy figure, and the module refuses to guess one.

Which accuracy figure matters: use the per-building RMSE or conformal
half-width from validate_buildings.py, not the per-pixel validation overlay
number build_city prints. On JAX_165 those are 4.25 m and 14.95 m respectively
-- the overlay scores every pixel, tree crowns and vehicles included, so it
describes a harsher quantity than the building surfaces compared here. Feeding
it in triples the threshold and hides real events.

Isolated pixels that clear the threshold are still not events. A collapsed
building is a contiguous region of the size of a building, so detections are
morphologically opened and filtered by area before being counted.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Two independent surfaces, so their difference carries sqrt(2) times the
# single-surface error. Anything below that is noise, not an event.
NOISE_MULTIPLIER = float(np.sqrt(2.0))

# How many multiples of the difference noise a pixel must clear before it is
# called change. 1.0 would flag a third of pure noise; 2.0 keeps the false
# positive rate low enough that a flagged region means something.
DEFAULT_SIGMA_FACTOR = 2.0

# A change region smaller than this is not a building event at any plausible
# scale, so it is dropped rather than reported.
MIN_EVENT_AREA_M2 = 40.0

# Structuring element for the opening, in metres. Sized to erase speckle
# without eating the corners off a real footprint.
OPEN_RADIUS_M = 1.0


# ---------------------------------------------------------------------------
# Raster loading and alignment
# ---------------------------------------------------------------------------

def load_dsm(path: str):
    """Return (array, transform, crs, gsd_m) for a DSM GeoTIFF."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        gsd = float(abs(src.transform.a))
        return arr, src.transform, src.crs, gsd


def align_to(src_arr, src_transform, src_crs,
             dst_shape, dst_transform, dst_crs):
    """Resample one DSM onto another's grid.

    Bilinear rather than nearest: the two grids rarely share pixel centres, and
    nearest-neighbour resampling of a height field quantises the surface into
    the source grid, which shows up in the difference as a chequer of +/- half a
    pixel of relief that has nothing to do with the event.
    """
    from rasterio.warp import Resampling, reproject

    out = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return out


# ---------------------------------------------------------------------------
# Differencing
# ---------------------------------------------------------------------------

def _remove_datum_offset(dz: np.ndarray, threshold_m: float) -> Tuple[np.ndarray, float]:
    """Subtract the constant vertical offset between the two builds.

    The offset is the median of the pixels that look stable at this threshold,
    not the median of everything. On a tile where a quarter of the area
    genuinely collapsed, the global median is pulled toward the collapse and the
    correction would then wrongly declare the intact three quarters to have
    risen.
    """
    finite = np.isfinite(dz)
    if not finite.any():
        return dz, 0.0

    # First pass: a robust centre, used only to decide what counts as stable.
    coarse = float(np.nanmedian(dz))
    stable = finite & (np.abs(dz - coarse) <= threshold_m)
    if stable.sum() < max(64, 0.01 * finite.sum()):
        stable = finite          # too little agreement to be selective

    offset = float(np.nanmedian(dz[stable]))
    return dz - offset, offset


def difference(before_path: str, after_path: str,
               accuracy_m: float,
               sigma_factor: float = DEFAULT_SIGMA_FACTOR) -> Dict:
    """Difference two DSMs and classify the result.

    ``accuracy_m`` is the per-surface height accuracy (RMSE against reference,
    or the conformal interval half-width) for these builds. It is required: the
    threshold is meaningless without it, and a default would silently turn noise
    into reported damage.
    """
    if not accuracy_m or accuracy_m <= 0:
        raise ValueError(
            "accuracy_m must be the measured per-surface height accuracy in "
            "metres (for example the validation RMSE reported by build_city). "
            "Change smaller than the measurement error cannot be distinguished "
            "from noise, so there is no safe default.")

    a_arr, a_tf, a_crs, a_gsd = load_dsm(after_path)
    b_arr, b_tf, b_crs, _ = load_dsm(before_path)

    if b_arr.shape != a_arr.shape or b_tf != a_tf or b_crs != a_crs:
        b_arr = align_to(b_arr, b_tf, b_crs, a_arr.shape, a_tf, a_crs)

    noise_m = accuracy_m * NOISE_MULTIPLIER
    threshold_m = noise_m * sigma_factor

    dz = a_arr - b_arr
    dz, offset_m = _remove_datum_offset(dz, threshold_m)

    valid = np.isfinite(dz)
    loss_raw = valid & (dz <= -threshold_m)
    gain_raw = valid & (dz >= threshold_m)

    loss = _clean(loss_raw, a_gsd)
    gain = _clean(gain_raw, a_gsd)

    px_area = a_gsd * a_gsd
    stats = {
        "accuracy_m": round(float(accuracy_m), 3),
        "difference_noise_m": round(float(noise_m), 3),
        "threshold_m": round(float(threshold_m), 3),
        "sigma_factor": float(sigma_factor),
        "datum_offset_removed_m": round(offset_m, 3),
        "gsd_m": round(float(a_gsd), 4),
        "valid_px": int(valid.sum()),
        "loss_area_m2": round(float(loss.sum() * px_area), 1),
        "gain_area_m2": round(float(gain.sum() * px_area), 1),
        "loss_frac": round(float(loss.sum() / max(valid.sum(), 1)), 4),
        "gain_frac": round(float(gain.sum() / max(valid.sum(), 1)), 4),
        "max_loss_m": (round(float(np.nanmin(dz[loss])), 2) if loss.any() else 0.0),
        "max_gain_m": (round(float(np.nanmax(dz[gain])), 2) if gain.any() else 0.0),
        "speckle_rejected_px": int((loss_raw.sum() - loss.sum())
                                   + (gain_raw.sum() - gain.sum())),
    }
    return {"dz": dz, "loss": loss, "gain": gain, "valid": valid,
            "transform": a_tf, "crs": a_crs, "gsd_m": a_gsd,
            "threshold_m": threshold_m, "stats": stats}


def _clean(mask: np.ndarray, gsd_m: float) -> np.ndarray:
    """Open the mask and drop regions too small to be a real event."""
    import cv2
    from scipy import ndimage

    if not mask.any():
        return mask

    r = max(1, int(round(OPEN_RADIUS_M / max(gsd_m, 1e-6))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k) > 0

    lab, n = ndimage.label(opened)
    if n == 0:
        return opened
    min_px = MIN_EVENT_AREA_M2 / max(gsd_m * gsd_m, 1e-9)
    counts = np.bincount(lab.ravel())
    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = counts[1:] >= min_px
    return keep[lab]


# ---------------------------------------------------------------------------
# Per-building attribution
# ---------------------------------------------------------------------------

def per_building(result: Dict, geojson_path: str,
                 min_covered_frac: float = 0.35) -> List[Dict]:
    """Attribute change to individual footprints from a build's GeoJSON.

    A footprint is judged on the MEDIAN difference inside it, not the extreme:
    one noisy pixel on a roof edge should not condemn a standing building. A
    footprint whose pixels are mostly outside the differenced area is skipped
    rather than reported on partial evidence.
    """
    import cv2
    from rasterio.transform import rowcol

    if not os.path.exists(geojson_path):
        return []

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    dz = result["dz"]
    valid = result["valid"]
    thr = result["threshold_m"]
    tf = result["transform"]
    h, w = dz.shape
    out: List[Dict] = []

    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        rows, cols = rowcol(tf, [p[0] for p in ring], [p[1] for p in ring])
        pts = np.stack([np.asarray(cols), np.asarray(rows)], axis=1).astype(np.int32)
        if pts.shape[0] < 3:
            continue

        x0, y0 = np.clip(pts.min(axis=0), [0, 0], [w - 1, h - 1])
        x1, y1 = np.clip(pts.max(axis=0) + 1, [1, 1], [w, h])
        if x1 <= x0 or y1 <= y0:
            continue

        sub = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(sub, [pts - np.array([x0, y0], dtype=np.int32)], 1)
        m = sub.astype(bool)
        if not m.any():
            continue

        win_dz = dz[y0:y1, x0:x1]
        win_ok = valid[y0:y1, x0:x1] & m
        if win_ok.sum() < min_covered_frac * m.sum():
            continue

        med = float(np.median(win_dz[win_ok]))
        if med <= -thr:
            verdict = "height lost"
        elif med >= thr:
            verdict = "height gained"
        else:
            verdict = "unchanged"

        props = feat.get("properties", {})
        out.append({
            "id": props.get("id", len(out)),
            "height_m": props.get("height_m"),
            "median_change_m": round(med, 2),
            "verdict": verdict,
            "pixels": int(win_ok.sum()),
        })
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def colourise(result: Dict) -> np.ndarray:
    """Render the change map.

    Loss and gain are given distinct hues and the unchanged majority is left
    near-neutral, shaded by the before-surface so the reader can still see what
    the scene is. A diverging ramp applied to raw dz would light the whole tile
    up with sub-threshold noise and imply precision the data does not have -- so
    only pixels that passed the threshold AND the area filter are coloured.
    """
    dz = result["dz"]
    loss, gain, valid = result["loss"], result["gain"], result["valid"]
    thr = result["threshold_m"]

    rgb = np.full(dz.shape + (3,), 32, dtype=np.uint8)
    rgb[valid] = 150                       # neutral grey for measured, unchanged

    # Saturation encodes magnitude, clipped at four thresholds so one extreme
    # outlier cannot flatten every other event to the same colour.
    span = max(thr * 4.0, 1e-6)
    if loss.any():
        t = np.clip(-dz[loss] / span, 0.0, 1.0)
        rgb[loss] = np.stack([
            (200 + 55 * t), (70 - 50 * t), (60 - 40 * t)], axis=-1).astype(np.uint8)
    if gain.any():
        t = np.clip(dz[gain] / span, 0.0, 1.0)
        rgb[gain] = np.stack([
            (60 - 40 * t), (150 + 90 * t), (200 - 60 * t)], axis=-1).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare two DepthWizard DSMs and report surface change.")
    ap.add_argument("before", help="DSM GeoTIFF from the earlier capture")
    ap.add_argument("after", help="DSM GeoTIFF from the later capture")
    ap.add_argument("--accuracy", type=float, required=True,
                    help="Per-surface height accuracy in metres. Use the "
                         "PER-BUILDING RMSE or conformal half-width from "
                         "validate_buildings.py (JAX_165: 4.25 m / 4.63 m), not "
                         "the per-pixel validation overlay figure printed by "
                         "build_city (14.95 m on the same tile). The overlay "
                         "scores every pixel including tree crowns, vehicles "
                         "and footprint edges, so it overstates the error on "
                         "the building surfaces this tool compares, and would "
                         "push the threshold far above any real event.")
    ap.add_argument("--sigma-factor", type=float, default=DEFAULT_SIGMA_FACTOR,
                    help="Multiples of difference noise a pixel must clear "
                         f"(default {DEFAULT_SIGMA_FACTOR}).")
    ap.add_argument("--buildings", default=None,
                    help="Optional buildings.geojson to attribute change per "
                         "footprint.")
    ap.add_argument("--out", default="change",
                    help="Output prefix (writes <prefix>.png and <prefix>.json).")
    a = ap.parse_args()

    res = difference(a.before, a.after, a.accuracy, a.sigma_factor)
    st = res["stats"]

    from PIL import Image
    Image.fromarray(colourise(res)).save(a.out + ".png")

    report = {"stats": st, "before": a.before, "after": a.after}
    if a.buildings:
        pb = per_building(res, a.buildings)
        report["buildings"] = pb
        report["buildings_summary"] = {
            "lost": sum(1 for b in pb if b["verdict"] == "height lost"),
            "gained": sum(1 for b in pb if b["verdict"] == "height gained"),
            "unchanged": sum(1 for b in pb if b["verdict"] == "unchanged"),
        }
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"detection threshold {st['threshold_m']} m "
          f"(= {st['sigma_factor']} x sqrt(2) x {st['accuracy_m']} m accuracy)")
    print(f"datum offset removed  {st['datum_offset_removed_m']:+.2f} m")
    print(f"height lost   {st['loss_area_m2']:.0f} m2  "
          f"({st['loss_frac']*100:.1f}% of measured area, "
          f"deepest {st['max_loss_m']:.1f} m)")
    print(f"height gained {st['gain_area_m2']:.0f} m2  "
          f"({st['gain_frac']*100:.1f}% of measured area, "
          f"tallest {st['max_gain_m']:.1f} m)")
    print(f"speckle rejected {st['speckle_rejected_px']} px "
          f"(below {MIN_EVENT_AREA_M2:.0f} m2 or removed by opening)")
    if a.buildings and report.get("buildings_summary"):
        s = report["buildings_summary"]
        print(f"buildings: {s['lost']} lost height, {s['gained']} gained, "
              f"{s['unchanged']} unchanged")
    print(f"wrote {a.out}.png and {a.out}.json")


if __name__ == "__main__":
    main()
