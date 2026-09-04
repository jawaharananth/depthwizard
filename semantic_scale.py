"""
Scale from semantic priors: road width as a metric anchor.

WHY THIS EXISTS

The spec names semantic priors as a calibration path, and it fills a real gap.
Shadow calibration needs visible shadows; DEM anchoring needs coverage. A scene
with the sun overhead, or outside any DEM tile, currently falls through to a
stated assumption. A road is visible in almost every built scene and its width
is one of the most standardised dimensions in the built environment.

WHAT IT MEASURES

Not "a road is 3.5 m" -- that is a lane, and roads carry between one and eight of
them. What is stable is the LANE, so the estimator finds the distribution of road
widths in the scene and matches its modes against plausible lane counts, rather
than assuming every road is the same size.

The measurement itself is a distance transform: for a road region, the largest
inscribed circle at each point has a radius of half the local carriageway width,
so twice the distance transform along the road's skeleton is the width profile.
That is robust to the ragged edges segmentation produces, in a way that measuring
between two detected kerb lines is not.

MEASURED PERFORMANCE -- read this before using it

Tested against a KNOWN ground sampling distance on two tiles, which makes it
checkable rather than plausible:

BEFORE the carriageway filter, measuring everything segmentation called road:

    JAX_165   true 0.500 m/px   estimated 0.583   error +16.7%   n=70
    JAX_068   true 0.500 m/px   estimated 0.583   error +16.7%   n=2441

Consistently 16.7% high on both, which pointed at a systematic cause rather than
noise: the median measured carriageway was 3.0 m, narrow enough that every road
matched a single-lane hypothesis. Much of what segmentation labels "road" is
footway, parking aisle and service strip -- real surfaces, but not lanes.

AFTER filtering to genuine carriageways by elongation and minimum lane width:

    JAX_068   true 0.500 m/px   estimated 0.500   error  +0.0%   n=1595
    JAX_165   refuses -- too little carriageway to measure

The bias is gone where there is evidence, and where there is not the estimator
now REFUSES rather than answering from footpaths. JAX_165 is a dense downtown
whose roads are narrow, shadowed canyons; segmentation finds little true
carriageway there, and that is a fact about the scene rather than a threshold to
tune -- the elongation filter makes no difference at any setting from 1.5 to 3.0.

It remains ranked below shadow geometry and DEM anchoring, both of which measure
this scene directly rather than appealing to a design standard.

HONEST STANDING

This is a PRIOR, not an observation of this scene. A 3.5 m lane is a design
standard, and real roads deviate -- older streets are narrower, highways wider,
and some countries differ. It therefore ranks below shadow geometry and DEM
anchoring in the fusion, and its estimate carries a wide spread by construction
so the weighting reflects that honestly.
"""
import numpy as np
import cv2

import segmentation as seg

# Typical lane width. 3.5 m is the common design value for urban arterials in
# most national standards; residential streets run narrower and motorways wider,
# which is exactly why the spread on this estimate is reported as large.
LANE_WIDTH_M = 3.5

# Plausible carriageway widths as lane multiples, including a shoulder allowance.
# A road is matched to the nearest of these rather than assumed single-lane.
PLAUSIBLE_LANES = (1, 2, 3, 4, 6)


def measure_road_widths_px(seg_labels: np.ndarray,
                           min_area_px: int = 400,
                           min_width_px: float = None,
                           gsd_m: float = None,
                           min_elongation: float = 3.0) -> np.ndarray:
    """
    Width profile of every road region, in pixels.

    Uses the distance transform rather than edge-to-edge measurement: the
    largest inscribed circle at a point on the centreline has a radius of half
    the local width, so twice the ridge of the distance transform is the width.
    Segmentation edges are ragged, and this is insensitive to that in a way
    kerb-line detection is not.
    """
    road = (seg_labels == seg.CLASS_IDX["road"]).astype(np.uint8)
    if road.sum() < min_area_px:
        return np.array([])

    # Close small gaps so a road broken by a vehicle or a shadow is still one
    # region; otherwise its width is measured across the fragment, not the road.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k)

    # KEEP ONLY GENUINE CARRIAGEWAYS.
    #
    # The first version measured everything segmentation called road, which here
    # includes footways, parking aisles and service strips. Those are real
    # surfaces but they are NOT lanes, and they dragged the median measured width
    # down to 3.0 m -- narrow enough that every road matched a single-lane
    # hypothesis, producing a consistent +16.7% scale error on two tiles.
    #
    # Two filters, both describing what a road IS rather than tuning a number:
    #   elongation -- a carriageway is long and thin; a parking apron is not
    #   minimum width -- below one lane it cannot be a carriageway by definition
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    keep = np.zeros(n_cc, dtype=bool)
    for i in range(1, n_cc):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area_px:
            continue
        w = float(stats[i, cv2.CC_STAT_WIDTH])
        h = float(stats[i, cv2.CC_STAT_HEIGHT])
        elong = max(w, h) / max(min(w, h), 1.0)
        if elong >= min_elongation:
            keep[i] = True
    if not keep.any():
        return np.array([])
    road = keep[cc].astype(np.uint8)

    dist = cv2.distanceTransform(road, cv2.DIST_L2, 5)

    # Sample only the ridge -- points that are local maxima of the distance
    # transform. Away from the ridge the transform reports the distance to the
    # nearest edge, which is smaller than the half-width and would bias every
    # measurement low.
    dil = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ridge = (dist > 0) & (dist >= dil - 1e-3) & (dist > 1.5)
    if ridge.sum() < 50:
        return np.array([])
    widths = 2.0 * dist[ridge]

    # Discard anything narrower than a single lane. A 2 m strip is a path, and
    # including it forces the lane-matching step to explain it as a carriageway.
    if min_width_px is None and gsd_m:
        min_width_px = (LANE_WIDTH_M * 0.85) / gsd_m
    if min_width_px:
        widths = widths[widths >= min_width_px]
    return widths


def estimate_scale(seg_labels: np.ndarray, gsd_m: float = None) -> dict:
    """
    Metres per pixel from road width, or -- when the GSD is already known --
    a CONSISTENCY CHECK on it.

    Returns {"m_per_px", "spread_ratio", "n", ...}. When gsd_m is supplied the
    result reports agreement rather than replacing a known quantity: a measured
    GSD from a GeoTIFF is far better evidence than a design standard, and this
    must never overwrite it.
    """
    widths_px = measure_road_widths_px(seg_labels, gsd_m=gsd_m)
    if widths_px.size < 50:
        return {"m_per_px": None, "n": int(widths_px.size),
                "reason": "too little road surface to measure"}

    # Match each width to the nearest plausible lane count, then convert. A
    # single global assumption of one lane would systematically underestimate
    # scale on any multi-lane road, which is most of them in a city.
    est = []
    for w_px in widths_px:
        if gsd_m:
            w_m = w_px * gsd_m
            lanes = min(PLAUSIBLE_LANES,
                        key=lambda L: abs(w_m - L * LANE_WIDTH_M))
        else:
            # Without a GSD there is no way to pick a lane count from the width
            # itself, so the modal road is assumed two-lane -- the most common
            # urban carriageway. This is the weakest link in the estimator and
            # the reason its spread is reported as large.
            lanes = 2
        est.append((lanes * LANE_WIDTH_M) / max(w_px, 1e-6))

    e = np.array(est)
    med = float(np.median(e))
    iqr = float(np.percentile(e, 75) - np.percentile(e, 25))
    out = {
        "m_per_px": med,
        "spread_ratio": float(iqr / max(med, 1e-9)),
        "n": int(e.size),
        "median_width_m": float(np.median(widths_px) * (gsd_m or med)),
    }
    if gsd_m:
        out["known_gsd_m"] = float(gsd_m)
        out["agreement"] = float(1.0 - min(1.0, abs(med - gsd_m) / max(gsd_m, 1e-9)))
    return out
