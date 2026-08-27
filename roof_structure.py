"""
Roof shape inference from the reconstructed height field.

Buildings are currently extruded as flat-top boxes. Real roofs are pitched,
hipped, or shed, and that silhouette is a large part of what makes a city
model read as buildings rather than blocks.

The governing constraint is evidence. A roof type is only assigned when the
height field inside the footprint actually supports it; otherwise the
building stays flat. Inventing pitched roofs everywhere would look better in
a screenshot and be wrong, and the height data here comes from a monocular
depth model whose per-building reliability was measured at r~0.08 -- so the
bar for claiming roof structure has to be high, not low.

Classification works in the footprint's own oriented frame (long axis vs
short axis), because roof ridges run parallel to a building's long axis in
the overwhelming majority of real construction.
"""
import numpy as np
import cv2

FLAT = "flat"
SHED = "shed"
GABLE = "gable"
HIP = "hip"
UNKNOWN = "unknown"


def _oriented_frame(poly: np.ndarray):
    """
    Returns (centre, long_axis_unit, short_axis_unit, half_long, half_short)
    for a 4-point footprint rectangle.
    """
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    a = np.radians(angle_deg)
    e0 = np.array([np.cos(a), np.sin(a)], dtype=np.float32)
    e1 = np.array([-np.sin(a), np.cos(a)], dtype=np.float32)

    if w >= h:
        return np.array([cx, cy], np.float32), e0, e1, w / 2.0, h / 2.0
    return np.array([cx, cy], np.float32), e1, e0, h / 2.0, w / 2.0


def _profile(heights: np.ndarray, coords: np.ndarray, axis: np.ndarray,
             half_extent: float, n_bins: int = 7):
    """
    Mean height binned along one axis of the footprint, normalised to
    [-1, 1] across the extent. Returns (bin_centres, mean_heights, counts).
    """
    if half_extent < 1e-6:
        return None, None, None
    t = np.clip((coords @ axis) / half_extent, -1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)

    means = np.full(n_bins, np.nan, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int32)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] >= 3:
            means[b] = float(heights[m].mean())

    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, means, counts


def _tent_score(centres: np.ndarray, means: np.ndarray) -> float:
    """
    How tent-shaped (peaked in the middle) a profile is, in [0, 1].

    Compares the fitted symmetric tent |t| model against a flat model. A gable
    seen across its ridge is high; a flat or monotonically sloping roof is low.
    """
    ok = ~np.isnan(means)
    if ok.sum() < 5:
        return 0.0
    t, y = centres[ok], means[ok]
    y = y - y.mean()

    tent = -np.abs(t)
    tent = tent - tent.mean()
    denom = float(np.dot(tent, tent))
    if denom < 1e-9:
        return 0.0

    alpha = float(np.dot(y, tent)) / denom
    if alpha <= 0:          # inverted (valley) -- not a roof shape we model
        return 0.0

    resid = y - alpha * tent
    ss_tot = float(np.dot(y, y))
    if ss_tot < 1e-9:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(resid, resid)) / ss_tot, 0.0, 1.0))


def _slope_score(centres: np.ndarray, means: np.ndarray):
    """Linear trend strength in [0,1] plus signed normalised gradient."""
    ok = ~np.isnan(means)
    if ok.sum() < 4:
        return 0.0, 0.0
    t, y = centres[ok], means[ok]
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-9:
        return 0.0, 0.0
    r2 = float(np.clip(1.0 - np.sum((y - pred) ** 2) / ss_tot, 0.0, 1.0))
    return r2, float(coef[0])


def detrend(dsm: np.ndarray) -> np.ndarray:
    """
    Remove the scene-wide linear trend from the height field.

    Monocular depth backbones trained on ground-level photography impose a
    front-to-back gradient on nadir imagery -- measured here as mean relative
    depth running 0.68 at the top of frame to 0.43 at the bottom. That
    gradient tilts *every* building by the same amount in the same direction,
    which a per-building slope test happily reports as a shed roof.

    Evidence this is real and not hypothetical: classifying without detrending
    labelled 65.5% of footprints as shed, with slope directions showing a
    circular concentration of R=0.318 -- a clear preferred direction where
    genuine roof orientations should be near-uniform.

    Subtracting the global plane leaves only *local* relief, so a slope has to
    differ from its surroundings to count as roof evidence. Terrain-scale
    slope is intentionally discarded here: this function is for roof shape
    classification only and is never applied to the exported elevation.
    """
    h, w = dsm.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx.ravel().astype(np.float32),
                          yy.ravel().astype(np.float32),
                          np.ones(h * w, dtype=np.float32)])
    z = dsm.ravel().astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return (z - A @ coef).reshape(h, w).astype(np.float32)


def classify_roof(poly: np.ndarray, dsm: np.ndarray,
                  min_relief_m: float = 0.8,
                  min_confidence: float = 0.45) -> dict:
    """
    Infer roof type for one footprint.

    min_relief_m: roof height variation must exceed this to claim any shape.
                  Below it the roof is flat within the noise of the elevation
                  source, and asserting a pitch would be fabrication.
    """
    poly_i = np.round(poly).astype(np.int32)
    x0, y0, w, h = cv2.boundingRect(poly_i)
    w, h = max(w, 1), max(h, 1)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly_i - [x0, y0]], 1)

    y1, x1 = min(y0 + h, dsm.shape[0]), min(x0 + w, dsm.shape[1])
    y0c, x0c = max(y0, 0), max(x0, 0)
    if y1 <= y0c or x1 <= x0c:
        return {"type": UNKNOWN, "confidence": 0.0, "reason": "footprint_outside_raster"}

    sub_dsm = dsm[y0c:y1, x0c:x1]
    sub_mask = mask[y0c - y0:y1 - y0, x0c - x0:x1 - x0].astype(bool)
    if sub_mask.sum() < 12:
        return {"type": UNKNOWN, "confidence": 0.0, "reason": "too_few_roof_pixels"}

    ys, xs = np.nonzero(sub_mask)
    heights = sub_dsm[sub_mask].astype(np.float32)

    # robust relief: ignore the extreme tails, which are usually edge bleed
    lo, hi = np.percentile(heights, [8, 92])
    relief = float(hi - lo)
    if relief < min_relief_m:
        return {"type": FLAT, "confidence": 1.0, "relief_m": relief,
                "reason": "relief_below_noise_floor"}

    centre, long_ax, short_ax, half_long, half_short = _oriented_frame(poly)
    coords = np.stack([xs + x0c - centre[0], ys + y0c - centre[1]], axis=1).astype(np.float32)

    c_s, m_s, _ = _profile(heights, coords, short_ax, half_short)   # across the ridge
    c_l, m_l, _ = _profile(heights, coords, long_ax, half_long)     # along the ridge

    if c_s is None or c_l is None:
        return {"type": UNKNOWN, "confidence": 0.0, "reason": "degenerate_footprint"}

    tent_across = _tent_score(c_s, m_s)
    tent_along = _tent_score(c_l, m_l)
    slope_r2_s, slope_s = _slope_score(c_s, m_s)
    slope_r2_l, slope_l = _slope_score(c_l, m_l)

    scores = {
        "tent_across": tent_across, "tent_along": tent_along,
        "slope_r2_short": slope_r2_s, "slope_r2_long": slope_r2_l,
        "relief_m": relief,
    }

    # hip: peaked across AND along the ridge (roof falls away on all four sides)
    if tent_across >= min_confidence and tent_along >= min_confidence:
        return {"type": HIP, "confidence": float(min(tent_across, tent_along)),
                "ridge_height_m": float(np.percentile(heights, 95)),
                "eave_height_m": float(np.percentile(heights, 12)), **scores}

    # gable: peaked across the ridge, flat along it
    if tent_across >= min_confidence and tent_along < min_confidence:
        return {"type": GABLE, "confidence": float(tent_across),
                "ridge_height_m": float(np.percentile(heights, 95)),
                "eave_height_m": float(np.percentile(heights, 12)), **scores}

    # shed: one consistent slope, no ridge.
    # Held to a stricter bar than gable/hip because a single uniform slope is
    # precisely the shape a residual global gradient counterfeits, whereas a
    # ridge is not. The slope must also be a meaningful fraction of the
    # building's own relief, not a barely-tilted flat roof.
    best_r2, best_slope, best_axis = ((slope_r2_s, slope_s, "short")
                                       if slope_r2_s >= slope_r2_l
                                       else (slope_r2_l, slope_l, "long"))
    shed_bar = min_confidence + 0.25
    slope_significant = abs(best_slope) > 0.35 * relief
    if best_r2 >= shed_bar and slope_significant:
        return {"type": SHED, "confidence": float(best_r2), "slope_axis": best_axis,
                "slope_sign": float(np.sign(best_slope)),
                "high_height_m": float(np.percentile(heights, 92)),
                "low_height_m": float(np.percentile(heights, 8)), **scores}

    # relief exists but matches no model -- stay flat rather than guess
    return {"type": FLAT, "confidence": 0.0, "reason": "relief_matches_no_roof_model", **scores}


if __name__ == "__main__":
    import sys
    import collections
    from PIL import Image
    from depth_model import DepthBackbone
    import segmentation as seg
    import mesh_generation as mg
    import dsm_refine

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    pil = Image.open(img_path).convert("RGB")
    image_np = np.array(pil)
    dsm = DepthBackbone().predict(pil)
    dsm = dsm_refine.refine_dsm(dsm, image_np) * 40.0
    dsm = detrend(dsm)
    seg_labels, _ = seg.segment(image_np)

    polys = mg._building_footprints(seg_labels)
    print(f"{len(polys)} footprints (global gradient removed before classification)")

    counts = collections.Counter()
    reasons = collections.Counter()
    confs = collections.defaultdict(list)
    for p in polys:
        r = classify_roof(p, dsm)
        counts[r["type"]] += 1
        confs[r["type"]].append(r.get("confidence", 0.0))
        if "reason" in r:
            reasons[r["reason"]] += 1

    print("\nroof types:")
    for t, n in counts.most_common():
        c = np.mean(confs[t]) if confs[t] else 0.0
        print(f"  {t:<9} {n:5d}  ({n/len(polys)*100:5.1f}%)  mean confidence {c:.2f}")
    if reasons:
        print("\nreasons:")
        for r, n in reasons.most_common():
            print(f"  {n:5d}  {r}")
