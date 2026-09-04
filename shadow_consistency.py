"""
Self-supervised shadow consistency: refine heights against the shadows the
image already contains.

THE IDEA

A height field plus a known sun direction predicts exactly where shadows must
fall. The image shows where they actually fall. Any mismatch is an error signal
that needs no labels, no ground truth and no training data -- the supervision is
the photograph itself.

This is a genuinely different constraint from the one the pipeline already uses.
`shadow_correction` measures the length of a shadow ONCE and converts it to a
height. This instead RENDERS the shadow the current heights imply and compares
the whole pattern, so a building that is too tall is penalised by its shadow
overshooting, and one that is too short by its shadow falling short -- including
where shadows land on neighbours rather than flat ground, which a length
measurement cannot handle at all.

WHAT IT DOES AND DOES NOT DO

It refines a SCALE and a per-building correction. It does not invent geometry:
a building with no visible shadow (occluded, north-facing gap, sun too high)
receives no correction and is reported as unconstrained rather than adjusted
toward whatever the optimiser found convenient.

METHOD

Shadow casting is done by ray-marching the height field along the solar
direction, which is the standard shadow-map construction in a raster:

    a pixel is shadowed if, walking toward the sun, the terrain ever rises
    above the straight line from that pixel toward the sun.

Comparison uses intersection-over-union against the detected shadow mask, which
is robust to the detector's absolute threshold in a way a per-pixel difference
is not.
"""
import math

import numpy as np
import cv2


def cast_shadows(height_m: np.ndarray, gsd_m: float,
                 sun_elev_deg: float, sun_az_deg: float,
                 max_distance_m: float = 250.0,
                 step_m: float = None) -> np.ndarray:
    """
    Which pixels the height field puts in shadow, for a given sun position.

    Marches toward the sun and asks whether anything rises above the sight line.
    The march is in METRES converted to pixels, so the result does not silently
    change meaning when the grid resolution does.
    """
    if sun_elev_deg <= 0:
        return np.zeros(height_m.shape, bool)

    h, w = height_m.shape
    if step_m is None:
        step_m = max(gsd_m, 0.5)
    n_steps = max(1, int(max_distance_m / step_m))

    rad_az = math.radians(sun_az_deg)
    # Toward the sun in image axes: x east, y south (row index grows southward).
    dx, dy = math.sin(rad_az), -math.cos(rad_az)
    tan_el = math.tan(math.radians(sun_elev_deg))

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    shadowed = np.zeros((h, w), bool)
    base = height_m.astype(np.float32)

    for i in range(1, n_steps + 1):
        d_m = i * step_m
        sx = xs + dx * (d_m / gsd_m)
        sy = ys + dy * (d_m / gsd_m)
        valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        if not valid.any():
            break
        xi = np.clip(sx, 0, w - 1).astype(np.int32)
        yi = np.clip(sy, 0, h - 1).astype(np.int32)
        # Height of the sight line toward the sun at this distance.
        ray = base + d_m * tan_el
        blocked = valid & (base[yi, xi] > ray)
        shadowed |= blocked
        # Once every remaining pixel is already shadowed there is nothing left
        # to learn from marching further.
        if shadowed.all():
            break
    return shadowed


def consistency(height_m: np.ndarray, observed_shadow: np.ndarray, gsd_m: float,
                sun_elev_deg: float, sun_az_deg: float,
                exclude: np.ndarray = None) -> dict:
    """
    Agreement between predicted and observed shadows, as IoU.

    `exclude` masks out surfaces whose darkness is not a cast shadow -- water
    and dense canopy read dark for their own reasons, and scoring them would
    reward a height field for the wrong reason.
    """
    pred = cast_shadows(height_m, gsd_m, sun_elev_deg, sun_az_deg)
    obs = observed_shadow.astype(bool)
    if exclude is not None:
        keep = ~exclude.astype(bool)
        pred, obs = pred & keep, obs & keep
    inter = int((pred & obs).sum())
    union = int((pred | obs).sum())
    return {
        "iou": float(inter / union) if union else 0.0,
        "predicted_frac": float(pred.mean()),
        "observed_frac": float(obs.mean()),
        "over_predicted": float((pred & ~obs).sum() / max(union, 1)),
        "under_predicted": float((obs & ~pred).sum() / max(union, 1)),
    }


def refine_scale(height_units: np.ndarray, observed_shadow: np.ndarray,
                 gsd_m: float, sun_elev_deg: float, sun_az_deg: float,
                 scale_guess: float, exclude: np.ndarray = None,
                 span: float = 0.6, n_probe: int = 9) -> dict:
    """
    Search for the scale whose predicted shadows best match the observed ones.

    A coarse-to-fine sweep rather than a gradient step: the objective is IoU on
    a binary mask, which is piecewise constant and has no useful gradient, so a
    direct search is both simpler and more honest about what is being optimised.

    Returns the refined scale and the IoU curve, so a caller can see whether the
    optimum is a genuine peak or a flat region where shadows say nothing.
    """
    lo, hi = scale_guess * (1 - span), scale_guess * (1 + span)
    best = {"scale": scale_guess, "iou": -1.0}
    curve = []
    for _ in range(2):                      # coarse, then refined around the peak
        probes = np.linspace(lo, hi, n_probe)
        for s in probes:
            r = consistency(height_units * s, observed_shadow, gsd_m,
                            sun_elev_deg, sun_az_deg, exclude)
            curve.append((float(s), r["iou"]))
            if r["iou"] > best["iou"]:
                best = {"scale": float(s), "iou": r["iou"]}
        step = (hi - lo) / (n_probe - 1)
        lo, hi = best["scale"] - step, best["scale"] + step

    ious = [c[1] for c in curve]
    peak_prominence = (max(ious) - float(np.median(ious))) if ious else 0.0
    return {
        "scale": best["scale"],
        "iou": best["iou"],
        "curve": curve,
        # A flat curve means shadows did not constrain the scale -- the caller
        # must not treat that optimum as a measurement.
        "constrained": bool(peak_prominence > 0.02),
        "peak_prominence": float(peak_prominence),
    }
