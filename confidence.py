"""
Per-pixel and per-building confidence.

WHY THIS IS ALMOST FREE

The plane sweep already computes, for every pixel, the best photometric
agreement it achieved across the views it triangulated from -- a normalised
cross-correlation in [-1, 1]. That is a real, physically meaningful measure of
how much the imagery actually constrains the height there, and it was being
written to disk and then ignored.

A textureless flat roof, a deep shadow, water, and a surface visible in only one
view all score low, and those are exactly the places the reconstruction should
not be trusted. Publishing one RMSE hides all of that; a confidence field says
WHERE the number applies.

Running several depth models and measuring their disagreement would be a second,
independent source of uncertainty, and a good one -- but it costs ~20 minutes of
CPU per tile, while this costs nothing because it is already computed.

WHAT THE NUMBERS MEAN

  ncc ~ 1.0    views agree strongly; height is well constrained
  ncc ~ 0.5    typical for real building surfaces here (scene mean 0.563)
  ncc < 0.3    weak evidence -- the plane sweep's own reject threshold
  ncc = -2.0   sentinel: no view pair ever saw this pixel
"""
import numpy as np
import cv2

# The plane sweep's own confidence threshold, reused so the map and the
# reconstruction agree about what "unreliable" means.
LOW_CONFIDENCE = 0.30


def to_image_grid(conf: np.ndarray, shape: tuple) -> np.ndarray:
    """Upsample the sweep's confidence to the image grid, clipped to [0, 1]."""
    up = cv2.resize(conf.astype(np.float32), (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    # -2.0 marks pixels no view pair covered; those are zero confidence, not
    # negative correlation, and must not be rescaled as if they were.
    up = np.where(up < -1.0, 0.0, up)
    return np.clip(up, 0.0, 1.0)


def colourise(conf01: np.ndarray) -> np.ndarray:
    """
    Confidence as an image: red where the reconstruction is weakly supported,
    green where it is well constrained.

    Deliberately NOT a rainbow. A red-to-green ramp has an unambiguous reading
    -- worse to better -- whereas a rainbow implies an ordering viewers have to
    be taught, and hides detail in its yellow band.
    """
    c = np.clip(conf01, 0.0, 1.0)
    rgb = np.zeros((*c.shape, 3), np.uint8)
    rgb[:, :, 0] = np.clip((1.0 - c) * 255 * 1.15, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(c * 255 * 1.10, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(60 + c * 40, 0, 255).astype(np.uint8)
    return rgb


def summarise(conf01: np.ndarray) -> dict:
    v = conf01[np.isfinite(conf01)]
    if v.size == 0:
        return {"mean": None}
    return {
        "mean": round(float(v.mean()), 3),
        "median": round(float(np.median(v)), 3),
        "frac_low": round(float((v < LOW_CONFIDENCE).mean()), 3),
        "frac_high": round(float((v > 0.7).mean()), 3),
    }


def building_confidence(conf01: np.ndarray, poly: np.ndarray) -> dict:
    """
    Confidence for one building, from the pixels inside its own footprint.

    The MEDIAN is used, not the mean: a single well-textured vent or parapet
    should not vouch for an otherwise featureless roof, and one dead pixel
    should not condemn a good one.
    """
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    x0, y0 = max(x, 0), max(y, 0)
    x1 = min(x + w, conf01.shape[1])
    y1 = min(y + h, conf01.shape[0])
    if x1 <= x0 or y1 <= y0:
        return {"confidence": None, "n_px": 0}
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32) - [x0, y0]], 1)
    mb = m.astype(bool)
    if mb.sum() < 4:
        return {"confidence": None, "n_px": int(mb.sum())}
    vals = conf01[y0:y1, x0:x1][mb]
    return {
        "confidence": round(float(np.median(vals)), 3),
        "frac_low": round(float((vals < LOW_CONFIDENCE).mean()), 3),
        "n_px": int(mb.sum()),
    }


def reliability_tier(building_conf: float, evidence: dict,
                     height_is_metric: bool) -> str:
    """
    A per-building reliability tag, decided by that building's OWN evidence.

    The scene-level tier (A/B/C) describes how the whole surface was scaled.
    It says nothing about whether any particular building was well observed --
    a scene can be Tier A while one roof inside it sits in deep shadow with no
    photometric support at all. This grades each building separately, which is
    the honest granularity: reconstruction quality is not uniform across a
    scene, so a single tag for the whole scene overstates the weak parts and
    understates the strong ones.
    """
    if building_conf is None:
        return "UNVERIFIED"
    ev = evidence or {}
    strong_signals = sum([
        building_conf >= 0.6,
        float(ev.get("height", 0)) >= 0.5,
        float(ev.get("edge", 0)) >= 0.3,
        float(ev.get("shadow", 0)) >= 0.5,
    ])
    if not height_is_metric:
        # Relative heights can be well OBSERVED but are never well MEASURED.
        return "OBSERVED" if strong_signals >= 2 else "WEAK"
    if building_conf < LOW_CONFIDENCE:
        return "WEAK"
    if strong_signals >= 3:
        return "HIGH"
    if strong_signals >= 2:
        return "MEDIUM"
    return "LOW"
