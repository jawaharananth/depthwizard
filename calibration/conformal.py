"""
Conformal prediction intervals for building heights.

WHAT THIS ADDS OVER A CONFIDENCE SCORE

The pipeline already reports a per-building confidence in [0, 1]. That number is
ordinal: it says one building is better supported than another, but not what it
means. "Confidence 0.6" does not tell anyone how wrong the height might be.

Split conformal prediction converts it into a statement with a guarantee:

    this building is 24.1 m, and the true height lies within +-2.3 m
    at least 90% of the time

The guarantee is distribution-free and finite-sample. It assumes only
EXCHANGEABILITY between the calibration set and the tiles it is applied to --
not Gaussian errors, not a correctly specified model, not anything about how the
heights were produced. That is what makes it applicable to a pipeline like this
one, which is a stack of heuristics and classical CV rather than a single
probabilistic model.

HOW IT WORKS

1. On a calibration set where LiDAR truth exists, compute each building's
   nonconformity score -- here the absolute height error, optionally normalised
   by that building's own confidence.
2. Take the ceil((n+1)(1-alpha))/n quantile of those scores. That specific
   quantile, not the plain (1-alpha) one, is what makes the coverage guarantee
   hold for finite n rather than only asymptotically.
3. On new buildings, the interval is the prediction plus or minus that quantile.

NORMALISED VARIANT

Using raw error gives every building the same interval width, which wastes the
confidence signal: a well-observed building deserves a tighter interval than one
in deep shadow. Dividing the score by an uncertainty proxy before taking the
quantile produces intervals that adapt per building while preserving the same
marginal coverage.

HONEST LIMIT

Coverage is MARGINAL, averaged over the population. It does not promise 90%
coverage within every subgroup: if low buildings are systematically worse, the
interval can under-cover there while the overall guarantee still holds. Subgroup
coverage is therefore measured and reported separately rather than assumed.

FITTED ONCE, COMMITTED, NEVER REFIT PER SCENE

The quantile lives in `calibration/conformal_quantile.json`, fitted by
`scripts/fit_conformal.py` on a named calibration tile where LiDAR truth
exists, and committed. The pipeline LOADS it; it does not fit.

That is not a convenience -- it is what makes the guarantee mean anything. A
quantile refitted per scene would be fitted on that scene's own errors, which
requires that scene's ground truth. At real evaluation there is none, so a
per-scene fit is either impossible or silently fitted against something that is
not truth. Worse, a quantile fitted and then evaluated on the same buildings is
circular: the quantile is chosen to make coverage come out right, so it always
does. `load_fitted()` therefore returns the committed artifact or None, and
every caller must treat None as "no interval", never as "fit one now".
"""
import json
import os

import numpy as np

ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "conformal_quantile.json")


def load_fitted(path: str = None) -> dict:
    """
    The committed quantile, or None when it has not been fitted yet.

    Returns the same dict shape calibrate() produces, plus the provenance
    fields fit_conformal.py records (which tile, when, held-out coverage), so a
    caller can say WHERE a published interval came from rather than presenting
    a bare number.

    Never fits. A missing or malformed artifact returns None, and the caller
    must then omit the interval -- see the module docstring for why refitting
    on the scene in hand would destroy the guarantee rather than restore it.
    """
    p = path or ARTIFACT_PATH
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            cal = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if cal.get("q") is None:
        return None
    return cal


def calibrate(errors: np.ndarray, alpha: float = 0.10,
              uncertainty: np.ndarray = None) -> dict:
    """
    Fit the conformal quantile on a set where truth is known.

    errors:      signed or absolute height errors, in metres
    alpha:       miscoverage rate; 0.10 gives a 90% interval
    uncertainty: optional per-building scale (larger = less certain). When
                 given, intervals adapt per building instead of being constant.
    """
    e = np.abs(np.asarray(errors, dtype=np.float64))
    e = e[np.isfinite(e)]
    n = e.size
    if n < 20:
        return {"q": None, "n": n,
                "reason": f"only {n} calibration points; too few for a "
                          f"meaningful {int((1-alpha)*100)}% quantile"}

    if uncertainty is not None:
        u = np.asarray(uncertainty, dtype=np.float64)
        u = u[np.isfinite(u)]
        if u.size == e.size:
            u = np.clip(u, 1e-3, None)
            scores = e / u
        else:
            u, scores = None, e
    else:
        u, scores = None, e

    # The finite-sample quantile level. Using the plain (1-alpha) empirical
    # quantile instead would give coverage only in the limit, and would
    # under-cover at the sample sizes available here.
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(scores, level))

    return {"q": q, "n": int(n), "alpha": float(alpha),
            "level": float(level), "normalised": u is not None,
            "median_abs_error": float(np.median(e))}


def interval(prediction_m: float, cal: dict,
             uncertainty: float = None) -> tuple:
    """Half-width interval for one building. Returns (low, high, half_width)."""
    if cal.get("q") is None:
        return (None, None, None)
    hw = cal["q"]
    if cal.get("normalised") and uncertainty is not None:
        hw = cal["q"] * max(float(uncertainty), 1e-3)
    return (prediction_m - hw, prediction_m + hw, hw)


def check_coverage(errors: np.ndarray, cal: dict,
                   uncertainty: np.ndarray = None) -> dict:
    """
    Empirical coverage on a held-out set.

    A conformal guarantee is only as good as the exchangeability assumption
    behind it, so it is checked rather than trusted. Coverage far below the
    nominal rate means the held-out data is not exchangeable with the
    calibration data -- a different city, a different sensor, a different season.
    """
    e = np.abs(np.asarray(errors, dtype=np.float64))
    if cal.get("q") is None:
        return {"coverage": None}
    if cal.get("normalised") and uncertainty is not None:
        hw = cal["q"] * np.clip(np.asarray(uncertainty, dtype=np.float64), 1e-3, None)
    else:
        hw = np.full_like(e, cal["q"])
    covered = e <= hw
    return {
        "coverage": float(covered.mean()),
        "nominal": float(1.0 - cal["alpha"]),
        "mean_half_width_m": float(np.mean(hw)),
        "n": int(e.size),
    }
