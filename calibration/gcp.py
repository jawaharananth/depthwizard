"""Scale from operator-supplied ground control: known heights on known buildings.

The problem statement names four routes to absolute scale: a DEM anchor, shadow
geometry, multi-view triangulation, and operator-supplied ground control. Three
are instrument routes -- they read the answer off data. This one is the fourth,
and it is different in kind: a person asserts that a particular structure is a
particular height, and the scene is scaled to agree.

That difference is carried through to the tier. A GCP-derived scale is NOT
Tier A. Tier A means an instrument fixed the datum and the pipeline can point at
the measurement; here the number's authority is the operator's, and a scene
scaled from a misremembered building height is confidently, precisely wrong with
no internal evidence of it. It gets its own tier so a reader is never misled
about where the metres came from.

WHY RATIO OF MEDIANS, NOT MEDIAN OF RATIOS

Both look like robust estimators. They are not equivalent, and the difference
matters at exactly the sizes an operator is likely to pick.

Each control point gives a ratio known/measured. Taking the median of those
ratios weights every point equally regardless of magnitude, so a 4 m shed
measured at 3 m contributes a ratio of 1.33 with the same authority as a 120 m
tower measured at 118 m contributing 1.017. The shed's ratio is dominated by an
error of one metre -- the pipeline's own noise floor -- and on a two-point fit
it can move the whole scene by 30%.

Ratio of medians -- median(known) / median(measured) -- forms the ratio once,
from two aggregates that are each already robust. A metre of error on a small
building is a metre inside a median, not a multiplier on the answer. The
estimator is scale-aware where median-of-ratios is not.

Both are reported. The solver uses ratio-of-medians and says what the other
would have given, because when the two disagree sharply the control points
themselves are inconsistent and the operator should see that rather than a
single confident number.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# A control point whose measured height is below this is refused. Near the
# pipeline's noise floor the ratio is mostly noise: at 1 m measured, half a
# metre of error is a 50% scale change.
MIN_MEASURED = 1.0

# Residual beyond this fraction of the known height marks a point as suspect in
# the report. Not auto-dropped -- the operator decides, because the outlier is
# as likely to be a mistyped height as a bad reconstruction.
SUSPECT_RESIDUAL_FRAC = 0.25

# Below this many points, or this narrow a height span, a power law and a flat
# ratio cannot be told apart -- fitting one anyway would report a confident
# exponent that is really just noise in log-space. A power law is worth
# reporting because monocular height error is often a height-dependent ratio
# rather than a fixed offset (tall structures in particular) -- but only where
# the data can actually distinguish the two forms.
MIN_POWER_LAW_POINTS = 4
MIN_POWER_LAW_SPAN = 3.0   # max(measured)/min(measured) ratio required


def _fit_power_law(meas: np.ndarray, known: np.ndarray) -> Optional[Dict]:
    """known = a * measured**b, fit by log-log least squares.

    Only worth reporting when the points actually span enough of a height
    range to separate "the error is a ratio" from "the error is an offset" --
    below MIN_POWER_LAW_POINTS/MIN_POWER_LAW_SPAN the two forms fit log-space
    data about equally well and the exponent is not a real measurement, just
    an artefact of which points happened to get clicked.
    """
    if len(meas) < MIN_POWER_LAW_POINTS:
        return None
    span = float(meas.max() / max(meas.min(), 1e-9))
    if span < MIN_POWER_LAW_SPAN:
        return None
    log_m, log_k = np.log(meas), np.log(known)
    A = np.column_stack([log_m, np.ones_like(log_m)])
    coef, *_ = np.linalg.lstsq(A, log_k, rcond=None)
    b, log_a = float(coef[0]), float(coef[1])
    a = float(np.exp(log_a))
    predicted = a * meas ** b
    resid = predicted - known
    # Compare against the flat ratio-of-medians prediction on the SAME points,
    # so "power law helps" is a measured comparison, not an assumption.
    flat_factor = float(np.median(known) / np.median(meas))
    flat_resid = meas * flat_factor - known
    return {
        "a": round(a, 6), "b": round(b, 4),
        "rmse_m": round(float(np.sqrt(np.mean(resid ** 2))), 3),
        "flat_rmse_m": round(float(np.sqrt(np.mean(flat_resid ** 2))), 3),
        "span": round(span, 2),
        "note": ("known = a * measured**b. b=1 means a power law found no "
                 "advantage over a flat ratio -- compare rmse_m against "
                 "flat_rmse_m before trusting the exponent."),
    }


def solve(points: Sequence[Dict]) -> Dict:
    """Solve a single scale factor from control points.

    Each point is ``{"measured": float, "known": float, "id": any}`` where
    ``measured`` is the height the pipeline currently reports for that building
    in whatever units the scene is in, and ``known`` is the operator's asserted
    height in metres.

    Returns a report carrying the factor, both estimators, and a per-point
    residual so a bad point can be identified and dropped.
    """
    usable, refused = [], []
    for p in points:
        m = float(p.get("measured", 0.0) or 0.0)
        k = float(p.get("known", 0.0) or 0.0)
        if m < MIN_MEASURED or k <= 0:
            refused.append({**p, "reason": (
                f"measured {m:.2f} below the {MIN_MEASURED:.1f} floor"
                if m < MIN_MEASURED else "known height must be positive")})
            continue
        usable.append({"id": p.get("id"), "measured": m, "known": k})

    if len(usable) < 2:
        return {"ok": False,
                "reason": f"need at least 2 usable control points, have {len(usable)}",
                "usable": usable, "refused": refused}

    meas = np.array([p["measured"] for p in usable], dtype=np.float64)
    known = np.array([p["known"] for p in usable], dtype=np.float64)

    factor = float(np.median(known) / np.median(meas))          # ratio of medians
    alt = float(np.median(known / meas))                        # median of ratios

    points_out = []
    for p in usable:
        predicted = p["measured"] * factor
        resid = predicted - p["known"]
        frac = abs(resid) / max(p["known"], 1e-9)
        points_out.append({
            "id": p["id"], "measured": round(p["measured"], 2),
            "known": round(p["known"], 2),
            "predicted_m": round(predicted, 2),
            "residual_m": round(resid, 2),
            "residual_frac": round(frac, 3),
            "suspect": bool(frac > SUSPECT_RESIDUAL_FRAC),
        })

    resid = np.array([p["residual_m"] for p in points_out], dtype=np.float64)
    disagreement = abs(factor - alt) / max(abs(factor), 1e-9)
    power_law = _fit_power_law(meas, known)

    # Leave-one-out cross-validation: refit the factor with each point held
    # out in turn, then score that point against the factor it had no part in
    # choosing. The in-sample residuals above are optimistic by construction
    # -- a 2-point fit has zero in-sample error on both points regardless of
    # whether either is trustworthy. LOO is the cheapest real check that a
    # fixed operator input can't game: a point can only look good here by
    # agreeing with the OTHER points, not by being included in its own fit.
    loo_resid = []
    if len(usable) >= 3:
        for i in range(len(usable)):
            held_out = usable[i]
            rest = usable[:i] + usable[i + 1:]
            rest_known = np.array([p["known"] for p in rest], dtype=np.float64)
            rest_meas = np.array([p["measured"] for p in rest], dtype=np.float64)
            loo_factor = float(np.median(rest_known) / np.median(rest_meas))
            loo_resid.append(held_out["measured"] * loo_factor - held_out["known"])
    loo_resid = np.array(loo_resid, dtype=np.float64)
    loo_rmse = float(np.sqrt(np.mean(loo_resid ** 2))) if loo_resid.size else None

    return {
        "ok": True,
        "factor": factor,
        "estimator": "ratio_of_medians",
        "alternative_median_of_ratios": alt,
        "estimator_disagreement_frac": round(float(disagreement), 3),
        # A wide gap between the two estimators means the control points do not
        # agree on one scale. Surfaced rather than hidden: the operator should
        # look at the points before trusting either number.
        "estimators_disagree": bool(disagreement > 0.10),
        "n_points": len(usable),
        "points": points_out,
        "residual_mae_m": round(float(np.mean(np.abs(resid))), 2),
        "residual_max_m": round(float(np.max(np.abs(resid))), 2),
        "n_suspect": int(sum(1 for p in points_out if p["suspect"])),
        "loo_rmse_m": round(loo_rmse, 2) if loo_rmse is not None else None,
        "loo_note": (None if loo_rmse is not None else
                     "need >=3 points for leave-one-out cross-validation "
                     "-- in-sample residuals above are optimistic with only 2"),
        "power_law": power_law,
        "power_law_note": (None if power_law is not None else
                           f"need >={MIN_POWER_LAW_POINTS} points spanning "
                           f">={MIN_POWER_LAW_SPAN:.0f}x in measured height to "
                           f"tell a power law apart from a flat ratio"),
        "refused": refused,
        "tier": TIER,
    }


# The tier string for a GCP-scaled scene. Deliberately distinct from
# "A (DEM-anchored ...)": both produce metres, but only one of them measured
# anything. A reader must be able to tell which they are holding.
TIER = "A* (operator ground control -- asserted, not instrument-measured)"


def apply_factor(records: Sequence[Dict], factor: float) -> List[Dict]:
    """Rescale building heights by a solved factor, in place-safe copies.

    Intervals are rescaled with the heights. A conformal band fitted in one
    scale is a statement about a proportion of that scale; carrying the raw
    metre width across a rescale would attach a band fitted at one size to
    heights at another.
    """
    out = []
    for r in records:
        r2 = dict(r)
        for key in ("height_m", "base_h", "roof_h"):
            if r2.get(key) is not None:
                r2[key] = round(float(r2[key]) * factor, 2)
        if r2.get("interval_half_width_m") is not None:
            r2["interval_half_width_m"] = round(
                float(r2["interval_half_width_m"]) * factor, 2)
        out.append(r2)
    return out


def format_report(res: Dict) -> str:
    """Human-readable solve report."""
    if not res.get("ok"):
        return f"GCP solve failed: {res.get('reason')}"
    lines = [
        f"GCP scale: x{res['factor']:.4f}  ({res['estimator']}, "
        f"{res['n_points']} points)",
        f"  residual MAE {res['residual_mae_m']:.2f} m, "
        f"max {res['residual_max_m']:.2f} m (in-sample, optimistic)",
    ]
    if res.get("loo_rmse_m") is not None:
        lines.append(f"  leave-one-out RMSE {res['loo_rmse_m']:.2f} m "
                     f"(each point scored against a factor it had no part in)")
    elif res.get("loo_note"):
        lines.append(f"  {res['loo_note']}")
    if res["estimators_disagree"]:
        lines.append(
            f"  WARNING: median-of-ratios would give x"
            f"{res['alternative_median_of_ratios']:.4f} "
            f"({res['estimator_disagreement_frac']*100:.0f}% apart) -- the "
            f"control points do not agree on one scale")
    pl = res.get("power_law")
    if pl is not None:
        better = pl["rmse_m"] < pl["flat_rmse_m"]
        lines.append(
            f"  power-law check: known = {pl['a']:.3f} * measured^{pl['b']:.3f} "
            f"-> RMSE {pl['rmse_m']:.2f} m vs flat-ratio RMSE {pl['flat_rmse_m']:.2f} m "
            f"({'power law fits better -- error may be a height-dependent ratio' if better else 'flat ratio already as good, no exponent worth trusting'})")
    elif res.get("power_law_note"):
        lines.append(f"  {res['power_law_note']}")
    for p in res["points"]:
        flag = "  SUSPECT" if p["suspect"] else ""
        lines.append(f"  point {p['id']}: measured {p['measured']:.1f} -> "
                     f"{p['predicted_m']:.1f} m vs known {p['known']:.1f} m  "
                     f"(residual {p['residual_m']:+.2f} m){flag}")
    for p in res.get("refused", []):
        lines.append(f"  refused {p.get('id')}: {p.get('reason')}")
    lines.append(f"  tier -> {res['tier']}")
    return "\n".join(lines)
