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
        f"max {res['residual_max_m']:.2f} m",
    ]
    if res["estimators_disagree"]:
        lines.append(
            f"  WARNING: median-of-ratios would give x"
            f"{res['alternative_median_of_ratios']:.4f} "
            f"({res['estimator_disagreement_frac']*100:.0f}% apart) -- the "
            f"control points do not agree on one scale")
    for p in res["points"]:
        flag = "  SUSPECT" if p["suspect"] else ""
        lines.append(f"  point {p['id']}: measured {p['measured']:.1f} -> "
                     f"{p['predicted_m']:.1f} m vs known {p['known']:.1f} m  "
                     f"(residual {p['residual_m']:+.2f} m){flag}")
    for p in res.get("refused", []):
        lines.append(f"  refused {p.get('id')}: {p.get('reason')}")
    lines.append(f"  tier -> {res['tier']}")
    return "\n".join(lines)
