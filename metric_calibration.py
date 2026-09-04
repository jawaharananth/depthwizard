"""
Metric scale by confidence-weighted fusion, not a fallback chain.

WHY FUSION RATHER THAN "PICK THE FIRST AVAILABLE"

The pipeline had a priority ladder: use the DEM if there is one, else shadows,
else assume. That throws away agreement. When two independent signals both put
the scale near the same value, the combined estimate deserves more confidence
than either alone -- and when they disagree, that disagreement is itself
information the caller should see rather than have silently resolved by
precedence.

Each signal contributes an estimate AND a weight derived from its own internal
consistency, so a signal that disagrees with itself cannot outvote one that does
not, regardless of where it sat in the old ladder.

A CAUTION LEARNED HERE, NOT ASSUMED

Naive fusion can be worse than picking one signal. Measured earlier in this
project: using MVS for both building shape and building height was worse than
using monocular for shape and MVS for height, because MVS is spatially noisy
even though it is metrically accurate. Combining signals is only right when they
are estimating THE SAME QUANTITY -- here, metres per unit -- which is why this
module fuses scales and not surfaces.

SIGNALS

  dem     a DEM offset fitted over pixels both surfaces call bare ground.
          Weight from the spread of that fit: a tight IQR means the two agree
          about terrain everywhere, not just on average.
  shadow  h = L * tan(sun elevation) per building. Weight from the spread of
          the per-building estimates.
  mvs     triangulated absolute metres. Scale is 1.0 by construction, and the
          weight is the mean photometric confidence.
  prior   a stated assumption about city form. Deliberately given a small fixed
          weight so it can break a tie but never override a measurement.
"""
import numpy as np

# The prior is an assumption, not an observation, so its weight is capped far
# below what any real signal earns. It exists to keep the fusion defined when
# nothing else is available, not to contribute to a measured answer.
PRIOR_WEIGHT = 0.05


def _weight_from_spread(spread_ratio: float) -> float:
    """
    Turn a relative spread into a weight.

    A signal whose own estimates disagree by as much as their median tells us
    almost nothing, so weight falls off as 1/(1+spread). This is deliberately
    gentle rather than an exponential: a moderately noisy signal should be
    down-weighted, not silenced, because on many scenes it is the only one
    present.
    """
    if spread_ratio is None or not np.isfinite(spread_ratio) or spread_ratio < 0:
        return 0.0
    return float(1.0 / (1.0 + max(spread_ratio, 0.0)))


def fuse(signals: dict) -> dict:
    """
    signals: {name: {"scale": float, "spread_ratio": float, "n": int}}

    Returns the fused scale, the weight each signal received, and an agreement
    figure so a caller can tell consensus from a lone voice.
    """
    usable = {}
    for name, s in (signals or {}).items():
        sc = s.get("scale")
        if sc is None or not np.isfinite(sc) or sc <= 0:
            continue
        if name == "prior":
            continue          # handled after the measured signals; see below
        w = _weight_from_spread(s.get("spread_ratio"))
        # Sample count MODULATES the weight; it must not dominate it.
        #
        # A first version multiplied by sqrt(n)/10, which let a noisy signal with
        # many samples outvote a clean one with few: shadow at spread 0.75 and
        # n=40 received weight 0.57 against MVS at spread 0.10 and n=6 on 0.35.
        # That inverts the whole point of weighting by internal consistency.
        # Bounded to [0.5, 1.0], more observations help but cannot rescue a
        # signal that disagrees with itself.
        n = max(1, int(s.get("n", 1)))
        w *= 0.5 + 0.5 * float(min(n, 50)) / 50.0
        if w > 0:
            usable[name] = {"scale": float(sc), "weight": float(w)}

    # The prior is an ASSUMPTION and only speaks when nothing was measured.
    #
    # Giving it a small permanent weight was wrong: with two measurements at
    # 1.00 and 1.06, a prior of 2.0 at weight 0.08 still dragged the fused
    # answer to 1.113. A stated assumption must never move a measured result,
    # only stand in for one that does not exist.
    if not usable and (signals or {}).get("prior", {}).get("scale"):
        pr = signals["prior"]
        usable["prior"] = {"scale": float(pr["scale"]), "weight": PRIOR_WEIGHT}

    if not usable:
        return {"scale": None, "reason": "no usable calibration signal",
                "contributions": {}}

    tot = sum(v["weight"] for v in usable.values())
    fused = sum(v["scale"] * v["weight"] for v in usable.values()) / tot

    # Agreement: how far the signals sit from the fused answer, relative to it.
    # Reported rather than acted on -- a caller may reasonably refuse a fusion
    # whose inputs disagree, and that decision is not this function's to make.
    if len(usable) > 1:
        dev = [abs(v["scale"] - fused) / max(fused, 1e-6) for v in usable.values()]
        agreement = float(1.0 - min(1.0, float(np.mean(dev))))
    else:
        agreement = None

    return {
        "scale": float(fused),
        "agreement": agreement,
        "n_signals": len(usable),
        "contributions": {k: {"scale": round(v["scale"], 3),
                              "weight": round(v["weight"] / tot, 3)}
                          for k, v in usable.items()},
    }


def describe(result: dict) -> str:
    if result.get("scale") is None:
        return f"no metric scale ({result.get('reason', 'unknown')})"
    parts = "  ".join(
        f"{k} {v['scale']:.3g} (w={v['weight']:.2f})"
        for k, v in result["contributions"].items())
    ag = result.get("agreement")
    tail = f", agreement {ag:.2f}" if ag is not None else " (single signal)"
    return f"scale {result['scale']:.4g} from {result['n_signals']} signal(s){tail}\n     {parts}"
