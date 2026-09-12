"""
Fit the conformal quantile ONCE and commit it.

This is the only place in the project allowed to fit a conformal quantile.
The pipeline loads the artifact this writes (calibration/conformal_quantile.json)
via calibration.conformal.load_fitted(); it never fits. See that module's
docstring for why -- in short, a per-scene fit needs that scene's ground truth,
which does not exist at evaluation, and fitting then scoring on the same
buildings is circular.

METHOD

Buildings are measured against LiDAR by validate_buildings.collect_rows(), the
same function that produces this project's published per-building accuracy, so
the interval is fitted on exactly the errors that accuracy describes.

The set is split in half:
  - the calibration half fits the quantile
  - the held-out half measures coverage, which is recorded in the artifact

Reporting coverage on the calibration half would be meaningless (the quantile
is chosen to make it come out right), so the number committed here is the
held-out one, and it is committed alongside the quantile rather than in a
separate document that can drift away from it.

BOTH VARIANTS ARE FITTED, ONE IS SELECTED

  constant            every building gets the same half-width
  confidence-adaptive half-width scales with 1/confidence, so a building the
                      photometry barely constrained earns wider bounds

The adaptive variant is preferred when it achieves at least nominal coverage,
since it spends width where the evidence is weak instead of spreading it
evenly. If it under-covers, the constant variant is selected instead -- a
tighter interval that does not hold is worse than a wide one that does.

Usage:
  python scripts/fit_conformal.py                 # JAX_165, alpha=0.10
  python scripts/fit_conformal.py JAX_165 0.10
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import validate_buildings as vb
from calibration import conformal as cf

ARTIFACT = os.path.join(ROOT, "calibration", "conformal_quantile.json")


def fit(tile: str = "JAX_165", alpha: float = 0.10, extent_m: float = 640.0,
        out_px: int = 2560, seed: int = 0) -> dict:
    a, truth_extent_m = vb.collect_rows(tile, extent_m=extent_m, out_px=out_px)
    ours, true, area, bconf = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    err = ours - true

    idx = np.arange(len(err))
    np.random.default_rng(seed).shuffle(idx)
    half = len(idx) // 2
    cal_i, test_i = idx[:half], idx[half:]

    # Low confidence must WIDEN the interval, so the scale is inverted.
    unc = 1.0 / np.clip(bconf, 0.05, 1.0)

    variants = {}
    for name, u_cal, u_test in (("constant", None, None),
                                ("confidence_adaptive", unc[cal_i], unc[test_i])):
        c = cf.calibrate(err[cal_i], alpha=alpha, uncertainty=u_cal)
        if c.get("q") is None:
            variants[name] = {"q": None, "reason": c.get("reason")}
            continue
        cov = cf.check_coverage(err[test_i], c, uncertainty=u_test)
        c["held_out_coverage"] = cov["coverage"]
        c["mean_half_width_m"] = cov["mean_half_width_m"]
        c["held_out_n"] = cov["n"]
        variants[name] = c

    # Prefer the adaptive variant only if it actually holds its nominal rate.
    # A tighter interval that under-covers is worse than a wider one that does
    # not, and this is the one decision in this script where the tempting
    # choice (always take the tighter number) is the wrong one.
    nominal = 1.0 - alpha
    adaptive = variants.get("confidence_adaptive", {})
    constant = variants.get("constant", {})
    if adaptive.get("q") is not None and (adaptive.get("held_out_coverage") or 0) >= nominal:
        selected = "confidence_adaptive"
    elif constant.get("q") is not None:
        selected = "constant"
    else:
        raise SystemExit("neither variant produced a usable quantile; "
                         f"constant: {constant.get('reason')}, "
                         f"adaptive: {adaptive.get('reason')}")

    chosen = dict(variants[selected])
    chosen.update({
        "variant": selected,
        "selection_reason": (
            "adaptive met nominal coverage on the held-out half"
            if selected == "confidence_adaptive" else
            "adaptive under-covered on the held-out half (or was unavailable); "
            "constant width selected because an interval that does not hold its "
            "rate is worse than a wider one that does"),
        "fitted_on": {
            "tile": tile, "extent_m": extent_m, "out_px": out_px,
            "n_buildings_total": int(len(err)),
            "n_calibration": int(len(cal_i)),
            "n_held_out": int(len(test_i)),
            "split_seed": seed,
            "height_source": "RPC plane-sweep MVS (validate_buildings.collect_rows)",
            "truth": "DFC2019 Track 3 airborne LiDAR",
        },
        "fitted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "fitted_by": "scripts/fit_conformal.py",
        "all_variants": variants,
        "usage_note": (
            "Loaded by calibration.conformal.load_fitted(). Applies ONLY to "
            "heights in real metres (Tier A/B). A Tier C scene's heights are "
            "in an assumed relative scale, so a metre band on them would be "
            "arithmetic on units that do not exist."),
    })
    return chosen


def main():
    tile = sys.argv[1] if len(sys.argv) > 1 else "JAX_165"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10

    print(f"fitting conformal quantile on {tile}, alpha={alpha} "
          f"({int((1-alpha)*100)}% nominal)\n")
    cal = fit(tile, alpha=alpha)

    print(f"  variant selected : {cal['variant']}")
    print(f"  reason           : {cal['selection_reason']}")
    print(f"  q                : {cal['q']:.4f}"
          + ("  (metres)" if cal["variant"] == "constant"
             else "  (metres per unit of 1/confidence)"))
    print(f"  mean half-width  : {cal['mean_half_width_m']:.2f} m")
    print(f"  held-out coverage: {cal['held_out_coverage']*100:.1f}% "
          f"(nominal {(1-alpha)*100:.0f}%, n={cal['held_out_n']})")
    print(f"  fitted on        : {cal['fitted_on']['n_buildings_total']} buildings, "
          f"{tile}")
    print()
    for name, v in cal["all_variants"].items():
        if v.get("q") is None:
            print(f"    {name:22s} unavailable: {v.get('reason')}")
        else:
            print(f"    {name:22s} half-width {v['mean_half_width_m']:5.2f} m   "
                  f"held-out coverage {v['held_out_coverage']*100:5.1f}%")

    with open(ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"\nwrote {ARTIFACT}")
    print("commit this file -- the pipeline loads it and must never refit.")


if __name__ == "__main__":
    main()
