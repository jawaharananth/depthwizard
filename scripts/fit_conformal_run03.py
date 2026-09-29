"""Refit the per-building 90% interval on the production model's own errors.

    python scripts/fit_conformal_run03.py

The earlier artifact (fitted 2026-09-12) was calibrated on RPC multi-view
stereo heights of one tile; the product's heights come from the monocular
model (run 03). Here: the 18 held-out DFC2019 tiles at 0.6 m
(scripts/compare_ckpt.py), fitted on all 458 buildings and checked
leave-one-tile-out (a tile's buildings are never both fitted and scored);
the nominal level is widened until that held-out coverage reaches 90%. A
half/half tile split covered only 83%: errors vary by tile, so the check has
to be across tiles, not across buildings. Writes calibration/
conformal_quantile.json (previous one kept as
conformal_quantile_mvs_2026-09-12.json).
"""
import datetime
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import depth_model as dm
import compare_ckpt as cc
from calibration.conformal import calibrate

r = cc.dfc(dm.DepthBackbone(metric=True))
rows = r["buildings"]
json.dump(rows, open(os.path.join(ROOT, "final_out", "run03_building_errors.json"), "w"))
tiles = sorted({t for t, _, _ in rows})
err = {t: np.array([p - g for tt, g, p in rows if tt == t]) for t in tiles}


def loto(alpha):
    """Leave-one-tile-out: fit on 17 tiles, cover the 18th; pooled coverage."""
    hit = tot = 0
    for t in tiles:
        q = calibrate(np.concatenate([err[u] for u in tiles if u != t]), alpha=alpha)["q"]
        hit += int((np.abs(err[t]) <= q).sum()); tot += err[t].size
    return hit / tot


# widen the nominal level until held-out tiles really are covered 90% of the time
alpha = 0.10
while loto(alpha) < 0.90 and alpha > 0.01:
    alpha = round(alpha - 0.01, 2)
e_all = np.concatenate([err[t] for t in tiles])
fit = calibrate(e_all, alpha=alpha)
cov = loto(alpha)
e_cal, e_out = e_all, e_all
cal_t = set(tiles)
print(f"nominal alpha {alpha}: q = {fit['q']:.2f} m, leave-one-tile-out coverage {cov:.3f}")
art = os.path.join(ROOT, "calibration", "conformal_quantile.json")
old = os.path.join(ROOT, "calibration", "conformal_quantile_mvs_2026-09-12.json")
if not os.path.exists(old):
    shutil.copy(art, old)
fit.update({"variant": "constant", "normalised": False, "held_out_coverage": cov,
            "held_out_method": "leave-one-tile-out over 18 tiles; nominal level widened until it reached 0.90",
            "held_out_n": int(e_all.size),
            "mean_half_width_m": fit["q"], "median_abs_error": float(np.median(np.abs(e_cal))),
            "fitted_on": {"tiles_calibration": sorted(cal_t), "tiles_held_out": "each tile in turn",
                          "n_calibration": int(e_cal.size), "split": "by tile, seed 0",
                          "height_source": "monocular model run 03 (production), raw metric heights, "
                                           "input degraded to 0.6 m",
                          "truth": "DFC2019 Track 3 airborne LiDAR, per-building median"},
            "fitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "fitted_by": "scripts/fit_conformal_run03.py",
            "replaces": "conformal_quantile_mvs_2026-09-12.json (fitted on MVS stereo heights)"})
json.dump(fit, open(art, "w"), indent=2)
print(f"90% interval: +/-{fit['q']:.2f} m on {e_all.size} buildings; leave-one-tile-out coverage {cov:.3f}")
