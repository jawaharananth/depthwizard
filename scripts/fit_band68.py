"""Height-aware 68 % (1-sigma) building band, same form as calibration/conformal_height_aware.json.

    python scripts/fit_band68.py

half-width = q68 * (1 + k * predicted height), k = 0.14 (the 90 % fit's k), q68 = 68th percentile of
|pred - LiDAR| / (1 + k * pred) over 458 held-out DFC2019 buildings (final_out/run03_building_errors.json).
Leave-one-tile-out coverage is recorded; claim limited to 8 m like the 90 % band.
Output: calibration/conformal_height_aware_68.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(ROOT, "final_out", "run03_building_errors.json"), encoding="utf-8"))
tid = np.array([r[0] for r in rows]); gt = np.array([r[1] for r in rows], float); hp = np.array([r[2] for r in rows], float)
K, LVL = 0.14, 0.68
s = np.abs(hp - gt) / (1 + K * hp)
q = float(np.quantile(s, LVL))
cov = np.concatenate([np.abs(hp[tid == t] - gt[tid == t]) <= np.quantile(s[tid != t], LVL) * (1 + K * hp[tid == t])
                      for t in sorted(set(tid.tolist()))])
art = {"method": "half-width = q * (1 + k * pred), 68 % level (1 sigma), building median height",
       "k": K, "q": round(q, 4), "level": LVL, "loto_coverage": round(float(cov.mean()), 4), "n": len(rows),
       "claim_max_height_m": 8, "data": "final_out/run03_building_errors.json", "fitted_by": "scripts/fit_band68.py"}
json.dump(art, open(os.path.join(ROOT, "calibration", "conformal_height_aware_68.json"), "w"), indent=1)
print(art)
