"""Does the inference backend change accuracy? Per-building RMSE on the 18
held-out DFC2019 tiles, at native and at ISRO's 0.6 m, for whichever backend
depth_model picks (set DW_NO_OV_GPU=1 for CPU).

    python scripts/backend_accuracy.py            # iGPU if available
    DW_NO_OV_GPU=1 python scripts/backend_accuracy.py

Same tiles, truth and building instances as scripts/resolution_test.py; it
does not overwrite that script's outputs. Writes final_out/backend_accuracy_<tag>.json.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import resolution_test as R  # noqa: E402

R.CONDITIONS = (("0.25 m native", 0.25, True), ("0.60 m, upsampled", 0.60, True))
_out = {}
_orig_dump = json.dump


def _capture(obj, fp, *a, **k):
    _out.update(obj)                      # keep the numbers, write nothing of resolution_test's


json.dump = _capture
_open = open


def _guarded_open(path, mode="r", *a, **k):
    if "w" in mode and os.path.basename(str(path)) in ("BENCHMARK_RESOLUTION.md", "resolution_test.json"):
        return _open(os.devnull, mode, *a, **k)
    return _open(path, mode, *a, **k)


import builtins  # noqa: E402
builtins.open = _guarded_open
t0 = time.time()
R.main()
builtins.open = _open
json.dump = _orig_dump
tag = "cpu" if os.environ.get("DW_NO_OV_GPU") else "igpu"
res = {k: {m: v[m] for m in ("rmse", "mae", "bldg_rmse", "bldg_bias", "n_buildings") if m in v}
       for k, v in _out.items()}
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(ROOT, "final_out", f"backend_accuracy_{tag}.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
