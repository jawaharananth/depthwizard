"""Repair cursor rasters written before hover_export._decode existed.

    python tools/repair_hover_nodata.py            # every scene + viewer/output

burn_buildings used to read hover_agl.u16 / hover_sigma.u16 back without
mapping the no-data code, so a masked image border became 655.35 m and was
re-encoded as 65534 (the largest real value). No real height or sigma is
655.34 m, so 65534 is restored to no-data and the panel facts recomputed.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hover_export as hx  # noqa: E402


def repair(d):
    p = os.path.join(d, "scene.json")
    if not os.path.exists(p):
        return None
    meta = json.load(open(p))
    hv = meta.get("hover")
    if not hv:
        return None
    w, h = hv["width"], hv["height"]
    fixed, arrays = 0, {}
    for key in ("agl", "sigma"):
        if not hv.get(key):
            continue
        f = os.path.join(d, hv[key])
        if not os.path.exists(f):
            continue
        q = np.fromfile(f, dtype="<u2").reshape(h, w)
        bad = q == hx.NODATA - 1
        if bad.any():
            q = np.where(bad, hx.NODATA, q).astype("<u2")
            q.tofile(f)
            fixed += int(bad.sum())
        arrays[key] = np.where(q == hx.NODATA, np.nan, q.astype(np.float32) / 100)
    if fixed:
        facts = meta.get("facts") or {}
        fresh = hx.scene_facts(arrays["agl"], arrays.get("sigma"), facts.get("pixel_m"))
        fresh.pop("ground_m", None); fresh.pop("pixel_m", None)
        meta["facts"] = {**facts, **fresh}
        json.dump(meta, open(p, "w"), indent=2)
    return fixed


if __name__ == "__main__":
    dirs = (sorted(glob.glob(os.path.join(ROOT, "scenes", "*"))) + [os.path.join(ROOT, "viewer", "output")]
            + sorted(glob.glob(os.path.join(ROOT, "cache", "uploads", "*"))))
    for d in dirs:
        n = repair(d)
        if n:
            print(f"{os.path.relpath(d, ROOT)}: {n} pixels restored to no-data")
