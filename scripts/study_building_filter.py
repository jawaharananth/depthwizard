"""Which exported buildings to keep? Rescore the per-building benchmark under
candidate rules, offline, from the 'new' builds' GeoJSON evidence fields.

    python scripts/study_building_filter.py

Every rule only ever REMOVES traced (non-real-outline) buildings; a real
Microsoft/OSM outline is always kept. Reads work/bench_buildings and
final_out/city_bb_new_<tile>_buildings.geojson; touches nothing.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.argv = sys.argv[:1]
import benchmark_buildings as bb      # noqa: E402
import dfc2019_height as DH           # noqa: E402


def traced(p):
    return p.get("outline_real", 0.0) < 0.5


RULES = {
    "all": lambda p: True,
    "real_only": lambda p: not traced(p),
    "h>=3": lambda p: not traced(p) or p["height_m"] >= 3,
    "veg<0.5": lambda p: not traced(p) or p.get("veg_seg", 0) < 0.5,
    "exg<0.4": lambda p: not traced(p) or p.get("exg_frac", 0) < 0.4,
    "rough<0.35": lambda p: not traced(p) or p.get("roughness", 0) < 0.35,
    "chm<3": lambda p: not traced(p) or p.get("chm_median_m", 0) < 3,
    "chm<3&exg<0.4": lambda p: not traced(p) or (p.get("chm_median_m", 0) < 3 and p.get("exg_frac", 0) < 0.4),
    "chm<3&rough<0.35": lambda p: not traced(p) or (p.get("chm_median_m", 0) < 3 and p.get("roughness", 0) < 0.35),
    "strict": lambda p: not traced(p) or (p.get("chm_median_m", 0) < 3 and p.get("exg_frac", 0) < 0.4
                                          and p.get("roughness", 0) < 0.35 and p["height_m"] >= 2.5),
}


def mapped_rule(inner, radius_m=30.0):
    """Apply `inner` to traced buildings only where a real outline stands
    within radius_m (the area is mapped); elsewhere keep traced ones."""
    def run(fc):
        from shapely.geometry import shape
        from shapely.strtree import STRtree
        real = [shape(f["geometry"]) for f in fc["features"] if not traced(f["properties"])]
        tree = STRtree(real) if real else None
        out = []
        for f in fc["features"]:
            p = f["properties"]
            if not traced(p):
                out.append(f); continue
            c = shape(f["geometry"]).centroid
            near = tree is not None and len(tree.query(c.buffer(radius_m))) > 0
            if not near or inner(p):
                out.append(f)
        return out
    return run


FC_RULES = {
    "mapped:rough<0.35": mapped_rule(lambda p: p.get("roughness", 0) < 0.35),
    "mapped:drop": mapped_rule(lambda p: False),
    "mapped:chm<3&rough<0.35": mapped_rule(lambda p: p.get("chm_median_m", 0) < 3 and p.get("roughness", 0) < 0.35),
}


def main():
    tiles = [t for t in sorted(DH.benchmark_tiles())
             if os.path.exists(os.path.join(ROOT, "final_out", f"city_bb_new_{t}_buildings.geojson"))]
    res = {}
    allr = {**{k: ("p", v) for k, v in RULES.items()}, **{k: ("fc", v) for k, v in FC_RULES.items()}}
    for name, (kind, rule) in allr.items():
        rows, fs = [], []
        for tile in tiles:
            src = os.path.join(ROOT, "final_out", f"city_bb_new_{tile}_buildings.geojson")
            fc = json.load(open(src))
            fc["features"] = ([f for f in fc["features"] if rule(f["properties"])] if kind == "p"
                              else rule(fc))
            tmp = os.path.join(bb.WORK, f"_study_{tile}.geojson")
            json.dump(fc, open(tmp, "w"))
            r, f_, _ = bb.score(tile, tmp)
            rows += r; fs.append(f_)
        s = bb.summarise(rows)
        s["false_share_mean"] = float(np.mean(fs))
        res[name] = s
        print(f"{name:18s} RMSE {s['rmse_m']:.2f}  median {s['median_abs_m']:.2f}  within2 {s['within_2m']:.2f}  "
              f"found {s['found']:.2f}  false {s['false_share_mean']:.2f}  bias {s['bias_m']:+.2f}", flush=True)
    json.dump({"tiles": tiles, "rules": res}, open(os.path.join(ROOT, "final_out", "study_building_filter.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
