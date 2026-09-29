"""Per-building height of what the viewer DRAWS, against airborne LiDAR.

    python scripts/benchmark_buildings.py new      # the server's upload settings
    python scripts/benchmark_buildings.py old      # previous upload settings
    python scripts/benchmark_buildings.py new JAX_004 JAX_017   # a subset

The model benchmarks (compare_ckpt, resolution_test) score the height map;
a judge sees building blocks. This scores the blocks: each of the 18
held-out DFC2019 tiles is degraded to ISRO's 0.6 m, uploaded through the
full production build exactly as the server runs it, the exported building
footprints are rasterised at their drawn heights onto the LiDAR grid, and
every real building (LiDAR class 6, >= 10 m^2) gets

    truth = median LiDAR height above ground inside it
    drawn = median of the drawn surface inside it (0 where nothing is drawn)

so a missed building, a merged block, a wrong height and a misplaced outline
all count. Nothing is fitted to the truth; it is read only in score().
Writes final_out/benchmark_buildings_<config>.json.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_city_image as bci           # noqa: E402
import dfc2019_height as DH              # noqa: E402
import dfc2019_loader as L               # noqa: E402
from rescore_baseline import ground_truth_ndsm   # noqa: E402

TRUTH = os.path.join(ROOT, "dfc2019_data", "truth", "Track3-Truth")
BLIND = os.path.join(ROOT, "final_out", "blind_input")
WORK = os.path.join(ROOT, "work", "bench_buildings")
# input imagery: "wv" = the WorldView blind tile degraded to 0.6 m (RGB);
# "naip" = real 0.6 m USDA NAIP with near-infrared (scripts/fetch_naip.py)
SRC = os.environ.get("BENCH_SRC", "wv")


def source(tile):
    return os.path.join(ROOT, "work", "naip", f"{tile}.tif") if SRC == "naip" else to_06(tile)

CONFIGS = {
    # what uploads were built with before 2026-09-29
    "old": dict(max_px=4096, pbr_maps=False, fetch_canopy=True),
    # server.upload_build_kwargs() now (PBR maps off: shading only, no heights)
    "final": dict(max_px=4096, pbr_maps=False, fetch_canopy=True, continuous_vegetation=True,
                  crown_radius_ratio=0.18, ground_grid=1400, fetch_outlines=True,
                  osm_structures=True),
    "new": dict(max_px=4096, pbr_maps=False, fetch_canopy=True, continuous_vegetation=True,
                crown_radius_ratio=0.18, ground_grid=1400, fetch_outlines=True,
                osm_structures=True),
}


def to_06(tile):
    """The blind GeoTIFF area-averaged to 0.6 m, as a Cartosat-2S product."""
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, f"{tile}_0p6.tif")
    if os.path.exists(out):
        return out
    with rasterio.open(os.path.join(BLIND, f"{tile}.tif")) as s:
        img = s.read()
        f = abs(s.transform.a) / 0.6
        w, h = int(round(s.width * f)), int(round(s.height * f))
        small = np.stack([cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA) for b in img])
        prof = s.profile.copy()
        prof.update(width=w, height=h, transform=s.transform * s.transform.scale(s.width / w, s.height / h))
        tags = s.tags()
    with rasterio.open(out, "w", **prof) as d:
        d.write(small)
        d.update_tags(**tags)
    return out


def build(tile, cfg):
    name = f"bb_{cfg}_{SRC}_{tile}" if SRC != "wv" else f"bb_{cfg}_{tile}"
    t = time.time()
    bci.build(source(tile), name, stage=False, **CONFIGS[cfg])
    return os.path.join(ROOT, "final_out", f"city_{name}_buildings.geojson"), time.time() - t


def score(tile, geojson):
    gt = L.load_tile(tile, TRUTH)
    d = gt["dsm"].copy()
    d[~gt["valid_mask"]] = np.nan
    t = ground_truth_ndsm(d, gt["cls"])
    H, W = t.shape
    # the loader's transform is anchored at the tile's lower edge; the input
    # GeoTIFF covers the same square, so its own bounds place the truth grid
    with rasterio.open(source(tile)) as s:
        grid = rasterio.transform.from_bounds(*s.bounds, W, H)
    fc = json.load(open(geojson))
    src_crs = (fc.get("crs") or {}).get("properties", {}).get("name")
    gcrs = gt["crs"] if not isinstance(gt["crs"], int) else f"EPSG:{gt['crs']}"
    shapes = []
    for f in fc["features"]:
        g = f["geometry"]
        if src_crs and rasterio.crs.CRS.from_user_input(src_crs) != rasterio.crs.CRS.from_user_input(gcrs):
            g = transform_geom(src_crs, gcrs, g)
        shapes.append((g, float(f["properties"]["height_m"])))
    drawn = (rasterize(shapes, out_shape=(H, W), transform=grid, fill=0.0,
                       dtype="float32", merge_alg=rasterio.enums.MergeAlg.replace)
             if shapes else np.zeros((H, W), np.float32))
    bld = ((gt["cls"] == 6) & np.isfinite(t)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bld, connectivity=4)
    px_m2 = abs(gt["transform"].a) ** 2
    rows = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] * px_m2 < 10:
            continue
        m = lab == i
        rows.append((float(np.median(t[m])), float(np.median(drawn[m])), st[i, cv2.CC_STAT_AREA] * px_m2))
    # drawn blocks that stand on no real building (false buildings)
    drawn_mask = drawn > 2.0
    false_share = float((drawn_mask & (bld == 0)).sum() / max(1, drawn_mask.sum()))
    return rows, false_share, len(fc["features"])


def summarise(rows):
    a = np.array(rows)
    if not len(a):
        return {}
    e = a[:, 1] - a[:, 0]
    tall = a[:, 0] >= 10
    return {"n_buildings": int(len(a)), "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "mae_m": float(np.mean(np.abs(e))), "bias_m": float(e.mean()),
            "median_abs_m": float(np.median(np.abs(e))),
            "within_2m": float(np.mean(np.abs(e) <= 2)),
            "found": float(np.mean(a[:, 1] >= 2.0)),
            "tall_n": int(tall.sum()),
            "tall_rmse_m": float(np.sqrt(np.mean(e[tall] ** 2))) if tall.any() else None,
            "tall_bias_m": float(e[tall].mean()) if tall.any() else None}


def one(cfg, tile):
    """Build and score one tile; the rows go to a file the parent reads."""
    gj, sec = build(tile, cfg)
    rows, fs, nb = score(tile, gj)
    json.dump({"rows": rows, "false_share": fs, "drawn_buildings": nb, "seconds": round(sec, 1)},
              open(os.path.join(WORK, f"{cfg}_{SRC}_{tile}.json"), "w"))


def main():
    import subprocess
    cfg = sys.argv[1]
    if len(sys.argv) > 3 and sys.argv[2] == "--one":
        one(cfg, sys.argv[3])
        return
    tiles = sys.argv[2:] or [t for t in sorted(DH.benchmark_tiles())
                             if os.path.exists(os.path.join(BLIND, f"{t}.tif"))]
    allrows, per = [], {}
    for tile in tiles:
        # each tile in its own process with a time limit: a native-library
        # hang in one build must not stall the whole benchmark
        out = os.path.join(WORK, f"{cfg}_{SRC}_{tile}.json")
        for attempt in (1, 2):
            if os.path.exists(out):
                break
            try:
                subprocess.run([sys.executable, __file__, cfg, "--one", tile], timeout=600,
                               stdout=open(os.path.join(WORK, f"{cfg}_{SRC}_{tile}.log"), "w"),
                               stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                print(f"{tile}: attempt {attempt} timed out", flush=True)
        if not os.path.exists(out):
            print(f"{tile}: FAILED (see {cfg}_{tile}.log)", flush=True)
            per[tile] = {"error": "failed"}
            continue
        r = json.load(open(out))
        rows = [tuple(x) for x in r["rows"]]
        allrows += rows
        s = summarise(rows)
        per[tile] = {**s, "false_share": r["false_share"], "drawn_buildings": r["drawn_buildings"],
                     "seconds": r["seconds"]}
        print(f"{tile}: {r['seconds']:5.0f}s  drawn {r['drawn_buildings']:4d}  truth {s.get('n_buildings', 0):3d}  "
              f"RMSE {s.get('rmse_m', float('nan')):5.2f}  bias {s.get('bias_m', float('nan')):+5.2f}  "
              f"found {s.get('found', 0):.2f}  false {r['false_share']:.2f}", flush=True)
    tot = summarise(allrows)
    fsh = [v["false_share"] for v in per.values() if "false_share" in v]
    tot["false_share_mean"] = float(np.mean(fsh)) if fsh else None
    res = {"config": cfg, "settings": CONFIGS[cfg], "all": tot, "tiles": per}
    json.dump(res, open(os.path.join(ROOT, "final_out", f"benchmark_buildings_{cfg}_{SRC}.json"), "w"), indent=2)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
