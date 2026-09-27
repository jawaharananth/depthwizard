"""Score our absolute DSM the way ISRO's FAQ says it will be scored.

    python scripts/isro_protocol_eval.py            # fetch, build, score
    python scripts/isro_protocol_eval.py --score    # re-score existing builds

ISRO (FAQ, 22 Sep 2026): GeoTIFF inputs are evaluated as an absolute DSM
"such as SRTM or Copernicus". So on real Indian scenes, one per landscape the
problem statement names, this builds our DSM from a single ~0.6 m image and
compares it with those two global surface models.

Not circular. The pipeline anchors its ground to one DEM; a DSM anchored to
Copernicus and then scored against Copernicus would be graded on its own
input. So every scene is built twice: anchored to Copernicus and scored
against SRTM, and anchored to SRTM and scored against Copernicus.

Scale. The references are 30 m surface models; nothing finer than 30 m exists
in them. Our DSM is block-averaged onto a 30 m grid first (what a 30 m sensor
would record of our surface), then compared cell by cell. This measures
whether our absolute surface agrees with the references at their own scale
-- datum, terrain, and the bulk of buildings and canopy -- not building-level
accuracy, which needs LiDAR (see BENCHMARK_BLIND.md).

Context row per scene: the two references scored against each other. Two
independent 30 m DSMs of the same ground disagree by this much; a single-image
DSM agreeing with one to about that level is as close as the reference allows.

Imagery: Esri World Imagery at zoom 18 (~0.57 m at these latitudes), fetched
with tools/fetch_scene.py. It is not Cartosat; the sensor gap is not measured.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dem_source  # noqa: E402

PY = sys.executable
WORK = os.path.join(ROOT, "final_out", "isro_eval")
ZOOM, NTILES = 18, 8            # 8x8 tiles = 2048 px, ~1.2 km square
CELL_M = 30.0

# name, landscape, lat, lon  (centres chosen for each landscape, not for results)
SCENES = [
    ("hyderabad_hitec", "urban", 17.4474, 78.3762),
    ("periyar_thekkady", "forested", 9.6031, 77.1615),
    ("shimla_ridge", "hilly", 31.1048, 77.1734),
    ("telangana_farmland", "sparse", 17.6200, 78.0500),
]
OTHER = {"glo30": "srtm", "srtm": "glo30"}


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


def fetch(name, lat, lon):
    out = os.path.join(WORK, f"{name}.tif")
    if not os.path.exists(out):
        x, y = tile_xy(lat, lon, ZOOM)
        subprocess.run([PY, os.path.join(ROOT, "tools", "fetch_scene.py"), out, "--zoom", str(ZOOM),
                        "--x0", str(x - NTILES // 2), "--y0", str(y - NTILES // 2),
                        "--n", str(NTILES)], check=True)
    return out


def build(img, name, anchor):
    tag = f"{name}_{anchor}"
    dsm = os.path.join(ROOT, "final_out", f"city_{tag}_DSM.tif")
    if not os.path.exists(dsm):
        log = os.path.join(WORK, f"{tag}.log")
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.run([PY, os.path.join(ROOT, "build_city_image.py"), img, "--name", tag,
                            "--no-stage", "--dem-source", anchor],
                           cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return dsm if os.path.exists(dsm) else None


def to_cells(dsm_path):
    """Block-average the DSM onto a ~30 m grid in its own CRS (valid pixels only)."""
    with rasterio.open(dsm_path) as s:
        a = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        tf, crs = s.transform, s.crs
        # 3857 map units are not ground metres: scale the cell to ~30 m on the ground.
        lat = rasterio.warp.transform(crs, "EPSG:4326", [s.bounds.left], [(s.bounds.top + s.bounds.bottom) / 2])[1][0]
        unit_per_m = 1 / math.cos(math.radians(lat)) if crs.to_epsg() == 3857 else 1.0
    cell = CELL_M * unit_per_m
    k = max(1, int(round(cell / abs(tf.a))))
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    blk = a[:H, :W].reshape(H // k, k, W // k, k)
    frac = np.isfinite(blk).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(blk, axis=(1, 3))
    mean[frac < 0.8] = np.nan                     # a cell must be mostly image
    return mean, tf.c, tf.f, abs(tf.a) * k, abs(tf.e) * k, crs


def ref_cells(src, x0, ytop, gx, gy, crs, shape):
    r = dem_source.sample_grid(x0, ytop, shape[0], gx, f"EPSG:{crs.to_epsg()}",
                               dem_source=src, width_px=shape[1], gsd_y_m=gy)
    return None if r.get("dem") is None else np.asarray(r["dem"], np.float32)


def stats(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 20:
        return None
    e = p[m] - t[m]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(e.mean()), "corr": float(np.corrcoef(p[m], t[m])[0, 1]),
            "n_cells": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="only re-score existing builds")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for name, land, lat, lon in SCENES:
        img = os.path.join(WORK, f"{name}.tif") if a.score else fetch(name, lat, lon)
        for anchor in ("glo30", "srtm"):
            dsm = (os.path.join(ROOT, "final_out", f"city_{name}_{anchor}_DSM.tif") if a.score
                   else build(img, name, anchor))
            if not dsm or not os.path.exists(dsm):
                print(f"{name} [{anchor}] build failed -- see {WORK}", flush=True)
                continue
            ours, x0, ytop, gx, gy, crs = to_cells(dsm)
            ref_src = OTHER[anchor]
            ref = ref_cells(ref_src, x0, ytop, gx, gy, crs, ours.shape)
            anc = ref_cells(anchor, x0, ytop, gx, gy, crs, ours.shape)
            if ref is None:
                print(f"{name}: {ref_src} unavailable", flush=True)
                continue
            row = {"scene": name, "landscape": land, "anchored_to": anchor, "scored_vs": ref_src,
                   "ours": stats(ours, ref),
                   "references_vs_each_other": stats(np.where(np.isfinite(ours), anc, np.nan), ref)
                   if anc is not None else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(WORK, "results.json"), "w"), indent=2)

    lines = ["| scene | landscape | anchored to -> scored vs | RMSE | MAE | bias | corr | cells | "
             "reference vs reference RMSE |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o, c = r["ours"], r["references_vs_each_other"]
        if not o:
            continue
        lines.append(f"| {r['scene']} | {r['landscape']} | {r['anchored_to']} -> {r['scored_vs']} | "
                     f"{o['rmse']:.2f} m | {o['mae']:.2f} m | {o['bias']:+.2f} m | {o['corr']:.3f} | "
                     f"{o['n_cells']} | {c['rmse']:.2f} m |" if c else "| - |")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(ROOT, "BENCHMARK_ISRO_PROTOCOL.md"), "w", encoding="utf-8") as fh:
        fh.write("# Absolute DSM vs SRTM / Copernicus over India (ISRO's stated protocol)\n\n"
                 + (__doc__.split("\n\n", 1)[1]) + "\n## Results\n\n" + txt +
                 "\n\nRegenerate: `python scripts/isro_protocol_eval.py`.\n")


if __name__ == "__main__":
    main()
