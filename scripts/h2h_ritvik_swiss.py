"""Head-to-head on Ritvik-Mod/SIH-2026-DepthX's own 10 swisstopo scenes.

    python scripts/h2h_ritvik_swiss.py [--gsd-in 0.33|0.6]

Reproduces that repo's published protocol (data/swisstopo/index.json +
heightmap/metrics.py, read, not copied): each scene is a north-up LV95
(EPSG:2056) square of 1024 px at 0.33 m centred on the listed lon/lat;
imagery is SWISSIMAGE 0.1 m resampled cubic, truth is swissSURFACE3D minus
swissALTI3D (0.5 m, bilinear) clipped at 0; metrics are MAE / RMSE / bias /
Pearson r over every pixel where both LiDAR products are valid, plus the
same on pixels whose truth is > 2 m ("building" proxy, as their code does
without a semantic mask).

--gsd-in 0.6 degrades the image to ISRO's 0.6 m first (area average) and
brings it back to the grid, as our pipeline sees a Cartosat-2S scene.
Their per-scene numbers (results/validation.json, key "agl") are printed
beside ours when their repo is found at --their-repo.
"""
import argparse
import json
import os
import sys
import urllib.request

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform as warp_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections"
COLL = {"img": "ch.swisstopo.swissimage-dop10", "dsm": "ch.swisstopo.swisssurface3d-raster",
        "dtm": "ch.swisstopo.swissalti3d"}
LV95 = "EPSG:2056"
CACHE = os.path.join(ROOT, "work", "h2h_ritvik")


def asset(coll, lon, lat, want):
    d = 0.002
    url = f"{STAC}/{coll}/items?bbox={lon-d:.5f},{lat-d:.5f},{lon+d:.5f},{lat+d:.5f}&limit=1"
    feats = json.loads(urllib.request.urlopen(url, timeout=90).read()).get("features", [])
    if not feats:
        return None
    for name, a in feats[0]["assets"].items():
        if want in name and name.endswith(".tif"):
            return a["href"]
    return None


def fetch(sc):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{sc['name']}.npz")
    if os.path.exists(out):
        z = np.load(out)
        return z["rgb"], z["agl"]
    n, gsd = sc["size_px"], sc["gsd_m"]
    x, y = warp_transform("EPSG:4326", LV95, [sc["lon"]], [sc["lat"]])
    half = n * gsd / 2
    tr = from_origin(x[0] - half, y[0] + half, gsd, gsd)
    hs = {k: asset(COLL[k], sc["lon"], sc["lat"], w) for k, w in (("img", "_0.1_"), ("dsm", "_0.5_"), ("dtm", "_0.5_"))}
    if not all(hs.values()):
        raise RuntimeError(f"missing product {hs}")
    rgb = np.zeros((3, n, n), np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(hs["img"]) as s:
            for i in range(3):
                reproject(rasterio.band(s, i + 1), rgb[i], src_transform=s.transform, src_crs=s.crs,
                          dst_transform=tr, dst_crs=LV95, resampling=Resampling.cubic)
        el = {}
        for k in ("dsm", "dtm"):
            a = np.zeros((n, n), np.float32)
            with rasterio.open(hs[k]) as s:
                reproject(rasterio.band(s, 1), a, src_transform=s.transform, src_crs=s.crs,
                          dst_transform=tr, dst_crs=LV95, resampling=Resampling.bilinear,
                          src_nodata=s.nodata, dst_nodata=np.nan)
            el[k] = a
    agl = np.maximum(el["dsm"] - el["dtm"], 0.0)
    agl[~(np.isfinite(el["dsm"]) & np.isfinite(el["dtm"]))] = np.nan
    rgb = np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    np.savez_compressed(out, rgb=rgb, agl=agl)
    return rgb, agl


def core(p, g, m):
    if m.sum() < 2:
        return None
    d = p[m] - g[m]
    return {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())),
            "bias": float(d.mean()), "r": float(np.corrcoef(p[m], g[m])[0, 1]), "n": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsd-in", type=float, default=0.33)
    ap.add_argument("--their-repo", default=os.environ.get("RITVIK_REPO", ""))
    a = ap.parse_args()
    theirs = {}
    if a.their_repo:
        scenes = json.load(open(os.path.join(a.their_repo, "data", "swisstopo", "index.json")))
        try:
            v = json.load(open(os.path.join(a.their_repo, "results", "validation.json")))["scenes"]
            theirs = {k: s.get("agl") for k, s in v.items()}
        except Exception:
            pass
    else:
        sys.exit("--their-repo is needed for the scene list")
    import depth_model as dm
    bb = dm.DepthBackbone(metric=True)
    rows, P, G, PB, GB = {}, [], [], [], []
    for sc in scenes:
        try:
            rgb, agl = fetch(sc)
        except Exception as e:
            print(f"{sc['name']}: skipped ({e})", flush=True); continue
        img = rgb
        if a.gsd_in > sc["gsd_m"] * 1.05:
            f = sc["gsd_m"] / a.gsd_in
            n = rgb.shape[0]
            small = cv2.resize(rgb, (int(n * f), int(n * f)), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
        pred = bb.predict_tiled(Image.fromarray(img), raw_metric=True).astype(np.float32)
        if pred.shape != agl.shape:
            pred = cv2.resize(pred, agl.shape[::-1], interpolation=cv2.INTER_LINEAR)
        pred = np.maximum(pred, 0)
        m = np.isfinite(agl)
        o, b = core(pred, agl, m), core(pred, agl, m & (agl > 2.0))
        rows[sc["name"]] = {"overall": o, "building_proxy": b, "theirs_agl": theirs.get(sc["name"])}
        P.append(pred[m]); G.append(agl[m]); PB.append(pred[m & (agl > 2)]); GB.append(agl[m & (agl > 2)])
        t = theirs.get(sc["name"]) or {}
        print(f"{sc['name']:13s} ours MAE {o['mae']:5.2f} RMSE {o['rmse']:5.2f} r {o['r']:.3f} | "
              f"theirs MAE {t.get('mae_m', float('nan')):5.2f} RMSE {t.get('rmse_m', float('nan')):5.2f} "
              f"r {t.get('r', float('nan')):.3f}", flush=True)
    p, g = np.concatenate(P), np.concatenate(G)
    pooled = core(p, g, np.ones_like(p, bool))
    pb, gbb = np.concatenate(PB), np.concatenate(GB)
    res = {"protocol": "Ritvik-Mod/SIH-2026-DepthX swisstopo 10 scenes, 1024 px @ 0.33 m, AGL clipped at 0",
           "gsd_in": a.gsd_in, "model": dm.DepthBackbone.metric_info().get("checkpoint"),
           "pooled": pooled, "pooled_building_proxy": core(pb, gbb, np.ones_like(pb, bool)),
           "mean_of_scene_mae": float(np.mean([r["overall"]["mae"] for r in rows.values()])),
           "scenes": rows}
    os.makedirs(os.path.join(ROOT, "results", "h2h"), exist_ok=True)
    out = os.path.join(ROOT, "results", "h2h", f"ritvik_swiss_gsd{a.gsd_in}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("pooled", "pooled_building_proxy", "mean_of_scene_mae")}, indent=1))


if __name__ == "__main__":
    main()
