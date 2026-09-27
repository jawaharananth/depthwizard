"""Four-landscape accuracy against airborne LiDAR, at ISRO's 0.6 m.

    python scripts/lidar_landscapes.py

The problem statement asks for "performance stability across urban, sparse,
hilly, and forested landscapes". DFC2019 (Jacksonville, Omaha) is flat, so it
cannot carry the hilly case. This benchmark uses Swiss federal open data, one
1 km tile per landscape, chosen by a named place rather than by result:

  * image:  SWISSIMAGE 10 cm orthophoto, area-averaged to 0.6 m (Cartosat-2S,
            per ISRO's FAQ), then run exactly as the pipeline runs a 0.6 m
            source: upsampled to the 0.3 m the model was trained at;
  * truth:  swissSURFACE3D Raster (LiDAR DSM, 0.5 m) minus swissALTI3D (LiDAR
            bare-earth DTM, 0.5 m) = height above ground, the quantity the
            model predicts. Both block-averaged to 1 m before scoring.

Nothing is fitted to the truth. The zero-prediction row per site ("everything
is ground") is the floor any estimate must beat. The photo and the LiDAR are
flown in different years where no same-year pair exists; the gap is reported
per site, and change in between (new buildings, tree growth) counts against us.

Data (c) swisstopo, open government data (opendata.swiss, free use with
attribution).
"""
import json
import os
import sys

import cv2
import numpy as np
import rasterio
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import depth_model as dm  # noqa: E402

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections"
WORK = os.path.join(ROOT, "data", "swiss_lidar")
IMG_GSD, TRAIN_GSD, SCORE_GSD = 0.6, 0.3, 1.0

# landscape, place, lat, lon
SITES = [
    ("urban", "Zurich old town", 47.3717, 8.5423),
    ("hilly", "Zermatt village", 46.0207, 7.7491),
    ("forested", "Sihlwald forest", 47.2530, 8.5560),
    ("sparse", "Grosses Moos farmland (Kerzers)", 46.9750, 7.1900),
]


def items(coll, lat, lon):
    r = requests.get(f"{STAC}/{coll}/items",
                     params={"bbox": f"{lon},{lat},{lon + 1e-4},{lat + 1e-4}", "limit": 50},
                     timeout=60)
    r.raise_for_status()
    return r.json()["features"]


def asset(feat, gsd):
    for k, v in feat["assets"].items():
        if k.endswith(".tif") and abs((v.get("eo:gsd") or v.get("gsd") or 0) - gsd) < 1e-6:
            return v["href"]
    return None


def year(feat):
    return int(feat["id"].split("_")[1])


def get(url):
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, url.rsplit("/", 1)[1])
    if not os.path.exists(p):
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(p + ".part", "wb") as f:
                for c in r.iter_content(1 << 20):
                    f.write(c)
        os.replace(p + ".part", p)
    return p


def pick(lat, lon):
    dsm = max(items("ch.swisstopo.swisssurface3d-raster", lat, lon), key=year)
    tile = dsm["id"].rsplit("_", 1)[1]
    dtm = [f for f in items("ch.swisstopo.swissalti3d", lat, lon) if f["id"].endswith(tile)]
    img = [f for f in items("ch.swisstopo.swissimage-dop10", lat, lon) if f["id"].endswith(tile)]
    dtm = min(dtm, key=lambda f: abs(year(f) - year(dsm)))
    img = min(img, key=lambda f: (abs(year(f) - year(dsm)), -year(f)))
    return tile, dsm, dtm, img


def block(a, k):
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:H, :W].reshape(H // k, k, W // k, k).mean(axis=(1, 3))


def stats(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    e = p[m] - t[m]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(e.mean()),
            "corr": float(np.corrcoef(p[m], t[m])[0, 1]) if np.std(p[m]) > 0 else float("nan"),
            "n": int(m.sum())}


def main():
    bb = dm.DepthBackbone(metric=True)
    rows = []
    for land, place, lat, lon in SITES:
        tile, fd, ft, fi = pick(lat, lon)
        dsm_p, dtm_p = get(asset(fd, 0.5)), get(asset(ft, 0.5))
        img_p = get(asset(fi, 0.1))
        with rasterio.open(dsm_p) as s:
            dsm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        with rasterio.open(dtm_p) as s:
            dtm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        truth = block(np.clip(dsm - dtm, 0, None), int(SCORE_GSD / 0.5))          # 1 m
        with rasterio.open(img_p) as s:
            rgb = s.read([1, 2, 3]).transpose(1, 2, 0)
            src_gsd = abs(s.transform.a)
        n06 = int(round(rgb.shape[1] * src_gsd / IMG_GSD))
        img06 = cv2.resize(rgb, (n06, n06), interpolation=cv2.INTER_AREA)            # 0.6 m sensor
        del rgb
        z = IMG_GSD / TRAIN_GSD
        work = cv2.resize(img06, (int(n06 * z), int(n06 * z)), interpolation=cv2.INTER_CUBIC)
        pred = bb.predict_tiled(Image.fromarray(work), raw_metric=True).astype(np.float32)
        pred = cv2.resize(np.maximum(pred, 0), truth.shape[::-1], interpolation=cv2.INTER_AREA)
        Image.fromarray(img06).save(os.path.join(WORK, f"{land}_{tile}_0.6m.jpg"), quality=85)
        row = {"landscape": land, "place": place, "tile": tile,
               "lidar_year": year(fd), "image_year": year(fi),
               "relief_m": float(np.nanpercentile(dtm, 99) - np.nanpercentile(dtm, 1)),
               "ours": stats(pred, truth), "zero": stats(np.zeros_like(truth), truth),
               "truth_p99_m": float(np.nanpercentile(truth, 99))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(WORK, "results.json"), "w"), indent=2)

    lines = ["| landscape | place (tile) | relief | LiDAR / photo year | RMSE | MAE | bias | corr | "
             "zero-prediction RMSE |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o, z0 = r["ours"], r["zero"]
        lines.append(f"| {r['landscape']} | {r['place']} ({r['tile']}) | {r['relief_m']:.0f} m | "
                     f"{r['lidar_year']} / {r['image_year']} | {o['rmse']:.2f} m | {o['mae']:.2f} m | "
                     f"{o['bias']:+.2f} m | {o['corr']:.3f} | {z0['rmse']:.2f} m |")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(ROOT, "BENCHMARK_LANDSCAPES.md"), "w", encoding="utf-8") as fh:
        fh.write("# Four landscapes vs airborne LiDAR at 0.6 m (Switzerland, swisstopo)\n\n"
                 + __doc__.split("\n\n", 1)[1].rsplit("\n\nData", 1)[0]
                 + "\n\n## Results (height above ground, 1 m cells)\n\n" + txt
                 + "\n\nData (c) swisstopo (open government data). Regenerate: "
                   "`python scripts/lidar_landscapes.py`.\n")


if __name__ == "__main__":
    main()
