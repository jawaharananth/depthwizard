"""Swiss federal open data as height-training tiles (run 04).

    python prepare_swiss.py OUT_DIR [--max-tiles N]

Why: every open 0.3 m height dataset we had (GAMUS, DFC2019) is US towns with
few tall buildings, few tall trees and no mountains, and the model compresses
exactly those (scripts/lidar_landscapes.py: bias -7 m in Zurich's old town and
Zermatt). swisstopo publishes the whole country as orthophotos plus airborne
LiDAR surface and terrain models, free to use with attribution.

Per 1 km LV95 tile:
  * RGB:   SWISSIMAGE via the federal WMS at 0.3 m (3334 x 3334), the scale
           the model trains at;
  * label: swissSURFACE3D Raster (LiDAR DSM, 0.5 m) minus swissALTI3D (LiDAR
           DTM, 0.5 m), clipped to 0..250 m, resampled onto the RGB grid.
Written as nine 1024 x 1024 tiles in the GAMUS h5 layout (image = uint8 RGB /
float16 metres) so gamus.GamusHeight reads them unchanged.

Held out: the four benchmark tiles of scripts/lidar_landscapes.py and every
tile touching them (a 1 km buffer) never enter training.

Data (c) swisstopo, open government data.
"""
import argparse
import concurrent.futures as cf
import io
import os
import time

import h5py
import numpy as np
import requests

STAC = "https://data.geo.admin.ch/api/stac/v0.9/search"
WMS = ("https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ch.swisstopo.swissimage"
       "&STYLES=&CRS=EPSG:2056&BBOX={e0},{n0},{e1},{n1}&WIDTH=3334&HEIGHT=3334&FORMAT=image/jpeg")
BENCH = {(2683, 1247), (2624, 1096), (2684, 1234), (2581, 1202)}
HOLD = {(e + de, n + dn) for e, n in BENCH for de in (-1, 0, 1) for dn in (-1, 0, 1)}

# (name, kind, lon0, lat0, lon1, lat1) -- spread over the landscapes the model is weakest on
REGIONS = [
    ("geneva", "urban", 6.12, 46.19, 6.17, 46.22), ("basel", "urban", 7.57, 47.54, 7.62, 47.57),
    ("lausanne", "urban", 6.61, 46.51, 6.66, 46.54), ("bern", "urban", 7.42, 46.94, 7.47, 46.97),
    ("zurich_west", "urban", 8.46, 47.38, 8.51, 47.41), ("winterthur", "urban", 8.71, 47.49, 8.75, 47.51),
    ("lugano", "hilly", 8.93, 45.99, 8.98, 46.02), ("davos", "hilly", 9.80, 46.78, 9.85, 46.81),
    ("grindelwald", "hilly", 8.02, 46.61, 8.07, 46.64), ("verbier", "hilly", 7.20, 46.09, 7.25, 46.11),
    ("st_moritz", "hilly", 9.82, 46.48, 9.87, 46.51), ("jura_forest", "forested", 6.20, 46.58, 6.25, 46.61),
    ("emmental_forest", "forested", 7.75, 46.95, 7.80, 46.98),
    ("ticino_forest", "forested", 8.80, 46.20, 8.85, 46.23),
    ("seeland_farms", "sparse", 7.25, 47.05, 7.30, 47.08), ("thurgau_farms", "sparse", 9.05, 47.55, 9.10, 47.58),
]
S = requests.Session()


def search(coll, bbox):
    """{(e_km, n_km): (year, tif_url_0.5m)} -- latest year per tile."""
    out, url = {}, STAC
    params = {"collections": coll, "bbox": ",".join(map(str, bbox)), "limit": 100}
    while url:
        r = S.get(url, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        for f in d["features"]:
            yr = int(f["id"].split("_")[1])
            e, n = map(int, f["id"].rsplit("_", 1)[1].split("-"))
            href = next((v["href"] for k, v in f["assets"].items()
                         if k.endswith(".tif") and abs((v.get("eo:gsd") or 0) - 0.5) < 1e-6), None)
            if href and (e, n) not in out or (href and yr > out[(e, n)][0]):
                out[(e, n)] = (yr, href)
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
        params = None
    return out


def _get(url, tries=5):
    for k in range(tries):
        try:
            r = S.get(url, timeout=120)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    return None


def build_tile(key, dsm_url, dtm_url, out_dir, kind):
    import cv2
    import rasterio
    from PIL import Image
    e, n = key
    stem0 = f"CH_{kind}_{e}_{n}"
    if os.path.exists(os.path.join(out_dir, f"{stem0}_8_AGL.h5")):
        return 9
    jpg = _get(WMS.format(e0=e * 1000, n0=n * 1000, e1=(e + 1) * 1000, n1=(n + 1) * 1000))
    dsm_b, dtm_b = _get(dsm_url), _get(dtm_url)
    if not (jpg and dsm_b and dtm_b):
        return 0
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    with rasterio.MemoryFile(dsm_b) as m, m.open() as s:
        dsm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
    with rasterio.MemoryFile(dtm_b) as m, m.open() as s:
        dtm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
    if dsm.shape != dtm.shape:
        return 0
    agl = np.clip(dsm - dtm, 0, 250)
    agl = cv2.resize(agl, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if np.isnan(agl).mean() > 0.2 or rgb.mean() < 5:
        return 0
    k = 0
    for i in range(3):
        for j in range(3):
            sl = np.s_[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024]
            st = f"{stem0}_{k}"
            with h5py.File(os.path.join(out_dir, f"{st}_RGB.h5"), "w") as h:
                h.create_dataset("image", data=rgb[sl], compression="gzip", compression_opts=3)
            with h5py.File(os.path.join(out_dir, f"{st}_AGL.h5"), "w") as h:
                h.create_dataset("image", data=agl[sl].astype(np.float16), compression="gzip",
                                 compression_opts=3)
            k += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max-tiles", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    jobs = []
    for name, kind, *bbox in REGIONS:
        dsm = search("ch.swisstopo.swisssurface3d-raster", bbox)
        dtm = search("ch.swisstopo.swissalti3d", bbox)
        keys = sorted(k for k in dsm if k in dtm and k not in HOLD)
        print(f"{name:16s} {kind:9s} {len(keys)} tiles", flush=True)
        jobs += [(k, dsm[k][1], dtm[k][1], kind) for k in keys]
    # Interleave regions so a capped run stays balanced.
    by_kind = {}
    for j in jobs:
        by_kind.setdefault(j[3], []).append(j)
    order = []
    while any(by_kind.values()):
        for kk in sorted(by_kind):
            if by_kind[kk]:
                order.append(by_kind[kk].pop(0))
    order = order[:a.max_tiles]
    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    n_sub = 0
    with cf.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(build_tile, k, du, tu, os.path.join(a.out, "train"), kind)
                for k, du, tu, kind in order]
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                n_sub += f.result()
            except Exception as err:
                print("  tile failed:", err, flush=True)
            if i % 25 == 0 or i == len(futs):
                print(f"  swiss {i}/{len(futs)} km tiles -> {n_sub} training tiles, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"swiss done: {n_sub} training tiles from {len(order)} km tiles", flush=True)


if __name__ == "__main__":
    main()
