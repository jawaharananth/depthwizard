"""
Train a lightweight height regressor on frozen AlphaEarth Foundations
embeddings, using real measured heights (Mumbai BMC floor records) as
labels, then apply it to predict real heights for Karnataka (RAMP
footprints, real shapes but no height data at all).

AlphaEarth (Google DeepMind, arXiv:2507.22291) ships a 64-dim embedding
per 10m pixel, trained across optical+radar+LiDAR+climate globally, engineered
specifically to transfer to places with no labels of their own. This is
NOT the foundation model itself -- that stays frozen, exactly as published.
Only the small regressor on top is trained here, on our own real labels.

Real, not mocked: every embedding read below is a live windowed read from
the actual public AlphaEarth COG tiles (source.coop mirror), no synthetic
data anywhere in this pipeline.
"""
import json

import cv2
import numpy as np
import rasterio
import rasterio.windows
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import dfc2019_loader as dfc

AEF_URL_TMPL = ("/vsicurl/https://s3.us-west-2.amazonaws.com/"
                "us-west-2.opendata.source.coop/{path}")

MUMBAI_TILE = "tge-labs/aef/v1/annual/2025/43N/x3tian92cwfwxlcga-0000008192-0000008192.tiff"
KARNATAKA_TILE = "tge-labs/aef/v1/annual/2024/43N/xmken1e3ej594cdqu-0000000000-0000008192.tiff"
JAX_TILE = "tge-labs/aef/v1/annual/2025/17N/xmre339lmy9yacsgn-0000000000-0000008192.tiff"
JAX_TRUTH_DIR = "dfc2019_data/truth/Track3-Truth"
JAX_N_TILES = 25   # real DFC2019 tiles to draw buildings from, not all 110 -- enough
                   # for real typology diversity without an hour of network reads

MUMBAI_BOUNDS = "mumbai_bmc/mosaic_bounds.txt"
MUMBAI_POLYGONS = "mumbai_bmc/polygons_px.json"
MUMBAI_MOSAIC_PX = 3072  # real mosaic pixel size, see mumbai_bmc build script

KARNATAKA_BOUNDS = "ramp_karnataka/mosaics/kn_a_bounds.txt"
KARNATAKA_POLYGONS = "ramp_karnataka/mosaics/kn_a_polygons.json"
KARNATAKA_MOSAIC_PX = 3072


def _read_bounds(path: str):
    with open(path) as f:
        w, s, e, n = (float(x) for x in f.read().split())
    return w, s, e, n


def _pixel_centroid(poly) -> tuple:
    arr = np.asarray(poly, np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _px_to_lonlat(px: float, py: float, bounds, mosaic_px: int) -> tuple:
    """
    Linear pixel-to-geographic mapping from the mosaic's own saved bounds.
    Valid because these mosaics were cut as simple axis-aligned crops (no
    rotation), the same assumption the rest of this project already makes
    for these two datasets.
    """
    w, s, e, n = bounds
    lon = w + (px / mosaic_px) * (e - w)
    lat = n - (py / mosaic_px) * (n - s)   # image row 0 is the north edge
    return lon, lat


def _sample_embeddings(tile_path: str, points_utm: list, epsg: int) -> np.ndarray:
    """
    Real windowed reads, one per point (small counts here -- hundreds, not
    millions -- so no batching complexity is worth adding). NaN row for any
    point the tile does not actually cover, an honest "no data" rather than
    a fabricated nearest-neighbour guess.
    """
    url = AEF_URL_TMPL.format(path=tile_path)
    out = np.full((len(points_utm), 64), np.nan, dtype=np.float32)
    with rasterio.Env(GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_CONNECTTIMEOUT=15):
        with rasterio.open(url) as src:
            inv = ~src.transform
            for i, (x, y) in enumerate(points_utm):
                c, r = inv * (x, y)
                c, r = int(c), int(r)
                if not (0 <= c < src.width and 0 <= r < src.height):
                    continue
                win = rasterio.windows.Window(c, r, 1, 1)
                val = src.read(window=win)[:, 0, 0]
                # Signed int8 dequantization: real range is -128..127,
                # mapped to [-1, 1] -- NOT the /127.5-1.0 uint8 formula,
                # which is for unsigned 0..255 data and was wrong on a
                # first attempt (verified by checking the dequantized
                # range fell partly outside [-1, 1]).
                out[i] = val.astype(np.float32) / 127.0
    return out


def build_mumbai_training_set():
    bounds = _read_bounds(MUMBAI_BOUNDS)
    d = json.load(open(MUMBAI_POLYGONS))
    polys, heights = d["polygons"], d["heights"]

    tf = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    points_utm, labels = [], []
    for poly, h in zip(polys, heights):
        if h is None:
            continue
        px, py = _pixel_centroid(poly)
        lon, lat = _px_to_lonlat(px, py, bounds, MUMBAI_MOSAIC_PX)
        x, y = tf.transform(lon, lat)
        points_utm.append((x, y))
        labels.append(h)

    print(f"  Mumbai: {len(labels)} real measured heights, "
          f"fetching real AlphaEarth embeddings for each...")
    emb = _sample_embeddings(MUMBAI_TILE, points_utm, 32643)
    labels = np.asarray(labels, np.float32)
    valid = np.isfinite(emb).all(axis=1)
    print(f"  {valid.sum()} of {len(labels)} points had real tile coverage")
    return emb[valid], labels[valid]


def build_jax_training_set():
    """
    Real per-building heights from DFC2019 (Jacksonville) LiDAR ground
    truth -- genuinely different building typology from dense Mumbai
    (suburban/mixed density), which is the actual diversity the Mumbai-only
    regressor was missing. DSM is absolute elevation, not height above
    ground, so each building's height is its own median DSM minus the
    SAME tile's median Ground-class (LAS code 2) elevation -- a per-tile
    ground reference, not per-building, because these tiles carry no
    separate DTM; real, if coarser than the per-footprint local-ground
    estimate the rest of this project uses where a real DTM exists.
    """
    tiles = dfc.list_available_tiles(JAX_TRUTH_DIR)[:JAX_N_TILES]
    points_utm, labels = [], []
    for prefix in tiles:
        try:
            t = dfc.load_tile(prefix, JAX_TRUTH_DIR)
        except FileNotFoundError:
            continue
        dsm, cls = t["dsm"], t["cls"]
        ground_px = cls == 2
        if ground_px.sum() < 50:
            continue
        tile_ground = float(np.median(dsm[ground_px]))

        building_mask = (cls == 6).astype(np.uint8)
        n, lab = cv2.connectedComponents(building_mask, connectivity=8)
        for comp_id in range(1, n):
            comp = lab == comp_id
            if comp.sum() < 20:   # ~5 sq m at 0.5 m/px -- real building, not noise
                continue
            h = float(np.median(dsm[comp])) - tile_ground
            if not (1.5 <= h <= 150.0):
                continue
            ys, xs = np.nonzero(comp)
            px, py = float(xs.mean()), float(ys.mean())
            utm_x, utm_y = t["transform"] * (px, py)
            points_utm.append((utm_x, utm_y))
            labels.append(h)

    print(f"  Jacksonville: {len(labels)} real buildings from {len(tiles)} "
          f"real LiDAR tiles, fetching real AlphaEarth embeddings...")
    emb = _sample_embeddings(JAX_TILE, points_utm, 32617)
    labels = np.asarray(labels, np.float32)
    valid = np.isfinite(emb).all(axis=1)
    print(f"  {valid.sum()} of {len(labels)} points had real tile coverage")
    return emb[valid], labels[valid]


def build_karnataka_query_set():
    bounds = _read_bounds(KARNATAKA_BOUNDS)
    d = json.load(open(KARNATAKA_POLYGONS))
    polys = d if isinstance(d, list) else d["polygons"]

    tf = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    points_utm = []
    for poly in polys:
        px, py = _pixel_centroid(poly)
        lon, lat = _px_to_lonlat(px, py, bounds, KARNATAKA_MOSAIC_PX)
        x, y = tf.transform(lon, lat)
        points_utm.append((x, y))

    print(f"  Karnataka: {len(points_utm)} footprints, "
          f"fetching real AlphaEarth embeddings for each...")
    emb = _sample_embeddings(KARNATAKA_TILE, points_utm, 32643)
    return emb, polys


if __name__ == "__main__":
    print("[1/4] Building Mumbai training set (real embeddings + real heights)")
    X_mum, y_mum = build_mumbai_training_set()
    print(f"  Mumbai set: X={X_mum.shape} y={y_mum.shape}, height range "
          f"{y_mum.min():.1f}-{y_mum.max():.1f} m")

    print("[2/4] Building Jacksonville training set (real LiDAR heights, "
          "different building typology from dense Mumbai)")
    X_jax, y_jax = build_jax_training_set()
    print(f"  Jacksonville set: X={X_jax.shape} y={y_jax.shape}, height range "
          f"{y_jax.min():.1f}-{y_jax.max():.1f} m")

    X = np.concatenate([X_mum, X_jax], axis=0)
    y = np.concatenate([y_mum, y_jax], axis=0)
    print(f"  combined training set: X={X.shape} y={y.shape}")

    print("[3/4] Cross-validating the regressor honestly before trusting it")
    # 5-fold CV on the ACTUAL training data, not a held-out claim without
    # one -- this is the number that matters, not R^2 on the training set
    # itself.
    reg = RandomForestRegressor(n_estimators=300, max_depth=12,
                                random_state=0, n_jobs=-1)
    cv_scores = cross_val_score(reg, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"  5-fold CV MAE: {-cv_scores.mean():.2f} +/- {cv_scores.std():.2f} m "
          f"(real held-out error, not a training-set fit)")

    reg.fit(X, y)

    print("[4/4] Predicting Karnataka heights from real AlphaEarth embeddings")
    Xk, kn_polys = build_karnataka_query_set()
    valid_k = np.isfinite(Xk).all(axis=1)
    print(f"  {valid_k.sum()} of {len(kn_polys)} Karnataka footprints have "
          f"real tile coverage")
    preds = np.full(len(kn_polys), np.nan, dtype=np.float32)
    preds[valid_k] = reg.predict(Xk[valid_k])
    preds = np.clip(preds, 2.0, 40.0)  # a real, physically sane building-height band

    out = {"heights_m": [None if not np.isfinite(p) else float(p) for p in preds]}
    with open("karnataka_alphaearth_heights.json", "w") as f:
        json.dump(out, f)
    finite = preds[np.isfinite(preds)]
    print(f"  predicted heights: median {np.median(finite):.1f}m, "
          f"std {finite.std():.1f}m, range {finite.min():.1f}-{finite.max():.1f}m")
    print("  saved: karnataka_alphaearth_heights.json")
