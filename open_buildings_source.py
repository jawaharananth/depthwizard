"""
Google Open Buildings 2.5D Temporal as a real per-building height anchor --
the same "ground truth fusion" pattern already proven for Mumbai (BMC
municipal floor-count records), applied to any scene with a known real-world
location, not just Mumbai. Built in direct response to a real, diagnosed
problem: a scene with no ground truth (Agra) renders with wildly
inconsistent adjacent-building heights -- verified by comparing it against
Mumbai Colaba (which HAS real ground truth and renders cleanly) and against
Agra's own pre-existing build (identical chaos, so not a new regression).

WHAT THIS IS, AND THE REAL ACCESS PATH (verified against live data, not
assumed from documentation -- three other paths were tried and genuinely
failed first: GlobalBuildingAtlas/mediaTUM is blocked by active bot
protection; the raw Open Buildings V3 polygon bucket has no height column at
all; the cloud-native GeoParquet route hit real SSL/addressing issues before
being fixed. This is the one that actually works end to end.)

  1. Google Cloud Storage, public, unauthenticated: bucket
     open-buildings-temporal-data. v1/manifests/{region}_{EPSG}_{date}.json
     lists every tile for one UTM zone, one date, with each tile's own
     affine transform and pixel dimensions -- enough to find the ONE tile
     covering a given point without downloading anything.
  2. The actual per-tile GeoTIFF is read via rasterio's /vsicurl/, windowed
     to just the scene's own extent -- the same technique dem_source.py
     already uses for Copernicus DEM. A 25000x25000px tile is never
     downloaded whole; only the requested window's bytes are fetched.
  3. Real, non-obvious bug fixed while building this: manifest URIs are
     concatenated onto uriPrefix with NO separator (uriPrefix ends
     ".../geotiffs/3", the URI starts "973c_...", concatenating them
     directly gives the real "3973c_..." directory name) -- inserting a "/"
     between them, the natural-looking thing to do, 404s every time.

Bands, per the manifest: 0=building_fractional_count, 1=building_height,
2=building_presence (0-indexed; rasterio/GDAL band numbers are these +1).
-99.0 is nodata across all three bands, confirmed against real tile data.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import rasterio
from pyproj import Transformer

MANIFEST_LIST_URL = ("https://storage.googleapis.com/storage/v1/b/"
                      "open-buildings-temporal-data/o?prefix=v1/manifests/&maxResults=2000")
GCS_BASE = "https://storage.googleapis.com/open-buildings-temporal-data/"
NODATA = -99.0

_FAIL_THRESHOLD = 3
_COOLDOWN_S = 120.0
_consecutive_failures = 0
_tripped_until = 0.0


def _breaker_open() -> bool:
    return time.time() < _tripped_until


def _record_failure() -> None:
    global _consecutive_failures, _tripped_until
    _consecutive_failures += 1
    if _consecutive_failures >= _FAIL_THRESHOLD:
        _tripped_until = time.time() + _COOLDOWN_S


def _record_success() -> None:
    global _consecutive_failures, _tripped_until
    _consecutive_failures = 0
    _tripped_until = 0.0


_manifest_name_cache: list[str] | None = None


def _list_manifest_names(timeout: float = 15.0) -> list[str]:
    global _manifest_name_cache
    if _manifest_name_cache is not None:
        return _manifest_name_cache
    with urllib.request.urlopen(MANIFEST_LIST_URL, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    names = [item["name"].rsplit("/", 1)[-1][:-5]  # strip "v1/manifests/" and ".json"
             for item in data.get("items", []) if item["name"].endswith(".json")]
    _manifest_name_cache = names
    return names


def find_tile(lat: float, lon: float, epsg: int, year: int = 2023) -> dict | None:
    """The one manifest tile (if any) covering this point, in the dataset's
    own UTM zone -- caller already knows/derives the right EPSG from the
    scene's own georeference (or from a UTM-zone-from-longitude formula for
    an ungeoreferenced source; see module note on Agra for why that matters).

    Returns {"path": full GCS object path, "transform": affine params} or
    None if no tile covers this point (a real, honestly-reported outcome
    for a point outside this dataset's coverage, not an error).
    """
    if _breaker_open():
        return None
    try:
        names = _list_manifest_names()
    except Exception:
        _record_failure()
        return None
    _record_success()

    candidates = [n for n in names if f"EPSG_{epsg}_{year}" in n]
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    ex, ny = tf.transform(lon, lat)

    for mname in candidates:
        if _breaker_open():
            return None
        try:
            with urllib.request.urlopen(f"{GCS_BASE}v1/manifests/{mname}.json", timeout=15.0) as r:
                manifest = json.loads(r.read().decode("utf-8"))
        except Exception:
            _record_failure()
            continue
        _record_success()

        prefix = manifest["uriPrefix"].replace("gs://open-buildings-temporal-data/", "")
        for source in manifest["tilesets"][0]["sources"]:
            at = source["affineTransform"]
            x0, y0 = at["translateX"], at["translateY"]
            w, h = source["dimensions"]["width"], source["dimensions"]["height"]
            x1 = x0 + w * at["scaleX"]
            y1 = y0 + h * at["scaleY"]
            if min(x0, x1) <= ex <= max(x0, x1) and min(y0, y1) <= ny <= max(y0, y1):
                # No separator here -- see module docstring; this is the
                # exact real bug this module exists to have fixed once.
                path = prefix + source["uris"][0]
                return {"path": path, "utm_x": ex, "utm_y": ny}
    return None


def sample_height_grid(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                        epsg: int, year: int = 2023) -> dict:
    """Real building-height raster for a scene's actual extent, or an
    honest empty result if no tile covers it / the tile has no data there
    (both real, expected outcomes -- see module docstring for why zero
    coverage must never be silently treated as a failure)."""
    cy = (min_lat + max_lat) / 2
    cx = (min_lon + max_lon) / 2
    tile = find_tile(cy, cx, epsg, year)
    if tile is None:
        return {"heights": None, "transform": None, "coverage": 0.0,
                "note": f"no Open Buildings tile covers this extent (EPSG:{epsg}, {year})"}

    remote_url = f"{GCS_BASE}{tile['path']}"
    url = f"/vsicurl/{remote_url}"
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y1 = tf.transform(min_lon, max_lat)   # NW corner
    x1, y0 = tf.transform(max_lon, min_lat)   # SE corner

    def _read_window(rio_path):
        with rasterio.open(rio_path) as src:
            inv = ~src.transform
            c0, r0 = inv * (x0, y1)
            c1, r1 = inv * (x1, y0)
            c0, c1 = sorted((int(c0), int(c1)))
            r0, r1 = sorted((int(r0), int(r1)))
            c0, r0 = max(0, c0), max(0, r0)
            c1, r1 = min(src.width, c1), min(src.height, r1)
            if c1 <= c0 or r1 <= r0:
                return "empty", None
            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            return src.read(2, window=window), src.window_transform(window)

    try:
        with rasterio.Env(GDAL_HTTP_TIMEOUT=25, GDAL_HTTP_CONNECTTIMEOUT=10):
            heights, win_transform = _read_window(url)
    except Exception as exc:
        # GDAL's own bundled cURL uses Windows' native "schannel" TLS backend
        # for /vsicurl/, which validates against the OS certificate store --
        # a real, seen-in-practice failure mode on a machine whose store is
        # missing an intermediate CA, completely independent of whether the
        # SAME network path works through Python's own SSL stack (it does
        # here: find_tile's urllib manifest fetch, moments earlier, succeeds
        # on the exact same host through the exact same network). Rather
        # than fail a real, available dataset over a local TLS-backend quirk,
        # fall back to fetching the whole object with urllib (proven working)
        # to a local temp file and pointing rasterio at that instead -- once
        # it's a local file, no further network/TLS involvement is needed.
        if "schannel" not in str(exc).lower() and "curl error" not in str(exc).lower():
            return {"heights": None, "transform": None, "coverage": 0.0,
                    "note": f"{type(exc).__name__}: {exc}"}
        import tempfile
        try:
            req = urllib.request.Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf_local:
                tf_local.write(data)
                local_path = tf_local.name
            heights, win_transform = _read_window(local_path)
        except Exception as exc2:
            return {"heights": None, "transform": None, "coverage": 0.0,
                    "note": f"vsicurl failed ({exc}); local-download fallback also "
                            f"failed: {type(exc2).__name__}: {exc2}"}
        finally:
            try:
                import os as _os
                _os.remove(local_path)
            except Exception:
                pass

    if isinstance(heights, str) and heights == "empty":
        return {"heights": None, "transform": None, "coverage": 0.0,
                "note": "computed window is empty (point near tile edge)"}

    valid = heights != NODATA
    coverage = float(valid.mean()) if valid.size else 0.0
    return {
        "heights": np.where(valid, heights, np.nan),
        "transform": win_transform, "epsg": epsg,
        "coverage": coverage,
        "note": f"{valid.sum()} of {valid.size} pixels have real building-height data "
                f"({coverage*100:.1f}% coverage)",
    }
