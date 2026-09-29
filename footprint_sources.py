"""Real building outlines for an uploaded, georeferenced image.

    import footprint_sources as fs
    polys = fs.fetch(src_transform, src_crs, width, height)   # pixel polygons, or []

The showcase places (HITEC City and the rest) are built on real outlines --
Microsoft Global ML Building Footprints, plus OpenStreetMap where Microsoft
has none -- instead of outlines traced from the image's own colours. This
gives an upload the same: one polygon per real building, in the upload's own
pixel grid, whatever its CRS (Web Mercator, UTM, lat/lon).

Both sources are fetched in parallel and fail soft: no network, a timeout or
an area neither covers returns [] and the build traces outlines from the
image as before. Microsoft tiles (one per ~80 km square) are cached on disk,
so a second upload over the same region costs nothing.

Co-registration with the upload is NOT assumed here: build_city_image
cross-correlates these outlines against the image's own building mask and
shifts them onto it before anything reads them (gt_align).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import threading
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache", "footprints")
MS_LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
UA = {"User-Agent": "DepthWizard-SIH26175/1.0"}
TIME_BUDGET_S = float(os.environ.get("DW_FOOTPRINT_BUDGET_S", "90"))
_first_pair = re.compile(r'"coordinates":\s*\[\s*\[\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)')
_lock = threading.Lock()


def _log(msg):
    print(f"      footprints: {msg}", flush=True)


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def _quadkey(x, y, z):
    s = ""
    for i in range(z, 0, -1):
        m = 1 << (i - 1)
        s += str((1 if x & m else 0) + (2 if y & m else 0))
    return s


def _get(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def _ms_links():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "ms_dataset_links.csv")
    if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 30 * 86400:
        data = _get(MS_LINKS, 60)
        with open(p + ".tmp", "wb") as f:
            f.write(data)
        os.replace(p + ".tmp", p)
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def microsoft(bbox):
    """Microsoft outlines (lon/lat rings) whose first vertex lies in bbox."""
    lon0, lat0, lon1, lat1 = bbox
    keys = {_quadkey(*_tile_xy(la, lo, 9), 9) for la in (lat0, lat1) for lo in (lon0, lon1)}
    rows = [r for r in _ms_links() if r["QuadKey"] in keys]
    rings = []
    for r in rows:
        p = os.path.join(CACHE, f"ms_{r['Location'].replace(' ', '_')}_{r['QuadKey']}.csv.gz")
        with _lock:
            if not os.path.exists(p):
                data = _get(r["Url"], 300)
                with open(p + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(p + ".tmp", p)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                m = _first_pair.search(line)
                if not m:
                    continue
                lo, la = float(m.group(1)), float(m.group(2))
                if not (lon0 <= lo <= lon1 and lat0 <= la <= lat1):
                    continue
                g = json.loads(line).get("geometry", {})
                if g.get("type") == "Polygon":
                    rings.append(g["coordinates"][0])
    return rings


def openstreetmap(bbox, timeout=25):
    lon0, lat0, lon1, lat1 = bbox
    q = f'[out:json][timeout:{timeout}];(way["building"]({lat0},{lon0},{lat1},{lon1}););out geom;'
    for host in OVERPASS:
        try:
            req = urllib.request.Request(host, data=("data=" + urllib.request.quote(q)).encode(), headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=timeout + 10).read())
            return [[[p["lon"], p["lat"]] for p in el.get("geometry") or []]
                    for el in data.get("elements", []) if len(el.get("geometry") or []) >= 4]
        except Exception as e:
            _log(f"OSM {host.split('/')[2]} failed ({type(e).__name__})")
    return None


def _to_pixels(rings, src_transform, src_crs):
    from rasterio.warp import transform as _tr
    if not rings:
        return []
    lons = [p[0] for r in rings for p in r]
    lats = [p[1] for r in rings for p in r]
    xs, ys = _tr("EPSG:4326", src_crs, lons, lats)
    inv = ~src_transform
    out, k = [], 0
    for r in rings:
        n = len(r)
        px = [list(inv * (xs[k + i], ys[k + i])) for i in range(n)]
        k += n
        out.append(px)
    return out


def _merge(primary, extra, W, H):
    """primary + every extra polygon whose centre no primary outline covers."""
    import cv2
    mask = np.zeros((H, W), np.uint8)
    for p in primary:
        cv2.fillPoly(mask, [np.round(np.asarray(p)).astype(np.int32)], 1)
    out = list(primary)
    for p in extra:
        c = np.mean(np.asarray(p), axis=0).astype(int)
        if 0 <= c[1] < H and 0 <= c[0] < W and mask[c[1], c[0]]:
            continue
        out.append(p)
    return out


def fetch(src_transform, src_crs, width, height):
    """Pixel polygons of real buildings inside the image, or [] on any failure."""
    if src_transform is None or not src_crs or os.environ.get("DW_NO_FOOTPRINTS"):
        return []
    try:
        from rasterio.warp import transform_bounds
        b = (src_transform.c, src_transform.f + src_transform.e * height,
             src_transform.c + src_transform.a * width, src_transform.f)
        lon0, lat0, lon1, lat1 = transform_bounds(src_crs, "EPSG:4326", min(b[0], b[2]), min(b[1], b[3]),
                                                  max(b[0], b[2]), max(b[1], b[3]))
    except Exception as e:
        _log(f"bounds unavailable ({e})")
        return []
    if (lon1 - lon0) * (lat1 - lat0) > 0.05:          # > ~5 x 5 km: not a place, skip
        _log("image too large for outline lookup")
        return []
    bbox = (lon0, lat0, lon1, lat1)
    got = {}

    def run(name, fn):
        try:
            got[name] = fn(bbox)
        except Exception as e:
            _log(f"{name} unavailable ({type(e).__name__}: {e})")

    t0 = time.time()
    th = [threading.Thread(target=run, args=("microsoft", microsoft), daemon=True),
          threading.Thread(target=run, args=("osm", openstreetmap), daemon=True)]
    for t in th:
        t.start()
    for t in th:
        t.join(max(1.0, TIME_BUDGET_S - (time.time() - t0)))
    ms = _to_pixels(got.get("microsoft") or [], src_transform, src_crs)
    osm = _to_pixels(got.get("osm") or [], src_transform, src_crs)

    def inside(p):
        a = np.asarray(p)
        return a[:, 0].max() >= 0 and a[:, 0].min() < width and a[:, 1].max() >= 0 and a[:, 1].min() < height

    ms = [p for p in ms if len(p) >= 4 and inside(p)]
    osm = [p for p in osm if len(p) >= 4 and inside(p)]
    polys = _merge(ms, osm, width, height)
    _log(f"{len(ms)} Microsoft + {len(polys) - len(ms)} OpenStreetMap outlines "
         f"({time.time() - t0:.0f} s)")
    return polys
