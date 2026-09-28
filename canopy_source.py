"""Real tree heights for any georeferenced scene: Meta / WRI canopy height
(CHMv2, 1 m, global, CC BY 4.0), read as a window straight from the public
cloud-optimised GeoTIFFs and warped onto the scene's own grid.

    chm = canopy_source.fetch(src_transform, src_crs, width, height)   # metres or None

The tiles are indexed by Web Mercator quadkey at zoom 10; a scene that
straddles tiles reads each and mosaics them. Any failure (no network, no
tile, timeout) returns None -- trees then come from the image alone, as
before -- because a missing extra must never stop a build.
"""
from __future__ import annotations

import math

import numpy as np

BASE = ("http://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/"
        "dinov3_global_chm_v2_ml3/chm/{qk}.tif")
_O = 20037508.342789244
_Z = 10


def _quadkey(x: int, y: int, z: int) -> str:
    q = ""
    for i in range(z, 0, -1):
        d, m = 0, 1 << (i - 1)
        if x & m:
            d += 1
        if y & m:
            d += 2
        q += str(d)
    return q


def fetch(src_transform, src_crs, width: int, height: int, timeout_s: int = 20):
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject, transform_bounds
        from rasterio.windows import from_bounds

        left, top = src_transform * (0, 0)
        right, bottom = src_transform * (width, height)
        l3, b3, r3, t3 = transform_bounds(src_crs, "EPSG:3857", min(left, right), min(top, bottom),
                                          max(left, right), max(top, bottom), densify_pts=21)
        t = 2 * _O / (1 << _Z)
        xs = range(int((l3 + _O) // t), int((r3 + _O) // t) + 1)
        ys = range(int((_O - t3) // t), int((_O - b3) // t) + 1)
        if len(xs) * len(ys) > 4:
            return None                                   # a scene this large is not an upload
        out = np.full((height, width), np.nan, np.float32)
        got = False
        with rasterio.Env(GDAL_HTTP_TIMEOUT=timeout_s, GDAL_HTTP_CONNECTTIMEOUT=8,
                          GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            for tx in xs:
                for ty in ys:
                    try:
                        with rasterio.open("/vsicurl/" + BASE.format(qk=_quadkey(tx, ty, _Z))) as r:
                            w = from_bounds(l3, b3, r3, t3, r.transform).round_offsets().round_lengths()
                            a = r.read(1, window=w, boundless=True, fill_value=0).astype(np.float32)
                            wt = r.window_transform(w)
                            dst = np.full((height, width), np.nan, np.float32)
                            reproject(a, dst, src_transform=wt, src_crs=r.crs, dst_transform=src_transform,
                                      dst_crs=src_crs, resampling=Resampling.bilinear, src_nodata=None,
                                      dst_nodata=np.nan)
                            fill = np.isfinite(dst) & ~np.isfinite(out)
                            out[fill] = dst[fill]
                            got = True
                    except Exception:
                        continue
        if not got or not np.isfinite(out).any():
            return None
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 120.0).astype(np.float32)
    except Exception:
        return None
