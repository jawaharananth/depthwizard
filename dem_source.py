"""
External DEM ingestion -- the Tier A anchor.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR

The problem statement names SRTM/DEM/GCPs as scale references, and until now
Tier A has never actually been exercised in this project: every scene has been
Tier B (shadow-calibrated) or Tier C (relative). This supplies the missing
anchor.

Be clear about what a 30 m DEM can and cannot do:

  IT CAN   fix the absolute datum and the broad terrain surface. Reconstructed
           elevations become real heights above sea level rather than a relative
           field with an assumed scale.

  IT CANNOT improve building heights. At 30 m posting a building occupies a
           fraction of one pixel, and global DEMs are deliberately smoothed
           toward bare earth. Anyone expecting building RMSE to fall because a
           DEM was added has misunderstood what was added.

SOURCE

Copernicus DEM GLO-30 from the AWS Open Data registry, read directly over HTTPS
through GDAL's /vsicurl/. No API key, no account, no extra dependency -- rasterio
is already required. It is chosen over SRTM because it is newer, has better
coastal and urban quality, and needs no Earthdata login; SRTM via the `elevation`
package would additionally require GDAL command-line binaries, which are not
reliably present on Windows.

A locally supplied DEM GeoTIFF always takes precedence when given, since a user
with survey-grade local data should never be overridden by a global model.

VERTICAL DATUM -- the part that is easy to get wrong

Copernicus DEM heights are orthometric, referenced to the EGM2008 geoid. The
DFC2019 ground truth is ellipsoidal WGS84. The two differ by the geoid
separation, which `fit_offset` recovers directly from the data: measured on
JAX_165 over 100,849 common ground pixels it is +33.4 m (IQR 5.9 m), i.e. the
Copernicus surface sits 33.4 m above the ellipsoidal truth. That is an order of
magnitude larger than any error this project is trying to measure, so mixing the
two datums silently would swamp every result.

(An earlier draft of this note said -23 m. That is the resulting ground
ELEVATION in Jacksonville, not the separation between the datums -- about 10 m
orthometric minus a 33.4 m geoid height. Two different quantities, and confusing
them is exactly the mistake this paragraph exists to prevent.)

`geoid_offset_m` makes the conversion explicit and refuses to guess.
"""
import os
import math

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

COP30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _cop30_tile_name(lon: float, lat: float) -> str:
    """
    Copernicus GLO-30 tiles are named by the integer degree of their SOUTH-WEST
    corner, so the coordinate must be floored, not rounded -- 30.9N lives in the
    N30 tile, not N31.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    la = int(math.floor(abs(lat))) if lat >= 0 else int(math.ceil(abs(lat)))
    lo = int(math.floor(abs(lon))) if lon >= 0 else int(math.ceil(abs(lon)))
    return f"Copernicus_DSM_COG_10_{ns}{la:02d}_00_{ew}{lo:03d}_00_DEM"


def sample_grid(utm_x0: float, utm_y_top: float, size_px: int, gsd_m: float,
                utm_epsg: str, dem_path: str = None,
                geoid_offset_m: float = None) -> dict:
    """
    Sample an external DEM onto the scene's UTM grid.

    geoid_offset_m is ADDED to the DEM's values to convert them into the target
    vertical datum. For Copernicus (EGM2008 orthometric) against WGS84
    ellipsoidal heights in Florida this is about -23. Passing None leaves the
    DEM in its own datum and says so in the result, rather than silently
    assuming a conversion.

    Returns {"dem": HxW metres or None, "source": str, "coverage": float, ...}.
    """
    xs = utm_x0 + (np.arange(size_px) + 0.5) * gsd_m
    ys = utm_y_top - (np.arange(size_px) + 0.5) * gsd_m
    gx, gy = np.meshgrid(xs, ys)
    lon_f, lat_f = warp_transform(utm_epsg, "EPSG:4326", gx.ravel(), gy.ravel())
    lon = np.array(lon_f).reshape(gx.shape)
    lat = np.array(lat_f).reshape(gx.shape)

    if dem_path:
        url, source = dem_path, f"local: {os.path.basename(dem_path)}"
    else:
        tile = _cop30_tile_name(float(lon.mean()), float(lat.mean()))
        url = f"/vsicurl/{COP30_BASE}/{tile}/{tile}.tif"
        source = f"Copernicus GLO-30 ({tile})"

    try:
        with rasterio.open(url) as src:
            # Invert the geotransform arithmetically rather than calling
            # src.index() per coordinate. index() is a Python-level call and a
            # 512x512 grid is 262144 of them -- it does not return in reasonable
            # time. The transform is affine, so one vectorised expression does
            # the same job.
            inv = ~src.transform
            cols_f = inv.a * lon.ravel() + inv.b * lat.ravel() + inv.c
            rows_f = inv.d * lon.ravel() + inv.e * lat.ravel() + inv.f
            rows = np.clip(rows_f.astype(np.int64), 0, src.height - 1)
            cols = np.clip(cols_f.astype(np.int64), 0, src.width - 1)
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1
            # One windowed read covering every sample, then index into it. A
            # per-pixel read over HTTP would be millions of range requests.
            block = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            dem = block[rows - r0, cols - c0].reshape(lon.shape)
            nodata = src.nodata
    except Exception as exc:
        return {"dem": None, "source": source, "coverage": 0.0,
                "error": f"{type(exc).__name__}: {exc}"}

    valid = np.isfinite(dem)
    if nodata is not None:
        valid &= (dem != nodata)
    # Copernicus uses very negative fill values over unmapped ocean.
    valid &= (dem > -400) & (dem < 9000)
    coverage = float(valid.mean())
    dem = np.where(valid, dem, np.nan)

    if geoid_offset_m is not None:
        dem = dem + float(geoid_offset_m)
        datum = f"converted with geoid offset {geoid_offset_m:+.1f} m"
    else:
        datum = "DEM's own vertical datum (EGM2008 for Copernicus) -- NOT converted"

    return {"dem": dem, "source": source, "coverage": coverage,
            "datum_note": datum, "geoid_offset_m": geoid_offset_m}


def fit_offset(our_terrain: np.ndarray, dem: np.ndarray,
               ground_mask: np.ndarray) -> dict:
    """
    Constant offset between our terrain and an external DEM, over ground only.

    This is what turns a relative surface into an absolute one without touching
    its shape: a single scalar, fitted where BOTH surfaces claim to describe
    bare earth. Buildings and canopy are excluded because the DEM does not model
    them, so including them would bias the offset upward by roughly the mean
    structure height.

    The median is used rather than the mean: DEM voids, water, and any
    misclassified rooftop are outliers, and one of them should not move the
    datum for the whole scene. The spread is returned so a caller can refuse a
    fit that clearly did not converge.
    """
    m = ground_mask & np.isfinite(dem) & np.isfinite(our_terrain)
    n = int(m.sum())
    if n < 500:
        return {"offset_m": None, "n": n, "reason": "too few common ground pixels"}
    d = dem[m] - our_terrain[m]
    med = float(np.median(d))
    iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
    return {"offset_m": med, "n": n, "iqr_m": iqr,
            "spread_ok": bool(iqr < 10.0)}


if __name__ == "__main__":
    import sys
    import dfc2019_loader as L

    tile = sys.argv[1] if len(sys.argv) > 1 else "JAX_165"
    c = L.parse_dsm_txt(f"dfc2019_data/truth/Track3-Truth/{tile}_DSM.txt")
    size, gsd = int(c["size_px"]), c["gsd_m"]
    r = sample_grid(c["utm_x"], c["utm_y"] + size * gsd, size, gsd, "EPSG:32617")
    print(f"{tile}: {r['source']}")
    if r["dem"] is None:
        print("  FAILED:", r.get("error"))
    else:
        d = r["dem"]
        print(f"  coverage {r['coverage']*100:.1f}%   "
              f"range {np.nanmin(d):.1f} .. {np.nanmax(d):.1f} m")
        print(f"  {r['datum_note']}")
        t = L.load_tile(tile, "dfc2019_data/truth/Track3-Truth")
        gt = t["dsm"].astype(np.float32)
        gnd = (t["cls"] == 2)
        off = fit_offset(gt, d, gnd)
        print(f"  offset vs LiDAR ground: {off.get('offset_m')} m "
              f"(n={off.get('n')}, IQR {off.get('iqr_m')})")
