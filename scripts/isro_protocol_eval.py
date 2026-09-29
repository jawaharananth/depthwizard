"""Score an absolute DSM the way ISRO says it will: against a public DEM.

    python scripts/isro_protocol_eval.py OUT_DSM.tif [--dem glo30|srtm]

ISRO's FAQ (IMG-PROCESS-SAC/SIH-DepthWizard-2026 issue 1): GeoTIFF outputs are
scored as an absolute DSM "such as SRTM or Copernicus" -- "make sure that for
GeoTIFF images, the values match DEM heights". A 30 m DEM cannot see a single
building, so the fair comparison is at the DEM's own grid: our DSM is
block-averaged to ~30 m and compared cell by cell. Native-grid figures (DEM
bilinearly upsampled) are reported beside them.

Reports RMSE, MAE, bias, NMAD, LE90 and Pearson r.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dem_source  # noqa: E402


def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    e = (a[m] - b[m]).astype(np.float64)
    med = np.median(e)
    return {"rmse_m": float(np.sqrt(np.mean(e ** 2))), "mae_m": float(np.mean(np.abs(e))),
            "bias_m": float(e.mean()), "nmad_m": float(1.4826 * np.median(np.abs(e - med))),
            "le90_m": float(np.percentile(np.abs(e), 90)),
            "r": float(np.corrcoef(a[m], b[m])[0, 1]) if np.std(a[m]) > 0 else None, "n": int(m.sum())}


def block(a, f):
    """Mean over f x f blocks, NaN-aware."""
    H, W = a.shape
    h, w = H // f, W // f
    a = a[:h * f, :w * f].reshape(h, f, w, f)
    v = np.isfinite(a)
    s = np.where(v, a, 0).sum(axis=(1, 3))
    n = v.sum(axis=(1, 3))
    return np.where(n > 0.5 * f * f, s / np.maximum(n, 1), np.nan)


def reference(path, dem):
    with rasterio.open(path) as s:
        dsm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        tf, crs = s.transform, s.crs
    H, W = dsm.shape
    d = dem_source.sample_grid(tf.c, tf.f, H, abs(tf.a), crs.to_string(), dem_source=dem,
                               width_px=W, gsd_y_m=abs(tf.e))
    ref = np.asarray(d["dem"], np.float32)
    if ref.shape != dsm.shape:
        ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_LINEAR)
    # metres per pixel on the ground (Web Mercator overstates by 1/cos(lat))
    gsd = abs(tf.a)
    if crs.to_epsg() == 3857:
        from pyproj import Transformer
        lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(tf.c + W * tf.a / 2, tf.f + H * tf.e / 2)[1]
        gsd *= np.cos(np.radians(lat))
    return dsm, ref, gsd


def evaluate(path, dem="glo30"):
    dsm, ref, gsd = reference(path, dem)
    f = max(1, int(round(30.0 / gsd)))
    return {"dsm": os.path.basename(path), "dem": dem, "gsd_m": round(gsd, 3), "block_px": f,
            "at_30m": stats(block(dsm, f), block(ref, f)), "native": stats(dsm, ref)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dsm", nargs="+")
    ap.add_argument("--dem", default="glo30")
    a = ap.parse_args()
    for p in a.dsm:
        print(json.dumps(evaluate(p, a.dem), indent=1))
