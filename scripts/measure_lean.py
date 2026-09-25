"""
Measure a scene's relief-displacement ("lean") vector: how far, in image
pixels, a point moves per metre of height above ground.

Two real, independent height rasters disagree in a specific, measurable way
on oblique imagery:
  * the monocular depth field is predicted FROM the oblique image, so a
    tower's height sits where its ROOF appears;
  * Open Buildings 2.5D is derived from near-nadir Sentinel-2, so the same
    tower's height sits over its BASE.
For each height band, the shift that best aligns the two band masks is
lean * band_height. A line fit through the origin across bands gives the
lean vector; residuals say whether the model actually holds.

    python scripts/measure_lean.py hitec_city.tif cache/<depth>.npy
"""
import sys

import cv2
import numpy as np
import rasterio
import rasterio.warp
from pyproj import Transformer

sys.path.insert(0, ".")
import open_buildings_source as obs


def ob_height_on_grid(tif_path, shape):
    with rasterio.open(tif_path) as ds:
        tf, crs = ds.transform, ds.crs.to_string()
        W0, H0 = ds.width, ds.height
    H, W = shape
    tf = tf * rasterio.Affine.scale(W0 / W, H0 / H)
    xs = [tf * p for p in [(0, 0), (W, 0), (0, H), (W, H)]]
    lons, lats = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        [p[0] for p in xs], [p[1] for p in xs])
    clon, clat = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    epsg = (32600 if clat >= 0 else 32700) + int((clon + 180) // 6) + 1
    r = obs.sample_height_grid(min(lons), min(lats), max(lons), max(lats), epsg=epsg)
    dst = np.full((H, W), np.nan, np.float32)
    rasterio.warp.reproject(source=r["heights"].astype(np.float32), destination=dst,
                            src_transform=r["transform"], src_crs=f"EPSG:{epsg}",
                            dst_transform=tf, dst_crs=crs,
                            resampling=rasterio.warp.Resampling.bilinear)
    return np.nan_to_num(dst, nan=0.0)


def band_shift(a, b, max_shift):
    """Integer (dx, dy) that best aligns mask b onto mask a, by FFT xcorr."""
    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    xc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    xc = np.fft.fftshift(xc)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = xc[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return ix - max_shift, iy - max_shift, float(win.max() / (np.abs(xc).mean() + 1e-9))


def measure(tif_path, depth_npy):
    depth = np.load(depth_npy).astype(np.float32)
    ground = cv2.erode(depth, np.ones((61, 61), np.uint8))
    ground = cv2.GaussianBlur(ground, (0, 0), 20)
    ndsm = np.maximum(depth - ground, 0)
    ob = ob_height_on_grid(tif_path, depth.shape)

    pts = []
    for lo, hi in [(15, 25), (25, 35), (35, 50), (50, 90)]:
        mid = 0.5 * (lo + hi)
        a = ((ndsm >= lo) & (ndsm < hi)).astype(np.float32)   # at roof (image)
        b = ((ob >= lo) & (ob < hi)).astype(np.float32)       # at base (nadir)
        if a.sum() < 500 or b.sum() < 500:
            continue
        dx, dy, peak = band_shift(a, b, max_shift=120)
        pts.append((mid, dx, dy, peak))
        print(f"  band {lo:>2}-{hi:<2} m: shift dx={dx:+4d} dy={dy:+4d} px  (peak {peak:.1f})")
    if len(pts) < 2:
        print("  not enough tall structure in both rasters to measure a lean")
        return None
    h = np.array([p[0] for p in pts])
    dx = np.array([p[1] for p in pts]); dy = np.array([p[2] for p in pts])
    vx = float((h * dx).sum() / (h * h).sum()); vy = float((h * dy).sum() / (h * h).sum())
    res = np.sqrt((dx - vx * h) ** 2 + (dy - vy * h) ** 2)
    print(f"  lean = ({vx:+.3f}, {vy:+.3f}) px per metre of height; "
          f"fit residual median {np.median(res):.1f} px")
    return vx, vy, float(np.median(res))


if __name__ == "__main__":
    measure(sys.argv[1], sys.argv[2])
