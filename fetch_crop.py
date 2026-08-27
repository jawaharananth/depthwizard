"""
Crop a window out of a remote cloud-optimised orthomosaic.

Reads over HTTP through GDAL's /vsicurl/, so only the bytes covering the window
are transferred -- the Varanasi mosaic is 181 MB and 38226x32998, and none of
that needs to land on disk to take a 600 m crop out of the middle.

Output is a real GeoTIFF with its CRS and transform preserved, so the ground
sampling distance downstream is MEASURED from the file rather than assumed.
"""
import argparse, os
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling


def crop(url, out_path, cx_frac, cy_frac, extent_m, out_px):
    with rasterio.open("/vsicurl/" + url) as s:
        gsd_x = abs(s.transform.a)
        # A geographic CRS reports degrees; convert to metres so extent means
        # the same thing regardless of how the source is projected.
        if s.crs and s.crs.is_geographic:
            import math
            lat = s.transform.f + s.transform.e * s.height / 2
            gsd_x = gsd_x * 111320.0 * math.cos(math.radians(lat))
        px_needed = int(extent_m / gsd_x)
        cx = int(s.width * cx_frac)
        cy = int(s.height * cy_frac)
        x0 = max(0, min(cx - px_needed // 2, s.width - px_needed))
        y0 = max(0, min(cy - px_needed // 2, s.height - px_needed))
        w = min(px_needed, s.width - x0)
        h = min(px_needed, s.height - y0)
        win = Window(x0, y0, w, h)
        print(f"source {s.width}x{s.height} @ {gsd_x*100:.1f} cm/px, crs {s.crs}")
        print(f"window ({x0},{y0}) {w}x{h} px = {w*gsd_x:.0f}x{h*gsd_x:.0f} m "
              f"-> {out_px}x{out_px}")

        data = s.read([1, 2, 3], window=win,
                      out_shape=(3, out_px, out_px), resampling=Resampling.average)
        out_gsd = (w * gsd_x) / out_px
        tr = s.window_transform(win)
        new_tr = rasterio.Affine(tr.a * w / out_px, tr.b, tr.c,
                                 tr.d, tr.e * h / out_px, tr.f)
        prof = {"driver": "GTiff", "width": out_px, "height": out_px, "count": 3,
                "dtype": "uint8", "crs": s.crs, "transform": new_tr,
                "compress": "deflate"}
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        with rasterio.open(out_path, "w", **prof) as d:
            d.write(data)
    filled = float((data.sum(axis=0) > 0).mean())
    print(f"wrote {out_path}  ({out_gsd*100:.1f} cm/px, {filled*100:.1f}% non-black)")
    return out_gsd, filled


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--cx", type=float, default=0.5)
    ap.add_argument("--cy", type=float, default=0.5)
    ap.add_argument("--extent", type=float, default=600.0)
    ap.add_argument("--px", type=int, default=2400)
    a = ap.parse_args()
    crop(a.url, a.out, a.cx, a.cy, a.extent, a.px)
