"""
Fetch the zoom-19 (native ~0.3 m) Esri World Imagery mosaic covering EXACTLY
the same bounds as an existing zoom-18 Web Mercator GeoTIFF, for use as a
higher-resolution REAL display texture.

    python tools/fetch_z19_display.py hitec_city.tif hitec_city_z19.png
"""
import io
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import rasterio
from PIL import Image

R = 6378137.0


def fetch(z, x, y, tries=4):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard-SIH26175/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def main(tif, out, zoom=19):
    with rasterio.open(tif) as ds:
        t, W, H = ds.transform, ds.width, ds.height
    mpp = 2 * math.pi * R / (256 * 2 ** zoom)
    x0 = round((t.c + math.pi * R) / (256 * mpp))
    y0 = round((math.pi * R - t.f) / (256 * mpp))
    nx = round(W * abs(t.a) / (256 * mpp))
    ny = round(H * abs(t.e) / (256 * mpp))
    # exactness check: the grid must land on the source bounds to < 1 px
    ox = -math.pi * R + x0 * 256 * mpp
    oy = math.pi * R - y0 * 256 * mpp
    print(f"z{zoom}: {nx}x{ny} tiles from ({x0},{y0}); origin error "
          f"{abs(ox - t.c):.2f} m, {abs(oy - t.f):.2f} m")
    mosaic = Image.new("RGB", (nx * 256, ny * 256))
    jobs = [(i, j) for j in range(ny) for i in range(nx)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for (i, j), tile in zip(jobs, ex.map(lambda ij: fetch(zoom, x0 + ij[0], y0 + ij[1]), jobs)):
            mosaic.paste(tile, (i * 256, j * 256))
    mosaic.save(out)
    print("wrote", out, mosaic.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
