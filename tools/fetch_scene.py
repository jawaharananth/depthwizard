"""
Fetch an Esri World Imagery mosaic for an exact XYZ tile block and write it as
a Web Mercator GeoTIFF (exact affine from the tile indices, no resampling).

    python tools/fetch_scene.py OUT.tif --zoom 18 --x0 188138 --y0 118157 --n 16
"""
import argparse
import io
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

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


def main(out, zoom, x0, y0, n):
    mpp = 2 * math.pi * R / (256 * 2 ** zoom)
    mosaic = Image.new("RGB", (n * 256, n * 256))
    jobs = [(i, j) for j in range(n) for i in range(n)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for (i, j), t in zip(jobs, ex.map(lambda ij: fetch(zoom, x0 + ij[0], y0 + ij[1]), jobs)):
            mosaic.paste(t, (i * 256, j * 256))
    arr = np.array(mosaic)
    tf = Affine(mpp, 0, -math.pi * R + x0 * 256 * mpp, 0, -mpp, math.pi * R - y0 * 256 * mpp)
    with rasterio.open(out, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=3, dtype="uint8", crs="EPSG:3857", transform=tf) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)
    mosaic.save(out.rsplit(".", 1)[0] + ".png")
    print(f"z{zoom} {n}x{n} tiles -> {out} ({arr.shape[1]}x{arr.shape[0]}, "
          f"{n * 256 * mpp:.0f} m square)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--x0", type=int, required=True)
    ap.add_argument("--y0", type=int, required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    main(a.out, a.zoom, a.x0, a.y0, a.n)
