"""
Fetch REAL surrounding imagery (Esri World Imagery, lower zoom) around an
existing Web Mercator scene GeoTIFF, for the viewer's flat context ring --
replacing any mirrored/placeholder backdrop with what is actually there.

Writes <out>.png and <out>.json (EPSG:3857 bounds of the mosaic).

    python tools/fetch_context.py hitec_city.tif hitec_context --zoom 16 --radius 8
"""
import argparse
import io
import json
import math
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


def main(tif, out, zoom, radius):
    with rasterio.open(tif) as ds:
        b = ds.bounds
    mpp = 2 * math.pi * R / (256 * 2 ** zoom)
    cx = ((b.left + b.right) / 2 + math.pi * R) / (256 * mpp)
    cy = (math.pi * R - (b.top + b.bottom) / 2) / (256 * mpp)
    x0, y0 = int(cx) - radius, int(cy) - radius
    n = 2 * radius + 1
    mosaic = Image.new("RGB", (n * 256, n * 256))
    jobs = [(i, j) for j in range(n) for i in range(n)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for (i, j), t in zip(jobs, ex.map(lambda ij: fetch(zoom, x0 + ij[0], y0 + ij[1]), jobs)):
            mosaic.paste(t, (i * 256, j * 256))
    mosaic.save(out + ".png")
    left = -math.pi * R + x0 * 256 * mpp
    top = math.pi * R - y0 * 256 * mpp
    meta = {"crs": "EPSG:3857", "zoom": zoom, "left": left, "top": top,
            "right": left + n * 256 * mpp, "bottom": top - n * 256 * mpp}
    json.dump(meta, open(out + ".json", "w"), indent=1)
    print(f"z{zoom} {n}x{n} tiles -> {out}.png ({n * 256 * mpp / 1000:.1f} km square)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tif"); ap.add_argument("out")
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--radius", type=int, default=8)
    a = ap.parse_args()
    main(a.tif, a.out, a.zoom, a.radius)
