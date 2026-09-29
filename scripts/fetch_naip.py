"""Real 0.6 m four-band imagery (R, G, B, NIR) over the DFC2019 LiDAR tiles.

    python scripts/fetch_naip.py test      # the 18 held-out benchmark tiles
    python scripts/fetch_naip.py train     # the training areas (for tuning)

USDA NAIP (Florida, 0.6 m, public domain; the 2019 capture where it exists)
from Microsoft's Planetary Computer, cut out server-side on exactly each
LiDAR tile's square and UTM grid. Cartosat-2S-like input from a real sensor:
native 0.6 m, real near-infrared -- not a degraded 0.3 m WorldView view.
Written to work/naip/<tile>.tif with the colour bands tagged and band 4
described as near infrared, so the build reads R, G, B and finds the NIR.
"""
import glob
import json
import os
import sys
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dfc2019_height as DH      # noqa: E402
import dfc2019_loader as L       # noqa: E402

TRUTH = os.path.join(ROOT, "dfc2019_data", "truth", "Track3-Truth")
OUT = os.path.join(ROOT, "work", "naip")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
CROP = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"


def tile_bounds(tile):
    """(crs, left, bottom, right, top) of the LiDAR tile; the loader's transform
    is anchored at the tile's lower edge."""
    gt = L.load_tile(tile, TRUTH)
    t, (H, W) = gt["transform"], gt["cls"].shape
    crs = gt["crs"] if not isinstance(gt["crs"], int) else f"EPSG:{gt['crs']}"
    return crs, t.c, t.f, t.c + W * t.a, t.f + H * abs(t.e)


def fetch(tile):
    out = os.path.join(OUT, f"{tile}.tif")
    if os.path.exists(out):
        return out
    crs, l, b, r, t = tile_bounds(tile)
    bbox = transform_bounds(crs, "EPSG:4326", l, b, r, t)
    q = {"collections": ["naip"], "bbox": list(bbox), "limit": 20}
    feats = json.loads(urllib.request.urlopen(urllib.request.Request(
        STAC, data=json.dumps(q).encode(), headers={"Content-Type": "application/json"}),
        timeout=60).read())["features"]
    feats = [f for f in feats if abs(float(f["properties"].get("gsd", 0)) - 0.6) < 0.01]
    feats.sort(key=lambda f: abs(int(f["properties"]["datetime"][:4]) - 2019))
    if not feats:
        raise RuntimeError("no 0.6 m NAIP over this tile")
    year = feats[0]["properties"]["datetime"][:4]
    feats = [f for f in feats if f["properties"]["datetime"][:4] == year]   # one capture, several sheets
    W = H = int(round((r - l) / 0.6))
    dst_tf = from_bounds(l, b, r, t, W, H)
    img = np.zeros((4, H, W), np.uint8)
    c = crs.replace(":", "%3A")
    for f in feats:
        u = (f"{CROP}/{l},{b},{r},{t}/{W}x{H}.tif?collection=naip&item={f['id']}"
             f"&assets=image&asset_bidx=image%7C1,2,3,4&coord_crs={c}&dst_crs={c}&resampling=average")
        data = urllib.request.urlopen(u, timeout=180).read()
        with rasterio.MemoryFile(data) as mf, mf.open() as s:
            a = s.read()
        mask = a[4] > 0 if a.shape[0] > 4 else a[:4].sum(0) > 0
        fill = (img.sum(0) == 0) & mask
        img[:, fill] = a[:4][:, fill]
        if (img.sum(0) > 0).mean() > 0.999:
            break
    os.makedirs(OUT, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=W, height=H, count=4, dtype="uint8",
                       crs=crs, transform=dst_tf, nodata=0) as d:
        d.write(img)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined]
        d.update_tags(SOURCE=f"USDA NAIP {year}, 0.6 m, via Microsoft Planetary Computer")
        d.set_band_description(4, "near infrared")
    return out


def main():
    which = sys.argv[1]
    if which == "test":
        tiles = sorted(DH.benchmark_tiles())
    else:
        tiles = sorted({os.path.basename(p).split("_v")[0].replace("DFC_", "")
                        for p in glob.glob(os.path.join(ROOT, "training", "dfc_tiles", "*", "*_RGB.h5"))})
    for tile in tiles:
        try:
            print(tile, fetch(tile), flush=True)
        except Exception as e:
            print(tile, "FAILED", type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main()
