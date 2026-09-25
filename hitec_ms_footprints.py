"""
Convert real Microsoft Global ML Building Footprints (India, quadkey 123301331,
2026-08-13 snapshot) into the pixel space of the existing hitec_city.tif mosaic,
to test whether real GT footprints fix the VERDICT POOR / recall 0.089 result
the automated-discovery path got on this same scene earlier this session.

No height attribute in the MS data for India (height=-1.0 sentinel) -- heights
still come from build_city_image's existing Open Buildings fusion / depth model,
same as before. This only replaces FOOTPRINT GEOMETRY, not height.
"""
import json
import math

import rasterio

R = 6378137.0
TIF = "hitec_city.tif"
MS_CSV = "nyc_fidi/_ms_hitec.csv"
OUT = "hitec_ms_polygons_px.json"

with rasterio.open(TIF) as ds:
    transform = ds.transform
    W, H = ds.width, ds.height
    minx, miny, maxx, maxy = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top


def lonlat_to_3857(lon, lat):
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


inv = ~transform
polygons = []
kept = 0
dropped_partial = 0
with open(MS_CSV) as f:
    for line in f:
        d = json.loads(line)
        ring = d["geometry"]["coordinates"][0]
        px_ring = []
        ok = True
        for lon, lat in ring:
            x, y = lonlat_to_3857(lon, lat)
            if not (minx <= x <= maxx and miny <= y <= maxy):
                ok = False
                break
            col, row = inv * (x, y)
            px_ring.append([col, row])
        if ok and len(px_ring) >= 4:
            polygons.append(px_ring)
            kept += 1
        elif not ok:
            dropped_partial += 1

print(f"real MS footprints kept (fully inside mosaic): {kept}")
print(f"dropped (outside/partial): {dropped_partial}")
json.dump({"polygons": polygons}, open(OUT, "w"))
print(f"written {OUT}")
