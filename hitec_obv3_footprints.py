"""
Convert real Google Open Buildings V3 polygons (S2 cell 3bd, live-streamed
and bbox-filtered from the real 4.8GB public GCS file) into the pixel space
of hitec_city.tif -- second, independent real footprint source to check
against MS Global ML Building Footprints' real coverage gap (281 polygons
vs this source's much larger real count in the same tight scene bbox).
"""
import csv
import json
import math

import rasterio
from shapely import wkt as shapely_wkt

R = 6378137.0
import sys
TIF = sys.argv[1] if len(sys.argv) > 1 else "hitec_city.tif"
OB_CSV = "nyc_fidi/_ob_v3_hitec.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "hitec_obv3_polygons_px.json"

with rasterio.open(TIF) as ds:
    transform = ds.transform
    minx, miny, maxx, maxy = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top

def lonlat_to_3857(lon, lat):
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y

inv = ~transform
polygons = []
areas = []
kept = 0
dropped_partial = 0
dropped_low_conf = 0

with open(OB_CSV, newline="") as f:
    r = csv.reader(f)
    for row in r:
        lat, lon, area_m2, conf, wkt_geom = row[0], row[1], row[2], row[3], row[4]
        conf = float(conf)
        if conf < 0.65:
            dropped_low_conf += 1
            continue
        # WKT: POLYGON((lon lat, lon lat, ...))
        # Outer ring only: a courtyard (hole) is filled by the prism.
        _g = shapely_wkt.loads(wkt_geom)
        if _g.geom_type != "Polygon":
            _g = max(_g.geoms, key=lambda g: g.area)
        pts = list(_g.exterior.coords)
        px_ring = []
        ok = True
        for plon, plat in pts:
            x, y = lonlat_to_3857(float(plon), float(plat))
            if not (minx <= x <= maxx and miny <= y <= maxy):
                ok = False
                break
            col, row_ = inv * (x, y)
            px_ring.append([col, row_])
        if ok and len(px_ring) >= 4:
            polygons.append(px_ring)
            areas.append(float(area_m2))
            kept += 1
        elif not ok:
            dropped_partial += 1

print(f"real Open Buildings V3 footprints kept (fully inside mosaic, conf>=0.65): {kept}")
print(f"dropped (outside/partial): {dropped_partial}, dropped (low confidence): {dropped_low_conf}")
if areas:
    areas.sort()
    print(f"area m2: min {areas[0]:.1f}  median {areas[len(areas)//2]:.1f}  max {areas[-1]:.1f}")
json.dump({"polygons": polygons}, open(OUT, "w"))
print(f"written {OUT}")
