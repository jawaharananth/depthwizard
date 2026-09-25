"""
Footprint regularisation with buildingregulariser (MIT), guarded.

Real footprints (Open Buildings / Microsoft polygons, traced on the image)
wobble by a pixel or two and neighbouring blocks that share a street line sit
a degree or two apart. buildingregulariser snaps each outline's edges to its
own dominant axis (plus 45 degrees), then rotates near-parallel neighbours onto
a common direction.

It is a SHAPE clean-up, never a new measurement, so every result is checked
against the original outline and thrown away when it moved too far:

    |area change| > 10 %   or   IoU(original, regularised) < 0.85
        -> keep the original polygon unchanged

Coordinates are converted to metres (x*gsd, -y*gsd) for the library so its
tolerances are physical, then back to the same pixel grid, so the output file
is a drop-in replacement for the input (same order, same length).

    python tools/regularise_footprints.py hitec2x2_polygons_px.json \
        hitec2x2_polygons_px_reg.json --gsd 0.5971642834779395
"""
import argparse
import json
import sys
import time

import numpy as np

MAX_AREA_CHANGE = 0.10
MIN_IOU = 0.85


def regularise(polygons: list, gsd_m: float, neighbor_alignment: bool = True,
               num_cores: int = 1) -> tuple:
    """Returns (out_polygons, report). out has the same length/order as the
    input; rejected or failed entries are the original ring, untouched."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    from buildingregulariser import regularize_geodataframe

    geoms, ok_idx = [], []
    for i, ring in enumerate(polygons):
        r = np.asarray(ring, float)
        if len(r) < 4:
            continue
        p = Polygon(np.column_stack([r[:, 0] * gsd_m, -r[:, 1] * gsd_m]))
        if not p.is_valid:
            p = make_valid(p)
            if p.geom_type != "Polygon":
                continue
        if p.area <= 0:
            continue
        geoms.append(p)
        ok_idx.append(i)
    gdf = gpd.GeoDataFrame({"src": ok_idx}, geometry=geoms, crs="EPSG:3857")
    t = time.time()
    reg = regularize_geodataframe(
        gdf, parallel_threshold=1.0, simplify=True, simplify_tolerance=0.5,
        allow_45_degree=True, allow_circles=True, num_cores=num_cores,
        neighbor_alignment=neighbor_alignment, neighbor_search_distance=30.0,
        neighbor_max_rotation=10)
    secs = time.time() - t

    out = [list(map(list, np.asarray(r, float))) for r in polygons]
    rep = {"input": len(polygons), "regularised": 0, "rejected_area": 0,
           "rejected_iou": 0, "failed": 0, "seconds": round(secs, 1)}
    by_src = {int(s): g for s, g in zip(reg["src"], reg.geometry)}
    ious, dareas = [], []
    for s, p0 in zip(ok_idx, geoms):
        g = by_src.get(s)
        if g is None or g.is_empty:
            rep["failed"] += 1
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        if g.geom_type != "Polygon" or not g.is_valid or g.area <= 0:
            rep["failed"] += 1
            continue
        da = abs(g.area - p0.area) / p0.area
        inter = g.intersection(p0).area
        iou = inter / (g.area + p0.area - inter)
        if da > MAX_AREA_CHANGE:
            rep["rejected_area"] += 1
            continue
        if iou < MIN_IOU:
            rep["rejected_iou"] += 1
            continue
        xy = np.asarray(g.exterior.coords, float)
        out[s] = [[float(x / gsd_m), float(-y / gsd_m)] for x, y in xy]
        rep["regularised"] += 1
        ious.append(iou)
        dareas.append(da)
    if ious:
        rep["iou_median"] = round(float(np.median(ious)), 4)
        rep["iou_p05"] = round(float(np.percentile(ious, 5)), 4)
        rep["area_change_median"] = round(float(np.median(dareas)), 4)
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gsd", type=float, required=True, help="metres per pixel")
    ap.add_argument("--no-neighbors", action="store_true")
    ap.add_argument("--cores", type=int, default=1)
    a = ap.parse_args()
    polys = json.load(open(a.src))["polygons"]
    out, rep = regularise(polys, a.gsd, not a.no_neighbors, a.cores)
    json.dump({"polygons": out, "regularisation": dict(
        rep, tool="buildingregulariser (MIT)",
        params="allow_45_degree, neighbor_alignment (30 m, 10 deg), "
               "simplify 0.5 m, parallel 1.0 m",
        guard=f"kept original where |dA|>{MAX_AREA_CHANGE:.0%} or IoU<{MIN_IOU}",
        source=a.src)}, open(a.dst, "w"))
    print(json.dumps(rep))


if __name__ == "__main__":
    sys.exit(main())
