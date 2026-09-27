"""Build a showcase place the way HITEC City and Periyar were built, for any point in India.

    python build_showcase.py mumbai_lower_parel --lat 19.0000 --lon 72.8260 --kind urban \
        --name "Lower Parel" --region "Mumbai, India"

kind: urban | hilly | forested | sparse

Everything real, fetched for the exact 2.45 km x 2.45 km square around the point:

  imagery     Esri World Imagery z18 (4096 px, ~0.6 m) + z19 (~0.3 m) of the
              same bounds (display texture and the model's input) + a z16 and
              z14 context ring, flat, labelled as context
  footprints  Microsoft Global ML Building Footprints (urban, hilly, sparse)
  heights     the fine-tuned model on the z19 imagery, fused with Google Open
              Buildings 2.5D heights where they exist (pipeline default)
  canopy      Meta / WRI CHMv2 canopy height, one continuous displaced surface
  terrain     Copernicus GLO-30; hills and forest stand on the DEM terrain
              (landscape mode), cities on the model's ground + DEM datum
  structures  flyovers, bridges, viaducts from OpenStreetMap
  after       rooftop structures, water surfaces, cursor-readout rasters,
              saved as scenes/<id> and listed in viewer/places.json

Downloads are cached under work/showcase/<id>/, so a rebuild only re-runs
the pipeline.
"""
import argparse
import csv
import gzip
import io
import json
import math
import os
import subprocess
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
R = 6378137.0
MS_LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
UA = {"User-Agent": "DepthWizard-SIH26175/1.0"}


def log(*a):
    print("[showcase]", *a, flush=True)


def run(args):
    log("$", " ".join(str(x) for x in args))
    r = subprocess.run([str(x) for x in args], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"step failed: {args[1] if len(args) > 1 else args}")


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


def quadkey(x, y, z):
    s = ""
    for i in range(z, 0, -1):
        m = 1 << (i - 1)
        s += str((1 if x & m else 0) + (2 if y & m else 0))
    return s


def ms_footprints(tif, work):
    """Microsoft Global ML footprints inside the scene, as pixel polygons."""
    import rasterio
    out = os.path.join(work, "ms_polygons_px.json")
    if os.path.exists(out):
        return json.load(open(out))["polygons"]
    with rasterio.open(tif) as ds:
        inv, (W, H) = ~ds.transform, (ds.width, ds.height)
        b = ds.bounds
    to_ll = lambda x, y: (math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))
    lon0, lat0 = to_ll(b.left, b.bottom)
    lon1, lat1 = to_ll(b.right, b.top)
    keys = set()
    for la in (lat0, lat1):
        for lo in (lon0, lon1):
            keys.add(quadkey(*tile_xy(la, lo, 9), 9))
    links = urllib.request.urlopen(urllib.request.Request(MS_LINKS, headers=UA), timeout=120).read().decode()
    urls = [r["Url"] for r in csv.DictReader(io.StringIO(links))
            if r["Location"] == "India" and r["QuadKey"] in keys]
    polys = []
    for u in urls:
        log(f"footprints: {u.rsplit('/', 2)[-2]}")
        raw = gzip.decompress(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=600).read())
        for line in raw.decode().splitlines():
            g = json.loads(line).get("geometry", {})
            if g.get("type") != "Polygon":
                continue
            ring = g["coordinates"][0]
            if not any(lon0 <= p[0] <= lon1 and lat0 <= p[1] <= lat1 for p in ring):
                continue
            px = []
            for lo, la in ring:
                x = R * math.radians(lo)
                y = R * math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
                c, r_ = inv * (x, y)
                px.append([float(c), float(r_)])
            polys.append(px)
    json.dump({"source": "Microsoft Global ML Building Footprints", "polygons": polys}, open(out, "w"))
    log(f"footprints: {len(polys)} buildings")
    return polys


def canopy(tif, work):
    """Meta/WRI CHMv2 canopy height on the scene grid (metres)."""
    import cv2
    import rasterio
    from rasterio.windows import from_bounds
    out = os.path.join(work, "chm.npy")
    if os.path.exists(out):
        return np.load(out)
    with rasterio.open(tif) as ds:
        b, W, H = ds.bounds, ds.width, ds.height
    o = 20037508.342789244
    t = 2 * o / 1024
    cx, cy = (b.left + b.right) / 2, (b.top + b.bottom) / 2
    qk = quadkey(int((cx + o) // t), int((o - cy) // t), 10)
    url = ("http://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/"
           f"dinov3_global_chm_v2_ml3/chm/{qk}.tif")
    try:
        with rasterio.open("/vsicurl/" + url) as r:
            w = from_bounds(b.left, b.bottom, b.right, b.top, r.transform)
            a = r.read(1, window=w, boundless=True, fill_value=0).astype(np.float32)
    except Exception as e:
        log(f"canopy map unavailable ({e}); building without it")
        return None
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.where(a > 0, cv2.GaussianBlur(a, (0, 0), 1.0), 0.0).astype(np.float32)
    np.save(out, a)
    log(f"canopy: tile {qk}, p99 {np.percentile(a, 99):.1f} m")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--kind", choices=["urban", "hilly", "forested", "sparse"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--fetch-only", action="store_true",
                    help="download and cache everything, build nothing (safe to run beside a build)")
    a = ap.parse_args()

    work = os.path.join(ROOT, "work", "showcase", a.id)
    os.makedirs(work, exist_ok=True)
    tif = os.path.join(work, f"{a.id}.tif")
    z19 = os.path.join(work, f"{a.id}_z19.png")
    ctx, far = os.path.join(work, f"{a.id}_context"), os.path.join(work, f"{a.id}_far")
    if not os.path.exists(tif):
        x, y = tile_xy(a.lat, a.lon, 18)
        run([PY, "tools/fetch_scene.py", tif, "--zoom", 18, "--x0", x - 8, "--y0", y - 8, "--n", 16])
    if not os.path.exists(z19):
        run([PY, "tools/fetch_z19_display.py", tif, z19])
    if not os.path.exists(ctx + ".png"):
        run([PY, "tools/fetch_context.py", tif, ctx, "--zoom", 16, "--radius", 8])
    if not os.path.exists(far + ".png"):
        run([PY, "tools/fetch_context.py", tif, far, "--zoom", 14, "--radius", 6])

    import cv2
    polys = [] if a.kind == "forested" else ms_footprints(tif, work)
    chm = canopy(tif, work)
    if a.fetch_only:
        log(f"fetched {a.id}: {len(polys)} footprints, canopy {'yes' if chm is not None else 'no'}")
        return
    mask = np.zeros((4096, 4096), np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.array(p, np.int32)], 1)

    sys.path.insert(0, ROOT)
    import build_city_image as bci
    landscape = a.kind in ("hilly", "forested")
    kw = dict(max_px=4096, use_dem=True, auto_zoom=False,
              gt_building_mask=mask.astype(bool), gt_building_polygons=polys,
              canopy_height_map=chm, continuous_vegetation=chm is not None,
              ground_grid=1400, pbr_maps=True, crown_radius_ratio=0.18,
              display_texture_path=z19, depth_image_path=z19,
              context_image=ctx, context_far_image=far,
              osm_structures=a.kind in ("urban", "hilly"))
    if landscape:
        # Hills and forest stand on the real terrain: the model's height above
        # ground carries no terrain, so the ground mesh is Copernicus GLO-30.
        kw.update(terrain_from_dem=True, canopy_mask_from_chm=chm is not None, canopy_blur_m=1.2)
    log(f"building {a.id} ({a.kind}) with {len(polys)} footprints, canopy {'yes' if chm is not None else 'no'}")
    bci.build(tif, a.id, **kw)

    run([PY, "-c", f"import scenes; scenes.save({a.id!r}, {a.name + ', ' + a.region!r})"])
    if polys:
        subprocess.run([PY, "tools/roof_details.py", a.id], cwd=ROOT)
    subprocess.run([PY, "tools/water_mask.py", "--scene", os.path.join("scenes", a.id)], cwd=ROOT)

    pj = os.path.join(ROOT, "viewer", "places.json")
    places = json.load(open(pj))
    places = [p for p in places if p["id"] != a.id]
    places.append({"id": a.id, "name": a.name, "region": a.region, "landscape": a.kind, "lidar": False})
    json.dump(places, open(pj, "w"), indent=2)
    log(f"done: scenes/{a.id}, listed in viewer/places.json")


if __name__ == "__main__":
    main()
