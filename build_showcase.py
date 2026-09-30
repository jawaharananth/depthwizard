"""Build a showcase place the way HITEC City and Periyar were built, for any point in India.

    python build_showcase.py mumbai_lower_parel --lat 19.0000 --lon 72.8260 --kind urban \
        --name "Lower Parel" --region "Mumbai, India"

kind: urban | hilly | forested | sparse

Everything real, fetched for the exact 2.45 km x 2.45 km square around the point:

  imagery     Esri World Imagery z18 (4096 px, ~0.6 m) + z19 (~0.3 m) of the
              same bounds (display texture and the model's input) + a z16 and
              z14 context ring, flat, labelled as context
  footprints  Microsoft Global ML Building Footprints + OpenStreetMap buildings
              where Microsoft has none (urban, hilly, sparse)
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


def z19_coverage(path):
    """Share of the 256 px tiles in a z19 mosaic that are real imagery: the
    placeholder tile repeats byte for byte, real tiles never do."""
    from collections import Counter
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("L")
    w, h = im.size
    small = np.asarray(im.resize((w // 16, h // 16), Image.BOX))
    tiles = small.reshape(h // 256, 16, w // 256, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    top = Counter(t.tobytes() for t in tiles).most_common(1)[0][1]
    return 1.0 - (top / len(tiles) if top > 1 else 0.0)


def esri_capture(lat, lon, work):
    """Capture date and satellite of the Esri World Imagery under a point, from
    its own metadata layers. The pass time is not published: sun-synchronous
    optical satellites cross near 10:30 local mean solar time, so that is
    assumed (and disclosed as assumed in the scene notes)."""
    cp = os.path.join(work, "capture.json")
    if os.path.exists(cp):
        return json.load(open(cp))
    d = 0.01
    url = ("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/identify?"
           f"geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326&layers=all&tolerance=2"
           f"&mapExtent={lon - d},{lat - d},{lon + d},{lat + d}&imageDisplay=800,800,96"
           "&returnGeometry=false&f=json")
    try:
        req = urllib.request.Request(url, headers=UA)
        res = json.load(urllib.request.urlopen(req, timeout=30)).get("results", [])
    except Exception as e:
        log(f"imagery metadata unavailable ({e}) -- sun left to the shadows")
        return None
    for r in res:
        at = r.get("attributes", {})
        s = str(at.get("SRC_DATE") or "")
        if "Resolution Metadata" in r.get("layerName", "") and len(s) == 8 and s.isdigit():
            date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
            h = 10.5 - lon / 15.0
            out = {"date": date, "utc": f"{date}T{int(h):02d}:{round(h % 1 * 60):02d}",
                   "sensor": at.get("SRC_DESC"), "res_m": at.get("SRC_RES"), "product": at.get("NICE_NAME")}
            json.dump(out, open(cp, "w"))
            return out
    return None


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


def osm_footprints(tif, work):
    """OpenStreetMap building outlines inside the scene, as pixel polygons.

    Microsoft's India footprints are thin in dense cores (0 inside Lower
    Parel's 2.4 km square), where OSM mappers have traced the towers.
    """
    import rasterio
    out = os.path.join(work, "osm_polygons_px.json")
    if os.path.exists(out):
        return json.load(open(out))["polygons"]
    with rasterio.open(tif) as ds:
        inv, b = ~ds.transform, ds.bounds
    to_ll = lambda x, y: (math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))
    lon0, lat0 = to_ll(b.left, b.bottom)
    lon1, lat1 = to_ll(b.right, b.top)
    q = f'[out:json][timeout:180];(way["building"]({lat0},{lon0},{lat1},{lon1}););out geom;'
    data = None
    for host in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            req = urllib.request.Request(host, data=("data=" + urllib.request.quote(q)).encode(), headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=240).read())
            break
        except Exception as e:
            log(f"OSM buildings: {host} failed ({e})")
    polys = []
    for el in (data or {}).get("elements", []):
        pts = el.get("geometry") or []
        if len(pts) < 4:
            continue
        px = []
        for p in pts:
            x = R * math.radians(p["lon"])
            y = R * math.log(math.tan(math.pi / 4 + math.radians(p["lat"]) / 2))
            c, r_ = inv * (x, y)
            px.append([float(c), float(r_)])
        polys.append(px)
    if data is not None:          # a failed query is not an answer: never cache it
        json.dump({"source": "OpenStreetMap contributors (ODbL)", "polygons": polys}, open(out, "w"))
    log(f"OSM buildings: {len(polys)}")
    return polys


def merge_footprints(primary, extra, size=4096):
    """primary + every extra polygon whose centre is not already covered."""
    import cv2
    mask = np.zeros((size, size), np.uint8)
    for p in primary:
        cv2.fillPoly(mask, [np.array(p, np.int32)], 1)
    out = list(primary)
    for p in extra:
        c = np.mean(np.array(p), axis=0).astype(int)
        if 0 <= c[1] < size and 0 <= c[0] < size and mask[c[1], c[0]]:
            continue
        out.append(p)
    return out


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

    # Esri has no zoom-19 capture over much of the hills; there it serves one
    # "map data not yet available" tile over and over, and the model reads that
    # as flat ground (Darjeeling came out 0.00 m everywhere). Such a mosaic is
    # not used at all: the build falls back to the trained-scale pass on z18.
    hires = z19 if z19_coverage(z19) >= 0.9 else None
    if hires is None:
        log(f"{a.id}: zoom-19 imagery is placeholder tiles ({z19_coverage(z19):.0%} real) -- using zoom 18")

    import cv2
    polys = [] if a.kind == "forested" else merge_footprints(ms_footprints(tif, work), osm_footprints(tif, work))
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
              display_texture_path=hires, depth_image_path=hires,
              context_image=ctx, context_far_image=far,
              osm_structures=a.kind in ("urban", "hilly"))
    cap = esri_capture(a.lat, a.lon, work)
    if cap:
        kw.update(capture_date=cap["date"], capture_utc=cap["utc"])
        log(f"{a.id}: imagery {cap['product']} {cap['sensor']} captured {cap['date']} (pass ~10:30 LMT assumed)")
    if landscape:
        # Hills and forest stand on the real terrain: the model's height above
        # ground carries no terrain, so the ground mesh is Copernicus GLO-30.
        kw.update(terrain_from_dem=True, canopy_mask_from_chm=chm is not None, canopy_blur_m=1.2)
    if a.kind == "hilly":
        # Hill towns: Microsoft's footprints are thin here. Use the upload
        # path's outlines instead -- Overture (OpenStreetMap + Microsoft +
        # Google Open Buildings), shifted onto this image by cross-correlation,
        # weak outlines kept as low relief rather than boxes, tree tops
        # filtered out -- and a display texture sharpened from the file's own
        # pixels when no real zoom-19 capture exists.
        import rasterio
        import footprint_sources
        with rasterio.open(tif) as _ds:
            ov = footprint_sources.fetch(_ds.transform, _ds.crs, _ds.width, _ds.height)
        if len(ov) > len(polys):
            log(f"{a.id}: {len(ov)} Overture outlines replace {len(polys)} Microsoft/OSM ones")
            polys = [p for p in ov if len(p) >= 4]
            mask = np.zeros((4096, 4096), np.uint8)
            for p in polys:
                cv2.fillPoly(mask, [np.array(p, np.int32)], 1)
            kw.update(gt_building_mask=mask.astype(bool), gt_building_polygons=polys)
        kw.update(sr_texture=hires is None)
    if a.kind == "hilly" and chm is not None:
        # The global canopy map calls a dense hill town's roofs and lanes
        # "trees" (Darjeeling: 8-39 m cones between houses). Keep canopy only
        # where the image itself looks like vegetation, and never on or right
        # beside a building.
        import rasterio
        with rasterio.open(tif) as _ds:
            rgb = _ds.read([1, 2, 3]).astype(np.float32)
        s = rgb.sum(0) + 1e-3
        exg = (2 * rgb[1] - rgb[0] - rgb[2]) / s
        veg = cv2.morphologyEx(((exg > 0.04) & (rgb[1] / s > 0.35)).astype(np.uint8),
                               cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        near_bldg = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
        keep = (veg > 0) & ~near_bldg
        before = float((chm > 2).mean())
        chm = np.where(keep, chm, 0.0).astype(np.float32)
        log(f"{a.id}: canopy kept on vegetation only: {before:.1%} -> {float((chm > 2).mean()):.1%} of pixels")
        kw.update(canopy_height_map=chm)
    log(f"building {a.id} ({a.kind}) with {len(polys)} footprints, canopy {'yes' if chm is not None else 'no'}")
    bci.build(tif, a.id, **kw)

    run([PY, "-c", f"import scenes; scenes.save({a.id!r}, {a.name + ', ' + a.region!r})"])
    # The DSM and sigma files travel with the scene (rooftop and water tools read them).
    import glob
    import shutil
    for f in glob.glob(os.path.join(ROOT, "final_out", f"city_{a.id}_*DSM.tif")) +             glob.glob(os.path.join(ROOT, "final_out", f"city_{a.id}_sigma.tif")):
        shutil.copy2(f, os.path.join(ROOT, "scenes", a.id, os.path.basename(f)))
    if polys or a.kind == "hilly":
        subprocess.run([PY, "tools/roof_details.py", a.id], cwd=ROOT)
    subprocess.run([PY, "tools/water_mask.py", "--scene", os.path.join("scenes", a.id)], cwd=ROOT)

    pj = os.path.join(ROOT, "viewer", "places.json")
    places = json.load(open(pj))
    entry = {"id": a.id, "name": a.name, "region": a.region, "landscape": a.kind, "lidar": False}
    idx = next((i for i, p in enumerate(places) if p["id"] == a.id), None)
    if idx is None:
        places.append(entry)
    else:
        places[idx] = {**places[idx], **entry}       # a rebuild keeps its place in the list
    json.dump(places, open(pj, "w"), indent=2)
    log(f"done: scenes/{a.id}, listed in viewer/places.json")


if __name__ == "__main__":
    main()
