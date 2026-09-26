"""
Water mask from REAL data: OpenStreetMap water geometry (Overpass, cached)
refined by the scene's own texture pixels.

    venv\\Scripts\\python.exe tools\\water_mask.py --scene scenes/hitec_2x2 [--viz DIR]
    venv\\Scripts\\python.exe tools\\water_mask.py --scene scenes/periyar_forest

Outputs (in the scene dir)
  water_mask.png   8-bit, --res x --res (default 2048), SAME bounds as the
                   texture (scene.json texture_file / terrain_texture.png):
                     255 = OSM water AND pixel colour matches this scene's water
                     160 = inside OSM water polygon, colour does NOT match
                           (weed/hyacinth cover, dry bed, bund, shadow...)
                      80 = outside OSM polygon but within 12 m of it and colour
                           matches (shoreline the OSM trace missed)
                       0 = not water
  water.json       {source, dates, colour model, mean real water colour,
                    polygons in world coords, per-OSM-feature coverage}

World frame (viewer/index.html realCoordsAt, build_city_image context
placement): x = (easting - affine[2]) / k, z = (northing - affine[5]) / k,
k = scene.json crs_units_per_m or 1.  x east, z north; the tile spans
x in [0, W], z in [-H, 0].
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
R_EARTH = 6378137.0
OVERPASS = "https://overpass-api.de/api/interpreter"


def lonlat_to_3857(lon, lat):
    x = np.radians(lon) * R_EARTH
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R_EARTH
    return x, y


def m3857_to_lonlat(x, y):
    return math.degrees(x / R_EARTH), math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2)


def fetch_osm(bbox, cache_path, refresh=False):
    if os.path.exists(cache_path) and not refresh:
        return json.load(open(cache_path)), True
    try:
        import truststore            # Windows cert store; plain certifi fails on this laptop
        truststore.inject_into_ssl()
    except Exception:
        pass
    import requests
    s, w, n, e = bbox
    q = (f"[out:json][timeout:120];("
         f"way[natural=water]({s},{w},{n},{e});relation[natural=water]({s},{w},{n},{e});"
         f"way[waterway]({s},{w},{n},{e});"
         f"way[landuse~\"reservoir|basin\"]({s},{w},{n},{e});relation[landuse~\"reservoir|basin\"]({s},{w},{n},{e});"
         f");out geom;")
    hdr = {"User-Agent": "DepthWizard-SIH/1.0 (water mask research)"}
    last = None
    for url in (OVERPASS, OVERPASS.replace("https://", "http://"), OVERPASS, OVERPASS):
        time.sleep(0 if last is None else 20)          # 504/429: back off and retry
        try:
            r = requests.post(url, data={"data": q}, headers=hdr, timeout=240)
            r.raise_for_status()
            j = r.json()
            j["_query"] = q
            j["_fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            json.dump(j, open(cache_path, "w"))
            return j, False
        except Exception as ex:          # try the next endpoint
            last = ex
    raise RuntimeError(f"Overpass failed: {last}")


def element_polygons(el):
    """shapely (Multi)Polygon in lon/lat for an area element, else None."""
    from shapely.geometry import Polygon, LineString
    from shapely.ops import polygonize, unary_union, linemerge
    if el["type"] == "way":
        g = el.get("geometry") or []
        if len(g) >= 4 and g[0] == g[-1]:
            p = Polygon([(q["lon"], q["lat"]) for q in g])
            return p.buffer(0) if not p.is_valid else p
        return None
    outers, inners = [], []
    for m in el.get("members", []):
        g = m.get("geometry")
        if not g or len(g) < 2:
            continue
        ls = LineString([(q["lon"], q["lat"]) for q in g])
        (inners if m.get("role") == "inner" else outers).append(ls)
    if not outers:
        return None
    outer = unary_union(list(polygonize(linemerge(outers))))
    if inners:
        inner = unary_union(list(polygonize(linemerge(inners))))
        outer = outer.difference(inner)
    return outer if not outer.is_empty else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scenes/hitec_2x2")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--viz", default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    from shapely.geometry import box, LineString, mapping
    from shapely.ops import transform as shp_transform

    sd = args.scene.rstrip("/\\")
    name = os.path.basename(sd)
    meta = json.load(open(os.path.join(sd, "scene.json")))
    a = meta["affine_transform"]
    k = meta.get("crs_units_per_m") or 1.0
    n_src = meta["resolution"][0]
    E0, N0 = a[2], a[5]
    E1, N1 = E0 + a[0] * n_src, N0 + a[4] * n_src
    lon0, lat1 = m3857_to_lonlat(E0, N1)
    lon1, lat0 = m3857_to_lonlat(E1, N0)
    pad = 0.01
    bbox = (lat1 - pad, lon0 - pad, lat0 + pad, lon1 + pad)
    osm, cached = fetch_osm(bbox, os.path.join("cache", f"osm_water_{name}.json"), args.refresh)
    print(f"OSM: {len(osm['elements'])} elements ({'cache' if cached else 'fetched'}), "
          f"base {osm.get('osm3s', {}).get('timestamp_osm_base')}")

    RES = args.res
    pix = a[0] * n_src / RES                       # map units per mask pixel

    def to_px(x, y):
        E, N = lonlat_to_3857(np.asarray(x), np.asarray(y))
        return (E - E0) / pix, (N0 - N) / pix

    tile = box(E0, N1, E1, N0)
    osm_mask = np.zeros((RES, RES), np.uint8)
    line_mask = np.zeros((RES, RES), np.uint8)
    feats = []
    for el in osm["elements"]:
        tags = el.get("tags", {})
        area = None
        if tags.get("natural") == "water" or tags.get("landuse") in ("reservoir", "basin"):
            area = element_polygons(el)
        if area is not None:
            a3857 = shp_transform(lambda x, y, z=None: lonlat_to_3857(np.asarray(x), np.asarray(y)), area)
            clip = a3857.intersection(tile)
            if clip.is_empty:
                continue
            polys = [clip] if clip.geom_type == "Polygon" else [g for g in getattr(clip, "geoms", []) if g.geom_type == "Polygon"]
            tmp = np.zeros_like(osm_mask)
            for p in polys:
                ext = np.c_[(np.array(p.exterior.coords)[:, 0] - E0) / pix, (N0 - np.array(p.exterior.coords)[:, 1]) / pix]
                cv2.fillPoly(tmp, [np.round(ext * 16).astype(np.int32)], 1, shift=4)
                for r in p.interiors:
                    h = np.c_[(np.array(r.coords)[:, 0] - E0) / pix, (N0 - np.array(r.coords)[:, 1]) / pix]
                    cv2.fillPoly(tmp, [np.round(h * 16).astype(np.int32)], 0, shift=4)
            osm_mask |= tmp
            feats.append({"osm": f"{el['type']}/{el['id']}", "name": tags.get("name"),
                          "tags": {kk: tags[kk] for kk in ("natural", "water", "landuse", "intermittent") if kk in tags},
                          "area_in_tile_m2": round(clip.area / (k * k), 1),
                          "area_total_m2": round(a3857.area / (k * k), 1),
                          "_px": tmp.astype(bool)})
        elif "waterway" in tags and el["type"] == "way":
            g = el.get("geometry") or []
            ls = shp_transform(lambda x, y, z=None: lonlat_to_3857(np.asarray(x), np.asarray(y)),
                               LineString([(q["lon"], q["lat"]) for q in g]))
            clip = ls.intersection(tile)
            if clip.is_empty:
                continue
            try:
                wd = float(tags.get("width", "0").split()[0])
            except ValueError:
                wd = 0.0
            wd = wd or {"river": 10.0, "canal": 6.0, "stream": 2.0, "drain": 2.0, "ditch": 1.5}.get(tags["waterway"], 2.0)
            thick = max(1, int(round(wd * k / pix)))
            for part in ([clip] if clip.geom_type == "LineString" else getattr(clip, "geoms", [])):
                c = np.array(part.coords)
                pts = np.c_[(c[:, 0] - E0) / pix, (N0 - c[:, 1]) / pix]
                cv2.polylines(line_mask, [np.round(pts).astype(np.int32)], False, 1, thick)
            feats.append({"osm": f"way/{el['id']}", "name": tags.get("name"),
                          "tags": {"waterway": tags["waterway"], **({"width": tags["width"]} if "width" in tags else {})},
                          "length_in_tile_m": round(clip.length / k, 1), "line": True})

    # ---- texture at mask resolution
    tex_name = meta.get("texture_file") or "terrain_texture.png"
    rgb = np.array(Image.open(os.path.join(sd, tex_name)).convert("RGB"))
    rgb = cv2.resize(rgb, (RES, RES), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(gray, (5, 5))
    lstd = np.sqrt(np.maximum(cv2.blur(gray * gray, (5, 5)) - mu * mu, 0))

    out = {"scene": name, "texture_file": tex_name, "texture_capture": meta.get("capture_time_utc"),
           "mask_file": "water_mask.png", "mask_res": [RES, RES],
           "mask_pixel_m": round(pix / k, 4),
           "mask_encoding": {"255": "OSM water + colour match", "160": "OSM water, colour mismatch",
                             "80": "colour-matched shoreline within 12 m outside OSM polygon", "0": "not water"},
           "world_frame": "x = (easting - affine[2]) / k, z = (northing - affine[5]) / k, k = "
                          f"{k} (scene.json crs_units_per_m or 1); x east, z north; mask pixel (col,row) "
                          f"-> x = (col+0.5)*{pix/k:.4f}, z = -(row+0.5)*{pix/k:.4f}",
           "source": {"geometry": "OpenStreetMap via Overpass API (natural=water, landuse=reservoir/basin, "
                                  "waterway=*), (c) OpenStreetMap contributors, ODbL",
                      "osm_timestamp": osm.get("osm3s", {}).get("timestamp_osm_base"),
                      "osm_cache": f"cache/osm_water_{name}.json",
                      "colour": f"scene texture {tex_name} pixels"},
           }

    ero = cv2.erode(osm_mask, np.ones((5, 5), np.uint8))
    if ero.sum() < 50:
        out.update(water_in_bounds=False,
                   note="No OSM water polygon inside the tile bounds; no water mask produced from "
                        "colour alone (without an in-scene water sample there is no real colour model, "
                        "and dark roofs/shadows would be false positives).",
                   osm_features=[{kk: vv for kk, vv in f.items() if kk != "_px"} for f in feats],
                   polygons_world=[], mean_water_rgb=None)
        final = np.zeros((RES, RES), np.uint8)
    else:
        # robust colour model of THIS scene's water (Lab), fitted inside eroded OSM polygons
        X = lab[ero > 0]
        X = X[lstd[ero > 0] < np.percentile(lstd[ero > 0], 90)]
        # An OSM lake polygon is the lake's mapped extent: in the dry season
        # much of it is exposed bed, weed or bund, and much of the land just
        # outside looks like that too.  So colours are clustered JOINTLY over
        # (a) pixels inside the eroded OSM water and (b) a land ring 20-80 m
        # outside it, equal sample sizes; a cluster is WATER when >= 80 % of
        # its members come from inside the lake.  Every pixel is then labelled
        # by its nearest cluster centre.
        rng = np.random.default_rng(0)
        ring = (cv2.dilate(osm_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(80 * k / pix) + 1,) * 2)) > 0) & \
               ~(cv2.dilate(osm_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(20 * k / pix) + 1,) * 2)) > 0)
        Xin = lab[ero > 0]
        Xout = lab[ring]
        nS = int(min(len(Xin), len(Xout), 150000))
        Xin = Xin[rng.choice(len(Xin), nS, replace=False)]
        Xout = Xout[rng.choice(len(Xout), nS, replace=False)]
        W = np.array([0.6, 1.0, 1.0], np.float32)          # de-weight L* (sun glint, ripples)
        KC = 12
        _, lk, cen = cv2.kmeans((np.r_[Xin, Xout] * W).astype(np.float32), KC, None,
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.3),
                                4, cv2.KMEANS_PP_CENTERS)
        lk = lk.ravel()
        n_in = np.bincount(lk[:nS], minlength=KC)
        n_out = np.bincount(lk[nS:], minlength=KC)
        purity = n_in / np.maximum(n_in + n_out, 1)
        water_c = np.where((purity >= 0.80) & (n_in >= 0.02 * nS))[0]
        cluster_info = {"k": KC, "centres_lab": [[round(float(v / w), 1) for v, w in zip(c, W)] for c in cen],
                        "inside_share": [round(float(p), 3) for p in purity],
                        "inside_px": n_in.tolist(), "ring_px": n_out.tolist(),
                        "water_clusters": water_c.tolist()}
        X = Xin[np.isin(lk[:nS], water_c)]
        m_ = X.mean(0) if len(X) else Xin.mean(0)
        C = np.cov(X.T) if len(X) > 10 else np.eye(3)
        # nearest-centre labelling, in chunks (RES^2 x KC distances)
        flat = (lab.reshape(-1, 3) * W)
        nearest = np.empty(len(flat), np.int16)
        for i in range(0, len(flat), 1 << 20):
            dd = ((flat[i:i + (1 << 20), None, :] - cen[None]) ** 2).sum(-1)
            nearest[i:i + (1 << 20)] = dd.argmin(1)
        colour_ok = np.isin(nearest, water_c).reshape(RES, RES)
        colour_ok = cv2.morphologyEx(colour_ok.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        buf = cv2.dilate(osm_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(12 * k / pix) + 1,) * 2))
        final = np.zeros((RES, RES), np.uint8)
        final[(osm_mask > 0)] = 160
        final[(osm_mask > 0) & colour_ok] = 255
        # shoreline extension: colour match, near OSM water, connected to confirmed water
        ext = (buf > 0) & (osm_mask == 0) & colour_ok
        n, lbl = cv2.connectedComponents(((final == 255) | ext).astype(np.uint8))
        conf_lbls = np.unique(lbl[final == 255])
        final[ext & np.isin(lbl, conf_lbls)] = 80
        # waterway lines: only where the texture confirms visible water
        line_ok = (line_mask > 0) & colour_ok & (final == 0)
        final[line_ok] = 80

        wat = final >= 255
        mean_rgb = rgb[wat].mean(0) if wat.any() else None
        med_rgb = np.median(rgb[wat], 0) if wat.any() else None
        # polygons in world coordinates (confirmed + shoreline)
        polys = []
        cnts, hier = cv2.findContours(((final == 255) | (final == 80)).astype(np.uint8), cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        s = pix / k
        for i, c in enumerate(cnts):
            if hier[0][i][3] != -1:
                continue                                         # holes attached below
            ar = cv2.contourArea(c) * s * s
            if ar < 20:
                continue
            c = cv2.approxPolyDP(c, 0.75, True).reshape(-1, 2)
            ring = [[round((x + 0.5) * s, 2), round(-(y + 0.5) * s, 2)] for x, y in c]
            holes = []
            j = hier[0][i][2]
            while j != -1:
                h = cv2.approxPolyDP(cnts[j], 0.75, True).reshape(-1, 2)
                if cv2.contourArea(cnts[j]) * s * s >= 20:
                    holes.append([[round((x + 0.5) * s, 2), round(-(y + 0.5) * s, 2)] for x, y in h])
                j = hier[0][j][0]
            polys.append({"area_m2": round(ar, 1), "exterior": ring, "holes": holes})
        polys.sort(key=lambda p: -p["area_m2"])
        for f in feats:
            if "_px" in f:
                px = f.pop("_px")
                f["colour_confirmed_fraction"] = round(float((final[px] == 255).mean()), 3) if px.any() else None
        out.update(
            water_in_bounds=True,
            colour_model={"space": "CIELAB (OpenCV 8-bit scaling)", "mean": [round(float(x), 2) for x in m_],
                          "cov": [[round(float(x), 2) for x in r] for r in C],
                          "rule": "joint k-means (k=12, L* weight 0.6) over eroded-OSM-water pixels and a "
                                  "20-80 m land ring; clusters with >=80 % lake members are water; pixels "
                                  "labelled by nearest centre",
                          "water_px_in_model": int(len(X)),
                          "clusters": cluster_info},
            mean_water_rgb=[round(float(x), 1) for x in mean_rgb] if mean_rgb is not None else None,
            median_water_rgb=[round(float(x), 1) for x in med_rgb] if med_rgb is not None else None,
            mean_water_hex=("#%02x%02x%02x" % tuple(int(round(x)) for x in mean_rgb)) if mean_rgb is not None else None,
            colour_note="sRGB of the texture as delivered (display-graded imagery, not reflectance)",
            area_m2={"confirmed_255": round(float((final == 255).sum()) * s * s, 1),
                     "osm_colour_mismatch_160": round(float((final == 160).sum()) * s * s, 1),
                     "shoreline_or_stream_80": round(float((final == 80).sum()) * s * s, 1)},
            osm_features=feats,
            polygons_world=polys,
        )
    for f in out.get("osm_features", []):
        f.pop("_px", None)
    Image.fromarray(final).save(os.path.join(sd, "water_mask.png"), optimize=True)
    json.dump(out, open(os.path.join(sd, "water.json"), "w"), indent=1)
    print(json.dumps({kk: out[kk] for kk in out if kk not in ("polygons_world", "osm_features", "colour_model")}, indent=1))
    print("OSM features in tile:", [(f["osm"], f.get("name"), f.get("area_in_tile_m2") or f.get("length_in_tile_m"),
                                    f.get("colour_confirmed_fraction")) for f in out.get("osm_features", [])])
    print("polygons:", len(out.get("polygons_world", [])))

    if args.viz:
        os.makedirs(args.viz, exist_ok=True)
        ov = rgb.copy()
        col = {255: (0, 120, 255), 160: (255, 60, 60), 80: (255, 255, 0)}
        for v_, c_ in col.items():
            m = final == v_
            ov[m] = (0.45 * ov[m] + 0.55 * np.array(c_)).astype(np.uint8)
        edge = cv2.morphologyEx(osm_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        ov[edge] = (255, 255, 255)
        ov[(line_mask > 0) & (final == 0)] = (200, 0, 200)
        Image.fromarray(np.concatenate([rgb, ov], 1)).resize((2048, 1024), Image.LANCZOS).save(
            os.path.join(args.viz, f"water_{name}.jpg"), quality=88)
        # zoom on the largest water body
        if out.get("polygons_world"):
            p = np.array(out["polygons_world"][0]["exterior"])
            cx, cy = int(p[:, 0].mean() / (pix / k)), int(-p[:, 1].mean() / (pix / k))
            r = 300
            x0, y0 = max(0, cx - r), max(0, cy - r)
            Image.fromarray(np.concatenate([rgb[y0:y0 + 2 * r, x0:x0 + 2 * r], ov[y0:y0 + 2 * r, x0:x0 + 2 * r]], 1)
                            ).save(os.path.join(args.viz, f"water_{name}_zoom.jpg"), quality=90)


if __name__ == "__main__":
    main()
