"""
LOD1 "context city" around a detailed scene: every real building in the
surrounding context imagery, extruded to its real Open Buildings height.

A lower tier than the core model, and labelled as such: footprints are
Microsoft Global ML Building Footprints, heights the Open Buildings 2.5D
raster (75th percentile inside the footprint plus a 4 m margin, which
tolerates the ~7 m co-registration offset measured on the core scene, and
no lean correction), roofs are textured from the real context imagery and
walls carry each building's measured median colour. Buildings with no height
data are left out rather than given a guessed one; heights under 3 m are
raised to 3 m (a single storey), which is disclosed.

World frame matches the core GLB: x = easting - tile_left, z = northing -
tile_top (north is +z), y up, base at 0 (the viewer lifts the group to the
ground-edge level, exactly as it does for the flat context imagery).

    python tools/build_context_city.py SCENE.tif CONTEXT_PREFIX MS.csv OUT.glb
"""
import json
import math
import sys

import cv2
import numpy as np
import rasterio
import rasterio.warp
from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, ".")
import city_model
import image_grade
import open_buildings_source as obs
from glb_export import _BufferBuilder, _pad4, FLOAT, UNSIGNED_INT, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER

R = 6378137.0
MIN_H = 3.0


def to3857(lon, lat):
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def main(scene_tif, ctx_prefix, ms_csv, out_glb):
    with rasterio.open(scene_tif) as ds:
        ox, oy = ds.transform.c, ds.transform.f
        core = ds.bounds
    ctx = json.load(open(ctx_prefix + ".json"))
    L, T, Rt, B = ctx["left"], ctx["top"], ctx["right"], ctx["bottom"]
    img = image_grade.grade(np.array(Image.open(ctx_prefix + ".png").convert("RGB")))
    Hc, Wc = img.shape[:2]
    mpp_x, mpp_y = (Rt - L) / Wc, (T - B) / Hc

    lon0, lon1 = math.degrees(L / R), math.degrees(Rt / R)
    lat_of = lambda y: math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    lat0, lat1 = lat_of(B), lat_of(T)

    # --- real heights: Open Buildings 2.5D, warped onto a 4 m grid in 3857 ---
    zone = int(((lon0 + lon1) / 2 + 180) // 6) + 1
    epsg = (32600 if (lat0 + lat1) / 2 >= 0 else 32700) + zone
    r = obs.sample_height_grid(lon0, lat0, lon1, lat1, epsg=epsg)
    print("Open Buildings:", r["note"])
    if r["heights"] is None:
        raise RuntimeError("no Open Buildings heights available for the context area: "
                           + r["note"])
    hres = 4.0
    hw, hh = int((Rt - L) / hres), int((T - B) / hres)
    htf = rasterio.Affine(hres, 0, L, 0, -hres, T)
    hgrid = np.full((hh, hw), np.nan, np.float32)
    rasterio.warp.reproject(source=r["heights"].astype(np.float32), destination=hgrid,
                            src_transform=r["transform"], src_crs=f"EPSG:{epsg}",
                            dst_transform=htf, dst_crs="EPSG:3857",
                            resampling=rasterio.warp.Resampling.bilinear)

    # --- footprints ---
    rv, rf, ruv = [], [], []   # roofs
    wv, wn, wc, wf = [], [], [], []   # walls
    kept = skipped_core = skipped_noh = 0
    heights = []
    with open(ms_csv) as f:
        for line in f:
            d = json.loads(line)
            ring = np.asarray(d["geometry"]["coordinates"][0], np.float64)
            if ring[:, 0].max() < lon0 or ring[:, 0].min() > lon1 or \
               ring[:, 1].max() < lat0 or ring[:, 1].min() > lat1:
                continue
            X, Y = to3857(ring[:, 0], ring[:, 1])
            cx, cy = X.mean(), Y.mean()
            if not (L < cx < Rt and B < cy < T):
                continue
            if core.left < cx < core.right and core.bottom < cy < core.top:
                skipped_core += 1
                continue
            poly = Polygon(np.c_[X, Y]).simplify(0.8)
            if poly.is_empty or not poly.is_valid or poly.area < 12:
                continue
            if poly.exterior.is_ccw:
                poly = Polygon(list(poly.exterior.coords)[::-1])   # clockwise (north-up)
            pts = np.asarray(poly.exterior.coords)[:-1]
            if len(pts) < 3:
                continue

            # height: 75th percentile inside footprint + 4 m margin
            buf = np.asarray(poly.buffer(4.0).exterior.coords)
            gx = (buf[:, 0] - L) / hres
            gy = (T - buf[:, 1]) / hres
            x0, y0 = int(max(gx.min(), 0)), int(max(gy.min(), 0))
            x1, y1 = int(min(gx.max() + 1, hw)), int(min(gy.max() + 1, hh))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.c_[gx - x0, gy - y0].astype(np.int32)], 1)
            vals = hgrid[y0:y1, x0:x1][m > 0]
            vals = vals[np.isfinite(vals) & (vals > -50)]
            if vals.size == 0:
                skipped_noh += 1
                continue
            h = max(float(np.percentile(vals, 75)), MIN_H)
            heights.append(h)

            # measured colour from the context imagery
            px = (pts[:, 0] - L) / mpp_x
            py = (T - pts[:, 1]) / mpp_y
            bx0, by0 = int(max(px.min(), 0)), int(max(py.min(), 0))
            bx1, by1 = int(min(px.max() + 1, Wc)), int(min(py.max() + 1, Hc))
            col = np.array([0.6, 0.6, 0.6], np.float32)
            if bx1 - bx0 >= 1 and by1 - by0 >= 1:
                mm = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
                cv2.fillPoly(mm, [np.c_[px - bx0, py - by0].astype(np.int32)], 1)
                sel = img[by0:by1, bx0:bx1][mm > 0]
                if len(sel):
                    col = np.median(sel, 0).astype(np.float32) / 255.0
            col_lin = np.power(col, 2.2)

            # geometry in the core scene's world frame
            wx = pts[:, 0] - ox
            wz = pts[:, 1] - oy
            n = len(pts)
            # roof
            base = len(rv)
            for i in range(n):
                rv.append((wx[i], h, wz[i]))
                ruv.append(((pts[i, 0] - L) / (Rt - L), (pts[i, 1] - B) / (T - B)))
            try:
                tris = city_model._ear_clip(np.c_[wx, -wz].astype(np.float64))
            except Exception:
                tris = [(0, i, i + 1) for i in range(1, n - 1)]
            for a, b, c in tris:
                rf.append((base + a, base + b, base + c))
            # walls (flat-shaded, own vertices)
            for i in range(n):
                j = (i + 1) % n
                p0 = (wx[i], 0.0, wz[i]); p1 = (wx[j], 0.0, wz[j])
                p2 = (wx[j], h, wz[j]); p3 = (wx[i], h, wz[i])
                ex, ez = wx[j] - wx[i], wz[j] - wz[i]
                nl = math.hypot(ex, ez) or 1.0
                nrm = (-ez / nl, 0.0, ex / nl)   # outward for a clockwise ring
                k = len(wv)
                wv.extend([p0, p1, p2, p3]); wn.extend([nrm] * 4); wc.extend([col_lin] * 4)
                wf.extend([(k, k + 1, k + 2), (k, k + 2, k + 3)])
            kept += 1

    print(f"context buildings: {kept} extruded, {skipped_core} left to the core model, "
          f"{skipped_noh} skipped (no height data); median height "
          f"{np.median(heights):.1f} m, max {np.max(heights):.1f} m")

    rv = np.asarray(rv, np.float32); rf = np.asarray(rf, np.uint32); ruv = np.asarray(ruv, np.float32)
    wv = np.asarray(wv, np.float32); wn = np.asarray(wn, np.float32)
    wc = np.asarray(wc, np.float32); wf = np.asarray(wf, np.uint32)
    # Roof winding must face up: check one triangle, flip all if needed.
    if len(rf):
        a, b, c = rv[rf[0, 0]], rv[rf[0, 1]], rv[rf[0, 2]]
        if np.cross(b - a, c - a)[1] < 0:
            rf = rf[:, ::-1].copy()
    rn = np.tile(np.array([[0, 1, 0]], np.float32), (len(rv), 1))
    # Wall winding must match its outward normal.
    if len(wf):
        a, b, c = wv[wf[0, 0]], wv[wf[0, 1]], wv[wf[0, 2]]
        if np.dot(np.cross(b - a, c - a), wn[wf[0, 0]]) < 0:
            wf = wf[:, ::-1].copy()

    buf = _BufferBuilder()
    meshes, nodes = [], []
    for name, v, nrm, extra, faces in (("ctx_roofs", rv, rn, ("TEXCOORD_0", ruv, "VEC2"), rf),
                                       ("ctx_walls", wv, wn, ("COLOR_0", wc, "VEC3"), wf)):
        if not len(faces):
            continue
        attrs = {
            "POSITION": buf.add_accessor(v, FLOAT, "VEC3", ARRAY_BUFFER, include_bounds=True),
            "NORMAL": buf.add_accessor(nrm, FLOAT, "VEC3", ARRAY_BUFFER),
            extra[0]: buf.add_accessor(extra[1], FLOAT, extra[2], ARRAY_BUFFER),
        }
        idx = buf.add_accessor(faces.reshape(-1), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        meshes.append({"name": name, "primitives": [{"attributes": attrs, "indices": idx}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    body = buf.data()
    gltf = {"asset": {"version": "2.0", "generator": "DepthWizard context city"},
            "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
            "nodes": nodes, "meshes": meshes,
            "buffers": [{"byteLength": len(body)}],
            "bufferViews": buf.buffer_views, "accessors": buf.accessors}
    js = _pad4(json.dumps(gltf).encode("utf-8"), b" ")
    bb = _pad4(body)
    with open(out_glb, "wb") as f:
        f.write(struct_header(12 + 8 + len(js) + 8 + len(bb)))
        f.write(len(js).to_bytes(4, "little") + b"JSON" + js)
        f.write(len(bb).to_bytes(4, "little") + b"BIN\x00" + bb)
    stats = {"buildings": kept, "median_height_m": float(np.median(heights)),
             "max_height_m": float(np.max(heights)), "min_height_clamp_m": MIN_H,
             "footprints": "Microsoft Global ML Building Footprints",
             "heights": "Google Open Buildings 2.5D (75th pct in footprint + 4 m)"}
    json.dump(stats, open(out_glb.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print("wrote", out_glb)


def struct_header(total):
    return b"glTF" + (2).to_bytes(4, "little") + total.to_bytes(4, "little")


if __name__ == "__main__":
    main(*sys.argv[1:5])
