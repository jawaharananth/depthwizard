"""Measure the image's own relief displacement from its tall buildings.

    python scripts/measure_image_lean.py scenes/hitec_2x2 archive/hitec_city_2x2/source/hitec2x2.tif

A tall building's roof appears displaced from its base along the sensor's
look direction, by an amount proportional to height. The footprints in
buildings.geojson sit at the building BASE, so for every building of 25 m or
more this searches the 2-D shift that puts the footprint outline on the
strongest image edges (the photographed roof outline), then fits

    roof_shift(h) = offset + lean * h

by least squares. This is the displacement of THIS image. The alignment lean
estimated while registering Open Buildings footprints is not: it is the lean
between Google's imagery and this one, and pointed the wrong way here
(NE vs the measured NW on HITEC).

Writes roof_lean_m_per_m / roof_offset_m (world x east, z north) into
scene.json, which the viewer uses to sample roofs and walls from where the
image actually photographed them.
"""
import json
import os
import sys

import cv2
import numpy as np
import rasterio

MIN_H = float(os.environ.get("DW_LEAN_MIN_H", "25"))
SEARCH_PX = 32
MAX_TOWERS = 300
MAX_RESIDUAL_M = 5.0     # HITEC City fits at 3.5 m; hill towns came out at 10 m
MAX_OFFSET_M = 12.0
MAX_LEAN = 0.6


def outline_points(px):
    pts = []
    for a, b in zip(px[:-1], px[1:]):
        n = max(2, int(np.hypot(*(b - a))))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + (b - a) * t)
    return np.vstack(pts)


def main(scene_dir, tif):
    g = json.load(open(os.path.join(scene_dir, "buildings.geojson")))
    with rasterio.open(tif) as src:
        gcrs = ((g.get("crs") or {}).get("properties") or {}).get("name")
        if src.crs is None or not gcrs or src.crs != rasterio.crs.CRS.from_user_input(gcrs):
            sys.exit(f"footprints ({gcrs}) and image ({src.crs}) are not in the same CRS")
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        T = src.transform
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = cv2.GaussianBlur(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                                    cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (0, 0), 1.0)
    H, W = mag.shape
    inv = ~T
    S = SEARCH_PX
    # Every shift at once, exactly: gather the edge strength under every
    # outline point at every offset (points x offsets), with the image padded
    # so out-of-image points read as absent. Same scores, same tie-breaking as
    # shifting the points one offset at a time -- which, 4,225 offsets per
    # tower in a Python loop, took 48 s of an upload build.
    magp = cv2.copyMakeBorder(mag, S, S, S, S, cv2.BORDER_CONSTANT, value=0)
    valp = cv2.copyMakeBorder(np.ones_like(mag), S, S, S, S, cv2.BORDER_CONSTANT, value=0)
    dyy, dxx = np.meshgrid(np.arange(-S, S + 1), np.arange(-S, S + 1), indexing="ij")
    dyy, dxx = dyy.ravel(), dxx.ravel()               # dy-major, as the old loop ran
    rows = []
    # the tallest towers carry the most lean signal; 300 of them fit the line
    # as well as thousands, at a bounded cost on a dense full-resolution scene
    feats = sorted((f for f in g["features"] if (f["properties"].get("height_m") or 0) >= MIN_H
                    and f["geometry"]["type"] == "Polygon"),
                   key=lambda f: -(f["properties"].get("height_m") or 0))[:MAX_TOWERS]
    for f in feats:
        h = f["properties"].get("height_m") or 0
        px = np.array([inv * (x, y) for x, y in f["geometry"]["coordinates"][0]], np.float32)
        pts = outline_points(px)
        if len(pts) < 40:
            continue
        qx = np.round(pts[:, 0:1] + dxx[None, :]).astype(np.int64) + S     # points x offsets
        qy = np.round(pts[:, 1:2] + dyy[None, :]).astype(np.int64) + S
        inside = (qx >= 0) & (qx < magp.shape[1]) & (qy >= 0) & (qy < magp.shape[0])
        qx, qy = np.clip(qx, 0, magp.shape[1] - 1), np.clip(qy, 0, magp.shape[0] - 1)
        ok_pt = inside & (valp[qy, qx] > 0)                              # inside the real image
        n_ok = ok_pt.sum(0)
        ok = n_ok >= 0.9 * len(pts)                                      # old: skip if ok.mean() < 0.9
        if not ok.any():
            continue
        s = np.where(ok_pt, magp[qy, qx], 0.0).sum(0, dtype=np.float64) / np.maximum(n_ok, 1)
        s = np.where(ok, s, -1.0)
        k = int(np.argmax(s))                                            # first max, dy-major like before
        rows.append((h, int(dxx[k]), int(dyy[k])))
    if len(rows) < 20:
        sys.exit(f"only {len(rows)} buildings >= {MIN_H} m -- too few to measure a lean")
    h, dx, dy = np.array(rows).T
    A = np.c_[h, np.ones_like(h)]
    (lx, cx), (ly, cy) = (np.linalg.lstsq(A, dx, rcond=None)[0],
                          np.linalg.lstsq(A, dy, rcond=None)[0])
    res = np.hypot(dx - (lx * h + cx), dy - (ly * h + cy))
    # Fragmentary footprints lock onto unrelated edges; refit on the buildings
    # that agree with the first fit so they cannot pull the line.
    keep = res <= max(6.0, np.percentile(res, 60))
    h, dx, dy = h[keep], dx[keep], dy[keep]
    A = np.c_[h, np.ones_like(h)]
    (lx, cx), (ly, cy) = (np.linalg.lstsq(A, dx, rcond=None)[0],
                          np.linalg.lstsq(A, dy, rcond=None)[0])
    res = np.hypot(dx - (lx * h + cx), dy - (ly * h + cy))
    rows = rows if keep.all() else [r for r, k in zip(rows, keep) if k]
    gsd_x, gsd_y = abs(T.a), abs(T.e)
    # Pixel (x right, y down) -> world (x east, z north).
    lean_w = [float(lx * gsd_x), float(-ly * gsd_y)]
    off_w = [float(cx * gsd_x), float(-cy * gsd_y)]
    mag_mm = float(np.hypot(*lean_w))
    az = float(np.degrees(np.arctan2(lean_w[0], lean_w[1])) % 360)
    print(f"{len(rows)} buildings >= {MIN_H:.0f} m")
    print(f"roof shift px = ({lx:+.3f} h {cx:+.1f}, {ly:+.3f} h {cy:+.1f}); "
          f"median residual {np.median(res):.1f} px")
    print(f"lean {mag_mm:.3f} m/m toward azimuth {az:.0f} deg; offset {off_w} m")

    meta_p = os.path.join(scene_dir, "scene.json")
    meta = json.load(open(meta_p))
    # A fit this loose samples walls from the wrong pixels (smeared facades on
    # steep hill towns, where footprints sit on slopes and edges are clutter):
    # better no photographed facade than a wrong one.
    res_m = float(np.median(res) * gsd_x)
    if res_m > MAX_RESIDUAL_M or float(np.hypot(*off_w)) > MAX_OFFSET_M or mag_mm > MAX_LEAN:
        for k in ("roof_lean_m_per_m", "roof_offset_m"):
            meta.pop(k, None)
        meta["roof_lean_note"] = (f"not used: fit too loose (median residual {res_m:.1f} m, "
                                  f"offset {np.hypot(*off_w):.1f} m, lean {mag_mm:.2f} m/m); "
                                  f"walls take each building's measured colour")
        json.dump(meta, open(meta_p, "w"), indent=1)
        sys.exit(meta["roof_lean_note"])
    meta["roof_lean_m_per_m"] = lean_w
    meta["roof_offset_m"] = off_w
    meta["roof_lean_note"] = (
        f"measured from {len(rows)} buildings >= {MIN_H:.0f} m: roof outline = "
        f"footprint + offset + lean*height (lean {mag_mm:.2f} m/m toward "
        f"{az:.0f} deg, median residual {np.median(res) * gsd_x:.1f} m)")
    json.dump(meta, open(meta_p, "w"), indent=1)
    print("wrote", meta_p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
