"""
Register each footprint polygon onto its roof as it appears in THIS image.

Measured on HITEC City 2x2 (tools/register_footprints.py --report): the Open
Buildings / Microsoft polygons are traced on different imagery, so against
the Esri z18/z19 capture they sit ~6-8 px (3.6-4.8 m) EAST of the real roofs
even for 1-2 storey houses, and further off (towards SE) the taller a
building is (another capture's lean). The prism roof is textured at the
polygon, so an unregistered roof shows a strip of ground/shadow and misses a
strip of its own roof.

Per building, the shift s maximising the mean image-gradient magnitude along
the outline is searched (cv2.matchTemplate of the rasterised outline against
|grad I|). A robust scene model s(h) = o + l*h (Huber, on confident peaks)
is the prior; each building takes its best shift inside a small window
around the prior (radius 3 px + 0.12*h), and keeps its original position
when the registered outline is not clearly better (gain < 1.08) or the
outline is too small to register.

Heights for the prior come from a previous build's buildings.geojson
(matched by centroid, <= 4 px) -- only the prior needs them; unmatched
polygons use h = 6 m.

    python tools/register_footprints.py in_px.json out_px.json \
        --image hitec2x2.png --heights scenes/hitec_2x2/buildings.geojson \
        --affine-from scenes/hitec_2x2/scene.json
"""
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MIN_GAIN = 1.08


def _grad(path):
    im = np.array(Image.open(path).convert("L")).astype(np.float32)
    im = cv2.GaussianBlur(im, (3, 3), 0.7)
    return np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))


def _score_map(gm, p, R):
    H, W = gm.shape
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int) + 1
    if x0 - R < 0 or y0 - R < 0 or x1 + R >= W or y1 + R >= H:
        return None
    tpl = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.float32)
    cv2.polylines(tpl, [np.round(p - [x0, y0]).astype(np.int32)], True, 1.0, 1)
    if tpl.sum() < 12:
        return None
    tpl /= tpl.sum()
    win = gm[y0 - R:y1 + R + 1, x0 - R:x1 + R + 1]
    return cv2.matchTemplate(win, tpl, cv2.TM_CCORR)      # index [R+dy, R+dx]


def register(polys, gm, heights):
    n = len(polys)
    P = [np.asarray(p, float) for p in polys]
    # pass 1: free search -> prior
    free = []
    for p, h in zip(P, heights):
        R = int(min(60, 0.9 * h + 12))
        sc = _score_map(gm, p, R)
        if sc is None:
            free.append(None); continue
        iy, ix = np.unravel_index(np.argmax(sc), sc.shape)
        gain = sc[iy, ix] / max(sc[R, R], 1e-6)
        peak = sc[iy, ix] / max(np.median(sc), 1e-6)
        free.append((ix - R, iy - R, gain, peak))
    from sklearn.linear_model import HuberRegressor
    ok = [i for i, r in enumerate(free) if r and r[3] > 1.8 and r[2] > 1.15]
    hs = np.array([heights[i] for i in ok])[:, None]
    fx = HuberRegressor(epsilon=1.35).fit(hs, [free[i][0] for i in ok])
    fy = HuberRegressor(epsilon=1.35).fit(hs, [free[i][1] for i in ok])
    prior = ((fx.intercept_, fx.coef_[0]), (fy.intercept_, fy.coef_[0]))
    # pass 2: constrained search around the prior
    out, shifts, gains, resid = [], [], [], []
    rep = {"input": n, "registered": 0, "kept_original": 0, "unregistrable": 0}
    for i, (p, h) in enumerate(zip(P, heights)):
        px_ = prior[0][0] + prior[0][1] * h
        py_ = prior[1][0] + prior[1][1] * h
        r = 3.0 + 0.12 * h
        R = int(np.ceil(max(abs(px_), abs(py_)) + r + 1))
        sc = _score_map(gm, p, R)
        if sc is None:
            out.append(polys[i]); shifts.append((0, 0)); rep["unregistrable"] += 1
            continue
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        allowed = np.hypot(xx - px_, yy - py_) <= r
        s2 = np.where(allowed, sc, -np.inf)
        iy, ix = np.unravel_index(np.argmax(s2), s2.shape)
        gain = sc[iy, ix] / max(sc[R, R], 1e-6)
        if gain < MIN_GAIN:
            out.append(polys[i]); shifts.append((0, 0)); rep["kept_original"] += 1
            continue
        dx, dy = int(ix - R), int(iy - R)
        out.append([[float(x + dx), float(y + dy)] for x, y in p])
        shifts.append((dx, dy)); gains.append(gain)
        if free[i]:
            resid.append((free[i][0] - px_, free[i][1] - py_, h))
        rep["registered"] += 1
    rep["prior"] = {"offset_px": [round(prior[0][0], 2), round(prior[1][0], 2)],
                    "per_m_px": [round(prior[0][1], 4), round(prior[1][1], 4)]}
    rep["median_gain"] = round(float(np.median(gains)), 3) if gains else None
    sh = np.array(shifts, float)
    rep["median_abs_shift_px"] = round(float(np.median(np.hypot(sh[:, 0], sh[:, 1]))), 2)
    # free-search scatter about the prior: the precision of an outline
    # registration, which also bounds any displacement-based height reading
    rr = np.array(resid)
    if len(rr):
        conf = np.hypot(rr[:, 0], rr[:, 1]) < 10
        rep["free_search_mad_px"] = round(float(np.median(np.abs(rr[conf, :2]))), 2)
    return out, shifts, rep


def heights_from_geojson(polys, gj_path, affine):
    a, _, c, _, e, f = affine
    gj = json.load(open(gj_path))
    C, Hs = [], []
    for ft in gj["features"]:
        r = np.array(ft["geometry"]["coordinates"][0], float)
        C.append([((r[:, 0] - c) / a).mean(), ((r[:, 1] - f) / e).mean()])
        Hs.append(float(ft["properties"].get("height_m") or 6.0))
    C = np.array(C)
    from scipy.spatial import cKDTree
    t = cKDTree(C)
    out, hit = [], 0
    for p in polys:
        q = np.asarray(p, float)
        if len(q) > 1 and np.allclose(q[0], q[-1]):
            q = q[:-1]
        d, j = t.query(q.mean(0))
        if d <= 4:
            out.append(Hs[j]); hit += 1
        else:
            out.append(6.0)
    return out, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--image", required=True)
    ap.add_argument("--heights", required=True)
    ap.add_argument("--affine-from", required=True)
    a = ap.parse_args()
    d = json.load(open(a.src))
    polys = d["polygons"]
    meta = json.load(open(a.affine_from))
    hs, hit = heights_from_geojson(polys, a.heights, meta["affine_transform"])
    gm = _grad(a.image)
    out, shifts, rep = register(polys, gm, hs)
    rep["heights_matched"] = hit
    d2 = dict(d)
    d2["polygons"] = out
    d2["registration"] = dict(rep, method="outline-gradient template match, "
                              "constrained to robust s(h)=o+l*h prior",
                              shifts_px=[list(s) for s in shifts])
    json.dump(d2, open(a.dst, "w"))
    print(json.dumps({k: v for k, v in rep.items()}))


if __name__ == "__main__":
    sys.exit(main())
