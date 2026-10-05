"""
Elevated structures -- flyovers, metro viaducts, skywalks -- as real 3D decks.

None of these are buildings, so no building-footprint dataset contains them,
and a single top-down image draws them as flat road. Their GEOMETRY comes
from OpenStreetMap, which maps them explicitly (bridge=* on highway/railway
ways, with lanes/width tags).

Their HEIGHT is measured from the image itself: with the capture sun known
(fitted to building shadows, see sun_from_shadows), a deck at height h throws
its shadow h*cot(elevation) away along the anti-sun direction, so the h
whose predicted shadow band is darkest is the deck's height. Measured per
~120 m segment, then pooled per real structure (OSM name, else class) with
outlier rejection -- two tracks of one viaduct, or two carriageways of one
flyover, must agree. Segments the geometry cannot constrain (a deck running
along the sun direction casts its shadow onto itself) take their structure's
pooled height; a structure with no usable measurement at all falls back to
the reconstructed surface, then to a class-typical height -- each counted
separately in the report.

Piers are INFERRED: OSM does not map pier positions, so they are placed at
the class-typical span along the deck and disclosed as such.
"""
import json
import math
import urllib.parse
import urllib.request

import numpy as np

OVERPASS = "https://overpass-api.de/api/interpreter"

CLASS_SPEC = {
    # width_m None -> from lanes/width tags
    "metro": {"width_m": 5.0, "deck_m": 1.8, "h_range": (8.0, 18.0), "default_h": 12.0,
              "span_m": 28.0, "pier_m": 1.8},
    "road": {"width_m": None, "deck_m": 1.5, "h_range": (4.5, 12.0), "default_h": 7.0,
             "span_m": 30.0, "pier_m": 1.6},
    "foot": {"width_m": 3.0, "deck_m": 0.8, "h_range": (4.0, 9.0), "default_h": 6.0,
             "span_m": 18.0, "pier_m": 0.6},
}
FOOT = {"footway", "path", "steps", "pedestrian", "cycleway", "corridor"}
SEG_M = 120.0
STEP_M = 2.0


SOLID_UNDER = True   # fill under elevated decks to the ground instead of piers


def fetch_bridges(lon0, lat0, lon1, lat1, cache_path=None):
    if cache_path:
        try:
            return json.load(open(cache_path, encoding="utf-8"))["elements"]
        except (OSError, ValueError, KeyError):
            pass
    bb = f"{lat0},{lon0},{lat1},{lon1}"
    q = (f'[out:json][timeout:60];(way["bridge"]["highway"]({bb});'
         f'way["bridge"]["railway"]({bb}););out tags geom;')
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "DepthWizard-SIH26175/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read().decode("utf-8"))
    if cache_path:
        json.dump(d, open(cache_path, "w", encoding="utf-8"))
    return d["elements"]


def _classify(tags):
    if tags.get("bridge") in (None, "no"):
        return None
    if tags.get("railway"):
        return "metro"
    hw = tags.get("highway")
    if not hw:
        return None
    return "foot" if hw in FOOT else "road"


def _road_width(tags):
    try:
        return float(str(tags["width"]).split()[0])
    except (KeyError, ValueError):
        pass
    try:
        lanes = int(str(tags.get("lanes", "")).split(";")[0])
    except ValueError:
        lanes = 2 if tags.get("oneway") == "yes" else 4
    return lanes * 3.5 + 1.2


def _resample(pts, step):
    seg = np.diff(pts, axis=0)
    d = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    if d[-1] < step:
        return pts, d
    t = np.arange(0, d[-1] + 1e-9, step)
    if d[-1] - t[-1] > 0.3 * step:
        t = np.r_[t, d[-1]]
    return np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], t


def _shadow_height(deck_poly_px, h_range, sun_az, sun_el, lean_px_per_m, gsd, dark,
                   occupied):
    """
    Deck height from its own shadow. Returns h, or None when the geometry does
    not constrain it or the evidence is weak (peak at the search edge, or too
    little contrast over the typical reading).
    """
    import cv2
    H, W = dark.shape
    sa = np.radians(sun_az + 180.0)
    d = np.array([np.sin(sa), -np.cos(sa)]) / gsd           # px per metre, anti-sun
    cot = 1.0 / np.tan(np.radians(sun_el))
    lean = np.asarray(lean_px_per_m if lean_px_per_m is not None else (0.0, 0.0))
    lo, hi = h_range
    hs = np.arange(max(1.0, lo * 0.5), hi * 1.3 + 1e-6, 0.5)
    ext = np.vstack([deck_poly_px, deck_poly_px + d * hs[-1] * cot])
    x0, y0 = np.maximum(np.floor(ext.min(0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(ext.max(0)).astype(int) + 3, [W, H])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    off = np.array([x0, y0])
    deck_m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(deck_m, [(deck_poly_px - off).astype(np.int32)], 1)
    scores = []
    for h in hs:
        sh = np.zeros_like(deck_m)
        cv2.fillPoly(sh, [(deck_poly_px + d * h * cot - off).astype(np.int32)], 1)
        img_deck = np.zeros_like(deck_m)
        cv2.fillPoly(img_deck, [(deck_poly_px + lean * h - off).astype(np.int32)], 1)
        sel = (sh > 0) & (deck_m == 0) & (img_deck == 0) & (occupied[y0:y1, x0:x1] == 0)
        n = int(sel.sum())
        scores.append(dark[y0:y1, x0:x1][sel].mean() if n >= 30 else np.nan)
    scores = np.array(scores)
    if np.all(np.isnan(scores)):
        return None
    i = int(np.nanargmax(scores))
    if i == 0 or i == len(hs) - 1:
        return None                                          # peak at the search edge
    base = float(np.nanmedian(scores))
    if scores[i] / max(base, 1e-3) < 1.15:
        return None
    h = float(hs[i])
    return h if lo <= h <= hi else None


def build_structures(ways, lonlat_to_px, dsm, ground, gsd, image_np,
                     sun_az=None, sun_el=None, lean_px_per_m=None, occupied=None):
    """
    Returns {"deck": (v, n, uv, f), "body": (v, n, col, f), "report": {...}}
    in the ground mesh's world frame (x = px*gsd, z = -py*gsd, y up).
    """
    H, W = dsm.shape
    ndsm = np.maximum(dsm - ground, 0.0)
    lum = image_np.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = (lum < np.percentile(lum, 18)).astype(np.float32)
    if occupied is None:
        occupied = np.zeros((H, W), np.uint8)
    step_px = max(1.0, STEP_M / gsd)
    seg_n = max(2, int(SEG_M / STEP_M))

    # ---- pass 1: centrelines and per-segment shadow measurements ----------
    items = []
    for w in ways:
        tags = w.get("tags", {})
        cls = _classify(tags)
        geom = w.get("geometry")
        if cls is None or not geom or len(geom) < 2:
            continue
        spec = CLASS_SPEC[cls]
        width_m = spec["width_m"] or _road_width(tags)
        px = np.array([lonlat_to_px(g["lon"], g["lat"]) for g in geom], np.float64)
        inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        if inside.sum() < 2:
            continue
        pts, dist_px = _resample(px[inside], step=step_px)
        if len(pts) < 2:
            continue
        n = len(pts)
        tan = np.gradient(pts, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        nor = np.c_[-tan[:, 1], tan[:, 0]]
        half_px = width_m / 2 / gsd
        starts = list(range(0, max(1, n - 1), seg_n))
        seg_h = []
        if sun_az is not None and sun_el is not None:
            for s0 in starts:
                s1 = min(n, s0 + seg_n + 1)
                if s1 - s0 < 3:
                    seg_h.append(None)
                    continue
                P, N_ = pts[s0:s1], nor[s0:s1]
                poly = np.vstack([P + N_ * half_px, (P - N_ * half_px)[::-1]])
                seg_h.append(_shadow_height(poly, spec["h_range"], sun_az, sun_el,
                                            lean_px_per_m, gsd, dark, occupied))
        else:
            seg_h = [None] * len(starts)
        # reconstructed-surface reading, as a later fallback
        hs = []
        for p, q in zip(pts, nor):
            samp = [p + q * half_px * f for f in (-0.5, -0.25, 0, 0.25, 0.5)]
            hs.append(float(np.median([ndsm[int(np.clip(s[1], 0, H - 1)),
                                            int(np.clip(s[0], 0, W - 1))] for s in samp])))
        items.append({"tags": tags, "cls": cls, "spec": spec, "pts": pts, "tan": tan,
                      "nor": nor, "half_px": half_px, "dist_px": dist_px, "n": n,
                      "starts": starts, "seg_h": seg_h, "surface_h": float(np.median(hs)),
                      "group": tags.get("name") or cls})

    # ---- pass 2: pool per real structure, reject outliers ----------------
    groups = {}
    for it in items:
        groups.setdefault(it["group"], []).extend([x for x in it["seg_h"] if x is not None])
    group_h = {}
    for g_, vals in groups.items():
        if vals:
            med = float(np.median(vals))
            keep = [v for v in vals if abs(v - med) <= 0.3 * med]
            group_h[g_] = float(np.median(keep)) if keep else med
    rep = {"structures": 0, "length_m": 0.0, "height_shadow": 0, "height_group": 0,
           "height_surface": 0, "height_default": 0, "segments_shadow_measured": 0,
           "segments_total": 0, "piers": 0, "by_class": {},
           "group_heights_m": {k: round(v, 1) for k, v in group_h.items()}}

    dv, dn, duv, df = [], [], [], []
    bv, bn, bc, bf = [], [], [], []

    def world(px, py, y):
        return (px * gsd, y, -py * gsd)

    def box(cx, cy, half_px, y0, y1, col):
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ring = [(cx + a * half_px, cy + b * half_px) for a, b in corners]
        for i in range(4):
            j = (i + 1) % 4
            (x0, z0), (x1, z1) = ring[i], ring[j]
            ex, ez = (x1 - x0) * gsd, -(z1 - z0) * gsd
            nl = math.hypot(ex, ez) or 1.0
            nrm = (-ez / nl, 0.0, ex / nl)
            k = len(bv)
            bv.extend([world(x0, z0, y0), world(x1, z1, y0), world(x1, z1, y1), world(x0, z0, y1)])
            bn.extend([nrm] * 4); bc.extend([col] * 4)
            bf.extend([(k, k + 1, k + 2), (k, k + 2, k + 3)])

    # ---- pass 3: heights per station, then geometry ---------------------
    for it in items:
        spec, n, pts, tan, nor = it["spec"], it["n"], it["pts"], it["tan"], it["nor"]
        half_px, starts, seg_h = it["half_px"], it["starts"], it["seg_h"]
        lo, hi = spec["h_range"]
        gh = group_h.get(it["group"])
        clean = [None if (x is None or (gh and abs(x - gh) > 0.3 * gh)) else x for x in seg_h]
        measured = [x for x in clean if x is not None]
        rep["segments_total"] += len(seg_h)
        rep["segments_shadow_measured"] += len(measured)
        if measured and len(measured) >= 0.5 * len(seg_h):
            fill = gh if gh else float(np.median(measured))
            mids = np.array([min(n - 1, s0 + seg_n // 2) for s0 in starts], float)
            vals = np.array([x if x is not None else fill for x in clean], float)
            h = np.interp(np.arange(n), mids, vals)
            kk = max(1, int(20.0 / STEP_M))
            h = np.array([np.mean(h[max(0, i - kk):i + kk + 1]) for i in range(n)])
            rep["height_shadow"] += 1
        elif gh:
            h = np.full(n, gh)
            rep["height_group"] += 1
        elif lo <= it["surface_h"] <= hi:
            h = np.full(n, it["surface_h"])
            rep["height_surface"] += 1
        else:
            h = np.full(n, spec["default_h"])
            rep["height_default"] += 1

        g = np.array([ground[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]
                      for p in pts])
        top = g + h
        # solid under the deck down to the ground (reads as one built mass, no gaps or thin walls)
        bot = (g - 0.5) if SOLID_UNDER else (top - spec["deck_m"])
        cols = [image_np[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))] for p in pts]
        col = np.power(np.median(np.array(cols, np.float32), 0) / 255.0, 2.2)

        L = pts + nor * half_px
        Rr = pts - nor * half_px
        base = len(dv)
        for i in range(n):
            for P in (L[i], Rr[i]):
                dv.append(world(P[0], P[1], top[i]))
                dn.append((0.0, 1.0, 0.0))
                duv.append((P[0] / max(W - 1, 1), 1.0 - P[1] / max(H - 1, 1)))
        for i in range(n - 1):
            a, b, c, d = base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3
            df.extend([(a, c, b), (b, c, d)])
        for side, S in ((1, L), (-1, Rr)):
            for i in range(n - 1):
                p0, p1 = S[i], S[i + 1]
                ex, ez = (p1[0] - p0[0]) * gsd, -(p1[1] - p0[1]) * gsd
                nl = math.hypot(ex, ez) or 1.0
                nrm = (side * ez / nl, 0.0, -side * ex / nl)
                kk = len(bv)
                bv.extend([world(p0[0], p0[1], bot[i]), world(p1[0], p1[1], bot[i + 1]),
                           world(p1[0], p1[1], top[i + 1]), world(p0[0], p0[1], top[i])])
                bn.extend([nrm] * 4); bc.extend([col] * 4)
                bf.extend([(kk, kk + 1, kk + 2), (kk, kk + 2, kk + 3)])
        for i in range(n - 1):
            kk = len(bv)
            bv.extend([world(L[i][0], L[i][1], bot[i]), world(Rr[i][0], Rr[i][1], bot[i]),
                       world(Rr[i + 1][0], Rr[i + 1][1], bot[i + 1]),
                       world(L[i + 1][0], L[i + 1][1], bot[i + 1])])
            bn.extend([(0.0, -1.0, 0.0)] * 4); bc.extend([col * 0.6] * 4)
            bf.extend([(kk, kk + 1, kk + 2), (kk, kk + 2, kk + 3)])
        for i, sgn in ((0, -1), (n - 1, 1)):
            kk = len(bv)
            t = tan[i] * sgn
            nrm = (t[0], 0.0, -t[1])
            bv.extend([world(L[i][0], L[i][1], bot[i]), world(Rr[i][0], Rr[i][1], bot[i]),
                       world(Rr[i][0], Rr[i][1], top[i]), world(L[i][0], L[i][1], top[i])])
            bn.extend([nrm] * 4); bc.extend([col] * 4)
            bf.extend([(kk, kk + 1, kk + 2), (kk, kk + 2, kk + 3)])

        length_m = it["dist_px"][-1] * gsd if len(it["dist_px"]) else 0.0
        span = spec["span_m"]
        s = span / 2
        while not SOLID_UNDER and s < length_m - 6.0:
            i = int(np.clip(round(s / STEP_M), 0, n - 1))
            box(pts[i][0], pts[i][1], spec["pier_m"] / 2 / gsd, g[i] - 0.5, bot[i], col * 0.85)
            rep["piers"] += 1
            s += span

        rep["structures"] += 1
        rep["length_m"] += length_m
        rep["by_class"][it["cls"]] = rep["by_class"].get(it["cls"], 0) + 1

    def arr(a, dt=np.float32):
        return np.asarray(a, dt)

    def orient(v, nrm, f):
        # Every face was built with an intended outward normal; flip any
        # triangle whose winding disagrees, so nothing is back-face culled.
        if not len(f):
            return f
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        gn = np.cross(b - a, c - a)
        flip = (gn * nrm[f[:, 0]]).sum(1) < 0
        f = f.copy()
        f[flip] = f[flip][:, ::-1]
        return f

    deck_v, deck_n = arr(dv), arr(dn)
    body_v, body_n = arr(bv), arr(bn)
    deck = (deck_v, deck_n, arr(duv), orient(deck_v, deck_n, arr(df, np.int64)).astype(np.uint32))
    body = (body_v, body_n, arr(bc), orient(body_v, body_n, arr(bf, np.int64)).astype(np.uint32))
    rep["length_m"] = round(rep["length_m"], 1)
    # legacy key some callers print
    rep["height_measured"] = rep["height_surface"]
    return {"deck": deck, "body": body, "report": rep}
