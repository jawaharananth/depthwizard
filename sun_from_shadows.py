"""
Sun azimuth + elevation from an image's own shadows, given KNOWN building
heights and the image's measured lean (relief displacement).

A building of height h whose roof appears at polygon P has its base at
P - lean*h. Its ground shadow is the sweep of that base along the anti-sun
direction by h*cot(elevation). The sun direction that makes the predicted
shadow regions darkest (all roofs and bases excluded) is the capture's sun.

With the capture DATE known, the search is over pass time instead, so the
answer is always a sun position that can occur there that day. HITEC City,
Hyderabad (Esri/Vantor capture 2026-01-27): 04:34 UTC, az 133.6 / el 39.5,
shadow contrast x1.36 (free 2-D fit: az 128 / el 44, x1.39).
"""
import cv2
import numpy as np


def fit_sun(image_rgb: np.ndarray, polys: list, heights: list,
            lean_px_per_m, gsd_m: float, min_h: float = 15.0,
            min_buildings: int = 20, ephemeris=None):
    """Returns dict(azimuth_deg, elevation_deg, contrast, n, capture_time_utc)
    or None. ephemeris: optional (lat, lon, date) when the capture date is known."""
    H, W = image_rgb.shape[:2]
    lum = image_rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = (lum < np.percentile(lum, 18)).astype(np.float32)
    lean = np.asarray(lean_px_per_m, np.float32)

    occupied = np.zeros((H, W), np.uint8)
    tall = []
    for p, h in zip(polys, heights):
        ring = np.asarray(p, np.float32)
        base = ring - lean * float(h)
        cv2.fillPoly(occupied, [ring.astype(np.int32)], 1)
        cv2.fillPoly(occupied, [base.astype(np.int32)], 1)
        if h >= min_h:
            tall.append((base, float(h)))
    if len(tall) < min_buildings:
        return None
    # The tallest cast the longest shadows, which constrain elevation most;
    # capping keeps the grid search bounded on large scenes.
    tall = sorted(tall, key=lambda t: -t[1])[:300]
    free = occupied == 0

    def score(az, el):
        sa = np.radians(az + 180.0)                       # shadow points away from sun
        d = np.array([np.sin(sa), -np.cos(sa)]) / gsd_m   # image x = east, y = south
        cot = 1.0 / np.tan(np.radians(el))
        num = den = 0.0
        for base, h in tall:
            hull = cv2.convexHull(np.vstack([base, base + d * h * cot])
                                  .astype(np.float32)).reshape(-1, 2)
            x0, y0 = np.maximum(np.floor(hull.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(hull.max(0)).astype(int) + 1, [W, H])
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [(hull - [x0, y0]).astype(np.int32)], 1)
            sel = (m > 0) & free[y0:y1, x0:x1]
            n = int(sel.sum())
            if n >= 20:
                num += float(dark[y0:y1, x0:x1][sel].sum())
                den += n
        return num / den if den else 0.0

    when = None
    if ephemeris is not None:
        # Capture DATE known: azimuth and elevation are then not independent
        # -- both follow from the one unknown, the pass time -- so search time
        # and let the solar ephemeris supply a physically consistent pair.
        import datetime as _dt
        from shadow_correction import sun_position
        lat, lon, date = ephemeris
        day = _dt.datetime(date.year, date.month, date.day)
        def at(mins):
            el, az = sun_position(lat, lon, day + _dt.timedelta(minutes=mins))
            return az, el
        cands = [m for m in range(0, 24 * 60, 4) if at(m)[1] > 15]
        s, m0 = max((score(*at(m)), m) for m in cands)
        s, m = max((score(*at(t)), t) for t in [m0 + d / 3 for d in range(-12, 13)])
        a, e = at(m)
        when = day + _dt.timedelta(minutes=m)
    else:
        s, a0, e0 = max((score(a, e), a, e) for a in range(0, 360, 6) for e in range(20, 81, 4))
        s, a, e = max((score(a, e), a, e) for a in range(a0 - 6, a0 + 7)
                      for e in range(max(10, e0 - 4), min(86, e0 + 5)))
    ortho = score(a + 90, e)
    # Require a real contrast over the same elevation rotated 90 deg, else the
    # shadows did not constrain anything and the answer is noise.
    if s < ortho * 1.2:
        return None
    return {"azimuth_deg": float(a % 360), "elevation_deg": float(e),
            "contrast": round(s / max(ortho, 1e-6), 2), "n": len(tall),
            "capture_time_utc": when.strftime("%Y-%m-%d %H:%M") if when else None}
