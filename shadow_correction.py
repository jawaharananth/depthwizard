"""
Shadow-based height cross-validation.

Independent physical check on AI-predicted building heights: given real sun
elevation/azimuth (computed from capture timestamp + lat/lon, NOT guessed)
and a measured shadow length, height = shadow_length * tan(sun_elevation).
Compared against the AI/depth-based height for the same building footprint.

This is a sanity check + optional nudge, not a replacement for the depth
model -- shadows are unreliable for very short/tall objects, overlapping
shadows, and low sun angles, so results are reported with an explicit
per-building confidence, not blindly trusted.
"""
import math
import numpy as np
import cv2

import segmentation as seg


# ---------------------------------------------------------------------------
# Sun position (NOAA / Jean Meeus simplified solar position algorithm).
# No external astronomy dependency: implemented directly so it works fully
# offline and is easy to unit-test against known reference points.
# ---------------------------------------------------------------------------

def sun_position(lat_deg: float, lon_deg: float, dt_utc) -> tuple:
    """
    dt_utc: timezone-aware or naive UTC datetime.
    Returns (elevation_deg, azimuth_deg). Azimuth is degrees clockwise from
    north (0=N, 90=E, 180=S, 270=W), standard convention.
    """
    import datetime as _dt

    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    jd = _julian_day(dt_utc)
    jc = (jd - 2451545.0) / 36525.0

    geom_mean_long_sun = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    geom_mean_anom_sun = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccent_earth_orbit = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    mean_anom_rad = math.radians(geom_mean_anom_sun)
    sun_eq_of_ctr = (math.sin(mean_anom_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
                     + math.sin(2 * mean_anom_rad) * (0.019993 - 0.000101 * jc)
                     + math.sin(3 * mean_anom_rad) * 0.000289)

    sun_true_long = geom_mean_long_sun + sun_eq_of_ctr
    omega = 125.04 - 1934.136 * jc
    sun_app_long = sun_true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    mean_obliq_ecliptic = 23 + (26 + ((21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813)))) / 60) / 60
    obliq_corr = mean_obliq_ecliptic + 0.00256 * math.cos(math.radians(omega))

    sun_declin = math.degrees(math.asin(
        math.sin(math.radians(obliq_corr)) * math.sin(math.radians(sun_app_long))))

    var_y = math.tan(math.radians(obliq_corr / 2)) ** 2
    eq_of_time = 4 * math.degrees(
        var_y * math.sin(2 * math.radians(geom_mean_long_sun))
        - 2 * eccent_earth_orbit * math.sin(mean_anom_rad)
        + 4 * eccent_earth_orbit * var_y * math.sin(mean_anom_rad) * math.cos(2 * math.radians(geom_mean_long_sun))
        - 0.5 * var_y * var_y * math.sin(4 * math.radians(geom_mean_long_sun))
        - 1.25 * eccent_earth_orbit * eccent_earth_orbit * math.sin(2 * mean_anom_rad))

    time_utc_minutes = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60.0
    true_solar_time = (time_utc_minutes + eq_of_time + 4 * lon_deg) % 1440

    hour_angle = true_solar_time / 4 - 180 if true_solar_time >= 0 else true_solar_time / 4 + 180

    lat_rad = math.radians(lat_deg)
    decl_rad = math.radians(sun_declin)
    ha_rad = math.radians(hour_angle)

    zenith_rad = math.acos(
        math.sin(lat_rad) * math.sin(decl_rad) + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad))
    elevation_deg = 90 - math.degrees(zenith_rad)

    az_denom = (math.cos(lat_rad) * math.sin(zenith_rad))
    if abs(az_denom) < 1e-9:
        azimuth_deg = 180.0
    else:
        az_cos = (math.sin(lat_rad) * math.cos(zenith_rad) - math.sin(decl_rad)) / az_denom
        az_cos = max(-1.0, min(1.0, az_cos))
        azimuth_deg = math.degrees(math.acos(az_cos))
        if hour_angle > 0:
            azimuth_deg = 360 - azimuth_deg

    return elevation_deg, azimuth_deg


def _julian_day(dt_utc) -> float:
    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day + (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def extract_capture_metadata(tif_path: str) -> dict:
    """
    Best-effort capture timestamp + lat/lon from a GeoTIFF's own tags/bounds.
    Returns None fields when genuinely absent -- never fabricates a time.
    """
    import rasterio

    result = {"datetime_utc": None, "lat": None, "lon": None}
    with rasterio.open(tif_path) as src:
        tags = src.tags()
        for key in ("TIFFTAG_DATETIME", "ACQUISITIONDATETIME", "acquisitionDateTime", "DATETIME"):
            if key in tags:
                result["_raw_datetime_tag"] = tags[key]
                break

        if src.crs is not None:
            bounds = src.bounds
            result["lat"] = (bounds.top + bounds.bottom) / 2.0
            result["lon"] = (bounds.left + bounds.right) / 2.0

    raw = result.get("_raw_datetime_tag")
    if raw:
        import datetime as _dt
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                result["datetime_utc"] = _dt.datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue

    return result


# ---------------------------------------------------------------------------
# Shadow detection + per-building height cross-check
# ---------------------------------------------------------------------------

def detect_shadow_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Shadow as a LOCAL contrast phenomenon, not a global brightness level.

    Otsu was used here and is badly wrong on a real ortho: it splits the
    histogram near its middle, so on JAX_068 it labelled 53.8% of a sunlit
    scene "dark". Everything downstream then measured noise -- shadow runs came
    out around 2.2 m against a true median building height of 11.5 m, roughly
    seven times short, and the scale calibration built on those runs made 37 m
    buildings 127 m tall.

    A shadowed patch is dark RELATIVE TO ITS SURROUNDINGS, which holds
    regardless of overall exposure or how much of the scene is in shade. The
    local reference is averaged over a window much wider than a shadow so it
    stays sunlit.
    """
    val = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    local = cv2.GaussianBlur(val, (0, 0), sigmaX=max(image_np.shape[0], 64) / 32.0)
    shadow = val < 0.62 * local

    # Real shadows are contiguous. Closing bridges the cars, kerbs and bright
    # specks that break a run; opening drops isolated dark pixels that would
    # otherwise start a false one.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m.astype(bool)


def cross_validate_heights(image_np: np.ndarray, seg_labels: np.ndarray, dsm: np.ndarray,
                            sun_elevation_deg: float, sun_azimuth_deg: float,
                            gsd_x_m: float, gsd_y_m: float, min_building_px: int = 30) -> list:
    """
    For each building footprint: AI height = max(dsm) - median(local ground dsm)
    within the footprint. Shadow height = measured shadow run length (in the
    anti-sun direction) * tan(sun_elevation).

    Returns a list of per-building dicts; buildings with no adjacent shadow
    found are reported with shadow_height_m=None rather than a fake value.
    """
    if sun_elevation_deg <= 0:
        raise ValueError("Sun below horizon (elevation <= 0) -- no shadows possible, cannot cross-validate")

    shadow_mask = detect_shadow_mask(image_np)
    building_mask = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    n, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(building_mask, connectivity=8)

    gsd_avg = (gsd_x_m + gsd_y_m) / 2.0
    # Shadows point AWAY from the sun: direction = azimuth + 180.
    shadow_dir_rad = math.radians((sun_azimuth_deg + 180) % 360)
    dx, dy = math.sin(shadow_dir_rad), -math.cos(shadow_dir_rad)  # image row/col unit vector, north-up assumption

    results = []
    ground_ref = float(np.percentile(dsm, 10))  # approximate local ground level

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_building_px:
            continue

        comp = cc_labels == i
        ai_height_m = float(np.percentile(dsm[comp], 95) - ground_ref)

        cy, cx = centroids[i][1], centroids[i][0]
        # Probe far enough to contain the tallest shadow the scene can cast,
        # expressed in metres and converted to pixels. A fixed 60-pixel probe
        # was 60 m on a 1 m grid but only 7.5 m on the 0.125 m orthos this
        # pipeline now produces -- shorter than almost every real shadow, so
        # every tall building came back with no shadow found. The bound is the
        # shadow of a 250 m structure at this sun elevation, plus the walk
        # across the footprint itself.
        max_shadow_m = 250.0 / max(math.tan(math.radians(sun_elevation_deg)), 0.05)
        span_px = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        max_probe_px = int(max_shadow_m / max(gsd_avg, 1e-6)) + int(span_px)
        shadow_run_px = 0
        shadow_start_step = None
        for step in range(1, max_probe_px):
            py, px = int(cy + dy * step), int(cx + dx * step)
            if not (0 <= py < shadow_mask.shape[0] and 0 <= px < shadow_mask.shape[1]):
                break
            if comp[py, px]:
                continue  # still inside the building footprint itself, keep walking to its edge
            if shadow_mask[py, px]:
                if shadow_start_step is None:
                    shadow_start_step = step
                shadow_run_px = step - shadow_start_step + 1
            elif shadow_start_step is not None:
                break  # shadow ended

        record = {
            "building_id": i,
            "area_px": int(area),
            "ai_height_m": round(ai_height_m, 2),
            "shadow_height_m": None,
            "diff_m": None,
            "relative_error_pct": None,
            "confidence": "low",
        }

        if shadow_run_px >= 3:
            shadow_length_m = shadow_run_px * gsd_avg
            shadow_height_m = shadow_length_m * math.tan(math.radians(sun_elevation_deg))
            diff = ai_height_m - shadow_height_m
            rel_err = abs(diff) / shadow_height_m * 100 if shadow_height_m > 0.1 else None

            record["shadow_height_m"] = round(shadow_height_m, 2)
            record["diff_m"] = round(diff, 2)
            record["relative_error_pct"] = round(rel_err, 1) if rel_err is not None else None
            record["confidence"] = "high" if shadow_run_px >= 8 else "medium"

        results.append(record)

    return results


def measure_shadow_lengths(image_np: np.ndarray, seg_labels: np.ndarray,
                           sun_azimuth_deg: float, gsd_m: float,
                           height_unitless: np.ndarray = None,
                           min_building_px: int = 200,
                           gap_tolerance_px: int = 6) -> list:
    """
    Shadow run length per building, measured from its anti-sun edge.

    Replaces a single ray fired from each footprint's centroid, which failed
    badly once segmentation started returning whole city blocks rather than
    edge fragments: the ray leaves a large blob far from the edge that
    actually casts the shadow, stops at the first dark gap, and returns a few
    metres. Measured on JAX_068, the median "shadow" came back at 1.9 m for
    buildings 10-37 m tall, and the resulting scale estimates ranged over a
    factor of 60.

    Two changes make it robust:
      * cast MANY rays, one per boundary pixel on the shadowed side, and take
        the median run -- a shadow broken by a car or a bright roof corner no
        longer decides the answer for the whole building;
      * tolerate small gaps along a run, since real shadows are interrupted by
        kerbs, vehicles and sunlit gaps between wings.

    Returns one dict per building with the median run in metres.
    """
    shadow = detect_shadow_mask(image_np)
    building = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(building, connectivity=8)

    rad = math.radians((sun_azimuth_deg + 180) % 360)
    dx, dy = math.sin(rad), -math.cos(rad)

    h, w = shadow.shape
    # Boundary pixels: in the footprint but with a non-footprint neighbour in
    # the shadow direction, i.e. the edge the shadow actually falls from.
    shifted = np.roll(np.roll(building, int(round(dy * 2)), axis=0),
                      int(round(dx * 2)), axis=1)
    edge = (building > 0) & (shifted == 0)

    max_probe = int(400.0 / max(gsd_m, 1e-6))
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_building_px:
            continue
        ys, xs = np.nonzero(edge & (lab == i))
        if ys.size == 0:
            continue
        if ys.size > 60:                      # cap the work on huge blocks
            sel = np.linspace(0, ys.size - 1, 60).astype(int)
            ys, xs = ys[sel], xs[sel]

        runs = []
        for y0, x0 in zip(ys, xs):
            run, gap, best = 0, 0, 0
            for step in range(1, max_probe):
                py, px = int(y0 + dy * step), int(x0 + dx * step)
                if not (0 <= py < h and 0 <= px < w):
                    break
                if lab[py, px] == i:
                    continue                  # still crossing the footprint
                if shadow[py, px]:
                    run += 1 + gap
                    gap = 0
                    best = max(best, run)
                else:
                    gap += 1
                    if gap > gap_tolerance_px:
                        break
            if best > 0:
                runs.append(best)

        if len(runs) < 5:
            continue
        rec = {
            "building_id": int(i),
            "area_px": int(stats[i, cv2.CC_STAT_AREA]),
            "shadow_len_m": float(np.median(runs)) * gsd_m,
            "n_rays": len(runs),
            "units": None,
        }
        if height_unitless is not None:
            # Height of this same building in field units: its roof against the
            # ground just outside its own footprint. Both must come from the
            # same building or the ratio is meaningless.
            x0b = stats[i, cv2.CC_STAT_LEFT]
            y0b = stats[i, cv2.CC_STAT_TOP]
            x1b = x0b + stats[i, cv2.CC_STAT_WIDTH]
            y1b = y0b + stats[i, cv2.CC_STAT_HEIGHT]
            pad = 12
            ys0, ys1 = max(0, y0b - pad), min(h, y1b + pad)
            xs0, xs1 = max(0, x0b - pad), min(w, x1b + pad)
            sub_lab = lab[ys0:ys1, xs0:xs1]
            sub_h = height_unitless[ys0:ys1, xs0:xs1]
            inside = sub_lab == i
            outside = (sub_lab == 0)
            if inside.sum() > 20 and outside.sum() > 20:
                rec["units"] = float(np.percentile(sub_h[inside], 90) -
                                     np.percentile(sub_h[outside], 50))
        out.append(rec)
    return out


def calibrate_scale(image_np: np.ndarray, seg_labels: np.ndarray,
                    height_unitless: np.ndarray, sun_elevation_deg: float,
                    sun_azimuth_deg: float, gsd_x_m: float, gsd_y_m: float) -> dict:
    """
    Metres per unit of a relative height field, from shadow geometry.

    A monocular depth field is defined only up to an unknown scale, so turning
    it into a surface model needs one number: how many metres a unit of it is
    worth. Taking that number from a constant -- which this project did, at an
    invented 200 -- makes every height in the scene an assertion.

    Shadows supply it physically. For each building with a clean shadow run,
    h = L * tan(sun elevation) is a metric height measured from the image
    itself; dividing by that building's height in field units gives one
    estimate of the scale. The median over all such buildings is the estimate
    used, since a handful of shadows will always be broken by an adjacent
    structure or a dark roof and would drag a mean badly.

    Returns the scale, the sample count, and the spread -- callers should
    refuse to treat the output as metric when `n` is small.
    """
    runs = measure_shadow_lengths(image_np, seg_labels, sun_azimuth_deg,
                                  (gsd_x_m + gsd_y_m) / 2.0,
                                  height_unitless=height_unitless)
    tan_el = math.tan(math.radians(sun_elevation_deg))

    # RATIO OF MEDIANS, not median of ratios.
    #
    # Per-building ratios divide a solid numerator (a shadow length, measured
    # in metres) by a noisy denominator (that building's height in unitless
    # depth, which for a low building is close to zero). Dividing first lets a
    # single near-zero denominator produce a ratio of hundreds, and the median
    # of a distribution like that is unstable: on JAX_068 it reported spreads
    # of 143% and then 686% of the median, and scaled 37 m buildings to 127 m.
    #
    # Taking the median of each quantity first and dividing once keeps both
    # sides robust, and is the standard estimator for a ratio of two noisy
    # positive quantities.
    tan_el = math.tan(math.radians(sun_elevation_deg))
    heights_m, units_list = [], []
    for r in runs:
        units = r.get("units")
        shadow_h = r["shadow_len_m"] * tan_el
        if units is None or units <= 1e-3 or shadow_h < 3.0:
            continue
        heights_m.append(shadow_h)
        units_list.append(units)

    if len(heights_m) < 8:
        return {"scale_m_per_unit": None, "n": len(heights_m),
                "reason": f"only {len(heights_m)} buildings with a usable shadow"}

    hm = np.array(heights_m)
    um = np.array(units_list)
    med_h = float(np.median(hm))
    med_u = float(np.median(um))
    med = med_h / max(med_u, 1e-6)

    # Stability from the numerator's own spread: how consistent the measured
    # building heights are. Reported so a caller can see whether the scene
    # gave a clean answer or a marginal one.
    spread = float((np.percentile(hm, 75) - np.percentile(hm, 25)) / max(med_h, 1e-6))

    return {
        "scale_m_per_unit": med,
        "n": len(heights_m),
        "median_building_height_m": round(med_h, 1),
        "spread_ratio": spread,
    }


if __name__ == "__main__":
    import sys
    import datetime as _dt

    lat, lon = float(sys.argv[1]) if len(sys.argv) > 1 else 28.6139, float(sys.argv[2]) if len(sys.argv) > 2 else 77.2090
    dt_utc = _dt.datetime(2026, 6, 21, 6, 0, 0)  # summer solstice, ~11:30am IST
    elev, az = sun_position(lat, lon, dt_utc)
    print(f"Sun position at lat={lat}, lon={lon}, {dt_utc} UTC: elevation={elev:.2f}deg, azimuth={az:.2f}deg")
