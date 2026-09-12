"""
Build a prism city model from a PLAIN image -- no DFC2019 truth, no RPC.

  python build_city_image.py path/to/nadir.jpg --gsd 0.3 --name tajmahal
  python build_city_image.py path/to/nadir.jpg --anchor-height 73 --name tajmahal

WHAT THIS PATH GIVES UP, AND WHY IT MUST SAY SO

The DFC2019 path has an RPC camera model, LiDAR ground truth, and per-view sun
metadata. A plain JPEG has none of that, so four things change:

  * no orthorectification -- the image is assumed already north-up and roughly
    nadir. If it is oblique, every result is wrong, so obliqueness is estimated
    and the build refuses rather than producing confident nonsense.
  * no ground truth -- nothing validates the heights. No accuracy figure can be
    quoted for output from this path, ever.
  * no sun metadata -- azimuth is estimated from the image's own shadows.
    Elevation cannot be recovered from a single image without a known height,
    so shadow calibration is unavailable unless an anchor is supplied.
  * no CRS -- exports carry no georeferencing. Attaching one would be a lie.

SCALE

Two honest options:

  --gsd METRES_PER_PIXEL
      If you know the ground sampling distance, everything downstream is metric.

  --anchor-height METRES
      Scale from a known landmark height: the tallest structure in the frame is
      declared to be this tall. This is an EXTERNAL FACT the operator supplies,
      not a measurement, and it is recorded as such in the scene metadata. It is
      how you get sensible absolute numbers out of a photograph whose scale is
      otherwise unknown.

Neither invents anything: one is a stated input, the other is a stated
assumption, and the scene is labelled accordingly.
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import cv2
from PIL import Image

import segmentation as seg
import progress as prog
import height_cache
import dsm_refine
import dtm as dtm_mod
import city_model
import image_grade
import mesh_repair
import model_audit
import geometry_validate as gval
import float_check as fcheck
import region_footprints
import building_discovery as bd
import shadow_correction
import overlay_rejection
import dem_source as dem_mod
from depth_model import DepthBackbone, orientation_check, backbone_tag
from glb_export import export_glb
import mesh_generation as mg

OUT_DIR = "final_out"
VIEWER_DIR = "viewer/output"
GROUND_GRID = 700


MIN_EDGE_PROBES = 20   # below this the direction scan has too little signal to trust


def estimate_sun_azimuth(image_np: np.ndarray, seg_labels: np.ndarray) -> float:
    """
    Sun azimuth from the image's own shadows.

    Shadows fall away from the sun, so the direction in which dark pixels
    cluster around structures gives the anti-solar direction directly. Every
    direction is scored by how much shadow lies a fixed distance from building
    EDGE pixels; the peak is the shadow direction, and the sun is opposite it.

    PROBES FROM THE EDGE, NOT FROM ANY BUILDING PIXEL -- and only where the
    probe lands off the building. An earlier version sampled all building
    pixels indiscriminately. Segmentation on a single image with no LiDAR
    labels calls 47-58% of the frame "building" (measured: precision 0.343
    at recall 0.932 against a held-out benchmark), so most sampled pixels
    were building INTERIOR, and stepping a fixed small distance from an
    interior pixel in most directions lands on the SAME building, or a
    neighbouring one -- not on ground or shadow. The resulting azimuth was
    not a clean sign error but noise of varying size: measured against real
    IMD sun angles on 3 DFC2019 tiles, the error was -178 deg (essentially
    flipped) on one, +57 deg on another, and -2.9 deg (fine) on a third --
    inconsistent because it tracked incidental building layout per tile, not
    a fixed bug. See scripts/validate_azimuth.py for the full measurement.

    This is measurable from a single image. Sun ELEVATION is not -- it needs a
    known height somewhere in the frame -- which is why this returns azimuth
    only and the caller must supply scale another way.

    Returns None -- refusing rather than guessing -- when there is not enough
    real edge/shadow signal to trust the result, same as every other
    refuse-loud path in this file.
    """
    shadow = shadow_correction.detect_shadow_mask(image_np)
    b = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    if b.sum() < 200:
        return None

    # Direction-independent building boundary: a building pixel with at least
    # one non-building 4/8-neighbour. This does not yet know the sun
    # direction (that is what is being solved for), so it cannot use the
    # directional edge trick measure_shadow_lengths() uses once azimuth is
    # already known -- a plain morphological boundary is the right tool here.
    eroded = cv2.erode(b, np.ones((3, 3), np.uint8))
    edge = (b > 0) & (eroded == 0)
    ys, xs = np.nonzero(edge)
    if ys.size < MIN_EDGE_PROBES:
        return None
    if ys.size > 4000:
        sel = np.linspace(0, ys.size - 1, 4000).astype(int)
        ys, xs = ys[sel], xs[sel]
    H, W = shadow.shape
    step = max(6, int(min(H, W) * 0.01))

    best_dir, best_score = None, -1.0
    for a in range(0, 360, 10):
        r = np.radians(a)
        dx, dy = np.sin(r), -np.cos(r)
        py = np.clip(np.round(ys + dy * step).astype(int), 0, H - 1)
        px = np.clip(np.round(xs + dx * step).astype(int), 0, W - 1)
        # Reject any probe that landed back on a building -- on this class of
        # heavily-over-segmented scene, a probe on building tells you nothing
        # about which way the sun is; it is exactly the corrupting signal
        # this fix removes.
        off_building = b[py, px] == 0
        if int(off_building.sum()) < MIN_EDGE_PROBES:
            continue
        score = float(shadow[py[off_building], px[off_building]].mean())
        if score > best_score:
            best_score, best_dir = score, a
    if best_dir is None:
        return None
    # Sun is opposite the shadow direction.
    return (best_dir + 180.0) % 360.0


def estimate_sun_elevation(image_np: np.ndarray, seg_labels: np.ndarray,
                           height_unitless: np.ndarray, sun_azimuth_deg: float,
                           gsd_m: float, anchor_height_m: float = None,
                           capture_dt_utc=None, lat_deg: float = None,
                           lon_deg: float = None) -> dict:
    """
    Sun elevation for this single image, from whichever source is available.

    Unlike azimuth, elevation cannot be read off shadow DIRECTION alone --
    h = L * tan(elevation) needs a real metre height on one side of the
    equation, and a bare image has none. Two honest sources exist:

      1. capture time + location (from GeoTIFF tags/CRS): the sun's position
         is then an astronomical fact, computed the same way ortho.py does it
         for DFC2019 tiles.
      2. --anchor-height: the operator's declared height of the tallest
         structure, combined with that structure's OWN measured shadow
         length, gives one elevation estimate directly from this image.

    If neither is available, this returns None with a reason instead of
    guessing -- an assumed elevation would make every downstream "measured"
    height an assertion wearing a measurement's clothes.
    """
    if capture_dt_utc is not None and lat_deg is not None and lon_deg is not None:
        elev, az_check = shadow_correction.sun_position(lat_deg, lon_deg, capture_dt_utc)
        return {"elevation_deg": elev, "source": "geotiff_datetime+location",
                "detail": f"lat={lat_deg:.4f} lon={lon_deg:.4f} "
                          f"time={capture_dt_utc.isoformat()} "
                          f"(computed azimuth {az_check:.0f} deg vs shadow-estimated "
                          f"{sun_azimuth_deg:.0f} deg)"}

    if anchor_height_m is not None:
        runs = shadow_correction.measure_shadow_lengths(
            image_np, seg_labels, sun_azimuth_deg, gsd_m,
            height_unitless=height_unitless)
        # The declared anchor is "the tallest structure in the frame" (per this
        # script's own docstring), so the candidate is the building with the
        # largest measured extent in field units -- not just any shadow.
        candidates = [r for r in runs if r.get("units") is not None
                     and r["units"] > 1e-3 and r["n_rays"] >= 3]
        if not candidates:
            return {"elevation_deg": None, "source": "anchor_shadow",
                    "reason": "anchor height supplied but no building shadow "
                              "cleared the ray-count/units threshold to measure it"}
        tallest = max(candidates, key=lambda r: r["units"])
        shadow_len_m = tallest["shadow_len_m"]
        if shadow_len_m < 1.0:
            return {"elevation_deg": None, "source": "anchor_shadow",
                    "reason": f"tallest structure's shadow is only "
                              f"{shadow_len_m:.1f} m -- too short to give a "
                              "stable elevation"}
        elev = math.degrees(math.atan(anchor_height_m / shadow_len_m))
        return {"elevation_deg": elev, "source": "anchor_shadow",
                "detail": f"tallest structure declared {anchor_height_m:.1f} m, "
                          f"its own shadow measured {shadow_len_m:.1f} m "
                          f"({tallest['n_rays']} rays) -> elevation {elev:.1f} deg"}

    return {"elevation_deg": None, "source": None,
            "reason": "no sun elevation source available -- need a GeoTIFF "
                      "with capture time + CRS, or --anchor-height"}


# Thresholds for the obliqueness screen. The separation they sit in is wide --
# nadir orthos measured 144 and 388 holes, oblique frames 0 to 2 -- so these are
# not tuned to the edge of anything.
SKY_MIN_SPAN = 0.60      # fraction of image width a sky region must cover
SKY_MAX_HOLES = 5        # above this the region is a roof mosaic, not sky
HOLE_MIN_AREA_PX = 40    # ignore speckle when counting holes


def check_nadir(image_np: np.ndarray) -> dict:
    """
    Obliqueness screen: is there real sky in this frame?

    The previous version asked whether the top of the image was bright and
    smooth. That is true of sky and equally true of every flat commercial
    rooftop, so it rejected the pipeline's own orthorectified tiles -- JAX_167
    scored 0.43 against a 0.25 threshold, and a downtown ortho scored 0.38. A
    screen that refuses the primary input format is worse than no screen.

    Brightness cannot separate the two, and neither can shape: a hazy downtown
    ortho produces a bright region that touches the top edge and spans the full
    width, exactly like sky. What does separate them is TOPOLOGY.

    Sky is one simple region. A field of bright roofs is perforated by streets,
    shadows, courtyards and vehicles, so its mask is riddled with holes.
    Measured over the available images the gap is two orders of magnitude:

        nadir orthos      144 and 388 holes
        oblique / ground  0, 1, 1 and 2 holes

    So sky is claimed only when a wide region touching the top edge is also
    nearly solid. Images with no wide bright region at the top -- most nadir
    imagery -- never reach that test at all.
    """
    h, w = image_np.shape[:2]
    half = image_np[: h // 2]

    gray = cv2.cvtColor(half, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex = cv2.blur(gray ** 2, (9, 9)) - cv2.blur(gray, (9, 9)) ** 2
    bright_flat = (gray > 150) & (tex < 60)

    hsv = cv2.cvtColor(half, cv2.COLOR_RGB2HSV)
    blueish = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 135) & (hsv[:, :, 1] > 40))

    mask = (bright_flat | blueish).astype(np.uint8)
    # Close small gaps so an aerial, a bird or a thin mast does not split the
    # sky into two components and halve its measured span.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best_i, span = None, 0.0
    for i in range(1, n):
        comp = lab == i
        if not comp[0].any():          # sky must touch the top edge
            continue
        s_i = comp.any(axis=0).sum() / float(w)
        if s_i > span:
            span, best_i = s_i, i

    holes = 0
    if best_i is not None:
        comp = (lab == best_i).astype(np.uint8)
        x = stats[best_i, cv2.CC_STAT_LEFT]
        y = stats[best_i, cv2.CC_STAT_TOP]
        bw = stats[best_i, cv2.CC_STAT_WIDTH]
        bh = stats[best_i, cv2.CC_STAT_HEIGHT]
        sub = comp[y:y + bh, x:x + bw]
        inv = (1 - sub).astype(np.uint8)
        hn, hlab, hstats, _ = cv2.connectedComponentsWithStats(inv, 4)
        for j in range(1, hn):
            if hstats[j, cv2.CC_STAT_AREA] <= HOLE_MIN_AREA_PX:
                continue
            ys, xs = np.where(hlab == j)
            # A background region touching the bounding box edge is outside the
            # component, not a hole in it.
            if (ys.min() == 0 or xs.min() == 0
                    or ys.max() == bh - 1 or xs.max() == bw - 1):
                continue
            holes += 1

    is_sky = (span >= SKY_MIN_SPAN) and (holes <= SKY_MAX_HOLES)
    return {"sky_span": round(float(span), 3),
            "sky_holes": int(holes),
            "likely_nadir": not is_sky}


def build(image_path: str, name: str, gsd_m: float = None, tracker=None,
          anchor_height_m: float = None, max_px: int = 2560,
          stage: bool = True, use_dem: bool = True, dem_path: str = None,
          dem_source: str = "glo30", return_arrays: bool = False) -> dict:
    """
    return_arrays: additive, off by default -- every existing caller (server.py,
    the CLI) is unaffected. When True, meta["_ndsm"] carries the predicted
    object-height-above-ground array (numpy, not JSON-safe), for a caller that
    needs the raw surface rather than just the written files -- e.g. a
    benchmark scoring against a ground-truth nDSM on the same grid. Callers
    passing return_arrays=True must also pass stage=False: the normal staging
    path json.dumps(meta) for scene.json, which a numpy array cannot survive.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"city_{name}")
    t0 = time.time()
    # A null tracker when none was supplied, so the five instrumentation
    # points below are bare calls rather than five guarded blocks -- each
    # guard would be another place the timing could silently be skipped.
    _tr = tracker if tracker is not None else prog.NullTracker()
    _tr.enter("load")

    # A GeoTIFF carries its own ground sampling distance and CRS. Reading them
    # is strictly better than accepting a --gsd flag: the horizontal scale
    # becomes MEASURED rather than asserted, and the export can legitimately
    # carry a CRS. PIL cannot open multiband GeoTIFFs at all, so this also
    # avoids the failure that used to crash the primary input format.
    src_crs = None
    # Captured for the Tier A/B attempts below: a DEM anchor needs the scene's
    # own UTM origin + EPSG (only available when the source is genuinely
    # projected, not merely geographic), and a shadow-elevation calibration
    # needs a real capture time + lat/lon when no --anchor-height is given.
    src_transform = None
    src_epsg_code = None
    src_is_projected_m = False
    capture_dt_utc = None
    src_lat = src_lon = None
    if image_path.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.open(image_path) as _s:
                if _s.crs is not None and _s.transform is not None:
                    m = abs(_s.transform.a)
                    if _s.crs.is_geographic:
                        lat = _s.transform.f + _s.transform.e * _s.height / 2
                        m = m * 111320.0 * math.cos(math.radians(lat))
                    else:
                        src_transform = _s.transform
                        src_epsg_code = _s.crs.to_epsg()
                        src_is_projected_m = src_epsg_code is not None
                    native_gsd = m
                    src_crs = str(_s.crs)
                    print(f"      georeferenced input: {src_crs}, "
                          f"{native_gsd*100:.1f} cm/px measured from the file")
                    if gsd_m is None:
                        gsd_m = native_gsd
                    # Centre-of-scene lat/lon, for a sun-position calculation --
                    # cheap and correct enough over a single tile's extent
                    # (sun elevation does not vary meaningfully across a few
                    # km), whether the CRS is geographic or projected.
                    try:
                        cx = _s.transform.c + _s.transform.a * _s.width / 2
                        cy = _s.transform.f + _s.transform.e * _s.height / 2
                        if _s.crs.is_geographic:
                            src_lon, src_lat = cx, cy
                        else:
                            from rasterio.warp import transform as _rwt
                            _lon, _lat = _rwt(_s.crs, "EPSG:4326", [cx], [cy])
                            src_lon, src_lat = _lon[0], _lat[0]
                    except Exception:
                        src_lat = src_lon = None
                    # Capture time, when the file actually carries one. Absent
                    # on most plain exports -- this is the uncommon case, not
                    # the default path.
                    try:
                        import datetime as _dtm
                        _tags = _s.tags()
                        _dtstr = (_tags.get("TIFFTAG_DATETIME")
                                 or _tags.get("EXIF DateTimeOriginal")
                                 or _tags.get("ACQUISITION_DATE"))
                        if _dtstr:
                            capture_dt_utc = _dtm.datetime.strptime(
                                _dtstr.strip(), "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        capture_dt_utc = None
                arr = np.transpose(_s.read([1, 2, 3]), (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
        except Exception as e:
            print(f"      (not readable as GeoTIFF: {e}; falling back to PIL)")
            pil = Image.open(image_path).convert("RGB")
    else:
        pil = Image.open(image_path).convert("RGB")
    if max(pil.size) > max_px:
        _sc = max_px / max(pil.size)
        if gsd_m is not None:
            gsd_m = gsd_m / _sc     # fewer pixels over the same ground
        pil = pil.resize((int(pil.size[0] * _sc), int(pil.size[1] * _sc)), Image.LANCZOS)
    image_np = np.array(pil)
    H, W = image_np.shape[:2]
    print(f"[1/5] {os.path.basename(image_path)}  {W}x{H}")
    if tracker is not None:
        tracker.megapixels = (W * H) / 1e6
    if min(W, H) < 800:
        raise SystemExit(
            f"image is {W}x{H} -- too small to resolve buildings. This pipeline "
            "needs roughly 2000px of nadir imagery; below ~800px a building is a "
            "handful of pixels and every footprint is noise.")

    nad = check_nadir(image_np)
    if not nad["likely_nadir"]:
        raise SystemExit(
            f"this does not look like a nadir (top-down) view -- a sky region "
            f"spanning {nad['sky_span']*100:.0f}% of the frame width sits above "
            f"the horizon. "
            "Every stage of this pipeline assumes a top-down view: depth is read as "
            "height, footprints as plan geometry, shadow length as building height. "
            "On an oblique or ground-level photo the output would be confident "
            "nonsense, so the build stops here rather than producing it.")

    _tr.leave(); _tr.enter("height_field")
    # The backbone tag is part of the key: a cached field from a different
    # model is not a cache hit, it is a different measurement.
    key = f"plainimg_{name}_{backbone_tag()}"
    height = height_cache.load(key, "tiled", H) if H == W else None
    if height is None:
        height = DepthBackbone().predict_tiled(pil)
        if H == W:
            height_cache.save(key, "tiled", H, height)
    print(f"[2/5] height field in {time.time()-t0:.0f}s")
    _tr.leave(); _tr.enter("scale")

    # Strip map-overlay graphics before anything reads the image as terrain.
    # A pin or label is opaque paint: segmentation calls it a building, the prism
    # builder extrudes a block from it, and the result is a phantom structure
    # standing on nothing.
    seg_pre, _ = seg.segment(image_np)
    image_np, ov = overlay_rejection.clean(
        image_np, veg_mask=(seg_pre == seg.CLASS_IDX["vegetation"]))
    if ov["count"]:
        print(f"      map-overlay graphics removed: {ov['count']} components, "
              f"{ov['coverage']*100:.2f}% of pixels inpainted")
        if ov["coverage"] > 0.03:
            print("      WARNING: heavy overlay coverage -- this looks like a "
                  "screenshot with labels on. Turn map labels OFF at the source; "
                  "inpainting restores appearance, not the data the graphic hid.")

    seg_labels, _ = seg.segment(image_np, height=height)
    oc = orientation_check(height, seg_labels)
    if oc.get("checked") and not oc["correct_orientation"]:
        raise SystemExit("height field inverted -- refusing to build")
    refined = dsm_refine.refine_dsm(height, image_np)

    sun_az = estimate_sun_azimuth(image_np, seg_labels)

    # ---- scale ----------------------------------------------------------
    rel = np.maximum(refined - dtm_mod.estimate_dtm(refined, seg_labels), 0.0)
    b = seg_labels == seg.CLASS_IDX["building"]
    p99 = float(np.percentile(rel[b], 99)) if b.sum() > 1000 else float(np.percentile(rel, 99))

    # Horizontal pixel pitch. When the source was georeferenced this is measured
    # from the file, so footprint areas and perimeters are real square metres
    # even when the VERTICAL scale below turns out to stay relative.
    px_m = gsd_m
    gsd = gsd_m if gsd_m else 0.25

    # TIER B ATTEMPT: the same physical shadow calibration build_city.py runs
    # on its monocular-fallback path, not a separate approximation.
    #
    # calibrate_scale needs sun ELEVATION, which azimuth alone cannot give --
    # see estimate_sun_elevation's docstring. Without a real elevation source
    # this attempt is skipped outright rather than guessed, and shadow_calib
    # below records exactly why so scene.json shows a refusal, not silence.
    shadow_calib = {"attempted": False}
    scale = tier = scale_source = None
    if sun_az is not None:
        elev_info = estimate_sun_elevation(
            image_np, seg_labels, refined, sun_az, gsd,
            anchor_height_m=anchor_height_m,
            capture_dt_utc=capture_dt_utc, lat_deg=src_lat, lon_deg=src_lon)
        if elev_info.get("elevation_deg") is not None:
            shadow_calib["attempted"] = True
            shadow_calib["elevation_source"] = elev_info["source"]
            shadow_calib["elevation_deg"] = round(elev_info["elevation_deg"], 2)
            cal = shadow_correction.calibrate_scale(
                image_np, seg_labels, refined,
                elev_info["elevation_deg"], sun_az, gsd, gsd)
            shadow_calib["n"] = cal.get("n", 0)
            shadow_calib["scale_m_per_unit"] = cal.get("scale_m_per_unit")
            shadow_calib["spread_ratio"] = cal.get("spread_ratio")
            if cal.get("scale_m_per_unit") is not None and cal.get("n", 0) >= 10:
                scale = cal["scale_m_per_unit"]
                tier = "B (shadow-calibrated)"
                scale_source = (f"shadow calibration: {cal['n']} buildings, "
                                f"{scale:.1f} m/unit, elevation from "
                                f"{elev_info['source']} "
                                f"({shadow_calib['elevation_deg']:.1f} deg)")
                print(f"[3/5] TIER B: scale {scale:.1f} m/unit from "
                      f"{cal['n']} shadow measurements "
                      f"(elevation via {elev_info['source']})")
            else:
                shadow_calib["refusal_reason"] = cal.get(
                    "reason", f"only {cal.get('n', 0)} usable shadow "
                              "measurements, below the 10-building floor")
                print(f"[3/5] Tier B refused: {shadow_calib['refusal_reason']} "
                      f"-- staying on Tier C")
        else:
            shadow_calib["refusal_reason"] = elev_info.get("reason", "unknown")
            print(f"[3/5] Tier B not attempted: {shadow_calib['refusal_reason']}")
    else:
        shadow_calib["refusal_reason"] = "no sun azimuth (insufficient building/shadow pixels)"
        print(f"[3/5] Tier B not attempted: {shadow_calib['refusal_reason']}")

    # TIER C: unchanged fallback, reached whenever Tier B was not attempted or
    # was refused above. Never silently promoted -- `tier` is only set to B
    # in the branch above, on an explicit pass of the same n>=10 floor
    # build_city.py uses.
    if tier is None:
        if anchor_height_m is not None:
            scale = anchor_height_m / max(p99, 1e-6)
            tier = "C (relative vertical, scaled to an operator-supplied landmark height)"
            scale_source = f"anchor: tallest structure declared {anchor_height_m} m"
        elif gsd_m is not None:
            # With a known GSD the horizontal scale is metric, but the VERTICAL
            # scale still is not: relative depth carries no metric information.
            # Assume the tallest structure is a plausible height rather than
            # pretending the depth field is metric.
            scale = 40.0 / max(p99, 1e-6)
            tier = "C (horizontal metric from measured GSD; vertical relative/assumed)"
            scale_source = (f"GSD {gsd_m:.3f} m/px measured from the file -- horizontal "
                            f"distances are real; vertical assumes tallest ~40 m")
        else:
            scale = 40.0 / max(p99, 1e-6)
            tier = "C (relative, no scale information at all)"
            scale_source = "none -- pixel units, vertical assumed 40 m tallest"
        print(f"[3/5] scale: {scale_source}")

    print(f"      sun azimuth estimated from shadows: "
          f"{'%.0f deg' % sun_az if sun_az is not None else 'not determinable'}")
    _tr.leave(); _tr.enter("buildings")

    dsm = refined * scale
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    dsm, n_prism = dsm_refine.prismify_buildings(dsm, seg_labels, terrain)
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    ground = city_model.flatten_ground(terrain, seg_labels, smooth_m=35.0, gsd_m=gsd)

    # TIER A ATTEMPT: anchor the absolute datum to an external DEM.
    #
    # Same fit as build_city.py's DEM step -- a 30 m DEM fixes WHERE the
    # surface sits vertically, fitted over pixels this pipeline's own
    # segmentation calls bare ground, so no external ground truth leaks in.
    # Only attempted when the source file is a GeoTIFF with a genuinely
    # projected (metric) CRS: without a real UTM origin there is no grid to
    # sample the DEM onto, and reprojecting a bare JPEG's assumed geometry
    # would anchor a datum to coordinates that were never real.
    dem_info = None
    if (use_dem or dem_path) and src_is_projected_m:
        d = dem_mod.sample_grid(
            src_transform.c, src_transform.f, ground.shape[0], gsd,
            f"EPSG:{src_epsg_code}", dem_path=dem_path, dem_source=dem_source)
        if d["dem"] is None:
            print(f"      Tier A not usable ({d.get('error')}) -- staying on tier {tier[0]}")
        else:
            gmask = ((seg_labels == seg.CLASS_IDX["bare_earth"]) |
                    (seg_labels == seg.CLASS_IDX["road"]))
            # FIT AGAINST RAW TERRAIN, NOT THE FLATTENED GROUND.
            #
            # `ground` is flatten_ground(terrain, smooth_m=35), a 35 m
            # smoothing pass whose job is to give the RENDERED terrain mesh a
            # clean bare-earth surface. That smoothing is a display decision,
            # and fitting a datum to it measures the smoother as much as the
            # DEM: it suppresses exactly the local relief the offset's IQR
            # spread guard exists to detect, so a scene whose terrain does not
            # actually track the DEM can pass the guard on a surface that was
            # smoothed into agreement. `terrain` is the unsmoothed
            # estimate_dtm() output -- the surface the pipeline actually
            # believes the bare earth to be -- and is what the offset belongs
            # to. The offset is still APPLIED to all three surfaces below.
            fit = dem_mod.fit_offset(terrain, d["dem"], gmask)
            if fit["offset_m"] is None or not fit.get("spread_ok", False):
                print(f"      Tier A refused ({fit.get('reason', 'spread too wide')}) "
                      f"-- staying on tier {tier[0]}")
                dem_info = {"attempted": True, "accepted": False,
                            "reason": fit.get("reason", "spread too wide")}
            else:
                ground = ground + fit["offset_m"]
                dsm = dsm + fit["offset_m"]
                terrain = terrain + fit["offset_m"]
                prior_tier = tier
                tier = "A (DEM-anchored absolute elevation)"
                dem_info = {"attempted": True, "accepted": True,
                            "source": d["source"], "offset_m": round(fit["offset_m"], 2),
                            "iqr_m": round(fit["iqr_m"], 2), "n_px": fit["n"],
                            "coverage": round(d["coverage"], 3),
                            "prior_tier": prior_tier}
                print(f"      TIER A: datum anchored to {d['source']}, "
                      f"offset {fit['offset_m']:+.2f} m from {fit['n']:,} ground pixels "
                      f"(IQR {fit['iqr_m']:.2f} m); was {prior_tier}")
    elif use_dem or dem_path:
        dem_info = {"attempted": True, "accepted": False,
                     "reason": "source is not a GeoTIFF with a projected metric CRS "
                               "-- no UTM grid to anchor a DEM onto"}
        print(f"      Tier A not attempted: {dem_info['reason']}")

    ndsm = np.maximum(dsm - ground, 0.0)

    # Re-fit the vertical scale AFTER squaring off the roofs.
    #
    # The scale was chosen from the raw depth field, but prismify_buildings then
    # lifts each footprint to a single roof height, which changes the very
    # distribution the scale was fitted to. Leaving it produced a campus of
    # three-storey blocks with a median height of 35.9 m. Re-fitting against the
    # final surface makes the stated assumption ("tallest structure ~ TARGET_TALL_M")
    # actually true of the geometry that gets exported.
    #
    # TIER C ONLY. This re-fit exists to enforce an ASSUMED tallest height --
    # on Tier B, `scale` came from calibrate_scale() measuring real buildings
    # against their own shadows, and there is no "target tallest" to enforce.
    # Running this on a Tier B scene would silently overwrite a measured scale
    # with one fitted to an invented assumption.
    b_mask = seg_labels == seg.CLASS_IDX["building"]
    if tier.startswith("C") and b_mask.sum() > 1000:
        TARGET_TALL_M = 40.0 if anchor_height_m is None else anchor_height_m
        p99_now = float(np.percentile(ndsm[b_mask], 99))
        if p99_now > 1e-6:
            refit = TARGET_TALL_M / p99_now
            # Scale the WHOLE surface about a common datum, terrain included.
            # Applying the factor only to the above-ground component leaves the
            # terrain at the old scale: measured here as 234 m of relief across a
            # flat 619 m campus, i.e. the site became a mountain with correctly
            # sized buildings perched on it.
            datum = float(np.min(ground))
            ground = datum + (ground - datum) * refit
            dsm = datum + (dsm - datum) * refit
            ndsm = np.maximum(dsm - ground, 0.0)
            scale *= refit
            print(f"      vertical scale re-fitted after roof squaring "
                  f"(x{refit:.3f}); tallest now {TARGET_TALL_M:.0f} m by assumption")

    shadow_mask = shadow_correction.detect_shadow_mask(image_np)
    disc = bd.discover(image_np, seg_labels, ndsm, gsd,
                       sun_azimuth_deg=sun_az, shadow_mask=shadow_mask,
                       min_area_m2=6.0)
    print("[4/5] building discovery")
    print(bd.format_report(disc["report"]))
    _tr.leave(); _tr.enter("export")

    # Footprints come from the SAME extractor the benchmarked path uses.
    #
    # This path previously traced building_discovery's segmentation contours
    # directly. Segmentation merges adjacent structures, so a contour is often a
    # whole city block rather than a building: measured on one uploaded tile the
    # largest was 9403 m2, and the roof triangles it produced spanned 126 m
    # across a 400 m scene. Extruded, those are the enormous flat plates that
    # made the render unusable -- and no per-polygon repair helps, because the
    # outline is a faithful trace of the wrong thing.
    #
    # region_footprints splits regions by watershed before tracing, which is why
    # it is the path with measured footprint numbers behind it (IoU 0.553,
    # recall 0.745). One extractor, used everywhere.
    #
    # min_height_m is in ndsm's units, which on this Tier C path are not metres
    # -- the surface was rescaled so the tallest structure reads TARGET_TALL_M,
    # so a 2 m threshold means "2 units on that assumed scale".
    rres = region_footprints.extract(
        image_np, ndsm, gsd, seg_labels=seg_labels,
        min_area_m2=8.0, min_height_m=1.5)
    footprints = rres["polygons"]
    # Drop the flattest share of candidates, judged against this scene's own
    # height distribution. These are the car parks and bare ground the height
    # field reads as slightly raised -- the buildings-on-roads problem.
    footprints, _thr = region_footprints.drop_low_regions(footprints, ndsm)
    if _thr is not None:
        print(f"      flat-region gate: kept {len(footprints)} "
              f"(cut below height {_thr:.1f})")
    print(f"      footprints from image regions: {rres['report']['retained']} "
          f"of {rres['report']['regions_examined']} regions")
    _rej = "  ".join(f"{k} {v}" for k, v in rres["report"]["rejected"].items() if v)
    if _rej:
        print(f"      rejected: {_rej}")

    # Provenance now tracks the tier actually reached: a height is MEASURED
    # only on B/A, where a physical shadow or DEM calibration produced the
    # scale, and stays INFERRED on C, where it is an assumption.
    _prov = bd.MEASURED if tier[0] in ("A", "B") else bd.INFERRED
    for rec in disc["instances"]:
        rec["provenance"] = _prov

    # Build, localise defects, repair the offenders, re-check. The loop is
    # bounded: each pass must strictly reduce the defect count or the strategy
    # escalates, and the last strategy removes the building, so it terminates.
    _gsmall_pre, _ = mg._resize_for_mesh(ground, seg_labels, GROUND_GRID)
    _cell_pre = gsd * (ground.shape[0] / _gsmall_pre.shape[0])
    # Roof colours are sampled from the WHITE-BALANCED image, not the raw one.
    # Correcting the ground texture alone would leave every roof carrying the
    # atmospheric blue cast while the terrain under it reads neutral -- the two
    # would not look like the same scene.
    _balanced = image_grade.white_balance(image_np)
    _rep = mesh_repair.repair_build(
        footprints, dsm, ground, gsd, image_np=_balanced, min_height_m=1.5,
        ground_small=_gsmall_pre, cell_m=_cell_pre)
    bverts, bfaces, binfo = _rep["verts"], _rep["faces"], _rep["binfo"]
    footprints = _rep["footprints"]
    _rr = _rep["report"]
    if _rr["converged"]:
        print(f"      repair: converged in {_rr['iterations']} pass(es), "
              f"{_rr['final_built']} buildings, no defects remaining")
    else:
        print(f"      repair: {_rr['iterations']} pass(es), "
              f"{_rr['final_built']} buildings, "
              f"{_rr['unrepaired']} could not be repaired")
    heights = np.array([r["height_m"] for r in binfo["buildings"]]) \
        if binfo["buildings"] else np.zeros(1)
    print(f"      {len(binfo['buildings'])} prisms; heights median "
          f"{np.median(heights):.1f}  max {heights.max():.1f} "
          f"({'m' if anchor_height_m else 'relative units'})")

    cverts, cfaces, n_canopy = city_model.build_canopy(
        seg_labels, dsm, ground, gsd, gsd, min_area_px=120)
    wverts, wfaces, n_water = city_model.build_water(seg_labels, ground, gsd, gsd)
    vverts, vfaces, n_veh = city_model.detect_vehicles(image_np, seg_labels, ground, gsd)

    gsmall, _ = mg._resize_for_mesh(ground, seg_labels, GROUND_GRID)
    gscale = ground.shape[0] / gsmall.shape[0]
    gverts, guvs, gfaces = mg.build_ground_mesh(gsmall, gsd * gscale, gsd * gscale)

    # The same checks the benchmark path runs. This path had none, so malformed
    # geometry shipped silently -- which is how a scene full of crossing plates
    # reached a viewer without anything objecting.
    _gv = [
        gval.validate(gverts, gfaces, "ground", expect_upward=True),
        gval.validate(bverts, bfaces, "buildings"),
        gval.validate(cverts, cfaces, "canopy"),
        gval.validate(wverts, wfaces, "water"),
        gval.validate(vverts, vfaces, "vehicles"),
    ]
    print("      geometry validation:")
    if not gval.report(_gv):
        print("      WARNING: a mesh failed a hard geometry check (see above)")
    _fc = [
        fcheck.check(bverts, gsmall, gsd * gscale, "buildings"),
        fcheck.check(cverts, gsmall, gsd * gscale, "canopy"),
        fcheck.check(vverts, gsmall, gsd * gscale, "vehicles"),
        fcheck.check(wverts, gsmall, gsd * gscale, "water", flat=True),
    ]
    if not fcheck.report(_fc):
        print("      WARNING: a class floats above the rendered terrain")

    # Second system: does the finished model agree with the image it came from?
    # Everything above checks the model against itself.
    try:
        _audit = model_audit.audit(
            footprints, binfo, image_np, ndsm,
            seg_labels == seg.CLASS_IDX["building"],
            sun_azimuth_deg=sun_az)
        print(model_audit.report(_audit))
    except Exception as _e:
        _audit = {"error": str(_e)}
        print(f"      model-vs-image audit failed: {_e}")

    # White balance first, then grade. The previous version boosted saturation
    # and applied an explicit cool gain to imagery that was ALREADY blue from
    # atmospheric scattering, taking R-B from -27.6 to -48.3 and saturation from
    # 30.7 to 49.9 -- a grey-blue monochrome scene. See image_grade.py.
    graded = image_grade.grade(image_np)
    Image.fromarray(graded).save(stem + "_texture.png")
    import io
    tex = io.BytesIO(); Image.fromarray(graded).save(tex, format="PNG")

    export_glb(stem + ".glb", gverts, guvs, gfaces, bverts, bfaces,
               texture_bytes=tex.getvalue(), building_uvs=None,
               building_colors=binfo.get("colors"),
               extra_meshes=[
                   ("canopy", cverts, cfaces, (0.31, 0.44, 0.28, 1.0)),
                   ("water", wverts, wfaces, (0.18, 0.37, 0.53, 1.0)),
                   ("vehicles", vverts, vfaces, (0.65, 0.70, 0.75, 1.0)),
               ])
    # SPEC-LITERAL OUTPUT NAMING: rDSM vs DSM.
    #
    # The problem statement distinguishes a RELATIVE surface (rDSM) from an
    # absolute metric one (DSM), and names them separately. The tier system
    # already made that distinction functionally, but the filename did not carry
    # it -- so a relative product and an absolute one landed on disk with
    # identical names and nothing in the filename to tell them apart.
    #
    # The suffix now states which it is. A file that leaves this machine says
    # what it is without anyone having to open its metadata.
    import dsm_export as _dx
    # HEIGHT_IS_METRIC now tracks the tier actually reached (A/B = measured,
    # C = assumed) rather than only "was some scale hint supplied" -- a Tier C
    # scene built with --anchor-height still has an ASSUMED vertical scale if
    # Tier B was attempted and refused, and must not be tagged as metric.
    _metric = (src_crs is not None) and (tier[0] in ("A", "B"))
    _suffix = "_DSM.tif" if _metric else "_rDSM.tif"
    try:
        _dx.export_dsm_geotiff_affine(
            dsm, stem + _suffix,
            transform=None, crs=src_crs if _metric else None,
            tags={"TIER": tier, "HEIGHT_IS_METRIC": str(_metric),
                  "PRODUCT": "DSM" if _metric else "rDSM",
                  "SOURCE": os.path.basename(image_path),
                  "PIPELINE": "DepthWizard"})
        print(f"      {'DSM' if _metric else 'rDSM'} written: "
              f"{os.path.basename(stem)}{_suffix}"
              f"{'' if _metric else '  (relative heights, no CRS)'}")
    except Exception as _e:
        print(f"      DSM export failed: {_e}")

    # PER-BUILDING EXPORT (GeoJSON + CSV).
    #
    # This path previously wrote none -- only the DFC2019 path did -- so a
    # scene built from an uploaded image had per-building evidence computed
    # and then thrown away, reachable nowhere. The viewer's building readout
    # reads this GeoJSON, so without it an uploaded scene could not be
    # inspected at all.
    #
    # A CRS is written only when the source genuinely had one. For a bare
    # JPEG the ring stays in PIXEL coordinates with crs null and a note
    # saying so, rather than a projection invented to make the file look
    # like the georeferenced one.
    from calibration import conformal as _cf
    _cal = _cf.load_fitted()
    _metric_tier = tier[0] in ("A", "B")
    _iv_half, _iv_nominal = {}, (None if _cal is None else 1.0 - _cal["alpha"])
    if _cal is not None and _metric_tier:
        for rec, prism in zip(disc["instances"], binfo["buildings"]):
            _u = None
            if _cal.get("normalised"):
                _u = 1.0 / max(float(rec.get("confidence") or 0.5), 0.05)
            _lo, _hi, _hw = _cf.interval(prism["height_m"], _cal, uncertainty=_u)
            if _hw is not None:
                _iv_half[rec["id"]] = round(float(_hw), 2)
    if _cal is None:
        print("      conformal interval: no fitted quantile committed "
              "(run scripts/fit_conformal.py) -- intervals omitted")
    elif not _metric_tier:
        print(f"      conformal interval: omitted on tier {tier[0]} -- heights "
              "are on an assumed relative scale, a metre band would be meaningless")
    else:
        print(f"      conformal interval: {_cal['variant']}, "
              f"{_iv_nominal*100:.0f}% nominal, {len(_iv_half)} buildings banded")

    import json as _json
    import csv as _csv
    _feats = []
    for rec, prism in zip(disc["instances"], binfo["buildings"]):
        _cnt = rec["contour"].reshape(-1, 2)
        if src_transform is not None:
            _ring = [list(src_transform * (float(_px), float(_py))) for _px, _py in _cnt]
        else:
            _ring = [[float(_px), float(_py)] for _px, _py in _cnt]
        if _ring and _ring[0] != _ring[-1]:
            _ring.append(_ring[0])
        _feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_ring]},
            "properties": {
                "id": rec["id"],
                "height_m": round(prism["height_m"], 2),
                "interval_half_width_m": _iv_half.get(rec["id"]),
                "coverage_nominal": _iv_nominal if rec["id"] in _iv_half else None,
                "area_m2": rec["area_m2"],
                "perimeter_m": rec["perimeter_m"],
                "size_class": rec["size_class"],
                "confidence": rec["confidence"],
                "provenance": rec["provenance"],
                "evidence": rec["evidence"],
                "height_is_metric": _metric_tier,
            },
        })
    with open(stem + "_buildings.geojson", "w") as f:
        _json.dump({
            "type": "FeatureCollection",
            "crs": ({"type": "name", "properties": {"name": src_crs}}
                    if src_transform is not None and src_crs else None),
            "coordinates_note": (
                "map coordinates in the source CRS" if src_transform is not None
                else "PIXEL coordinates -- the source carried no georeferencing, "
                     "so no projection is claimed"),
            "features": _feats}, f)
    with open(stem + "_buildings.csv", "w", newline="") as f:
        _w = _csv.writer(f)
        _w.writerow(["id", "height_m", "interval_half_width_m", "coverage_nominal",
                     "area_m2", "perimeter_m", "size_class", "confidence",
                     "provenance", "height_is_metric",
                     "ev_height", "ev_edge", "ev_texture", "ev_shadow"])
        for rec, prism in zip(disc["instances"], binfo["buildings"]):
            _e = rec["evidence"]
            _w.writerow([rec["id"], round(prism["height_m"], 2),
                         _iv_half.get(rec["id"], ""),
                         _iv_nominal if rec["id"] in _iv_half else "",
                         rec["area_m2"], rec["perimeter_m"], rec["size_class"],
                         rec["confidence"], rec["provenance"], _metric_tier,
                         _e["height"], _e["edge"], _e["texture"], _e["shadow"]])
    print(f"      per-building export: {len(_feats)} buildings to GeoJSON + CSV"
          + ("" if src_transform is not None else "  (pixel coords, no CRS)"))

    print(f"[5/5] {len(gfaces)} ground + {len(bfaces)} building faces, "
          f"{os.path.getsize(stem + '.glb')/1e6:.1f} MB")
    _tr.leave()

    meta = {
        "source_image": os.path.abspath(image_path),
        "resolution": [W, H],
        "tier": tier,
        "scale_source": scale_source,
        "gsd_m": px_m,
        "sun_azimuth_deg": round(sun_az, 1) if sun_az is not None else None,
        "sun_elevation_deg": shadow_calib.get("elevation_deg"),
        "sun_note": ("elevation from " + shadow_calib["elevation_source"]
                    if shadow_calib.get("elevation_source")
                    else "elevation not recoverable from a single image without a "
                         "known height or capture time+location"),
        "crs": src_crs,
        "ground_truth": "NONE -- no LiDAR available for this image; no accuracy "
                        "figure can be quoted for this scene",
        "nadir_check": nad,
        "overlays_removed": ov["count"],
        "overlay_coverage": round(ov["coverage"], 5),
        "model": "prism city (flat roofs, vertical walls)",
        "buildings_extruded": len(binfo["buildings"]),
        "canopy_volumes": n_canopy, "water_bodies": n_water, "vehicles": n_veh,
        "provenance": _prov,
        "height_is_metric": tier[0] in ("A", "B"),
        "shadow_calibration": shadow_calib,
        "dem_anchor": dem_info,
        "conformal": (None if _cal is None else {
            "applied": bool(_iv_half),
            "variant": _cal["variant"],
            "coverage_nominal": _iv_nominal,
            "half_width_m": _cal["q"] if _cal["variant"] == "constant" else None,
            "fitted_on": _cal["fitted_on"]["tile"],
            "held_out_coverage": _cal.get("held_out_coverage"),
            "omitted_reason": (None if _metric_tier else
                               f"tier {tier[0]}: heights are relative, not metres"),
        }),
        "discovery": disc["report"],
        "build_seconds": round(time.time() - t0, 1),
    }
    if return_arrays:
        assert not stage, "return_arrays=True requires stage=False (meta is not JSON-safe with an array in it)"
        meta["_ndsm"] = ndsm
        # The internal metres-per-raw-depth-unit actually applied for this
        # build, whatever tier was reached (Tier B: calibrate_scale's own
        # output, untouched; Tier C: the assumed-tallest scale, possibly
        # re-fitted after roof squaring). A caller that needs to compare this
        # pipeline's own calibration against a truth-fit affine -- without
        # applying that fit to anything -- needs this number to convert
        # between "metres in the finished scene" and "units in the raw field"
        # on the same basis calibrate_scale itself uses.
        meta["_effective_scale_m_per_unit"] = float(scale)

    if stage:
        os.makedirs(VIEWER_DIR, exist_ok=True)
        for stale in ("terrain_ao.png", "terrain_normal.png", "terrain_roughness.png",
                      "terrain_metalness.png", "terrain_heatmap.png"):
            p = os.path.join(VIEWER_DIR, stale)
            if os.path.exists(p):
                os.remove(p)
        shutil.copy2(stem + ".glb", os.path.join(VIEWER_DIR, "terrain.glb"))
        shutil.copy2(stem + "_texture.png", os.path.join(VIEWER_DIR, "terrain_texture.png"))
        # This path used to DELETE buildings.geojson here, because it never
        # produced one and a previous scene's file would otherwise be read as
        # this scene's. It produces one now, so it is staged rather than
        # removed -- that is what makes the viewer's per-building readout work
        # on an uploaded image.
        shutil.copy2(stem + "_buildings.geojson",
                     os.path.join(VIEWER_DIR, "buildings.geojson"))
        with open(os.path.join(VIEWER_DIR, "scene.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"      staged to {VIEWER_DIR}  ({meta['build_seconds']}s)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--name", default="scene")
    ap.add_argument("--gsd", type=float, default=None, help="metres per pixel, if known")
    ap.add_argument("--anchor-height", type=float, default=None,
                    help="known height in metres of the tallest structure in frame")
    ap.add_argument("--px", type=int, default=2560)
    ap.add_argument("--no-stage", action="store_true")
    # DEM anchoring is ON by default. It needs no operator input and no
    # ground truth -- a global 30 m DEM is fetched for the scene's own
    # coordinates -- so the reason to make it opt-in no longer holds: a
    # georeferenced input should get an absolute datum unless someone says
    # otherwise. It still refuses itself (loudly) on a non-projected source
    # or a wide offset spread, so defaulting it on cannot silently promote a
    # scene that has not earned Tier A.
    ap.add_argument("--no-dem", dest="dem", action="store_false",
                    help="skip the Copernicus/SRTM datum anchor (Tier A); "
                         "the anchor is attempted by default on any GeoTIFF "
                         "with a projected metric CRS")
    ap.set_defaults(dem=True)
    ap.add_argument("--dem-source", choices=["glo30", "srtm"], default="glo30",
                    help="which global DEM to anchor against")
    ap.add_argument("--dem-path", default=None,
                    help="use a local DEM GeoTIFF instead of the global model")
    a = ap.parse_args()
    build(a.image, a.name, gsd_m=a.gsd, anchor_height_m=a.anchor_height,
          max_px=a.px, stage=not a.no_stage, use_dem=a.dem,
          dem_path=a.dem_path, dem_source=a.dem_source)
