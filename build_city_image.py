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
import rasterio.warp
from pyproj import Transformer
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
import open_buildings_source as obs
from depth_model import DepthBackbone, orientation_check, backbone_tag
from glb_export import export_glb
import mesh_generation as mg

OUT_DIR = "final_out"
VIEWER_DIR = "viewer/output"
GROUND_GRID = 700
# Relief (1st-99th percentile of the 30 m-smoothed terrain map) from which a
# scene's 3D ground follows the terrain instead of a flat plane.
RELIEF_MODE_M = 20.0
# Smallest side an image may have: 256 px at 0.6 m is ~150 m, a few blocks.
MIN_SIDE_PX = 256


MIN_EDGE_PROBES = 20   # below this the direction scan has too little signal to trust

# A footprint's real-height sample is only trusted if at least this share of
# its own pixels have real data -- otherwise a building that only clips the
# corner of real coverage gets a height guessed from a handful of pixels,
# which is worse than not fusing at all.
OPEN_BUILDINGS_MIN_FOOTPRINT_COVERAGE = 0.3
# Below this scene-wide coverage, fetching and warping a whole raster for a
# handful of usable pixels isn't worth it -- report the real "no/low
# coverage" outcome and move on rather than pretend the attempt helped.
OPEN_BUILDINGS_MIN_SCENE_COVERAGE = 0.05
# The metric depth model's output stops growing near this height (measured on
# HITEC City: 99.9th percentile 11.4 m over 50-77 m towers), so above it the
# depth estimate says "tall", not "how tall".
DEPTH_SATURATION_M = 8.0


def _poly_crop_mask(poly, H: int, W: int):
    """(window, mask) for a polygon rasterised over its image-clipped bounding
    box, or None when that box is empty. mask equals the full-image fillPoly
    mask restricted to `window`, which holds every pixel it could set."""
    pts = np.asarray(poly, np.int32)
    flat = pts.reshape(-1, 2)
    x0, y0 = max(int(flat[:, 0].min()), 0), max(int(flat[:, 1].min()), 0)
    x1, y1 = min(int(flat[:, 0].max()) + 1, W), min(int(flat[:, 1].max()) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [pts - np.array([x0, y0], np.int32)], 1)
    return (slice(y0, y1), slice(x0, x1)), mask.astype(bool)


def _analysis_stretch(img: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
    """Per-band 2-98th percentile stretch over the valid pixels (uint8 RGB)."""
    out = np.empty_like(img)
    for b in range(img.shape[2]):
        band = img[..., b].astype(np.float32)
        v = band[valid] if (valid is not None and valid.shape == band.shape and valid.any()) else band
        lo, hi = np.percentile(v[::7] if v.size > 4_000_000 else v, [2, 98])
        out[..., b] = np.clip((band - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return out


def _mask_shift(a: np.ndarray, b: np.ndarray, max_shift: int, fa=None):
    """(dx, dy, peak): integer shift that best aligns mask b onto mask a.
    fa: a's transform, when the caller aligns several b against one a."""
    import scipy.fft as _sfft                      # same double-precision FFT, all cores
    if fa is None:
        fa = _sfft.rfft2(a - a.mean(), workers=-1)
    fb = _sfft.rfft2(b - b.mean(), workers=-1)
    xc = np.fft.fftshift(_sfft.irfft2(fa * np.conj(fb), s=a.shape, workers=-1))
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = xc[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return ix - max_shift, iy - max_shift, float(win.max() / (np.abs(xc).mean() + 1e-9))


def estimate_ob_alignment(ob_grid: np.ndarray, footprints: list, gsd_m: float):
    """
    Per-height pixel shift between where a building's ROOF appears in this
    image and where its height sits in the Open Buildings raster:

        shift(h) = offset + lean * h        (pixels)

    Two separate real effects, both measured on HITEC City, Hyderabad:
      * lean -- the image is oblique, so a roof is displaced from its base
        in proportion to height (~0.5 px north per metre there), while the
        Open Buildings raster (from near-nadir Sentinel-2) puts height over
        the base. Polygons traced on the image sit at the roof.
      * offset -- plain co-registration error between the two sources
        (~18 px / 11 m east there), independent of height.
    Without both, a tower's height lands on the sheds at its foot and the
    tower's own roof samples the empty ground north of it.

    Fit by cross-correlating each Open Buildings height band against the
    rasterised roof polygons. Returns None when the fit is not trustworthy.
    """
    H, W = ob_grid.shape
    roofs = np.zeros((H, W), np.float32)
    min_px = 60.0 / max(gsd_m * gsd_m, 1e-6)
    for p in footprints:
        a = np.asarray(p, np.int32)
        if cv2.contourArea(a) >= min_px:
            cv2.fillPoly(roofs, [a], 1.0)
    if roofs.sum() < 2000:
        return None
    max_shift = int(min(160, 60.0 / max(gsd_m, 1e-6)))
    hs, dxs, dys, ws = [], [], [], []
    import scipy.fft as _sfft
    _fa = _sfft.rfft2(roofs - roofs.mean(), workers=-1)      # the roofs, transformed once
    for lo, hi in [(8, 16), (16, 24), (24, 34), (34, 46), (46, 60), (60, 120)]:
        band = ((ob_grid >= lo) & (ob_grid < hi)).astype(np.float32)
        if band.sum() < 1500:
            continue
        dx, dy, peak = _mask_shift(roofs, band, max_shift, fa=_fa)
        hs.append(0.5 * (lo + hi)); dxs.append(dx); dys.append(dy); ws.append(peak)
    if len(hs) < 3:
        return None
    A = np.stack([np.ones(len(hs)), np.array(hs)], 1)
    Wt = np.diag(ws)
    cx = np.linalg.lstsq(Wt @ A, Wt @ np.array(dxs, float), rcond=None)[0]
    cy = np.linalg.lstsq(Wt @ A, Wt @ np.array(dys, float), rcond=None)[0]
    res = np.hypot(np.array(dxs) - A @ cx, np.array(dys) - A @ cy)
    lean = float(np.hypot(cx[1], cy[1]))
    if np.median(res) > 10 or lean > 2.0:
        return None
    return {"offset_px": (float(cx[0]), float(cy[0])),
            "lean_px_per_m": (float(cx[1]), float(cy[1])),
            "residual_px": float(np.median(res)), "bands": len(hs)}


def fuse_open_buildings_heights(dsm: np.ndarray, ground: np.ndarray,
                                footprints: list, src_transform, src_crs: str,
                                info: dict = None) -> int:
    """
    Burns real Google Open Buildings 2.5D per-building heights into the
    surface model -- the same "splice the real signal into dsm before
    anything downstream reads it" technique already used for Mumbai's real
    municipal floor-count records (see build()'s ground-truth fusion,
    search "REAL BUG FIX"). That fix only ever helps a scene with a
    per-city record on hand; this is the same fix generalised to ANY scene
    with a real georeference, sourced from a public dataset instead.

    Returns the number of footprints whose height was fused in. Zero
    coverage (no tile here, or a tile with no data at this location) is a
    real, honestly reported outcome -- see open_buildings_source.py's own
    module docstring -- not a failure to retry or hide.
    """
    if src_transform is None or src_crs is None:
        return 0

    H, W = dsm.shape
    r, epsg = ob_fetch(dsm.shape, src_transform, src_crs)
    print(f"      Open Buildings height fusion: {r['note']}")
    if r["heights"] is None or r["coverage"] < OPEN_BUILDINGS_MIN_SCENE_COVERAGE:
        return 0
    return _fuse_ob_grid(dsm, ground, footprints, src_transform, src_crs, info, r, epsg)


def ob_fetch(shape, src_transform, src_crs):
    """The Open Buildings heights for a scene's extent (fetched once: see
    open_buildings_source.sample_height_grid), and the UTM zone they are in."""
    H, W = shape
    corners_px = [(0, 0), (W, 0), (0, H), (W, H)]
    corners_crs = [src_transform * p for p in corners_px]
    xs = [c[0] for c in corners_crs]
    ys = [c[1] for c in corners_crs]
    tf_to_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lons, lats = tf_to_ll.transform(xs, ys)

    # Open Buildings tiles are indexed by their OWN real UTM zone, which has
    # nothing to do with whatever CRS the source image happens to carry.
    # A prior version of this function extracted the EPSG digits straight
    # out of src_crs and used those -- silently correct only by coincidence
    # whenever the source already happened to be in the right UTM zone
    # (true for every scene built so far), and a real, confirmed false
    # negative otherwise: a Web Mercator (EPSG:3857) source for Varanasi
    # reported "no tile covers this extent" even though the real UTM-44N
    # tile for that exact location was independently confirmed to exist.
    # The scene's own centre longitude/latitude determines the real UTM
    # zone regardless of the source's CRS.
    _clon = (min(lons) + max(lons)) / 2
    _clat = (min(lats) + max(lats)) / 2
    _zone = int((_clon + 180) // 6) + 1
    epsg = (32600 if _clat >= 0 else 32700) + _zone

    return obs.sample_height_grid(min(lons), min(lats), max(lons), max(lats), epsg=epsg), epsg


def _fuse_ob_grid(dsm, ground, footprints, src_transform, src_crs, info, r, epsg) -> int:
    H, W = dsm.shape
    # Warp the fetched grid onto THIS image's own pixel grid. Both
    # transforms are real Affines in a real CRS -- a genuine reprojection,
    # not a resize or a guess at alignment.
    dst = np.full((H, W), np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=r["heights"].astype(np.float32), destination=dst,
        src_transform=r["transform"], src_crs=f"EPSG:{epsg}",
        dst_transform=src_transform, dst_crs=src_crs,
        resampling=rasterio.warp.Resampling.bilinear)

    _gsd = abs(float(src_transform.a))
    align = estimate_ob_alignment(np.nan_to_num(dst, nan=0.0), footprints, _gsd)
    if info is not None:
        info["alignment"] = align
    if align is not None:
        print(f"      Open Buildings alignment: offset "
              f"({align['offset_px'][0]:+.1f}, {align['offset_px'][1]:+.1f}) px, lean "
              f"({align['lean_px_per_m'][0]:+.3f}, {align['lean_px_per_m'][1]:+.3f}) px/m "
              f"(fit residual {align['residual_px']:.1f} px over {align['bands']} bands)")
    h_max = float(np.nanmax(dst)) if np.isfinite(dst).any() else 0.0
    cand = np.arange(0.0, h_max + 1.0, 1.0)

    n_fused = n_rejected = 0
    for poly in footprints:
        crop = _poly_crop_mask(poly, H, W)
        if crop is None:
            continue
        win, mask = crop
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        ys += win[0].start
        xs += win[1].start
        if align is None:
            vals = dst[win][mask]
            vals = vals[np.isfinite(vals)]
            if vals.size < max(4, int(OPEN_BUILDINGS_MIN_FOOTPRINT_COVERAGE * mask.sum())):
                continue
            real_h = float(np.median(vals))
        else:
            # The height must explain its own sampling location: sample the
            # raster at where a building of height h, with this roof, would
            # have its base, and keep the tallest h that reproduces itself.
            # A shed at a tower's foot has no such h (at its own height the
            # base sample is the tower; at the tower's height it is empty
            # ground), so it is left to the depth model rather than
            # inheriting a neighbour's 70 m.
            (ox, oy), (vx, vy) = align["offset_px"], align["lean_px_per_m"]
            min_ok = max(4, int(OPEN_BUILDINGS_MIN_FOOTPRINT_COVERAGE * ys.size))
            # Every candidate height at once: neighbouring heights round to
            # the same pixel shift, so each distinct shift's sample median is
            # computed once, for all shifts in one array (same rounding --
            # round-half-even -- and same candidate order as a loop).
            sxy = np.stack([np.rint(ox + vx * cand), np.rint(oy + vy * cand)], 1).astype(np.int64)
            uniq, inv = np.unique(sxy, axis=0, return_inverse=True)
            yy = ys[None, :] - uniq[:, 1:2]
            xx = xs[None, :] - uniq[:, 0:1]
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            vals = np.where(ok, dst[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)], np.nan)
            n_fin = np.isfinite(vals).sum(1)
            use = (ok.sum(1) >= min_ok) & (n_fin >= 4)
            med = np.full(len(uniq), np.nan)
            if use.any():
                med[use] = np.nanmedian(vals[use], axis=1)
            f_all = med[np.asarray(inv).ravel()]
            keep = np.isfinite(f_all) & (np.abs(f_all - cand) <= np.maximum(2.0, 0.12 * cand))
            consistent = [float(v) for v in f_all[keep]]
            real_h = None
            if consistent:
                # Tie-break with the depth model's own metric height, which is
                # reliable for low structures but saturates near 10 m: below
                # that, a coincidental match with a neighbour's base must not
                # turn a house into a tower.
                d = float(np.median((dsm[win] - ground[win])[mask]))
                real_h = (max(consistent) if d >= DEPTH_SATURATION_M
                          else min(consistent, key=lambda c: abs(c - d)))
            if real_h is None:
                n_rejected += 1
                continue
        if real_h < 1.0:
            continue
        local_ground = float(np.median(ground[win][mask]))
        dsm[win][mask] = local_ground + real_h
        n_fused += 1
    if align is not None and n_rejected:
        print(f"      {n_rejected} footprint(s) had no self-consistent Open Buildings "
              f"height (e.g. a shed at a tower's foot) -- depth-model height kept")
    return n_fused


def calibrate_to_reference_heights(dsm: np.ndarray, ground: np.ndarray,
                                   pre_rel: np.ndarray, footprints: list,
                                   roof_dsm: np.ndarray = None,
                                   min_controls: int = 20,
                                   relative: bool = True):
    """
    SCALE CALIBRATION FROM REAL REFERENCE HEIGHTS. Every footprint whose
    height was replaced by a real one (Open Buildings fusion) is a ground
    control: its real height against the depth field's own (relative)
    reading there (`pre_rel` = dsm - ground captured BEFORE fusion). A robust
    fit of that ratio (median log-ratio over footprint bboxes) converts the
    relative field to metres for every building WITHOUT a real height -- the
    "minimal ground control" calibration, with many controls instead of a
    handful. dsm (and roof_dsm) are modified in place outside the fused
    footprints. Returns the calibration record, or None when fewer than
    min_controls usable controls exist (nothing is changed then).
    """
    _post = (dsm - ground)
    _fused_mask = np.zeros(dsm.shape[:2], bool)
    _ratios = []
    for _fp in footprints:
        _pp = np.asarray(_fp, np.int32).reshape(-1, 2)
        _x0, _y0 = np.maximum(_pp.min(0), 0)
        _x1, _y1 = np.minimum(_pp.max(0) + 1, [dsm.shape[1], dsm.shape[0]])
        if _x1 - _x0 < 2 or _y1 - _y0 < 2:
            continue
        _m = np.zeros((_y1 - _y0, _x1 - _x0), np.uint8)
        cv2.fillPoly(_m, [_pp - [_x0, _y0]], 1)
        _m = _m.astype(bool)
        if not _m.any():
            continue
        _a = float(np.median(pre_rel[_y0:_y1, _x0:_x1][_m]))
        _b = float(np.median(_post[_y0:_y1, _x0:_x1][_m]))
        if abs(_b - _a) > 0.25:
            _fused_mask[_y0:_y1, _x0:_x1] |= _m
            if _a > 1.0 and _b > 2.0:
                _ratios.append(_b / _a)
    if len(_ratios) < min_controls:
        if _ratios:
            print(f"      reference-height calibration skipped: {len(_ratios)} usable "
                  f"control(s) < {min_controls}")
        return None
    _lr = np.log(np.asarray(_ratios))
    _k = float(np.exp(np.median(_lr)))
    _iqr = float(np.percentile(_lr, 75) - np.percentile(_lr, 25))
    if not relative:
        # METRIC DEPTH GATE: the field is already in metres. Only rescale when
        # the controls agree tightly (IQR < 0.7) that it is badly off
        # (ratio outside [0.67, 1.5]); a wide-spread x3-4 ratio is noise in
        # the controls, not a scale error, and must not be applied.
        _reason = None
        if 0.67 <= _k <= 1.5:
            _reason = f"metric depth: ratio x{_k:.3f} within [0.67, 1.5]"
        elif _iqr >= 0.7:
            _reason = f"metric depth: log-ratio IQR {_iqr:.2f} >= 0.7 (controls disagree)"
        if _reason is not None:
            print(f"      reference-height calibration skipped -- {_reason}")
            return {"applied": False, "n_controls": len(_ratios),
                    "scale": round(_k, 4), "log_ratio_iqr": round(_iqr, 3),
                    "depth": "metric", "skipped_reason": _reason}
    _rel = dsm - ground
    _nf = ~_fused_mask
    dsm[_nf] = ground[_nf] + _rel[_nf] * _k
    if roof_dsm is not None:
        roof_dsm[_nf] = ground[_nf] + (roof_dsm - ground)[_nf] * _k
    print(f"      scale calibrated to {len(_ratios)} real building heights: "
          f"x{_k:.3f} (log-ratio IQR {_iqr:.2f}) applied to every building "
          f"without a real height")
    return {"applied": True, "n_controls": len(_ratios), "scale": round(_k, 4),
            "log_ratio_iqr": round(_iqr, 3),
            "depth": "relative" if relative else "metric"}


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
        # Near-overhead sun is numerically unstable, not just imprecise: height
        # = shadow_length * tan(elevation), and tan(elevation) explodes as
        # elevation approaches 90 deg. Measured on a real scene: a shadow of
        # a few pixels gave elevation 87.7 deg, and applying that tangent to
        # every OTHER building's shadow produced a median height of 170 m in
        # a scene of single-storey rural housing. The anchor building's own
        # height is still right by construction (that is what it was declared
        # to be) -- it is every other building's derived height that inherits
        # the blown-up tangent. Refusing here is the same "refuse loudly
        # rather than guess" rule the metric-model plausibility check applies.
        if elev > 80.0:
            return {"elevation_deg": None, "source": "anchor_shadow",
                    "reason": f"derived elevation {elev:.1f} deg is too close to "
                              "overhead to calibrate other buildings safely -- "
                              f"the anchor's own shadow ({shadow_len_m:.1f} m) is "
                              "short enough that tan(elevation) amplifies any "
                              "measurement noise into an implausible scale"}
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


def _pair_prisms_to_instances(prisms, instances, footprints, max_px=40.0):
    """Match each built prism to the discovery record describing the same building.

    These two lists come from DIFFERENT extractors -- prisms are built from
    region_footprints.extract(), the discovery records from
    building_discovery.discover() -- so they have different lengths and
    different orderings. zip() over them pairs each height with an unrelated
    building's evidence and silently drops the tail: measured on Varanasi, 951
    prisms against 522 records, so 429 prisms vanished from the export and every
    one that survived carried another structure's confidence, area and
    provenance.

    This is the same failure that once made a validation script report a 5x
    regression, and the same answer applies: never pair two independently
    filtered lists by position.

    Returns a list, parallel to `prisms`, holding the matched record or None.
    Unmatched prisms are exported with their own geometry and height and no
    evidence fields, which is the honest result -- the evidence genuinely does
    not exist for them.
    """
    import numpy as _np
    if not instances:
        return [None] * len(prisms)
    cent = []
    for r in instances:
        c = r.get("centroid_px") or r.get("centroid")
        if c is None:
            cnt = r.get("contour")
            c = (float(_np.mean(cnt[:, 0, 0])), float(_np.mean(cnt[:, 0, 1])))                 if cnt is not None else (0.0, 0.0)
        cent.append((float(c[0]), float(c[1])))
    cent = _np.asarray(cent, dtype=_np.float64)

    out, taken = [], set()
    for pr in prisms:
        pi = pr.get("poly_index")
        if pi is None or pi >= len(footprints):
            out.append(None)
            continue
        fp = _np.asarray(footprints[pi], dtype=_np.float64)
        cx, cy = fp[:, 0].mean(), fp[:, 1].mean()
        d2 = (cent[:, 0] - cx) ** 2 + (cent[:, 1] - cy) ** 2
        order = _np.argsort(d2)
        pick = None
        for j in order[:4]:
            if j in taken:
                continue
            if d2[j] <= max_px * max_px:
                pick = int(j)
            break
        if pick is not None:
            taken.add(pick)
            out.append(instances[pick])
        else:
            out.append(None)
    return out


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
    for i in np.unique(lab[0]):        # sky must touch the top edge: only those are tested
        if i == 0:
            continue
        x, y, bw_, bh_ = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        s_i = (lab[y:y + bh_, x:x + bw_] == i).any(axis=0).sum() / float(w)
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
          dem_source: str = "glo30", return_arrays: bool = False,
          use_metric: bool = True, gt_building_mask: np.ndarray = None,
          gt_building_polygons: list = None,
          gt_building_heights: list = None,
          skip_canopy: bool = False,
          canopy_height_map: np.ndarray = None,
          vlm_judge: bool = False,
          organic_roofs: bool = False,
          multiscale_fusion: bool = False,
          continuous_vegetation: bool = False,
          supplement_discovery: bool = False,
          ground_grid: int = GROUND_GRID,
          pbr_maps: bool = False,
          crown_radius_ratio: float = 0.32,
          sr_texture: bool = False,
          display_texture_path: str = None,
          context_image: str = None,
          auto_zoom: bool = True,
          capture_date: str = None,
          osm_structures: bool = False,
          context_city_footprints: str = None,
          depth_image_path: str = None,
          context_far_image: str = None,
          require_open_buildings: bool = False,
          terrain_from_dem: bool = False,
          dem_canopy_penetration: float = 0.6,
          canopy_mask_from_chm: bool = False,
          capture_utc: str = None,
          canopy_blur_m: float = None,
          fix_projection_scale: bool = None,
          gsd_assumed: bool = False,
          fetch_canopy: bool = False) -> dict:
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
    _ob_info = {}
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
    # Map units per ground metre of the source CRS at the scene centre: 1 for
    # UTM, ~1/cos(lat) for Web Mercator (EPSG:3857 overstates distance by
    # ~5% at Hyderabad, ~32% at New York). World/mesh coordinates are ground
    # metres; anything expressed as a source-CRS offset (context imagery
    # placement) is divided by this.
    _crs_units_per_m = 1.0
    _gt_grid_bound = (gt_building_mask is not None or gt_building_polygons is not None
                      or depth_image_path is not None or display_texture_path is not None
                      or canopy_height_map is not None)
    _valid_src = None          # False where the input carries no image (nodata, border, alpha)
    if image_path.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.open(image_path) as _s:
                import input_bands as _ib
                _band_idx, _band_how = _ib.rgb_band_plan(_s)
                if _band_how == "grey":
                    _band_idx = [1]
                _raw = _s.read(_band_idx)
                _valid_src = _ib.dataset_valid(_s, _band_idx) & ~_ib.border_collar(_raw)
                if _band_how != "assumed-rgb" or _s.count >= 4:
                    print(f"      bands: R,G,B = {_band_idx} of {_s.count} ({_band_how})")
                _s_crs, _s_tf = _s.crs, _s.transform
                _has_geo = (_s_crs is not None and _s_tf is not None
                            and not _s_tf.is_identity)
                if _has_geo and _s_crs.is_geographic and not _gt_grid_bound:
                    # A lat/lon GeoTIFF has no metric grid: reproject it once,
                    # here, onto its local UTM zone so every downstream step
                    # (DEM anchor, Open Buildings fusion, DSM export, slope)
                    # works on real metres with a real geotransform. Skipped
                    # only when a caller supplies other rasters bound to the
                    # FILE's pixel grid (GT footprints, CHM, hi-res textures),
                    # which a reprojection would silently misregister.
                    from rasterio.warp import (calculate_default_transform as _cdt,
                                               reproject as _rpj, Resampling as _Rs)
                    from rasterio.crs import CRS as _CRS
                    _clon = _s_tf.c + _s_tf.a * _s.width / 2
                    _clat = _s_tf.f + _s_tf.e * _s.height / 2
                    _utm = _CRS.from_epsg((32600 if _clat >= 0 else 32700)
                                          + int((_clon + 180) // 6) + 1)
                    _res = abs(_s_tf.a) * 111320.0 * math.cos(math.radians(_clat))
                    _tf2, _w2, _h2 = _cdt(_s_crs, _utm, _s.width, _s.height,
                                          *_s.bounds, resolution=_res)
                    _dst = np.zeros((_raw.shape[0], _h2, _w2), _raw.dtype)
                    for _b in range(_raw.shape[0]):
                        _rpj(_raw[_b], _dst[_b], src_transform=_s_tf, src_crs=_s_crs,
                             dst_transform=_tf2, dst_crs=_utm, resampling=_Rs.bilinear,
                             src_nodata=None, dst_nodata=0)
                    # The rotated footprint leaves corners with no source pixel;
                    # carry the valid mask through the same warp so they stay
                    # masked instead of becoming black "ground".
                    _vdst = np.zeros((_h2, _w2), np.uint8)
                    _rpj(_valid_src.astype(np.uint8), _vdst, src_transform=_s_tf,
                         src_crs=_s_crs, dst_transform=_tf2, dst_crs=_utm,
                         resampling=_Rs.nearest, src_nodata=None, dst_nodata=0)
                    _valid_src = _vdst > 0
                    print(f"      geographic input ({_s_crs}) reprojected to {_utm} "
                          f"at {_res*100:.1f} cm/px: {_s.width}x{_s.height} -> {_w2}x{_h2}")
                    _raw, _s_crs, _s_tf = _dst, _utm, _tf2
                if _has_geo:
                    m = abs(_s_tf.a)
                    if _s_crs.is_geographic:
                        lat = _s_tf.f + _s_tf.e * _s.height / 2
                        m = m * 111320.0 * math.cos(math.radians(lat))
                    else:
                        src_transform = _s_tf
                        src_epsg_code = _s_crs.to_epsg()
                        src_is_projected_m = src_epsg_code is not None
                        # Conformal projections other than UTM (Web Mercator
                        # above all) have a scale factor far from 1: the
                        # transform's pixel size is map units, not ground
                        # metres. Only corrected for plain uploads -- the
                        # showcase GT-footprint builds keep their original
                        # map-unit behaviour (their context imagery, viewer
                        # coordinate readout and published numbers assume it);
                        # opt-in via fix_projection_scale=True.
                        try:
                            from pyproj import Proj as _Proj, Transformer as _Tr
                            _cx = _s_tf.c + _s_tf.a * _raw.shape[2] / 2
                            _cy = _s_tf.f + _s_tf.e * _raw.shape[1] / 2
                            _lo, _la = _Tr.from_crs(_s_crs.to_wkt(), "EPSG:4326",
                                                    always_xy=True).transform(_cx, _cy)
                            _k = float(_Proj(_s_crs.to_wkt()).get_factors(_lo, _la).meridional_scale)
                        except Exception:
                            _k = 1.0
                        _fix = (fix_projection_scale if fix_projection_scale is not None
                                else not _gt_grid_bound)
                        if abs(_k - 1.0) > 0.01:
                            if _fix:
                                _crs_units_per_m = _k
                                m = m / _k
                                print(f"      projection scale factor {_k:.4f} at scene centre "
                                      f"({_s_crs}): pixel size corrected to ground metres")
                            else:
                                print(f"      NOTE: projection scale factor {_k:.4f} ({_s_crs}) "
                                      f"not corrected (GT-grid build keeps map-unit metres)")
                    native_gsd = m
                    src_crs = str(_s_crs)
                    print(f"      georeferenced input: {src_crs}, "
                          f"{native_gsd*100:.1f} cm/px measured from the file")
                    if gsd_m is None:
                        gsd_m = native_gsd
                    # Centre-of-scene lat/lon, for a sun-position calculation --
                    # cheap and correct enough over a single tile's extent
                    # (sun elevation does not vary meaningfully across a few
                    # km), whether the CRS is geographic or projected.
                    try:
                        cx = _s_tf.c + _s_tf.a * _raw.shape[2] / 2
                        cy = _s_tf.f + _s_tf.e * _raw.shape[1] / 2
                        if _s_crs.is_geographic:
                            src_lon, src_lat = cx, cy
                        else:
                            from rasterio.warp import transform as _rwt
                            _lon, _lat = _rwt(_s_crs, "EPSG:4326", [cx], [cy])
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
                if _raw.dtype != np.uint8:
                    # 16-bit / float imagery (e.g. 11-bit Cartosat/WorldView
                    # DNs): clipping at 255 blows nearly every pixel to white.
                    # Stretch each band over its own 2nd-98th percentile of
                    # valid (non-zero) pixels instead.
                    _out = np.zeros(_raw.shape, np.uint8)
                    for _b in range(_raw.shape[0]):
                        _band = _raw[_b].astype(np.float32)
                        _v = _band[np.isfinite(_band) & (_band != 0) & _valid_src]
                        if _v.size == 0:
                            continue
                        _lo, _hi = np.percentile(_v, [2, 98])
                        _out[_b] = np.clip((_band - _lo) / max(_hi - _lo, 1e-6) * 255.0,
                                           0, 255).astype(np.uint8)
                    print(f"      {_raw.dtype} input stretched to 8-bit "
                          f"(per-band 2-98th percentile)")
                    _raw = _out
                if _raw.shape[0] == 1:
                    # Panchromatic: replicate to three channels.
                    _raw = np.repeat(_raw, 3, axis=0)
                arr = np.ascontiguousarray(np.transpose(_raw, (1, 2, 0)))
                if not _valid_src.all():
                    print(f"      nodata / black border: {100 * (~_valid_src).mean():.1f}% "
                          f"of pixels carry no image; masked in every output")
                    arr = _ib.fill_invalid(arr, _valid_src)
                pil = Image.fromarray(arr)
        except Exception as e:
            print(f"      (not readable as GeoTIFF: {e}; falling back to PIL)")
            pil = Image.open(image_path).convert("RGB")
            _valid_src = None
    else:
        _pim = Image.open(image_path)
        if _pim.mode in ("RGBA", "LA") or (_pim.mode == "P" and "transparency" in _pim.info):
            _pim = _pim.convert("RGBA")
            _alpha = np.asarray(_pim)[..., 3]
            if (_alpha == 0).any() and (_alpha > 0).any():
                import input_bands as _ib
                _valid_src = _alpha > 0
                print(f"      transparent pixels: {100 * (~_valid_src).mean():.1f}%; "
                      f"masked in every output")
                pil = Image.fromarray(_ib.fill_invalid(np.asarray(_pim.convert("RGB")),
                                                       _valid_src))
            else:
                pil = _pim.convert("RGB")
        else:
            pil = _pim.convert("RGB")
    # Pixel size the georeference (if any) currently describes; the transform
    # is rescaled ONCE below to whatever grid auto-zoom/max_px leave, so every
    # later consumer (Open Buildings warp, DEM grid, DSM/GeoJSON export, OSM
    # structures) reads src_transform as the WORKING grid's affine.
    _geo_w, _geo_h = pil.size
    # The source grid, kept for the full-resolution DSM product (a Cartosat
    # scene is far larger than the 3D scene's working grid).
    _pil_src, _tf_src, _gsd_src = pil, src_transform, gsd_m
    # Auto-zoom: upsample toward the depth model's own training resolution
    # when the source is coarser than it. GAMUS/DFC2019, the real datasets
    # this project's checkpoint is tuned on, are 0.3 m/px; ISRO's own imagery
    # is typically 0.5-1.0 m/px, and a token at that scale already averages a
    # building with its surroundings before the head ever sees it -- a real,
    # measured effect elsewhere (a real competitor's own probe: per-building
    # RMSE 1.35->2.26 m and bias -0.16->-1.67 m at 0.6 m vs 0.3 m, recovered
    # to 1.40 m by upsampling back to the trained scale first). Capped at 3x:
    # beyond that a pixel is being invented, not recovered, and the model was
    # never shown ground that coarse to begin with.
    NATIVE_TRAIN_GSD_M = 0.3
    # Skipped when max_px would immediately shrink the upsampled image back
    # (upsample then downsample is a lossy round trip, not a gain), or when
    # the caller turns it off.
    _input_gsd_m = gsd_m   # what the file itself has -- the panel reports this, not the working grid
    _would_undo = gsd_m is not None and max(pil.size) * min(3.0, gsd_m / NATIVE_TRAIN_GSD_M) > max_px * 1.05
    if auto_zoom and not _would_undo and gsd_m is not None and gsd_m > NATIVE_TRAIN_GSD_M * 1.15:
        _zoom = min(3.0, gsd_m / NATIVE_TRAIN_GSD_M)
        print(f"      auto-zoom: input is {gsd_m*100:.1f} cm/px against the "
              f"{NATIVE_TRAIN_GSD_M*100:.0f} cm the checkpoint is tuned at "
              f"-> upsampling x{_zoom:.2f} before inference")
        gsd_m = gsd_m / _zoom
        pil = pil.resize((round(pil.size[0] * _zoom), round(pil.size[1] * _zoom)),
                         Image.BICUBIC)

    if max(pil.size) > max_px:
        _sc = max_px / max(pil.size)
        if gsd_m is not None:
            gsd_m = gsd_m / _sc     # fewer pixels over the same ground
        pil = pil.resize((int(pil.size[0] * _sc), int(pil.size[1] * _sc)), Image.LANCZOS)
    if src_transform is not None and pil.size != (_geo_w, _geo_h):
        from affine import Affine as _Aff
        src_transform = src_transform * _Aff.scale(_geo_w / pil.size[0], _geo_h / pil.size[1])
    image_np = np.array(pil)
    H, W = image_np.shape[:2]
    _gsd_work = gsd_m          # ground sample the 3D scene's model pass actually saw
    # The valid mask on the working grid (same resizes as the image).
    _valid = None
    if _valid_src is not None and not _valid_src.all():
        _valid = cv2.resize(_valid_src.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST) > 0
        if not _valid.any():
            raise SystemExit("the image has no valid pixels: it is entirely nodata, "
                             "transparent or black border, so there is nothing to "
                             "estimate heights for.")
    print(f"[1/5] {os.path.basename(image_path)}  {W}x{H}")
    if tracker is not None:
        tracker.megapixels = (W * H) / 1e6
    # The terrain tile and the Open Buildings heights depend only on where the
    # image is, so both downloads start now and run beside the height model
    # (12-20 s of network each, previously after it). Each is collected where
    # it was always used; any failure there falls back to fetching in line.
    _prefetch = {}
    if src_transform is not None and src_crs is not None and not os.environ.get("DW_NO_PREFETCH"):
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _pfx = _TPE(max_workers=2, thread_name_prefix="dw-prefetch")
        if (use_dem or dem_path) and src_is_projected_m:
            _dem_args = (src_transform.c, src_transform.f, H, abs(src_transform.a), f"EPSG:{src_epsg_code}")
            _dem_kw = dict(dem_path=dem_path, dem_source=dem_source, width_px=W,
                           gsd_y_m=abs(src_transform.e))
            _prefetch["dem"] = (_dem_args, _dem_kw, _pfx.submit(dem_mod.sample_grid, *_dem_args, **_dem_kw))
        _prefetch["ob"] = _pfx.submit(ob_fetch, (H, W), src_transform, src_crs)
        if fetch_canopy and canopy_height_map is None:
            # real tree heights (Meta/WRI CHMv2, 1 m) for the trees in 3D
            import canopy_source as _cs
            _prefetch["chm"] = _pfx.submit(_cs.fetch, src_transform, src_crs, W, H)
        _pfx.shutdown(wait=False)
    if min(W, H) < MIN_SIDE_PX:
        raise SystemExit(
            f"image is {W}x{H} pixels -- too small to build from: at least "
            f"{MIN_SIDE_PX} x {MIN_SIDE_PX} is needed (about 150 m at 0.6 m). "
            "Upload a larger crop of the scene.")

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
    # use_metric=False must mean "behave as if no metric checkpoint exists",
    # not just "ignore it later" -- DepthBackbone(metric=False) enforces that
    # at construction, so _bb.is_metric below can never disagree with the
    # depth_is_metric check further down (line ~621), which reads use_metric
    # again independently. Letting the two diverge would mean this backbone
    # returns real metres (is_metric True) while the pipeline treats them as
    # an uncalibrated Tier C field and multiplies in an invented "tallest
    # ~40m" scale on top of numbers that were already correct.
    _bb = DepthBackbone(metric=(None if use_metric else False))
    # raw_metric=True changes what this field actually contains -- real
    # metres instead of a [0,1] relative field -- so it has to be part of the
    # cache key too, not just the backbone tag. Without this, a field cached
    # BEFORE this fix existed (silently [0,1]-normalised despite a metric
    # checkpoint being active) would be loaded here as if it were the
    # now-correct real-metres field, and everything downstream that assumes
    # `depth_is_metric => height is already in metres` would be wrong in the
    # specific, silent way this whole fix exists to stop.
    # The metric checkpoint's own fingerprint belongs in the key: a retrained
    # checkpoint otherwise reads the previous model's cached field back
    # (measured: a new GAMUS checkpoint scored bit-identically to the old one
    # on all 10 blind DFC tiles because every height came from this cache).
    _ckid = ""
    if _bb.is_metric:
        try:
            with open(os.path.join(DepthBackbone.METRIC_CKPT, DepthBackbone.MARKER)) as _f:
                _ckid = "_" + str(json.load(_f).get("canary", {}).get("hash", "nofp"))[:12]
        except Exception:
            _ckid = "_nofp"
    key = (f"plainimg_{name}_{backbone_tag()}"
          + ("_metric" + _ckid if _bb.is_metric else "")
          + ("_fused" if multiscale_fusion else "")
          + ("_hires" if depth_image_path else "")
          # The pixels themselves: a same-named input with different content
          # (re-read band order, masked border) must not reuse old heights.
          + "_px" + __import__("hashlib").md5(
              np.ascontiguousarray(image_np[::7, ::7]).tobytes()).hexdigest()[:10])
    # TRAINED-SCALE PASS. The metric model is run on the SOURCE pixels,
    # resampled to the 0.3 m it was trained at, whenever the working grid is
    # coarser than that or smaller than the source: the product DSM is written
    # from it at the source grid, and the 3D scene's heights are it, averaged
    # onto the working grid. Running the model on the working grid instead
    # (a 4096 px 0.57 m scene shrunk to 2560 px = 0.91 m) under-called HITEC
    # to p99 2.3 m against ~9 m at the trained scale; 0.6 m input costs
    # +2.9 % per building upsampled vs +7.3 % not (BENCHMARK_RESOLUTION.md).
    _agl_full, _full_zoom = None, 1.0
    FULLRES_MAX_MPX = float(os.environ.get("DW_FULLRES_MAX_MPX", "80"))
    _trained_scale_pass = bool(
        _bb.is_metric and not depth_image_path and _pil_src is not None
        and _gsd_src is not None and _gsd_work is not None
        and (max(_gsd_src, _gsd_work) > NATIVE_TRAIN_GSD_M * 1.15
             or max(_pil_src.size) > max(W, H) * 1.2)
        and not os.environ.get("DW_NO_FULLRES"))
    if _trained_scale_pass:
        key += "_ts"
        _Wf, _Hf = _pil_src.size
        if _gsd_src > NATIVE_TRAIN_GSD_M * 1.15:
            _full_zoom = min(3.0, _gsd_src / NATIVE_TRAIN_GSD_M)
            _full_zoom = max(1.0, min(_full_zoom, math.sqrt(FULLRES_MAX_MPX * 1e6 / (_Wf * _Hf))))
        _cp = os.path.join(height_cache.CACHE_DIR, f"{key}_full.npy")
        if os.path.exists(_cp):
            _agl_full = np.load(_cp)
        else:
            _tz = time.time()
            print(f"      trained-scale pass: {_Wf}x{_Hf} source pixels at x{_full_zoom:.2f} "
                  f"(~{_gsd_src / _full_zoom:.2f} m); 3D scene grid {W}x{H}")
            _src_in = (_pil_src if _full_zoom < 1.05 else
                       _pil_src.resize((round(_Wf * _full_zoom), round(_Hf * _full_zoom)),
                                       Image.BICUBIC))
            _agl_full = np.maximum(_bb.predict_tiled(_src_in, raw_metric=True), 0.0)
            del _src_in
            _agl_full = cv2.resize(_agl_full.astype(np.float32), (_Wf, _Hf),
                                   interpolation=cv2.INTER_AREA)
            os.makedirs(height_cache.CACHE_DIR, exist_ok=True)
            np.save(_cp, _agl_full)
            print(f"      trained-scale pass in {time.time() - _tz:.0f}s")
    height = height_cache.load(key, "tiled", H) if H == W else None
    if height is None:
        if depth_image_path:
            # Inference on a REAL higher-resolution image of the same bounds
            # (e.g. zoom-19, ~0.3 m -- the checkpoint's training scale), then
            # area-averaged onto the working grid. Measured on HITEC City at
            # native 0.6 m: the metric head's scene p99 was 1.95 m and the
            # pipeline rightly refused it; the model needs its native scale,
            # and upsampling 0.6 m pixels does not supply real detail.
            _hp = Image.open(depth_image_path).convert("RGB")
            print(f"      depth inference on {_hp.size[0]}x{_hp.size[1]} real imagery "
                  f"(same bounds), resampled to the {W}x{H} working grid")
            _hh = _bb.predict_tiled(_hp, raw_metric=_bb.is_metric)
            del _hp
            height = cv2.resize(_hh.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
            del _hh
        elif _trained_scale_pass:
            height = cv2.resize(_agl_full, (W, H), interpolation=cv2.INTER_AREA)
        elif multiscale_fusion:
            height = _bb.predict_fused(pil, raw_metric=_bb.is_metric)
        else:
            height = _bb.predict_tiled(pil, raw_metric=_bb.is_metric)
        if H == W:
            height_cache.save(key, "tiled", H, height)
    print(f"[2/5] height field in {time.time()-t0:.0f}s")
    if _valid is not None:
        # No image, no height: masked pixels stand at ground level from here on
        # (so nothing is extruded there) and are NaN / cut in every output.
        height = np.where(_valid, height, 0.0).astype(np.float32)
    # A metric checkpoint predicts height above ground directly. That field is
    # the elevation PRODUCT (DSM file, benchmark): measured blind on 10
    # DFC2019 tiles, the raw field scores RMSE 2.82 m / MAE 1.59 m / corr
    # 0.787, while the same field after the mesh post-processing below
    # (refinement, roof squaring, a second ground subtraction) scored
    # 4.36 m / 2.36 m / 0.493. Post-processing still shapes the 3D MESH.
    _agl_raw = np.asarray(height, np.float32).copy() if _bb.is_metric else None
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

    # Buildings are found on a contrast-normalised copy: the same per-band
    # 2-98th percentile stretch a 16-bit file already gets on reading. On one
    # HITEC scene the unstretched 8-bit file gave 318 buildings and the same
    # pixels read from 16-bit gave 1,629 (~1,900 per km2 are there). The 3D
    # texture keeps the original colours; only segmentation and outlines see this.
    image_an = _analysis_stretch(image_np, _valid)
    seg_labels, _ = seg.segment(image_an, height=height)
    if _valid is not None:
        seg_labels = seg_labels.copy()
        seg_labels[~_valid] = seg.CLASS_IDX["bare_earth"]
    if gt_building_mask is not None:
        # Every scale computation below (initial p99, DTM ground-fit, the
        # roof-squaring refit) reads seg_labels' building channel, not just
        # footprint discovery. Left un-patched, the "tallest structure" that
        # anchors the whole scene's vertical scale is measured under the
        # STOCK classifier's ~0.2-precision mask -- noise and canopy fragments
        # included -- even on a build whose footprints come from ground truth.
        # Splicing the real mask in here, once, before anything reads it,
        # is what actually makes height and DTM ground-fit trust the GT too.
        _gt0 = gt_building_mask
        if _gt0.shape != seg_labels.shape:
            _gt0 = cv2.resize(_gt0.astype(np.uint8),
                              (seg_labels.shape[1], seg_labels.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        seg_labels = seg_labels.copy()
        seg_labels[_gt0 > 0] = seg.CLASS_IDX["building"]
    oc = orientation_check(height, seg_labels)
    if oc.get("checked") and not oc["correct_orientation"]:
        if _bb.is_metric:
            # The inversion this guards against is a RELATIVE-depth failure
            # (depth read as height). A metric model predicts height above
            # ground directly and cannot come out upside down; on a steep
            # hillside the building-vs-ground comparison is weak anyway
            # (the ground class is mostly slope, footprints shift with
            # relief). Reported, not fatal. Darjeeling tripped it.
            print(f"      orientation check weak on this scene: buildings mean "
                  f"{oc['building_mean']:.2f} m vs ground {oc['ground_mean']:.2f} m "
                  f"(metric model, not an inversion) -- continuing")
        else:
            raise SystemExit("height field inverted -- refusing to build")
    refined = dsm_refine.refine_dsm(height, image_np)

    sun_az = estimate_sun_azimuth(image_np, seg_labels)
    # With a capture time and a location the sun's azimuth is an astronomical
    # fact; the shadow-direction heuristic above is only a fallback. On
    # DFC2019 JAX_004 it returned 40 deg against a true 108 deg (the IMD's own
    # meanSunAz), so every shadow ray was cast the wrong way.
    if capture_dt_utc is not None and src_lat is not None and src_lon is not None:
        _el_eph, _az_eph = shadow_correction.sun_position(src_lat, src_lon, capture_dt_utc)
        if _el_eph > 0:
            if sun_az is not None and abs((sun_az - _az_eph + 180) % 360 - 180) > 20:
                print(f"      sun azimuth: ephemeris {_az_eph:.0f} deg replaces the "
                      f"shadow estimate {sun_az:.0f} deg")
            sun_az = float(_az_eph)

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
    # Does the backbone in use predict metres, or a relative field?
    depth_is_metric = use_metric and DepthBackbone.metric_available()

    if tier is None and depth_is_metric:
        # A METRIC BACKBONE SKIPS THE SCALE QUESTION ENTIRELY.
        #
        # This is the whole return on fine-tuning. The stock backbone emits a
        # relative field, so every branch below has to invent a scale, and
        # measured blind that invention lands 24-353% from what each tile needs
        # -- roughly three quarters of the blind error. A model trained on GAMUS
        # AGL predicts metres, so there is nothing to invent: the scale is 1.0
        # and the "tallest ~ 40 m" assumption never runs.
        scale = 1.0
        _mi = DepthBackbone.metric_info()
        tier = "B (metric depth model -- heights predicted directly in metres)"
        scale_source = (
            f"metric depth model: heights predicted in metres directly "
            f"(GAMUS-tuned; licensed at MAE "
            f"{_mi.get('finetuned_mae_m_no_affine', '?')} m against a baseline "
            f"of {_mi.get('baseline_mae_m_with_affine', '?')} m that was given "
            f"the optimal scale)")
        print(f"[3/5] scale: {scale_source}")

        # SANITY-CHECK THE METRIC MODEL BEFORE TRUSTING ITS METRES.
        #
        # Scale is 1.0 here, so a model that predicts near-zero everywhere
        # produces a scene with no buildings and no error message -- measured
        # exactly that way with a deliberately under-trained checkpoint: 0
        # prisms, heights 0.0. A licensed checkpoint is not a guarantee of a
        # sane prediction on THIS image, only of having beaten a baseline on
        # GAMUS, so the field is checked against the scene in front of it.
        _p99_metric = float(np.percentile(rel, 99))
        # Only a field that is essentially flat is refused. A LOW field (a
        # suburb of one-storey houses under trees) is still metres, and the
        # fallback below would multiply this same field by 40/p99 -- measured
        # on DFC2019 JAX_004 as roughly x400, i.e. 500 m houses. Under-called
        # metres are a smaller error than an invented scale.
        if 0.5 <= _p99_metric < 3.0:
            print(f"      metric model 99th percentile is only {_p99_metric:.2f} m on "
                  f"this scene -- kept (low-rise scene or under-call), not rescaled")
        if _p99_metric < 0.5:
            print(f"      REFUSED: the metric model's 99th percentile height is "
                  f"{_p99_metric:.2f} m across this whole scene. That is not a "
                  f"plausible city, so the checkpoint's metres are rejected "
                  f"rather than used -- it is licensed on GAMUS but is not "
                  f"transferring to this image. Falling back to the stock "
                  f"backbone's relative field, scaled the same way Tier C "
                  f"always has been.")
            # Undo everything the metric branch set, so the Tier C block
            # below runs exactly as if depth_is_metric had been False from
            # the start -- a rejected reading must not leave a half-applied
            # scale=1.0 behind for the next block to inherit by accident.
            tier = None
            scale = None
            scale_source = None
            _agl_raw = None

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
        # An assumed scale must never manufacture skyscrapers from noise: a
        # near-flat field divided into "tallest = 40 m" can multiply by
        # hundreds. Bound the result to a physically plausible tallest point.
        if np.isfinite(scale) and float(np.max(rel)) * scale > 250.0:
            scale = 250.0 / max(float(np.max(rel)), 1e-6)
            scale_source += "; capped so no point exceeds 250 m"
        print(f"[3/5] scale: {scale_source}")

    print(f"      sun azimuth estimated from shadows: "
          f"{'%.0f deg' % sun_az if sun_az is not None else 'not determinable'}")
    _tr.leave(); _tr.enter("buildings")

    dsm = refined * scale
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    # roof_dsm: a SEPARATE copy of the surface, captured before buildings are
    # squared to one value each, kept only for build_prisms's own roof-height
    # sampling. Everything else below (terrain estimation, ground, footprint
    # discovery, the model-vs-image audit) keeps using the squared `dsm`
    # exactly as it always has -- that squaring is real and load-bearing
    # beyond just roof shape, confirmed directly (DFC2019 JAX_004): skipping
    # it outright dropped surviving buildings 58->27 and collapsed the
    # audit's height-agreement correlation 0.928->0.522, because
    # prismify_buildings also stabilises what the terrain/repair loop sees.
    # roof_dsm exists so build_prisms can see the UN-squared signal for the
    # one thing it's actually needed for -- fitting a real per-footprint
    # roof plane -- without that raw signal ever reaching terrain, ground,
    # or discovery, where it measurably hurt.
    roof_dsm = dsm.copy() if organic_roofs else None
    dsm, n_prism = dsm_refine.prismify_buildings(dsm, seg_labels, terrain)
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    # 15 m, not the previous 35 m: flatten_ground's own docstring says the
    # radius only needs to be WIDER THAN A BUILDING to remove its residue --
    # Karnataka's median footprint is ~11 m per side (119.5 sq m), so 15 m
    # already clears that with margin. 35 m was nearly 3x that with no
    # comment justifying the extra width, and it was smoothing away real,
    # measured terrain relief along with the residue: the exported ground
    # mesh at 35 m read as visibly flat next to the buildings even though
    # real elevation variation (std 1.35 m across the scene) existed in the
    # unsmoothed terrain underneath it.
    ground = city_model.flatten_ground(terrain, seg_labels, smooth_m=15.0, gsd_m=gsd)

    # TIER A ATTEMPT: anchor the absolute datum to an external DEM.
    #
    # Same fit as build_city.py's DEM step -- a 30 m DEM fixes WHERE the
    # surface sits vertically, fitted over pixels this pipeline's own
    # segmentation calls bare ground, so no external ground truth leaks in.
    # Only attempted when the source file is a GeoTIFF with a genuinely
    # projected (metric) CRS: without a real UTM origin there is no grid to
    # sample the DEM onto, and reprojecting a bare JPEG's assumed geometry
    # would anchor a datum to coordinates that were never real.
    def _sample_dem():
        # The working grid exactly: origin + per-axis pixel size in the
        # source CRS's own units, rows x cols (non-square scenes included).
        args = (src_transform.c, src_transform.f, ground.shape[0], abs(src_transform.a),
                f"EPSG:{src_epsg_code}")
        kw = dict(dem_path=dem_path, dem_source=dem_source,
                  width_px=ground.shape[1], gsd_y_m=abs(src_transform.e))
        pf = _prefetch.get("dem")
        if pf is not None and pf[0] == args and pf[1] == kw:
            try:
                return pf[2].result()
            except Exception:
                pass                           # fetch it in line instead
        return dem_mod.sample_grid(*args, **kw)

    dem_info = None
    d = None
    _datum_fit = None
    _terrain_datum = None
    if (use_dem or dem_path) and src_is_projected_m:
        d = _sample_dem()
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
            _datum_fit = fit if fit.get("offset_m") is not None else None
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

    # HILLS IN 3D. The metric model reads height ABOVE GROUND and carries no
    # terrain, so on its own a hillside is drawn flat (the DSM export already
    # stands on the terrain; the 3D scene did not). When the terrain map shows
    # real relief across the scene, the 3D ground becomes that terrain --
    # smoothed to the DEM's own 30 m and corrected for the share of buildings
    # and trees the DEM already counts (anchoring.py, k measured on LiDAR) --
    # and the model's heights stand on it. A flat scene is left as it was.
    if (not terrain_from_dem and _bb.is_metric and d is not None and d.get("dem") is not None
            and tier.startswith("A") and not os.environ.get("DW_FLAT_GROUND")):
        import anchoring as _anc
        _dm = np.asarray(d["dem"], np.float32)
        if _dm.shape != ground.shape:
            _dm = cv2.resize(_dm, (ground.shape[1], ground.shape[0]), interpolation=cv2.INTER_LINEAR)
        _sm = _anc.smooth(_dm, gsd)
        _fin = np.isfinite(_sm)
        _relief = float(np.percentile(_sm[_fin], 99) - np.percentile(_sm[_fin], 1)) if _fin.any() else 0.0
        if _relief >= RELIEF_MODE_M:
            _nd = np.maximum(dsm - ground, 0.0)
            _k = _anc.DEFAULT_K.get(dem_source, 0.4)
            _, _g = _anc.absolute_surface(np.where(_fin, _dm, np.nanmedian(_dm)), _nd, gsd, _k)
            _g = np.where(np.isfinite(_g), _g, np.nanmedian(_g)).astype(np.float32)
            terrain = _g + (terrain - ground)
            dsm = _g + _nd
            ground = _g
            dem_info = {**(dem_info or {}), "relief_mode": True, "relief_m": round(_relief, 1), "k": _k}
            print(f"      relief {_relief:.0f} m: 3D ground follows the terrain map "
                  f"({d.get('source')}, k = {_k}); model heights stand on it")

    if terrain_from_dem and (use_dem or dem_path) and src_is_projected_m:
        # Landscape mode: the terrain IS the DEM. The monocular depth field
        # has no bare-earth signal under a closed canopy. Copernicus GLO-30 is
        # an X-band SURFACE model -- over dense forest its phase centre sits
        # partway into the canopy (~0.4-0.7 of its height in tropical forest
        # studies) -- so it is corrected by that fraction of the real canopy
        # height, smoothed to the DEM's own ~30 m scale, before the canopy is
        # added back on top. An approximation of bare earth, disclosed as such.
        _d = d if (d is not None and d.get("dem") is not None) else \
            _sample_dem()
        if _d.get("dem") is not None:
            _dem = _d["dem"].astype(np.float32)
            if _dem.shape != ground.shape:
                _dem = cv2.resize(_dem, (ground.shape[1], ground.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            _pen = 0.0
            if canopy_height_map is not None:
                _cr = cv2.resize(canopy_height_map.astype(np.float32),
                                 (ground.shape[1], ground.shape[0]), interpolation=cv2.INTER_LINEAR)
                _dem = _dem - dem_canopy_penetration * cv2.GaussianBlur(
                    _cr, (0, 0), sigmaX=max(1.0, 60.0 / gsd))
                _pen = dem_canopy_penetration
            _dem = cv2.GaussianBlur(_dem, (0, 0), sigmaX=max(1.0, 12.0 / gsd))
            # The scene is built with its low point at ~0 (viewer convenience);
            # the removed datum is added back on DSM export -- elevation, EGM2008.
            _terrain_datum = float(np.percentile(_dem, 1))
            _dem = _dem - _terrain_datum
            ground = _dem
            terrain = _dem.copy()
            dsm = _dem.copy()
            tier = ("A (terrain: Copernicus GLO-30, canopy-corrected x"
                    f"{_pen:.1f}; canopy: real canopy-height map)")
            dem_info = {"attempted": True, "accepted": True, "source": _d.get("source"),
                        "mode": "terrain_from_dem", "canopy_penetration_correction": _pen,
                        "relief_m": round(float(np.percentile(_dem, 99) - np.percentile(_dem, 1)), 1)}
            print(f"      terrain from DEM ({_d.get('source')}), canopy-penetration "
                  f"correction x{_pen:.1f}; relief {dem_info['relief_m']} m")
        else:
            print(f"      terrain_from_dem: DEM unavailable ({_d.get('error')})")

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
    if gt_building_polygons is not None:
        # One step better than extract_from_mask: identity comes from the
        # annotation, not a re-traced raster contour, so two buildings that
        # touch (a common Indian row-house pattern) never fuse into one --
        # see extract_from_polygons's docstring for the measured 1,751-to-834
        # collapse that rasterising first causes.
        assert gt_building_mask is not None, (
            "gt_building_polygons needs gt_building_mask alongside it, purely "
            "to know the pixel grid the polygon coordinates were computed on")
        _sy = ndsm.shape[0] / gt_building_mask.shape[0]
        _sx = ndsm.shape[1] / gt_building_mask.shape[1]
        _scaled_polys = [np.asarray(p, np.float32) * [_sx, _sy]
                         for p in gt_building_polygons]
        rres = region_footprints.extract_from_polygons(
            _scaled_polys, ndsm, gsd, known_heights=gt_building_heights)
        footprints = rres["polygons"]
        print(f"      footprints from GROUND TRUTH polygons: {rres['report']['retained']} "
              f"of {rres['report']['regions_examined']} individually-labelled buildings")
        if gt_building_heights is not None:
            print(f"      of those, {rres['report']['measured_heights']} carry a REAL "
                  f"measured height (e.g. municipal floor-count record), not an estimate")
        _rej = "  ".join(f"{k} {v}" for k, v in rres["report"]["rejected"].items() if v)
        if _rej:
            print(f"      rejected: {_rej}")

        # REAL BUG FIX, found by tracing why Mumbai's exported heights never
        # showed the signature of real floor-count data (multiples of a
        # storey height): rres["records"] carries each footprint's real
        # measured height when one exists, but until this fix only
        # rres["polygons"] (the SHAPES) was ever kept -- the measured
        # heights were computed, printed as a summary count above, and then
        # discarded. mesh_repair.repair_build (called further down) samples
        # HEIGHT from `dsm` directly, with no known-heights parameter of its
        # own, so a real measurement that never touches `dsm` never reaches
        # the final geometry or export no matter how correctly it was
        # computed here. Fix: burn each measured footprint's real height
        # into `dsm` now, the same "splice the real signal in before
        # anything reads it" technique already used above for the
        # segmentation mask -- so every later consumer of `dsm` (mesh
        # generation, ndsm, the audit) sees the real number as if it had
        # always been there, rather than needing its own separate
        # known-heights plumbing.
        _records = rres["records"]
        _n_burned = 0
        for _i, _rec in enumerate(_records):
            if _rec.get("height_provenance") != "measured" or _i >= len(footprints):
                continue
            _crop = _poly_crop_mask(footprints[_i], *dsm.shape[:2])
            if _crop is None or not _crop[1].any():
                continue
            _win, _fp_mask = _crop
            _local_ground = float(np.median(ground[_win][_fp_mask]))
            dsm[_win][_fp_mask] = _local_ground + _rec["height"]
            if roof_dsm is not None:
                roof_dsm[_win][_fp_mask] = _local_ground + _rec["height"]
            _n_burned += 1
        if _n_burned:
            ndsm = np.maximum(dsm - ground, 0.0)
            print(f"      {_n_burned} real measured height(s) burned into the surface "
                  f"model -- these now drive the actual geometry, not just the report")

        # Real Open Buildings heights for whichever footprints did NOT already
        # get a real measured height above. A GT-polygon source can carry real
        # SHAPES with no height attribute at all (e.g. Microsoft Global ML
        # Building Footprints -- India ships height=-1 sentinels for every
        # record), which previously meant this branch never called
        # fuse_open_buildings_heights at all -- only the fully-automated
        # discovery branch did. Restricted to the unmeasured subset so a real
        # municipal measurement from above is never overwritten by a coarser
        # public-dataset estimate.
        _measured_idx = {
            _i for _i, _rec in enumerate(_records)
            if _rec.get("height_provenance") == "measured"
        }
        _unmeasured_fp = [fp for _i, fp in enumerate(footprints) if _i not in _measured_idx]
        if _unmeasured_fp:
            _pre_rel = (dsm - ground).astype(np.float32)
            _n_ob_fused2 = fuse_open_buildings_heights(
                dsm, ground, _unmeasured_fp, src_transform, src_crs,
                info=_ob_info)
            if roof_dsm is not None and _n_ob_fused2:
                fuse_open_buildings_heights(roof_dsm, ground, _unmeasured_fp,
                                            src_transform, src_crs)
            if _n_ob_fused2:
                ndsm = np.maximum(dsm - ground, 0.0)
                print(f"      {_n_ob_fused2} real Open Buildings height(s) fused "
                      f"into the surface model (remaining unmeasured GT footprints)")
                # SCALE CALIBRATION FROM REAL REFERENCE HEIGHTS. Every fused
                # footprint is a ground-control height: its real height against
                # the depth field's own (relative) reading there. A robust fit
                # of that ratio converts the relative field to metres for every
                # building WITHOUT a real height -- the "minimal ground control"
                # calibration, with thousands of controls instead of a handful.
                _cal_ob = calibrate_to_reference_heights(
                    dsm, ground, _pre_rel, _unmeasured_fp, roof_dsm=roof_dsm,
                    relative=not (_bb.is_metric and "metric depth model" in (scale_source or "")))   # the field itself, not the tier label: Tier A (terrain-anchored) is metric too
                if _cal_ob is not None:
                    _ob_info["calibration"] = _cal_ob
                if _cal_ob is not None and _cal_ob["applied"]:
                    ndsm = np.maximum(dsm - ground, 0.0)
                    if tier[0] == "C":
                        tier = (f"B (relative depth calibrated to {_cal_ob['n_controls']} real "
                                f"building heights)")
            elif require_open_buildings:
                # Refuse rather than stage a scene whose heights silently fell
                # back to an assumption (e.g. a network outage mid-build).
                raise SystemExit("require_open_buildings: no real Open Buildings heights "
                                 "could be fused -- refusing to build/stage")

        if supplement_discovery:
            # A GT footprint layer is a real source, but not necessarily an
            # EXHAUSTIVE one -- measured directly on HITEC City, Hyderabad:
            # Microsoft Global ML Building Footprints supplied 281 real
            # polygons for a scene where independent image-region discovery
            # (the same watershed extractor the fully-automated branch uses)
            # finds 472 candidate shapes from the same real pixels. Run that
            # same discovery here and add only the regions that land on NO
            # existing GT footprint, so real GT shapes are never replaced or
            # duplicated by a coarser discovered one -- this only fills gaps.
            _drres = region_footprints.extract(
                image_an, ndsm, gsd, seg_labels=seg_labels,
                min_area_m2=6.0, min_height_m=1.5)
            _discovered, _sthr = region_footprints.drop_low_regions(
                _drres["polygons"], ndsm)
            _gt_mask_all = np.zeros(dsm.shape[:2], np.uint8)
            for _fp in footprints:
                cv2.fillPoly(_gt_mask_all, [np.asarray(_fp, np.int32)], 1)
            _gt_mask_all = _gt_mask_all.astype(bool)
            _added_fps = []
            for _d in _discovered:
                _crop = _poly_crop_mask(_d, *dsm.shape[:2])
                if _crop is None or not _crop[1].any():
                    continue
                _win, _dm = _crop
                _overlap = (_gt_mask_all[_win] & _dm).sum() / _dm.sum()
                if _overlap < 0.1:
                    _added_fps.append(_d)
            print(f"      supplemental discovery: {len(_discovered)} candidate region(s) "
                  f"examined, {len(_added_fps)} added (no existing GT footprint there)")
            if _added_fps:
                footprints = footprints + _added_fps
                _n_ob_fused3 = fuse_open_buildings_heights(
                    dsm, ground, _added_fps, src_transform, src_crs)
                if roof_dsm is not None and _n_ob_fused3:
                    fuse_open_buildings_heights(roof_dsm, ground, _added_fps,
                                                src_transform, src_crs)
                ndsm = np.maximum(dsm - ground, 0.0)
                if _n_ob_fused3:
                    print(f"      {_n_ob_fused3} real Open Buildings height(s) fused "
                          f"into the supplemental footprints")
    elif gt_building_mask is not None:
        # Existence and shape are already decided by the annotator -- see
        # extract_from_mask's docstring for why the height/size gates below
        # would otherwise throw away real, human-verified buildings.
        _gt = gt_building_mask
        if _gt.shape != ndsm.shape:
            _gt = cv2.resize(_gt.astype(np.uint8), (ndsm.shape[1], ndsm.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        rres = region_footprints.extract_from_mask(_gt, ndsm, gsd)
        footprints = rres["polygons"]
        print(f"      footprints from GROUND TRUTH mask: {rres['report']['retained']} "
              f"of {rres['report']['regions_examined']} labelled buildings")
        _rej = "  ".join(f"{k} {v}" for k, v in rres["report"]["rejected"].items() if v)
        if _rej:
            print(f"      rejected: {_rej}")
    else:
        # recall 0.745). One extractor, used everywhere.
        #
        # min_height_m is in ndsm's units, which on this Tier C path are not
        # metres -- the surface was rescaled so the tallest structure reads
        # TARGET_TALL_M, so a 2 m threshold means "2 units on that assumed scale".
        rres = region_footprints.extract(
            image_an, ndsm, gsd, seg_labels=seg_labels,
            min_area_m2=6.0, min_height_m=1.5)
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

        # Real per-building heights where a public dataset has them for
        # this exact location -- only reachable on a real georeference
        # (src_transform/src_crs), which this branch is not guaranteed to
        # have (a plain PNG has neither). Genuinely no-op, not a silent
        # skip, when neither exists.
        _pre_rel_up = (dsm - ground).astype(np.float32)
        _n_ob_fused = fuse_open_buildings_heights(
            dsm, ground, footprints, src_transform, src_crs)
        if roof_dsm is not None and _n_ob_fused:
            # Same real measured heights, mirrored into the roof-sampling
            # copy so a real Open Buildings height is never lost behind an
            # organic-roof fit -- a measured number always wins over a plane
            # fitted to noise.
            fuse_open_buildings_heights(roof_dsm, ground, footprints,
                                        src_transform, src_crs)
        if _n_ob_fused:
            # Same refresh Mumbai's ground-truth fusion already needs (see
            # the gt_building_polygons branch above): dsm was mutated in
            # place, but ndsm was computed from the OLD dsm and numpy does
            # not track that dependency -- every downstream reader of ndsm
            # (the model-vs-image audit, in particular) would otherwise
            # keep comparing against pre-fusion heights and see a
            # correlation collapse that has nothing to do with whether the
            # fusion itself is right.
            ndsm = np.maximum(dsm - ground, 0.0)
            print(f"      {_n_ob_fused} real Open Buildings height(s) fused "
                  f"into the surface model")
            # Same reference-height calibration as the GT-footprint branch:
            # every fused footprint is a ground control for the buildings
            # Open Buildings does not cover.
            _cal_ob = calibrate_to_reference_heights(
                dsm, ground, _pre_rel_up, footprints, roof_dsm=roof_dsm,
                relative=not (_bb.is_metric and "metric depth model" in (scale_source or "")))   # the field itself, not the tier label: Tier A (terrain-anchored) is metric too
            if _cal_ob is not None:
                _ob_info["calibration"] = _cal_ob
            if _cal_ob is not None and _cal_ob["applied"]:
                ndsm = np.maximum(dsm - ground, 0.0)
                if tier[0] == "C":
                    tier = (f"B (relative depth calibrated to {_cal_ob['n_controls']} real "
                            f"building heights)")

    # Provenance now tracks the tier actually reached: a height is MEASURED
    # only on B/A, where a physical shadow or DEM calibration produced the
    # scale, and stays INFERRED on C, where it is an assumption.
    _prov = bd.MEASURED if tier[0] in ("A", "B") else bd.INFERRED
    for rec in disc["instances"]:
        rec["provenance"] = _prov

    # Build, localise defects, repair the offenders, re-check. The loop is
    # bounded: each pass must strictly reduce the defect count or the strategy
    # escalates, and the last strategy removes the building, so it terminates.
    _gsmall_pre, _ = mg._resize_for_mesh(ground, seg_labels, ground_grid)
    _cell_pre = gsd * (ground.shape[0] / _gsmall_pre.shape[0])
    # Roof colours are sampled from the WHITE-BALANCED image, not the raw one.
    # Correcting the ground texture alone would leave every roof carrying the
    # atmospheric blue cast while the terrain under it reads neutral -- the two
    # would not look like the same scene.
    # Same graded image the ground/roof texture is written from, so a wall's
    # sampled colour matches the textured roof above it.
    _balanced = image_grade.grade(image_np)
    _al = _ob_info.get("alignment")
    _rep = mesh_repair.repair_build(
        footprints, dsm, ground, gsd, image_np=_balanced, min_height_m=1.5,
        ground_small=_gsmall_pre, cell_m=_cell_pre, roof_dsm=roof_dsm,
        lean_px_per_m=_al["lean_px_per_m"] if _al else None)
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
          f"({'m' if (anchor_height_m or depth_is_metric or not tier.startswith('C')) else 'relative units'})")

    # Sun from the photo's own shadows, now that per-building heights and the
    # lean are known: lights the render the way the image was actually lit,
    # so rendered shadows fall where the photographed ones already are.
    _sun_fit = None
    # Gated on a real height alignment (Open Buildings heights in metres), not
    # on the depth tier: the tall buildings that constrain the fit carry
    # fused real heights even when the depth field itself is only relative.
    if _al and binfo["buildings"]:
        import sun_from_shadows
        _polys = [footprints[r["poly_index"]] for r in binfo["buildings"]
                  if r["poly_index"] < len(footprints)]
        _hs = [r["height_m"] for r in binfo["buildings"] if r["poly_index"] < len(footprints)]
        _eph = None
        if capture_date and src_lat is not None:
            import datetime as _dtm
            _eph = (src_lat, src_lon, _dtm.date.fromisoformat(capture_date))
        _sun_fit = sun_from_shadows.fit_sun(image_np, _polys, _hs,
                                            _al["lean_px_per_m"], gsd, ephemeris=_eph)
        if _sun_fit:
            print(f"      sun from shadows of {_sun_fit['n']} buildings with known heights: "
                  f"azimuth {_sun_fit['azimuth_deg']:.0f} deg, elevation "
                  f"{_sun_fit['elevation_deg']:.0f} deg (contrast x{_sun_fit['contrast']})"
                  + (f", pass {_sun_fit['capture_time_utc']} UTC" if _sun_fit.get('capture_time_utc') else ""))
        else:
            print("      sun from shadows: not constrained enough -- left unset")

    if _sun_fit is None and capture_utc and src_lat is not None:
        # No buildings to fit shadows to (e.g. a forest): real capture DATE
        # from the imagery's metadata plus an ASSUMED pass time, through the
        # solar ephemeris. Disclosed as assumed in the scene notes.
        import datetime as _dtm
        from shadow_correction import sun_position as _sp
        _t = _dtm.datetime.fromisoformat(capture_utc)
        _el, _az = _sp(src_lat, src_lon, _t)
        _sun_fit = {"azimuth_deg": float(_az), "elevation_deg": float(_el), "n": 0,
                    "contrast": None, "capture_time_utc": _t.strftime("%Y-%m-%d %H:%M"),
                    "assumed_time": True}
        print(f"      sun from ephemeris at {capture_utc} UTC (date from imagery metadata, "
              f"pass time assumed): azimuth {_az:.0f} deg, elevation {_el:.0f} deg")

    normal_surface = None
    if canopy_height_map is None and _prefetch.get("chm") is not None:
        try:
            canopy_height_map = _prefetch["chm"].result(timeout=60)
        except Exception:
            canopy_height_map = None
        if canopy_height_map is not None:
            print(f"      tree heights from the Meta/WRI canopy map (CHMv2): "
                  f"p90 {np.percentile(canopy_height_map[canopy_height_map > 1], 90) if (canopy_height_map > 1).any() else 0:.1f} m")
        else:
            print("      canopy map unavailable -- trees from the image alone")
    if canopy_height_map is not None:
        # Real per-pixel vegetation height (Meta HighResCanopyHeight / CHMv2,
        # DINOv2 backbone, US-LiDAR-trained) replaces the scene's own noisy
        # monocular ndsm for vegetation pixels only. This is why Karnataka's
        # canopy used to be dropped entirely (skip_canopy): the monocular dsm
        # had no real depth signal for vegetation, just noise, and no cap
        # could rescue it. CHM gives an independent, real height estimate --
        # out-of-domain for Indian vegetation (US-forest-trained, unverified
        # numerically here, only visually checked against two real crops) but
        # not noise, so it replaces dsm-ground rather than filtering it.
        _chm = canopy_height_map
        if _chm.shape != dsm.shape:
            _chm = cv2.resize(_chm.astype(np.float32), (dsm.shape[1], dsm.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        _veg = seg_labels == seg.CLASS_IDX["vegetation"]
        if canopy_mask_from_chm:
            # A validated canopy-height PRODUCT already says where trees are
            # (0 over water, grass and bare ground); colour segmentation
            # misses shaded canopy, which would punch holes in a forest.
            _veg = _chm > 1.0
        dsm[_veg] = ground[_veg] + np.maximum(_chm[_veg], 0.0)
        ndsm = np.maximum(dsm - ground, 0.0)
        if continuous_vegetation:
            # Opt-in. The first versions of this looked wrong (history below);
            # the height-banded smoothing further down is what fixed it, and
            # it is now the preferred look. The idea: fold vegetation into the ground mesh's
            # own displacement instead of separate tree objects, since only
            # buildings have real vertical walls to smear into a ramp under
            # continuous displacement. Real per-pixel CHM drives the bump,
            # smoothed at a real crown-radius sigma (0.32x height, the same
            # ratio build_trees uses) specifically to fix an early version's
            # aliased needle-spikes. Even smoothed, a real rendered test
            # still reads as jagged mountain ridges, not canopy -- a coarse
            # heightfield mesh cannot fake foliage silhouette the way a
            # dedicated crown primitive can, and no further smoothing sigma
            # fixes that; it is the wrong tool for this specific subject, not
            # a tuning problem. build_trees' individual cone instances are
            # the one that actually looks right for vegetation. Terrain
            # itself is a heightfield for the same reason trees should not
            # be: bare earth has no vertical walls and no crown silhouette to
            # lose, so nothing here argues against continuous ground -- it
            # only argues against extending that same technique to trees.
            # ground_display is a SEPARATE array from `ground`: building base
            # heights, footprint discovery and the audit all keep reading
            # the unbumped `ground` regardless of this flag.
            # Raw per-pixel bumps aliased into needle spikes when downsampled
            # straight to the coarse mesh grid (confirmed visually: a real
            # test build rendered as sharp isolated rock-like peaks, not
            # canopy) -- a narrow real crown compressed to one grid vertex
            # with flat neighbours a metre or two away is a near-vertical
            # slope regardless of how fine the grid is. Real terrain
            # pipelines low-pass filter a heightfield before decimating it
            # for exactly this reason. The blur radius is the same real
            # crown-to-height ratio build_trees already uses (0.32x height),
            # so a bump spreads over roughly the footprint a real crown's
            # canopy actually covers, not just its single tallest pixel.
            # A SINGLE blur radius, sized from the scene-median vegetation
            # height, was the actual bug behind the "jagged mountain" look:
            # Karnataka's median vegetation pixel is short scrub (1.9 m), so
            # the one radius it produced (~1 m) was right for scrub and far
            # too narrow for the real, rare tall trees also in this scene
            # (15-20 m, ~1.7% of vegetation pixels) -- an isolated tall real
            # peak under-smoothed relative to its OWN height is exactly what
            # renders as a narrow spike instead of a proportioned crown.
            # Fix: bin vegetation by height and blur each band at ITS OWN
            # real crown radius (still the same 0.32x-height ratio), then
            # combine by max so a tall band's wider spread is never clipped
            # by a shorter band's narrower one where they overlap. Same
            # banding idea build_canopy already used for the blob-prism
            # path, applied here to a continuous smoothing radius instead of
            # a discrete extrusion height.
            _veg_vals = _chm[_veg]
            _bands_m = [1.0, 3.0, 6.0, 10.0, 15.0, float(np.nanmax(_veg_vals)) + 1.0] \
                if _veg_vals.size else [1.0]
            _bump_full = np.zeros(ground.shape, np.float32)
            _bump_full[_veg] = np.maximum(_veg_vals, 0.0)
            _bump_smooth = np.zeros(ground.shape, np.float32)
            _lo = 0.0
            if canopy_blur_m:
                # Closed canopy (a forest): one continuous surface already;
                # the per-band max below leaves steps at band boundaries there.
                _bump_smooth = cv2.GaussianBlur(_bump_full, (0, 0),
                                                sigmaX=max(0.5, canopy_blur_m / max(gsd, 1e-6)))
                _bands_m = []
            for _hi in _bands_m:
                _band_mask = _veg & (_chm >= _lo) & (_chm < _hi)
                if not _band_mask.any():
                    _lo = _hi
                    continue
                _band_h = float(np.median(_chm[_band_mask]))
                _radius_m = max(1.0, _band_h * crown_radius_ratio)
                _sigma_px = max(1.0, _radius_m / max(gsd, 1e-6))
                _band_full = np.where(_band_mask, _bump_full, 0.0).astype(np.float32)
                _band_smooth = cv2.GaussianBlur(_band_full, (0, 0), sigmaX=_sigma_px)
                _bump_smooth = np.maximum(_bump_smooth, _band_smooth)
                _lo = _hi
            # Keep the terrain flat only under the prisms actually built, with
            # a soft ~1 m falloff. Excluding the SEGMENTATION building class
            # instead left hard cliffs wherever segmentation said "building"
            # but no prism exists to hide the step (measured visually on
            # HITEC City: straight-edged walls cutting through groves).
            _prism_mask = np.zeros(ground.shape, np.uint8)
            for _fp in footprints:
                cv2.fillPoly(_prism_mask, [np.asarray(_fp, np.int32)], 1)
            _feather_px = max(1.0, 1.0 / max(gsd, 1e-6))
            _keep = 1.0 - np.clip(cv2.GaussianBlur(
                _prism_mask.astype(np.float32), (0, 0), sigmaX=_feather_px) * 2.0, 0.0, 1.0)
            _keep[_prism_mask > 0] = 0.0
            ground_display = ground + _bump_smooth * _keep
            # Leaf-clump scale detail for the NORMAL MAP only: the mesh stays
            # on the smoothed crown surface, but per-texel lighting follows the
            # real CHM at ~0.6 m, so crowns shade as clumped foliage rather
            # than a melted blob.
            _detail = cv2.GaussianBlur(np.where(_veg, np.maximum(_chm, 0.0), 0.0)
                                       .astype(np.float32), (0, 0),
                                       sigmaX=max(1.0, 0.6 / max(gsd, 1e-6)))
            normal_surface = np.maximum(ground_display,
                                        ground + 0.5 * (_bump_smooth + _detail) * _keep)
            cverts, cfaces, n_canopy = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
            _veg_h_med = float(np.nanmedian(_veg_vals)) if _veg_vals.size else 0.0
            print(f"      vegetation folded into the continuous ground mesh, "
                  f"height-banded smoothing (median height {_veg_h_med:.1f} m, "
                  f"max {float(np.nanmax(_veg_vals)) if _veg_vals.size else 0:.1f} m) "
                  f"-- no separate tree objects")
        else:
            ground_display = ground
            # Individual trees, not a merged blob: a real CHM gives a genuine
            # per-pixel height raster, so individual local maxima in it are real
            # tree crowns, not an invented distribution -- the blob-per-patch
            # path (build_canopy) exists for scenes with no CHM, where per-pixel
            # height is monocular noise and only a coarse, capped blob is honest.
            cverts, cfaces, n_canopy = city_model.build_trees(
                seg_labels, _chm, ground, gsd, gsd)
            print(f"      {n_canopy} individual trees placed from real CHM local "
                  f"maxima (median canopy height "
                  f"{np.nanmedian(_chm[_veg]) if _veg.any() else 0.0:.1f} m)")
    elif skip_canopy:
        # This scene's canopy depth signal is noise-dominated (measured on
        # Karnataka kn_a: 95th percentile of vegetation height was 23.3 m,
        # contaminated well past any per-patch cap worth applying) and the
        # narrow-footprint taper reads as spikes regardless of height. Rather
        # than ship a wrong-looking canopy, this scene has none -- an honest
        # gap, not a wrong claim.
        ground_display = ground
        cverts, cfaces, n_canopy = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    else:
        ground_display = ground
        cverts, cfaces, n_canopy = city_model.build_canopy(
            seg_labels, dsm, ground, gsd, gsd, min_area_px=120)
    if terrain_from_dem:
        # Landscape mode: water lies in the DEM terrain itself (textured, and
        # near-mirror via the roughness map); the separate segmentation-driven
        # water mesh came out blocky and off the real shoreline there.
        wverts, wfaces, n_water = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    else:
        wverts, wfaces, n_water = city_model.build_water(seg_labels, ground, gsd, gsd)
    if terrain_from_dem:
        # Landscape mode: the vehicle detector fires on canopy texture in a
        # forest (thousands of phantom "cars"), so it is not run there.
        vverts, vfaces, n_veh = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    else:
        vverts, vfaces, n_veh = city_model.detect_vehicles(image_np, seg_labels, ground, gsd)

    gsmall, _ = mg._resize_for_mesh(ground_display, seg_labels, ground_grid)
    gscale = ground_display.shape[0] / gsmall.shape[0]
    gverts, guvs, gfaces = mg.build_ground_mesh(gsmall, gsd * gscale, gsd * gscale)
    if _valid is not None:
        # Cut the ground where the input had no image, so a nodata collar is a
        # hole rather than a flat slab. A face stays only if all three corners
        # are on image.
        _vm = cv2.resize(_valid.astype(np.uint8), (gsmall.shape[1], gsmall.shape[0]),
                         interpolation=cv2.INTER_NEAREST).ravel() > 0
        _n0 = len(gfaces)
        gfaces = gfaces[_vm[gfaces].all(axis=1)]
        print(f"      ground mesh: {_n0 - len(gfaces)} faces cut over nodata")

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
        if gt_building_mask is not None:
            # Score against the real ground truth, not the stock classifier --
            # otherwise the audit grades GT-driven footprints against the very
            # segmentation they were built specifically to bypass.
            _audit_truth = gt_building_mask
            if _audit_truth.shape != ndsm.shape:
                _audit_truth = cv2.resize(
                    _audit_truth.astype(np.uint8), (ndsm.shape[1], ndsm.shape[0]),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            _audit_truth = seg_labels == seg.CLASS_IDX["building"]
        if not binfo["buildings"] and not _audit_truth.any():
            # Nothing to audit (a landscape with no buildings): report that,
            # rather than scoring an empty model as a failed one.
            _audit = {"verdict": "N/A", "note": "no buildings in scene -- building audit not applicable"}
            print("      model-vs-image audit: not applicable (no buildings in scene)")
        else:
            _audit = model_audit.audit(
                footprints, binfo, image_np, ndsm,
                _audit_truth,
                sun_azimuth_deg=sun_az)
            print(model_audit.report(_audit))
    except Exception as _e:
        _audit = {"error": str(_e)}
        print(f"      model-vs-image audit failed: {_e}")

    # Layer 2: an independent VLM judge, opt-in (real API cost per call) and
    # reporting-only -- see vlm_audit.py's module docstring for why it never
    # writes back into dsm/heights/footprints. Off by default so it never
    # fires silently on a build that didn't ask for it.
    if vlm_judge:
        try:
            import vlm_audit
            _vaudit = vlm_audit.audit(
                footprints, binfo, image_np, ndsm.shape,
                out_dir=os.path.dirname(stem) or ".",
                name=os.path.basename(stem))
            print(vlm_audit.report(_vaudit))
        except Exception as _e:
            print(f"      VLM judge audit skipped: {_e}")

    # White balance first, then grade. The previous version boosted saturation
    # and applied an explicit cool gain to imagery that was ALREADY blue from
    # atmospheric scattering, taking R-B from -27.6 to -48.3 and saturation from
    # 30.7 to 49.9 -- a grey-blue monochrome scene. See image_grade.py.
    graded = image_grade.grade(image_np)
    Image.fromarray(graded).save(stem + "_texture.png")

    # SLOPE + HEIGHT HEATMAP layers (the viewer's SLOPE / HEATMAP buttons),
    # same files/meta build_city.py writes. Slope is metric, on the bare-earth
    # surface (`ground`; in landscape mode that IS the DEM terrain).
    slope_stats = None
    try:
        import terrain_analysis as _terra
        _slope = _terra.slope_degrees(ground, gsd)
        slope_stats = _terra.slope_stats(_slope)
        Image.fromarray(_terra.colourise_slope(_slope)).save(stem + "_slope.png")
        print(f"      slope: median {slope_stats['median_deg']}deg, "
              f"{slope_stats['frac_flat']*100:.0f}% flat, "
              f"{slope_stats['frac_steep']*100:.0f}% steep (>15deg)")
    except Exception as _e:
        print(f"      slope layer failed: {_e}")
    try:
        mg._save_heatmap_texture(dsm, stem + "_heatmap.png")
    except Exception as _e:
        print(f"      heatmap layer failed: {_e}")

    if pbr_maps:
        # Baked at full texture resolution from the same surface the ground
        # mesh displaces (vegetation bumps included), so per-texel shading
        # resolves crown shape far below the mesh's vertex spacing. Normal
        # strength 1.0: the relief here is real metres, not exaggerated.
        import terrain_maps as _tm
        mg._save_normal_map(normal_surface if normal_surface is not None else ground_display,
                            gsd, gsd, stem + "_normal.png", strength=1.0)
        # Physical (unstretched) sky-view factor with a light curvature term:
        # the stretched default maps this mostly-open scene's small occluded
        # tail to near-black, which painted every tree clump as an ink blot.
        _ao_surf = ground_display.copy()
        _pm = np.zeros(ground_display.shape, np.uint8)
        for _fp in footprints:
            cv2.fillPoly(_pm, [np.asarray(_fp, np.int32)], 1)
        _ao_surf[_pm > 0] = dsm[_pm > 0]
        _pbr = _tm.bake_terrain_maps(_ao_surf, seg_labels, gsd,
                                     ao_path=stem + "_ao.png",
                                     roughness_path=stem + "_roughness.png",
                                     quality="high", stretch=False,
                                     curvature_weight=0.2)
        print(f"      PBR maps baked: normal + AO (svf mean "
              f"{_pbr['svf_mean_physical']:.3f}) + roughness")

    _tex_display = None
    if display_texture_path:
        # A higher-resolution REAL image of exactly the same bounds (e.g. the
        # zoom-19 tiles of the same Web Mercator grid): same grade as the
        # working texture, used only for rendering -- all measurements keep
        # the working image.
        _hi = np.array(Image.open(display_texture_path).convert("RGB"))
        Image.fromarray(image_grade.grade(_hi)).save(stem + "_texture_display.jpg", quality=92)
        _tex_display = {"method": f"native higher-resolution imagery "
                                  f"({_hi.shape[1]}x{_hi.shape[0]}, same bounds)",
                        "scale": 1, "measurements_use": "working-resolution pixels"}
        print(f"      display texture: real {_hi.shape[1]}x{_hi.shape[0]} imagery, same bounds")
    elif sr_texture:
        # Display-only: synthesised detail for close-range rendering. The GLB,
        # footprints, colours and audit all keep the original pixels.
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
        import sr_texture as _sr
        _t_sr = time.time()
        Image.fromarray(_sr.upscale(graded)).save(stem + "_texture_display.jpg", quality=92)
        _tex_display = {"method": "Real-ESRGAN x2plus (display only)", "scale": 2,
                        "measurements_use": "original pixels"}
        print(f"      display texture super-resolved x2 (Real-ESRGAN) in "
              f"{time.time() - _t_sr:.0f}s -- display only")
    import io
    tex = io.BytesIO(); Image.fromarray(graded).save(tex, format="PNG")

    export_glb(stem + ".glb", gverts, guvs, gfaces, bverts, bfaces,
               texture_bytes=tex.getvalue(), building_uvs=binfo.get("uvs"),
               building_colors=binfo.get("colors"),
               extra_meshes=[
                   ("canopy", cverts, cfaces, (0.31, 0.44, 0.28, 1.0)),
                   ("water", wverts, wfaces, (0.18, 0.37, 0.53, 1.0)),
                   ("vehicles", vverts, vfaces, (0.65, 0.70, 0.75, 1.0)),
               ])

    _structs = None
    if osm_structures and src_transform is not None and src_crs:
        # Flyovers, metro viaducts, skywalks: real OSM centrelines, deck
        # heights measured from this scene's own surface (see osm_structures).
        import osm_structures as _osm
        from glb_export import export_meshes_glb
        # src_transform already describes the working grid (rescaled at load).
        _fw, _fh = W, H
        _sx = _sy = 1.0
        _inv = ~src_transform
        _to_src = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        _to_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

        def _ll2px(lon, lat):
            X, Y = _to_src.transform(lon, lat)
            c, r = _inv * (X, Y)
            return c * _sx, r * _sy
        _b = [src_transform * p for p in [(0, 0), (_fw, _fh)]]
        (_lo0, _la1), (_lo1, _la0) = [_to_ll.transform(*p) for p in _b]
        try:
            _ways = _osm.fetch_bridges(_lo0, _la0, _lo1, _la1, cache_path=stem + "_osm_bridges.json")
            _occ = np.zeros(dsm.shape[:2], np.uint8)
            for _fp in footprints:
                cv2.fillPoly(_occ, [np.asarray(_fp, np.int32)], 1)
            _st = _osm.build_structures(
                _ways, _ll2px, dsm, ground, gsd, graded,
                sun_az=_sun_fit["azimuth_deg"] if _sun_fit else None,
                sun_el=_sun_fit["elevation_deg"] if _sun_fit else None,
                lean_px_per_m=_al["lean_px_per_m"] if _al else None, occupied=_occ)
            dv_, dn_, duv_, df_ = _st["deck"]
            bv_, bn_, bc_, bf_ = _st["body"]
            export_meshes_glb(stem + "_structures.glb",
                              [("struct_deck", dv_, dn_, "TEXCOORD_0", duv_, df_),
                               ("struct_body", bv_, bn_, "COLOR_0", bc_, bf_)])
            _structs = dict(_st["report"], file="structures.glb",
                            source="OpenStreetMap bridge/viaduct ways (geometry, lanes); deck "
                                   "heights measured from each deck's own shadow under the "
                                   "fitted sun where constrained, else class-typical; piers "
                                   "inferred at class-typical spans")
            r_ = _st["report"]
            print(f"      elevated structures: {r_['structures']} ({r_['by_class']}), "
                  f"{r_['length_m']:.0f} m of deck; height from own shadow on "
                  f"{r_.get('height_shadow', 0)} ({r_.get('segments_shadow_measured', 0)}/"
                  f"{r_.get('segments_total', 0)} segments measured), from surface on "
                  f"{r_['height_measured']}, class-typical on {r_['height_default']}; "
                  f"{r_['piers']} piers (inferred spacing)")
        except Exception as _e:
            print(f"      elevated structures skipped: {_e}")
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
    # A DSM (as opposed to an rDSM) must hold ABSOLUTE elevation: a metric
    # vertical scale alone is not enough if no external datum was reached
    # (DEM fetch failed / refused, no terrain_from_dem) -- that surface is
    # height above local ground, i.e. relative to an arbitrary zero.
    _abs_datum = _metric and (tier[0] == "A" or _terrain_datum is not None
                              or _datum_fit is not None)
    _suffix = "_DSM.tif" if _abs_datum else "_rDSM.tif"
    # Absolute elevation for the exported DSM: Tier A already carries the DEM
    # datum; otherwise a georeferenced metric scene takes the DEM's median
    # offset over bare ground as its DATUM (shape agreement not required for
    # a datum), so the file holds elevation, not height above local ground.
    _product_agl = None
    # Real building heights fused in (Open Buildings) or given with the
    # footprints are better than the model where they exist, so those scenes
    # keep the fused surface; the raw field is the product where the model
    # alone produced the heights.
    _fused_real = bool(gt_building_polygons) or any(
        locals().get(_k) for _k in ("_n_ob_fused", "_n_ob_fused2", "_n_ob_fused3"))
    if (_agl_raw is not None and _metric and not _fused_real
            and "metric depth model" in (scale_source or "") + str(tier)):
        _product_agl = np.maximum(_agl_raw if _agl_raw.shape == dsm.shape else
                                  cv2.resize(_agl_raw, (dsm.shape[1], dsm.shape[0]),
                                             interpolation=cv2.INTER_AREA), 0.0).astype(np.float32)
    # Surface = the pipeline's ground (which carries the datum work above)
    # plus the model's own height above ground, when that is what we have.
    _dsm_base = (np.minimum(dsm, ground) + _product_agl) if _product_agl is not None else dsm
    _dsm_out, _datum = _dsm_base, ("DEM-anchored (Tier A)" if tier[0] == "A" else "local ground = 0")
    if _metric and _terrain_datum is not None:
        _dsm_out = _dsm_base + _terrain_datum
        _datum = "Copernicus GLO-30 (EGM2008 geoid), terrain_from_dem"
    elif _metric and tier[0] != "A" and _datum_fit is not None:
        _dsm_out = _dsm_base + float(_datum_fit["offset_m"])
        _datum = (f"Copernicus/SRTM median offset {_datum_fit['offset_m']:+.2f} m over "
                  f"{_datum_fit.get('n', 0)} bare-ground px (datum only)")
    # The working grid's exact affine. src_transform was rescaled to the
    # working grid at load, so it is used as-is. The HORIZONTAL georeference
    # is real whenever the source had one, so it is written for an rDSM too;
    # only the vertical is relative there. A CRS without a geotransform is a
    # file no GIS can place, which is what this used to write.
    if _valid is not None and _dsm_out.shape == _valid.shape:
        _dsm_out = np.where(_valid, _dsm_out, np.nan).astype(np.float32)   # written as nodata
    _georef = src_transform is not None and src_crs is not None \
        and dsm.shape == (H, W)
    _tf_out = src_transform if _georef else None
    # FULL-RESOLUTION PRODUCT. A Cartosat scene is many times the 3D scene's
    # working grid; scoring a DSM that was averaged down to it throws away the
    # resolution the evaluation is at. The model's height above ground is
    # re-run on the source pixels; the ground under it (DEM-anchored, smooth
    # at 30 m) is resampled from the working grid. The 3D scene is unchanged.
    #
    # The model is run at the scale it was trained at: a 0.6 m source is
    # upsampled 2x first. Measured on the 18 held-out DFC2019 tiles
    # (BENCHMARK_RESOLUTION.md), 0.6 m costs +2.9 % per-building RMSE this way
    # and +7.3 % fed at its own pixel count. Bounded by FULLRES_MAX_MPX so a
    # huge scene still finishes on a CPU; the zoom shrinks to fit.
    _full = None
    _anchor_info = None
    if _product_agl is not None:
        # The product's height above ground: the trained-scale pass on the
        # source grid when there was one, else the working grid's own field.
        _agl_p = _agl_full if _agl_full is not None else _product_agl
        _shape_p = _agl_p.shape
        _is_full = _agl_full is not None and _shape_p != _product_agl.shape
        _valid_p = _valid_src if _is_full else _valid
        # ABSOLUTE SURFACE (anchoring.py): terrain from the global DEM, not the
        # model's own flat "ground" -- a metric model predicts height above
        # ground and knows nothing of the hill it stands on. Measured against
        # airborne LiDAR elevations in scripts/lidar_landscapes.py.
        _dem_w = d["dem"] if (d is not None and d.get("dem") is not None) else None
        if _dem_w is None and _metric and _georef and (use_dem or dem_path) and src_is_projected_m                 and not terrain_from_dem:
            try:
                _dd = _sample_dem()
                _dem_w = _dd.get("dem")
            except Exception:
                _dem_w = None
        if (_dem_w is not None and _metric and _georef and not terrain_from_dem
                and not os.environ.get("DW_OLD_ANCHOR")):
            import anchoring as _anc
            _k = _anc.DEFAULT_K.get(dem_source, 0.5)
            _dem_p = np.asarray(_dem_w, np.float32)
            if _dem_p.shape != _shape_p:
                _dem_p = cv2.resize(_dem_p, (_shape_p[1], _shape_p[0]), interpolation=cv2.INTER_LINEAR)
            _gsd_p = (_gsd_src if _is_full else _gsd_work) or px_m or 1.0
            _dsm_new, _ = _anc.absolute_surface(_dem_p, _agl_p, _gsd_p, _k)
            _datum = (f"{dem_source.upper()} terrain (EGM2008/EGM96 as published) at 30 m, "
                      f"minus {_k:.2f} x model height (DEM surface fraction); + model height")
            _anchor_info = {"method": "dem_terrain_plus_model_height", "dem": dem_source, "k": _k}
            _abs_datum = True
            _suffix = "_DSM.tif"
        else:
            _base_lo = (_dsm_out - _product_agl).astype(np.float32)
            _base_lo = np.where(np.isfinite(_base_lo), _base_lo, np.nanmedian(_base_lo))
            _dsm_new = (cv2.resize(_base_lo, (_shape_p[1], _shape_p[0]), interpolation=cv2.INTER_LINEAR)
                        if _base_lo.shape != _shape_p else _base_lo) + _agl_p
        if _valid_p is not None and _valid_p.shape == _shape_p:
            _dsm_new = np.where(_valid_p, _dsm_new, np.nan)
        _dsm_out = _dsm_new.astype(np.float32)
        if _is_full:
            _full = {"shape": _shape_p, "tf": _tf_src if _georef else None}
            print(f"      DSM product at the source grid: {_shape_p[1]}x{_shape_p[0]} "
                  f"(3D scene uses {W}x{H})")
    _tf_dsm = _full["tf"] if _full else _tf_out
    try:
        _dx.export_dsm_geotiff_affine(
            _dsm_out, stem + _suffix,
            transform=_tf_dsm, crs=src_crs if _georef else None,
            tags={"TIER": tier, "HEIGHT_IS_METRIC": str(_metric),
                  "PRODUCT": "DSM" if _abs_datum else "rDSM",
                  "VERTICAL_DATUM": (_datum if _abs_datum else
                                     "local ground = 0 (no absolute datum reached)"
                                     if _metric else "relative"),
                  "SOURCE": os.path.basename(image_path),
                  "PIPELINE": "DepthWizard"})
        print(f"      {'DSM' if _abs_datum else 'rDSM'} written: "
              f"{os.path.basename(stem)}{_suffix}"
              f"{'' if _abs_datum else '  (no absolute datum)'}"
              f"{'' if _georef else '  (no CRS)'}")
    except Exception as _e:
        print(f"      DSM export failed: {_e}")

    # PER-PIXEL UNCERTAINTY (1 sigma, metres).
    #
    # The spread of the model across 4 rotated views, mapped to metres by a
    # calibration fitted on DFC2019 validation views and checked on held-out
    # ones (checkpoints/height_best/sigma_calibration.json):
    #     sigma = sqrt((k * tta_std)^2 + s0^2)
    # Computed at <= 1024 px to keep uploads quick, then upsampled.
    _sigma, _sigma_info = None, None
    _cal_p = os.path.join(DepthBackbone.METRIC_CKPT, "sigma_calibration.json")
    # Needs a metric height field, not a georeference: a PNG's heights are
    # metres too, and its cursor must show the same +/-.
    if ((_metric or "metric depth model" in (scale_source or "")) and _bb.is_metric
            and os.path.exists(_cal_p) and not os.environ.get("DW_NO_SIGMA")):
        try:
            _cal = json.load(open(_cal_p))
            _sw = min(1.0, 1024.0 / max(W, H))
            _ps = pil.resize((max(64, int(W * _sw)), max(64, int(H * _sw))), Image.BILINEAR)
            _, _std = _bb.predict_tta(_ps, raw_metric=True, views=4)
            _std = cv2.resize(_std.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            # Height term (scripts/fit_sigma.py): error grows with the height
            # predicted -- tall structures are under-called -- so sigma does too.
            _hgt = np.maximum(np.nan_to_num(_agl_raw if _agl_raw is not None else height), 0.0)
            if _hgt.shape != _std.shape:
                _hgt = cv2.resize(_hgt.astype(np.float32), (_std.shape[1], _std.shape[0]),
                                  interpolation=cv2.INTER_AREA)
            _sigma = np.sqrt((_cal["k"] * _std) ** 2 + _cal["s0"] ** 2
                             + (_cal.get("c_height", 0.0) * _hgt) ** 2).astype(np.float32)
            if _valid is not None and _sigma.shape == _valid.shape:
                _sigma = np.where(_valid, _sigma, np.nan).astype(np.float32)
            _sigma_info = {"file": os.path.basename(stem) + "_sigma.tif",
                           "definition": "1-sigma height uncertainty, metres",
                           "method": f"4-view rotation spread, calibrated sqrt(({_cal['k']:.2f}*std)^2 + "
                                     f"{_cal['s0']:.2f}^2 + ({_cal.get('c_height', 0.0):.2f}*height)^2) on {_cal.get('fit_views')} DFC2019 val views",
                           "heldout_ece": round(float(_cal.get("ece_heldout", float("nan"))), 3),
                           "heldout_rank_corr": round(float(_cal.get("rank_corr", float("nan"))), 3),
                           "median_m": round(float(np.nanmedian(_sigma)), 2)}
            _sig_file = _sigma
            if _full:
                # Same grid as the DSM file, so the two overlay exactly in GIS.
                _sig_file = cv2.resize(np.nan_to_num(_sigma, nan=0.0),
                                       (_full["shape"][1], _full["shape"][0]),
                                       interpolation=cv2.INTER_LINEAR)
                if _valid_src is not None and _valid_src.shape == _sig_file.shape:
                    _sig_file = np.where(_valid_src, _sig_file, np.nan)
                _sig_file = _sig_file.astype(np.float32)
            _dx.export_dsm_geotiff_affine(
                _sig_file, stem + "_sigma.tif", transform=_tf_dsm,
                crs=src_crs if _georef else None,
                tags={"PRODUCT": "height uncertainty (1 sigma, metres)",
                      "METHOD": _sigma_info["method"], "PIPELINE": "DepthWizard"})
            print(f"      uncertainty written: {_sigma_info['file']} "
                  f"(median +/-{_sigma_info['median_m']} m)")
        except Exception as _e:
            _sigma = None
            print(f"      uncertainty skipped: {_e}")

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
    # Pair prisms to discovery records SPATIALLY, never by position: the two
    # lists come from different extractors with different lengths.
    _paired = _pair_prisms_to_instances(
        binfo["buildings"], disc.get("instances") or [], footprints)
    _n_paired = sum(1 for r in _paired if r is not None)
    if len(binfo["buildings"]) != len(disc.get("instances") or []):
        print(f"      pairing: {_n_paired} of {len(binfo['buildings'])} prisms "
              f"matched to a discovery record "
              f"({len(disc.get('instances') or [])} records available)")

    from calibration import conformal as _cf
    _cal = _cf.load_fitted()
    _metric_tier = tier[0] in ("A", "B")
    _iv_half, _iv_nominal = {}, (None if _cal is None else 1.0 - _cal["alpha"])
    if _cal is not None and _metric_tier:
        for _i, prism in enumerate(binfo["buildings"]):
            rec = _paired[_i]
            _u = None
            if _cal.get("normalised"):
                _u = 1.0 / max(float((rec or {}).get("confidence") or 0.5), 0.05)
            _lo, _hi, _hw = _cf.interval(prism["height_m"], _cal, uncertainty=_u)
            if _hw is not None:
                _iv_half[_i] = round(float(_hw), 2)
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
    # Geometry comes from the footprint that was actually EXTRUDED, not from
    # the discovery contour. Those are different extractors: exporting the
    # discovery outline described buildings that are not the ones in the model,
    # so the GeoJSON the viewer and damage.py read did not match the mesh.
    for _i, prism in enumerate(binfo["buildings"]):
        rec = _paired[_i] or {}
        _pi = prism.get("poly_index")
        if _pi is None or _pi >= len(footprints):
            continue
        _cnt = np.asarray(footprints[_pi]).reshape(-1, 2)
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
                "id": rec.get("id", _i),
                "height_m": round(prism["height_m"], 2),
                "interval_half_width_m": _iv_half.get(_i),
                "coverage_nominal": _iv_nominal if _i in _iv_half else None,
                # Evidence fields are null on a prism that matched no discovery
                # record. That is the honest export: the evidence genuinely does
                # not exist for it, and carrying a neighbour's would be worse
                # than carrying nothing.
                "area_m2": rec.get("area_m2"),
                "perimeter_m": rec.get("perimeter_m"),
                "size_class": rec.get("size_class"),
                "confidence": rec.get("confidence"),
                "provenance": rec.get("provenance"),
                "evidence": rec.get("evidence"),
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
        for _i, prism in enumerate(binfo["buildings"]):
            rec = _paired[_i] or {}
            _e = rec.get("evidence") or {}
            _w.writerow([rec.get("id", _i), round(prism["height_m"], 2),
                         _iv_half.get(_i, ""),
                         _iv_nominal if _i in _iv_half else "",
                         rec.get("area_m2", ""), rec.get("perimeter_m", ""),
                         rec.get("size_class", ""), rec.get("confidence", ""),
                         rec.get("provenance", ""), _metric_tier,
                         _e.get("height", ""), _e.get("edge", ""),
                         _e.get("texture", ""), _e.get("shadow", "")])
    print(f"      per-building export: {len(_feats)} buildings to GeoJSON + CSV"
          + ("" if src_transform is not None else "  (pixel coords, no CRS)"))

    print(f"[5/5] {len(gfaces)} ground + {len(bfaces)} building faces, "
          f"{os.path.getsize(stem + '.glb')/1e6:.1f} MB")
    _tr.leave()

    meta = {
        "source_image": os.path.abspath(image_path),
        "resolution": [W, H],
        "nodata_fraction": (round(float((~_valid).mean()), 5) if _valid is not None else 0.0),
        "anchoring": _anchor_info,
        "bands": ({"rgb": list(_band_idx), "how": _band_how}
                  if "_band_how" in locals() else None),
        "tier": tier,
        "scale_source": scale_source,
        "gsd_m": px_m,
        "sun_azimuth_deg": (_sun_fit["azimuth_deg"] if _sun_fit else
                            round(sun_az, 1) if sun_az is not None else None),
        "sun_elevation_deg": (_sun_fit["elevation_deg"] if _sun_fit
                              else shadow_calib.get("elevation_deg")),
        "capture_time_utc": _sun_fit.get("capture_time_utc") if _sun_fit else None,
        # Local MEAN solar hour of the pass (UTC + longitude/15), for the
        # viewer's time-of-day slider to mark where the capture sits.
        "capture_local_solar_hour": (
            round((int(_sun_fit["capture_time_utc"][11:13]) + int(_sun_fit["capture_time_utc"][14:16]) / 60
                   + src_lon / 15.0) % 24, 2)
            if _sun_fit and _sun_fit.get("capture_time_utc") and src_lon is not None else None),
        "sun_note": ("solar ephemeris at the capture date with an assumed pass time"
                     if (_sun_fit and _sun_fit.get("assumed_time")) else
                     f"fitted to the shadows of {_sun_fit['n']} buildings with known heights"
                     + (f" (solar ephemeris, capture {_sun_fit['capture_time_utc']} UTC)"
                        if _sun_fit.get("capture_time_utc") else "") if _sun_fit else
                     "elevation from " + shadow_calib["elevation_source"]
                    if shadow_calib.get("elevation_source")
                    else "elevation not recoverable from a single image without a "
                         "known height or capture time+location"),
        "crs": src_crs,
        # Source-CRS map units per ground metre at the scene centre (1 for
        # UTM; ~1/cos(lat) for Web Mercator when the scale was corrected).
        # World/mesh x/z are ground metres, so a map offset = world * this.
        "crs_units_per_m": round(_crs_units_per_m, 6),
        # The affine transform (rasterio's 6-coefficient convention: a,b,c,
        # d,e,f, applied as easting = a*col + b*row + c, northing = d*col +
        # e*row + f) was already used above to place building footprints in
        # real map coordinates for the GeoJSON export, but was never itself
        # exported -- so the viewer had no way to convert a point it hovers
        # on the mesh back into a real coordinate. Only meaningful alongside
        # a non-null crs; a scene with no georeferencing carries this as null
        # rather than a transform that would silently imply one exists.
        "affine_transform": (
            [src_transform.a, src_transform.b, src_transform.c,
             src_transform.d, src_transform.e, src_transform.f]
            if src_transform is not None else None),
        "slope": slope_stats,
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
        # World-space roof displacement per metre of height (x, z), measured
        # from the image's own relief displacement. The viewer uses it to find
        # each sensor-facing wall's real facade pixels in the texture.
        "texture_display": _tex_display,
        "lean_m_per_m": (None if not _ob_info.get("alignment") else [
            _ob_info["alignment"]["lean_px_per_m"][0] * gsd,
            -_ob_info["alignment"]["lean_px_per_m"][1] * gsd]),
        # Reference-height (Open Buildings) scale calibration: applied, or
        # skipped with the gate's reason (metric depth, see the helper).
        "height_calibration": _ob_info.get("calibration"),
    }
    if return_arrays:
        assert not stage, "return_arrays=True requires stage=False (meta is not JSON-safe with an array in it)"
        meta["_ndsm"] = _product_agl if _product_agl is not None else ndsm
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
                      "terrain_metalness.png", "terrain_heatmap.png", "terrain_slope.png"):
            p = os.path.join(VIEWER_DIR, stale)
            if os.path.exists(p):
                os.remove(p)
        for _suf, _dst in (("_slope.png", "terrain_slope.png"),
                           ("_heatmap.png", "terrain_heatmap.png")):
            if os.path.exists(stem + _suf):
                shutil.copy2(stem + _suf, os.path.join(VIEWER_DIR, _dst))
        # the viewer's copy is meshopt-compressed (~1/3 the bytes, same geometry
        # to ~2 cm); the full-precision GLB stays in final_out/
        import glb_compress
        _how = glb_compress.compress(stem + ".glb", os.path.join(VIEWER_DIR, "terrain.glb"))
        print(f"      3D model for the viewer: {_how} "
              f"({os.path.getsize(os.path.join(VIEWER_DIR, 'terrain.glb')) / 1e6:.0f} MB)")
        # The PNG texture is always staged (other tools read it); a display
        # texture ships as JPEG -- photographic content, ~10x smaller -- and
        # scene.json names it so the viewer loads that instead.
        shutil.copy2(stem + "_texture.png", os.path.join(VIEWER_DIR, "terrain_texture.png"))
        _disp_dst = os.path.join(VIEWER_DIR, "terrain_texture_display.jpg")
        if os.path.exists(_disp_dst):
            os.remove(_disp_dst)
        if _tex_display:
            shutil.copy2(stem + "_texture_display.jpg", _disp_dst)
            meta["texture_file"] = "terrain_texture_display.jpg"
        _ctx_dst = os.path.join(VIEWER_DIR, "terrain_context.jpg")
        if os.path.exists(_ctx_dst):
            os.remove(_ctx_dst)
        if context_image and src_transform is not None:
            # REAL surrounding imagery (flat, not modelled), placed by its true
            # map coordinates. World x/z are metres from this tile's top-left
            # corner in the source CRS (x east, z = northing - top).
            _cm = json.load(open(context_image + ".json"))
            _ctx = np.array(Image.open(context_image + ".png").convert("RGB"))
            Image.fromarray(image_grade.grade(_ctx)).save(_ctx_dst, quality=90)
            _ox, _oy = src_transform.c, src_transform.f
            _k = _crs_units_per_m   # source-CRS offsets -> ground metres
            meta["context"] = {
                "x0": (_cm["left"] - _ox) / _k, "x1": (_cm["right"] - _ox) / _k,
                "z0": (_cm["bottom"] - _oy) / _k, "z1": (_cm["top"] - _oy) / _k,
                "source": f"Esri World Imagery zoom {_cm['zoom']} -- real "
                          f"surroundings, flat, not modelled in 3D"}
        _far_dst = os.path.join(VIEWER_DIR, "terrain_context_far.jpg")
        if os.path.exists(_far_dst):
            os.remove(_far_dst)
        if context_far_image and src_transform is not None:
            _fm = json.load(open(context_far_image + ".json"))
            _far = np.array(Image.open(context_far_image + ".png").convert("RGB"))
            Image.fromarray(image_grade.grade(_far)).save(_far_dst, quality=88)
            meta["context_far"] = {
                "x0": (_fm["left"] - src_transform.c) / _crs_units_per_m,
                "x1": (_fm["right"] - src_transform.c) / _crs_units_per_m,
                "z0": (_fm["bottom"] - src_transform.f) / _crs_units_per_m,
                "z1": (_fm["top"] - src_transform.f) / _crs_units_per_m,
                "source": f"Esri World Imagery zoom {_fm['zoom']} -- real, flat, horizon context"}
        for _stale in ("structures.glb", "context_city.glb"):
            if os.path.exists(os.path.join(VIEWER_DIR, _stale)):
                os.remove(os.path.join(VIEWER_DIR, _stale))
        if _structs:
            shutil.copy2(stem + "_structures.glb", os.path.join(VIEWER_DIR, "structures.glb"))
            meta["structures"] = _structs
        if context_city_footprints and context_image and src_transform is not None:
            # LOD1 real buildings across the surrounding context imagery
            # (a lower, labelled tier -- see tools/build_context_city.py).
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
            import build_context_city as _bcc
            _cc = stem + "_context_city.glb"
            try:
                _bcc.main(image_path, context_image, context_city_footprints, _cc)
                shutil.copy2(_cc, os.path.join(VIEWER_DIR, "context_city.glb"))
                meta["context_city"] = dict(json.load(open(_cc.rsplit(".", 1)[0] + ".json")),
                                            file="context_city.glb")
            except Exception as _e:
                print(f"      context city skipped: {_e}")
        if _sigma is not None:
            # CONFIDENCE view: colour ramp of +/-sigma (0-8 m), plus a raw
            # channel the hover readout decodes as sigma = value/255*20 m.
            _sd = cv2.resize(np.nan_to_num(_sigma, nan=0.0), (min(W, 2048), min(H, 2048)),
                             interpolation=cv2.INTER_AREA)
            _ramp = cv2.applyColorMap(np.clip(_sd / 8.0 * 255, 0, 255).astype(np.uint8),
                                      cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(VIEWER_DIR, "terrain_confidence.png"), _ramp)
            cv2.imwrite(os.path.join(VIEWER_DIR, "terrain_confidence_raw.png"),
                        np.clip(_sd / 20.0 * 255, 0, 255).astype(np.uint8))
            meta["uncertainty"] = dict(_sigma_info, raw_encoding="sigma_m = value/255*20",
                                       ramp="turbo, 0-8 m")
        else:
            for _stale in ("terrain_confidence.png", "terrain_confidence_raw.png"):
                _sp = os.path.join(VIEWER_DIR, _stale)
                if os.path.exists(_sp):
                    os.remove(_sp)
        if pbr_maps:
            for _suf, _dst in (("_normal.png", "terrain_normal.png"),
                               ("_ao.png", "terrain_ao.png"),
                               ("_roughness.png", "terrain_roughness.png")):
                shutil.copy2(stem + _suf, os.path.join(VIEWER_DIR, _dst))
        # This path used to DELETE buildings.geojson here, because it never
        # produced one and a previous scene's file would otherwise be read as
        # this scene's. It produces one now, so it is staged rather than
        # removed -- that is what makes the viewer's per-building readout work
        # on an uploaded image.
        shutil.copy2(stem + "_buildings.geojson",
                     os.path.join(VIEWER_DIR, "buildings.geojson"))
        # Tell the viewer which elevation product /api/dsm will serve.
        if os.path.exists(stem + _suffix):
            meta["dsm_file"] = os.path.basename(stem + _suffix)
            meta["dsm_product"] = "DSM" if _abs_datum else "rDSM"
        # Cursor readout: the product's own height above ground and its
        # 1-sigma uncertainty, as exact rasters (hover_export.py).
        try:
            import hover_export as _hx
            _hv_agl = (_product_agl if _product_agl is not None else
                       (_agl_raw if _agl_raw is not None else None))
            if _hv_agl is not None:
                _hv_agl = np.asarray(_hv_agl, np.float32)
                if _hv_agl.shape != (H, W):
                    _hv_agl = cv2.resize(_hv_agl, (W, H), interpolation=cv2.INTER_AREA)
                if _valid is not None:
                    _hv_agl = np.where(_valid, _hv_agl, np.nan)
                meta["hover"] = _hx.write(VIEWER_DIR, _hv_agl, _sigma, locals().get("ground_display"))
                _centre = (_hx.lonlat_centre(src_transform, src_crs, W, H)
                           if (src_transform is not None and src_crs is not None) else None)
                meta["facts"] = _hx.scene_facts(_hv_agl, _sigma, px_m, {
                    "centre_lonlat": _centre,
                    "crs": str(src_crs) if src_crs is not None else None,
                    "elevation_reference": _datum if _abs_datum else None,
                    "pixel_assumed": True if (gsd_assumed and not _georef) else None,
                    # the uploaded file's pixel, not the x2 grid it is upsampled to
                    **({"pixel_m": round(float(_input_gsd_m), 3)} if _input_gsd_m else {}),
                    "model": json.load(open(os.path.join(DepthBackbone.METRIC_CKPT, "metric_ok.json")))
                    .get("checkpoint", "metric") if _bb.is_metric else "relative",
                })
        except Exception as _e:
            print(f"      hover rasters skipped: {_e}")
        with open(os.path.join(VIEWER_DIR, "scene.json"), "w") as f:
            json.dump(meta, f, indent=2)
        try:
            if meta.get("hover") and meta.get("affine_transform"):
                import hover_export as _hx
                _nb = _hx.burn_buildings(VIEWER_DIR, os.path.join(VIEWER_DIR, "buildings.geojson"),
                                         meta["affine_transform"], [W, H])
                print(f"      cursor readout: {_nb} buildings at their drawn heights")
        except Exception as _e:
            print(f"      building heights not written into the readout: {_e}")
        # The viewer samples roofs and walls where this image photographed
        # them (footprint + lean * height). The alignment lean above is the
        # lean between the footprint source's imagery and this image, so the
        # image's own displacement is measured separately from its towers.
        if image_path.lower().endswith((".tif", ".tiff")) and meta.get("buildings_extruded"):
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "scripts"))
                import measure_image_lean as _mil
                _mil.main(VIEWER_DIR, image_path)
            except (Exception, SystemExit) as _e:
                print(f"      image lean not measured: {_e}")
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
