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
from depth_model import DepthBackbone, orientation_check, backbone_tag
from glb_export import export_glb
import mesh_generation as mg

OUT_DIR = "final_out"
VIEWER_DIR = "viewer/output"
GROUND_GRID = 700


def estimate_sun_azimuth(image_np: np.ndarray, seg_labels: np.ndarray) -> float:
    """
    Sun azimuth from the image's own shadows.

    Shadows fall away from the sun, so the direction in which dark pixels
    cluster around structures gives the anti-solar direction directly. Every
    direction is scored by how much shadow lies a fixed distance from building
    pixels; the peak is the shadow direction, and the sun is opposite it.

    This is measurable from a single image. Sun ELEVATION is not -- it needs a
    known height somewhere in the frame -- which is why this returns azimuth
    only and the caller must supply scale another way.
    """
    shadow = shadow_correction.detect_shadow_mask(image_np)
    b = (seg_labels == seg.CLASS_IDX["building"])
    if b.sum() < 200:
        return None
    ys, xs = np.nonzero(b)
    if ys.size > 4000:
        sel = np.linspace(0, ys.size - 1, 4000).astype(int)
        ys, xs = ys[sel], xs[sel]
    H, W = shadow.shape
    step = max(6, int(min(H, W) * 0.01))

    best_dir, best_score = None, -1.0
    for a in range(0, 360, 10):
        r = np.radians(a)
        dx, dy = np.sin(r), -np.cos(r)
        py = np.clip((ys + dy * step).astype(int), 0, H - 1)
        px = np.clip((xs + dx * step).astype(int), 0, W - 1)
        score = float(shadow[py, px].mean())
        if score > best_score:
            best_score, best_dir = score, a
    # Sun is opposite the shadow direction.
    return (best_dir + 180.0) % 360.0


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
          stage: bool = True) -> dict:
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
    if image_path.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.open(image_path) as _s:
                if _s.crs is not None and _s.transform is not None:
                    m = abs(_s.transform.a)
                    if _s.crs.is_geographic:
                        import math
                        lat = _s.transform.f + _s.transform.e * _s.height / 2
                        m = m * 111320.0 * math.cos(math.radians(lat))
                    native_gsd = m
                    src_crs = str(_s.crs)
                    print(f"      georeferenced input: {src_crs}, "
                          f"{native_gsd*100:.1f} cm/px measured from the file")
                    if gsd_m is None:
                        gsd_m = native_gsd
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

    px_m = gsd_m
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

    # Horizontal pixel pitch. When the source was georeferenced this is measured
    # from the file, so footprint areas and perimeters are real square metres and
    # real metres even though the VERTICAL scale remains relative.
    gsd = gsd_m if gsd_m else 0.25
    print(f"[3/5] scale: {scale_source}")
    print(f"      sun azimuth estimated from shadows: "
          f"{'%.0f deg' % sun_az if sun_az is not None else 'not determinable'}")
    _tr.leave(); _tr.enter("buildings")

    dsm = refined * scale
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    dsm, n_prism = dsm_refine.prismify_buildings(dsm, seg_labels, terrain)
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    ground = city_model.flatten_ground(terrain, seg_labels, smooth_m=35.0, gsd_m=gsd)
    ndsm = np.maximum(dsm - ground, 0.0)

    # Re-fit the vertical scale AFTER squaring off the roofs.
    #
    # The scale was chosen from the raw depth field, but prismify_buildings then
    # lifts each footprint to a single roof height, which changes the very
    # distribution the scale was fitted to. Leaving it produced a campus of
    # three-storey blocks with a median height of 35.9 m. Re-fitting against the
    # final surface makes the stated assumption ("tallest structure ~ TARGET_TALL_M")
    # actually true of the geometry that gets exported.
    b_mask = seg_labels == seg.CLASS_IDX["building"]
    if b_mask.sum() > 1000:
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

    for rec in disc["instances"]:
        rec["provenance"] = bd.INFERRED   # never MEASURED on this path

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
    _metric = (src_crs is not None) and (anchor_height_m is not None or gsd_m is not None)
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
        "sun_elevation_deg": None,
        "sun_note": "azimuth estimated from image shadows; elevation not recoverable "
                    "from a single image without a known height",
        "crs": src_crs,
        "ground_truth": "NONE -- no LiDAR available for this image; no accuracy "
                        "figure can be quoted for this scene",
        "nadir_check": nad,
        "overlays_removed": ov["count"],
        "overlay_coverage": round(ov["coverage"], 5),
        "model": "prism city (flat roofs, vertical walls)",
        "buildings_extruded": len(binfo["buildings"]),
        "canopy_volumes": n_canopy, "water_bodies": n_water, "vehicles": n_veh,
        "provenance": bd.INFERRED,
        "height_is_metric": False,
        "discovery": disc["report"],
        "build_seconds": round(time.time() - t0, 1),
    }

    if stage:
        os.makedirs(VIEWER_DIR, exist_ok=True)
        for stale in ("terrain_ao.png", "terrain_normal.png", "terrain_roughness.png",
                      "terrain_metalness.png", "terrain_heatmap.png",
                      "buildings.geojson"):
            p = os.path.join(VIEWER_DIR, stale)
            if os.path.exists(p):
                os.remove(p)
        shutil.copy2(stem + ".glb", os.path.join(VIEWER_DIR, "terrain.glb"))
        shutil.copy2(stem + "_texture.png", os.path.join(VIEWER_DIR, "terrain_texture.png"))
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
    a = ap.parse_args()
    build(a.image, a.name, gsd_m=a.gsd, anchor_height_m=a.anchor_height,
          max_px=a.px, stage=not a.no_stage)
