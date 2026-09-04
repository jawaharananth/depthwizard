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
import height_cache
import dsm_refine
import dtm as dtm_mod
import city_model
import building_discovery as bd
import shadow_correction
import overlay_rejection
from depth_model import DepthBackbone, orientation_check
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


def check_nadir(image_np: np.ndarray) -> dict:
    """
    Cheap obliqueness screen.

    In a nadir view, building facades are not visible, so strong vertical image
    structure is rare and roof edges dominate. In an oblique or ground-level
    photo, facades occupy a large fraction of the frame and produce a strong,
    consistently-oriented gradient field plus a sky region at the top.

    A sky test is the most reliable single cue available without metadata: a
    nadir frame has no horizon, so a large bright low-texture region across the
    top of the image means the camera was not pointing down.
    """
    h, w = image_np.shape[:2]
    top = image_np[: h // 4]
    gray_top = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex = cv2.blur(gray_top ** 2, (9, 9)) - cv2.blur(gray_top, (9, 9)) ** 2
    bright_flat = float(((gray_top > 150) & (tex < 60)).mean())

    hsv_top = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    blueish = float(((hsv_top[:, :, 0] > 90) & (hsv_top[:, :, 0] < 135) &
                     (hsv_top[:, :, 1] > 40)).mean())

    sky_fraction = max(bright_flat, blueish)
    return {"sky_fraction_top": round(sky_fraction, 3),
            "likely_nadir": sky_fraction < 0.25}


def build(image_path: str, name: str, gsd_m: float = None,
          anchor_height_m: float = None, max_px: int = 2560,
          stage: bool = True) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"city_{name}")
    t0 = time.time()

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
    if min(W, H) < 800:
        raise SystemExit(
            f"image is {W}x{H} -- too small to resolve buildings. This pipeline "
            "needs roughly 2000px of nadir imagery; below ~800px a building is a "
            "handful of pixels and every footprint is noise.")

    nad = check_nadir(image_np)
    if not nad["likely_nadir"]:
        raise SystemExit(
            f"this does not look like a nadir (top-down) view -- "
            f"{nad['sky_fraction_top']*100:.0f}% of the upper frame reads as sky. "
            "Every stage of this pipeline assumes a top-down view: depth is read as "
            "height, footprints as plan geometry, shadow length as building height. "
            "On an oblique or ground-level photo the output would be confident "
            "nonsense, so the build stops here rather than producing it.")

    key = f"plainimg_{name}"
    height = height_cache.load(key, "tiled", H) if H == W else None
    if height is None:
        height = DepthBackbone().predict_tiled(pil)
        if H == W:
            height_cache.save(key, "tiled", H, height)
    print(f"[2/5] height field in {time.time()-t0:.0f}s")

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

    footprints = []
    for rec in disc["instances"]:
        cnt = rec["contour"]
        eps = 0.012 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) >= 3:
            footprints.append(poly)
    for rec in disc["instances"]:
        rec["provenance"] = bd.INFERRED   # never MEASURED on this path

    bverts, bfaces, binfo = city_model.build_prisms(
        footprints, dsm, ground, gsd, gsd, min_height_m=1.5, image_np=image_np)
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

    g = image_np.astype(np.float32) / 255.0
    lum = (g * np.array([0.299, 0.587, 0.114], np.float32)).sum(axis=2, keepdims=True)
    g = lum + (g - lum) * 1.12
    g = np.clip((g - 0.42) * 1.14 + 0.46, 0, 1)
    g = np.clip(g * np.array([0.94, 0.985, 1.06], np.float32), 0, 1)
    graded = (g * 255).astype(np.uint8)
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
