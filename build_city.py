"""
Build a clean prism city model from one satellite tile and stage it.

  python build_city.py JAX_068

Differs from build_scene.py in one structural way: buildings are separate
extruded volumes rather than bumps in a heightfield, and the ground is bare
earth beneath them. See city_model.py for why that distinction decides how the
result can possibly look.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import cv2
from PIL import Image

import ortho
import segmentation as seg
import height_cache
import dsm_refine
import dtm as dtm_mod
import city_model
import building_discovery as bd
import region_footprints as rf
import mvs_height
import dem_source
import dfc2019_loader as L
import shadow_correction
from depth_model import DepthBackbone, orientation_check
from glb_export import export_glb
import mesh_generation as mg

RGB_DIR = "dfc2019_data/rgb/Track3-RGB-1"
TRUTH_DIR = "dfc2019_data/truth/Track3-Truth"
METADATA_DIR = "dfc2019_data/metadata/Track3-Metadata"
OUT_DIR = "final_out"
VIEWER_DIR = "viewer/output"

ASSUMED_TALL_M = 40.0     # only used when shadow calibration fails; see build_scene.py
GROUND_GRID = 700         # ground is smooth bare earth, so it needs few vertices


def build(tile: str, out_px: int = 2048, stage: bool = True,
          extent_m: float = None, no_mvs: bool = False, n_views: int = 6,
          use_dem: bool = False, dem_path: str = None) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"city_{tile.lower()}")
    t_start = time.time()

    o = ortho.orthorectify(tile, TRUTH_DIR, RGB_DIR, METADATA_DIR,
                           out_px=out_px, extent_m=extent_m)
    image_np = o["image"]
    gsd = o["gsd_m"]
    print(f"[1/5] {tile} {out_px}x{out_px} @ {gsd:.3f} m/px  "
          f"{o['extent_m']:.0f}x{o['extent_m']:.0f} m ground, coverage {o['coverage']*100:.1f}%")
    print(f"      view {os.path.basename(o['rgb_path'])} ({o['off_nadir_deg']} deg off-nadir); "
          f"LiDAR truth covers only the central {o['truth_extent_m']:.0f} m")

    # HEIGHT SOURCE: RPC multi-view stereo, with monocular depth as fallback.
    #
    # Measured on JAX_165 against LiDAR, per building on ground-truth footprints:
    #   monocular + shadow scale   MAE 13.75 m   corr 0.24
    #   MVS, 6 near-nadir views    MAE  7.24 m   corr 0.532
    # MVS triangulates through the RPC cameras, so heights are MEASURED in
    # absolute WGS84 metres and need no scale calibration at all.
    use_mvs = not no_mvs
    mvs = None
    if use_mvs:
        try:
            t_m = time.time()
            mvs = mvs_height.compute(tile, TRUTH_DIR, RGB_DIR, METADATA_DIR,
                                     extent_m=o["extent_m"], n_views=n_views)
            print(f"[2/5] MVS height from {mvs['views']} views "
                  f"({'cached' if mvs['cached'] else f'{time.time()-t_m:.0f}s'}), "
                  f"{mvs['gsd_m']:.2f} m grid")
        except Exception as exc:
            print(f"[2/5] MVS unavailable ({exc}); falling back to monocular depth")
            mvs = None

    mvs_dsm = None
    if mvs is not None:
        # MVS is kept ASIDE, not substituted for the working surface. It supplies
        # the final height of each prism and nothing else -- see the fusion note
        # below for the measurements behind that split.
        mvs_dsm = mvs_height.to_image_grid(mvs["dsm"], image_np)
        tier = "A (MVS-triangulated heights, absolute metres)"
        cal = {"n": mvs["views"], "method": "rpc_plane_sweep"}
        height = None
        scale = 1.0      # MVS is already metric; nothing is rescaled

    # FUSION: monocular for SHAPE, MVS for MAGNITUDE.
    #
    # Each signal is better at a different thing, measured on JAX_165 against
    # LiDAR over the 256 m the truth covers:
    #
    #                  nDSM shape corr      height MAE
    #   monocular            0.413            8.53 m
    #   MVS                  0.187            6.53 m
    #
    # MVS triangulates real metres but is spatially noisy: sweeping its
    # elevated-region threshold across the whole range, precision against the
    # LiDAR building mask never exceeded 0.49 at ANY setting, including at
    # exactly the right coverage. The monocular field reached 0.606 at the same
    # coverage. Its surface is smooth and wrong in magnitude; MVS is right in
    # magnitude and rough in outline.
    #
    # So the monocular field decides WHERE buildings are -- segmentation and
    # footprint extraction both run on it -- and MVS supplies only HOW TALL each
    # finished building is.
    #
    # Using MVS for both, which the first version did, pushed the building mask
    # to 81.9% of frame against a true 45.9%. Using MVS for footprint extraction
    # too was also worse: IoU 0.472 against 0.635, at recall 0.496 against 0.756,
    # because MVS under-measures low buildings and they fall below the height cut
    # before shape is ever considered.
    #
    # Height is taken as a PERCENTILE OVER EACH FOOTPRINT, not per pixel. MVS
    # noise is per-pixel and largely independent, so averaging over the hundreds
    # of pixels inside a footprint suppresses it while keeping the metric
    # accuracy that made MVS worth using (MAE 6.53 m against monocular's 8.53 m).
    if mvs is not None:
        seg_key = f"tiled_e{int(o['extent_m'])}"
        height = height_cache.load(tile, seg_key, out_px)
        if height is None:
            print("      (monocular shape cue not cached; computing it -- "
                  "heights still come from MVS)")
            height = DepthBackbone().predict_tiled(pil, verbose=False)
            height_cache.save(tile, seg_key, out_px, height)
        else:
            print("      shape from monocular (cached); heights from MVS")
        refined = dsm_refine.refine_dsm(height, image_np)
        # Segment ONCE. This block called seg.segment three times on a 2560x2560
        # image -- the shape cue, the shadow calibration, and the fallback each
        # recomputed it, and the build stalled before reaching the mesh stage.
        _seg_tmp, _ = seg.segment(image_np, height=height)
        cal_s = shadow_correction.calibrate_scale(
            image_np, _seg_tmp, refined, o["sun_elev_deg"], o["sun_azimuth_deg"],
            gsd, gsd)
        sc = cal_s.get("scale_m_per_unit")
        if sc is None or cal_s.get("n", 0) < 10:
            sc = ASSUMED_TALL_M / max(float(np.percentile(np.maximum(
                refined - dtm_mod.estimate_dtm(refined, _seg_tmp), 0.0), 99)), 1e-6)
        dsm = refined * sc

    seg_labels, _ = seg.segment(image_np, height=height)
    oc = orientation_check(height, seg_labels)
    if oc.get("checked") and not oc["correct_orientation"]:
        raise SystemExit(
            f"height field is inverted: buildings {oc['building_mean']:.4f} below "
            f"ground {oc['ground_mean']:.4f} -- refusing to build a scene with "
            "every structure in a pit")
    print(f"[3/5] segmentation  building "
          f"{(seg_labels == seg.CLASS_IDX['building']).mean()*100:.1f}% of frame")

    if mvs is None:
        refined = dsm_refine.refine_dsm(height, image_np)

    if mvs is None:
        # Monocular path only: a relative field needs a metres-per-unit scale.
        # The MVS path already produced absolute metres and must not be rescaled.
        cal = shadow_correction.calibrate_scale(
            image_np, seg_labels, refined, o["sun_elev_deg"], o["sun_azimuth_deg"],
            gsd, gsd)
        scale = cal["scale_m_per_unit"]
        if scale is None or cal["n"] < 10:
            rel = np.maximum(refined - dtm_mod.estimate_dtm(refined, seg_labels), 0.0)
            b = seg_labels == seg.CLASS_IDX["building"]
            p99 = float(np.percentile(rel[b], 99)) if b.sum() > 1000 else float(np.percentile(rel, 99))
            scale = ASSUMED_TALL_M / max(p99, 1e-6)
            tier = "C (relative, assumed vertical scale)"
            print(f"[3/5] scale NOT measured ({cal.get('reason', 'unstable')}); "
                  f"{scale:.0f} m/unit assuming tallest ~{ASSUMED_TALL_M:.0f} m -- RELATIVE")
        else:
            tier = "B (shadow-calibrated)"
            print(f"[3/5] scale {scale:.1f} m/unit from {cal['n']} shadow measurements")
        dsm = refined * scale
    else:
        print(f"[3/5] heights from MVS, already absolute: "
              f"{np.nanmin(mvs_dsm):.1f} to {np.nanmax(mvs_dsm):.1f} m "
              f"(monocular shape cue carries an arbitrary scale and is not used "
              f"for height)")

    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    ground = city_model.flatten_ground(terrain, seg_labels, smooth_m=35.0, gsd_m=gsd)

    # TIER A: anchor the absolute datum to an external DEM.
    #
    # A 30 m DEM fixes WHERE the surface sits vertically; it cannot improve
    # building heights, because at that posting a building is a fraction of a
    # pixel and global DEMs are smoothed toward bare earth. The offset is fitted
    # over pixels BOTH surfaces call bare ground, using this pipeline's own
    # segmentation rather than the LiDAR labels, so no ground truth leaks into
    # the input.
    dem_info = None
    if use_dem or dem_path:
        coords = L.parse_dsm_txt(os.path.join(TRUTH_DIR, f"{tile}_DSM.txt"))
        t_ext = coords["size_px"] * coords["gsd_m"]
        pad = (o["extent_m"] - t_ext) / 2.0
        epsg = "EPSG:32617" if tile.startswith("JAX") else "EPSG:32615"
        d = dem_source.sample_grid(
            coords["utm_x"] - pad, coords["utm_y"] + t_ext + pad,
            out_px, o["extent_m"] / out_px, epsg, dem_path=dem_path)
        if d["dem"] is None:
            print(f"[3b] DEM unavailable ({d.get('error')}) -- staying on tier {tier[0]}")
        else:
            gmask = (seg_labels == seg.CLASS_IDX["bare_earth"]) |                     (seg_labels == seg.CLASS_IDX["road"])
            # Fit against whichever surface is ABSOLUTE. On the MVS path the
            # monocular surface is only a shape cue -- it carries an arbitrary
            # scale (measured spanning -97 to 228 m on this tile) and fitting a
            # DEM to it produces a spread so wide the guard rightly rejects it.
            # The MVS surface is the one in real metres, so it is the one the
            # datum belongs to.
            anchor_surface = (dtm_mod.estimate_dtm(mvs_dsm, seg_labels)
                              if mvs_dsm is not None else ground)
            fit = dem_source.fit_offset(anchor_surface, d["dem"], gmask)
            if fit["offset_m"] is None or not fit.get("spread_ok", False):
                print(f"[3b] DEM offset not usable ({fit.get('reason', 'spread too wide')}) "
                      f"-- staying on tier {tier[0]}")
            else:
                # Shift the surface the heights actually come from.
                if mvs_dsm is not None:
                    mvs_dsm = mvs_dsm + fit["offset_m"]
                else:
                    dsm = dsm + fit["offset_m"]
                    terrain = terrain + fit["offset_m"]
                    ground = ground + fit["offset_m"]
                tier = "A (DEM-anchored absolute elevation)"
                dem_info = {"source": d["source"], "offset_m": round(fit["offset_m"], 2),
                            "iqr_m": round(fit["iqr_m"], 2), "n_px": fit["n"],
                            "coverage": round(d["coverage"], 3)}
                print(f"[3b] TIER A: datum anchored to {d['source']}")
                print(f"     offset {fit['offset_m']:+.2f} m from {fit['n']:,} ground "
                      f"pixels (IQR {fit['iqr_m']:.2f} m)")


    ndsm = np.maximum(dsm - ground, 0.0)

    # High-recall multi-scale discovery, replacing the single-pass single-threshold
    # extraction. One pass has one effective object size; sheds and city blocks are
    # not found by the same kernel.
    # Footprints from IMAGE REGIONS, not from a threshold on the depth field.
    #
    # Measured A/B against LiDAR on the central 256 m of this tile:
    #   depth threshold : IoU 0.443  recall 0.607  precision 0.621
    #   image regions   : IoU 0.635  recall 0.756  precision 0.798
    # and false positives on ground fell 17.6% -> 7.6%, on trees 9.2% -> 6.3%.
    #
    # The old mask had the right AMOUNT of building (46.0% vs 45.9% truth) in
    # the wrong PLACES, and an exhaustive +-10 m shift search proved it was not
    # a registration error. Depth is too smooth to carry a roof outline; the
    # outline is in the image.
    min_h = 2.0 if tier[0] in ("A", "B") else 0.02 * float(np.percentile(ndsm, 99))
    rres = rf.extract(image_np, ndsm, gsd, seg_labels=seg_labels, min_area_m2=8.0,
                      min_height_m=min_h)
    footprints = rres["polygons"]
    print(f"[4/5] footprints from image regions: {rres['report']['retained']} of "
          f"{rres['report']['regions_examined']} regions")
    print(f"      rejected: " + "  ".join(f"{k} {v}" for k, v in
                                          rres["report"]["rejected"].items() if v))

    # Discovery still runs, for the evidence/provenance records and the
    # small-object accounting -- it is the reporting path, not the geometry path.
    shadow_mask = shadow_correction.detect_shadow_mask(image_np)
    disc = bd.discover(image_np, seg_labels, ndsm, gsd,
                       sun_azimuth_deg=o["sun_azimuth_deg"],
                       shadow_mask=shadow_mask, min_area_m2=6.0)
    print(bd.format_report(disc["report"]))
    prov = bd.MEASURED if tier[0] in ("A", "B") else bd.INFERRED
    for rec in disc["instances"]:
        rec["provenance"] = prov

    # Prism heights come from the MVS surface when it is available; the
    # monocular surface is used for everything up to this point. build_prisms
    # takes a percentile inside each footprint, so this is where MVS's per-pixel
    # noise gets averaged away and only its metric accuracy survives.
    height_dsm = mvs_dsm if mvs_dsm is not None else dsm
    # Ground for HEIGHT measurement comes from a morphological opening of the
    # surface itself, never from the segmentation mask.
    #
    # The mask-based DTM fills under buildings from the nearest unmasked pixel,
    # and at ~0.6 recall that neighbour is often another roof segmentation
    # missed, so roof height propagates into the terrain. Measured on JAX_165
    # against LiDAR: its ground sat +15.78 m too high under buildings, which
    # collapsed measured building height to 5.03 m against a true 19.84 m. The
    # heights were never wrong -- the datum beneath them was.
    #
    #                          ground err   building height
    #   mask-based DTM          +13.99 m       5.03 m
    #   opening (this)           -3.04 m      22.84 m     (LiDAR 19.84 m)
    height_ground = dtm_mod.ground_from_dsm(height_dsm, gsd)

    bverts, bfaces, binfo = city_model.build_prisms(
        footprints, height_dsm, height_ground, gsd, gsd, min_height_m=1.5,
        image_np=image_np)

    # Provenance (section 24): a height is only MEASURED when the scene carried a
    # metric scale. On a Tier C scene it remains INFERRED, whatever it looks like.
    prov = bd.MEASURED if tier[0] in ("A", "B") else bd.INFERRED
    for rec in disc["instances"]:
        rec["provenance"] = prov

    heights = np.array([r["height_m"] for r in binfo["buildings"]]) if binfo["buildings"] else np.zeros(1)
    print(f"      {len(binfo['buildings'])} prisms extruded from {len(footprints)} footprints "
          f"({binfo['skipped']} below minimum height)")
    print(f"      heights: median {np.median(heights):.1f} m  p90 {np.percentile(heights,90):.1f} m  "
          f"max {heights.max():.1f} m")

    # Ground mesh: smooth bare earth, so a coarse grid carries it with no loss.
    # The ground MESH must use the same surface the buildings are based on.
    #
    # These had drifted apart: building bases moved to the morphological ground
    # (correct, ~0 m error) while the rendered terrain stayed on the mask-based
    # DTM, which measures +14 m too high. The result is a terrain surface
    # floating above its own buildings -- they read as buried, with flat ground
    # on top of them. Two different notions of "ground" in one scene is never
    # right; there is only one ground.
    gsmall, _ = mg._resize_for_mesh(height_ground, seg_labels, GROUND_GRID)
    gscale = ground.shape[0] / gsmall.shape[0]
    gverts, guvs, gfaces = mg.build_ground_mesh(gsmall, gsd * gscale, gsd * gscale)

    # Gentle contrast and saturation lift on the ground texture.
    #
    # WorldView imagery is captured flat on purpose -- it is measurement data,
    # graded for dynamic range rather than for looking good -- so dropped into
    # a lit 3D scene it reads as washed-out grey. This is a display grade, not
    # a change to anything measured: the texture is decoration on the ground
    # mesh and no height, footprint or metric is derived from it.
    g = image_np.astype(np.float32) / 255.0
    lum = (g * np.array([0.299, 0.587, 0.114], np.float32)).sum(axis=2, keepdims=True)
    g = lum + (g - lum) * 1.12                      # saturation, restrained
    # Contrast about a point ABOVE mid-grey, with a shadow lift. Pivoting at
    # 0.5 on imagery whose median sits near 0.49 drove the shaded half of the
    # scene toward black, and the saturation boost then turned it blue.
    g = np.clip((g - 0.42) * 1.14 + 0.46, 0, 1)
    # Cool cast on the ground so terrain, buildings and sky read as one palette.
    # Applied as a per-channel gain, which shifts colour temperature without
    # touching relative brightness -- roads stay darker than rooftops.
    g = np.clip(g * np.array([0.94, 0.985, 1.06], np.float32), 0, 1)
    graded = (g * 255).astype(np.uint8)

    Image.fromarray(graded).save(stem + "_texture.png")
    import io
    tex = io.BytesIO()
    Image.fromarray(graded).save(tex, format="PNG")

    # Vegetation and water as their own meshes. A prism city with neither reads
    # as a model of a car park; both are already segmented and were being
    # discarded, and both are what supplies colour in the reference imagery.
    # Split buildings into height classes so each can carry its own material.
    #
    # A city rendered in one material reads as extruded plastic: a glass tower,
    # a concrete block and a low retail unit all return light identically, so
    # the only thing distinguishing them is silhouette. Building stock actually
    # correlates strongly with height -- tall structures at this scale are
    # curtain-walled, low ones are masonry or render -- so height is a defensible
    # proxy for material rather than an arbitrary recolouring.
    #
    # The split happens here because a glTF primitive carries exactly one
    # material. Three meshes is the simplest way to get three material responses
    # without per-vertex shader work.
    CLASS_BOUNDS = [("building_low", 0.0, 15.0),
                    ("building_mid", 15.0, 40.0),
                    ("building_tall", 40.0, 1e9)]
    bclass = []
    for name, lo, hi in CLASS_BOUNDS:
        vsel, fsel, csel = [], [], []
        base = 0
        for rec in binfo["buildings"]:
            if not (lo <= rec["height_m"] < hi):
                continue
            off, cnt = rec["vertex_offset"], rec["vertex_count"]
            vsel.append(bverts[off:off + cnt])
            cols = binfo.get("colors")
            if cols is not None and len(cols) >= off + cnt:
                csel.append(cols[off:off + cnt])
            fmask = (bfaces >= off) & (bfaces < off + cnt)
            keep = fmask.all(axis=1)
            fsel.append(bfaces[keep] - off + base)
            base += cnt
        if vsel:
            bclass.append((name,
                           np.concatenate(vsel),
                           np.concatenate(fsel),
                           np.concatenate(csel) if csel else None))
    # Count buildings, not vertices/8. Prisms have a variable vertex count --
    # a footprint may have any number of corners -- so dividing by 8 assumed a
    # fixed box and drifted as soon as polygons stopped being rectangles.
    _cls_counts = {n: 0 for n, _, _ in CLASS_BOUNDS}
    for rec in binfo["buildings"]:
        for n, lo, hi in CLASS_BOUNDS:
            if lo <= rec["height_m"] < hi:
                _cls_counts[n] += 1
                break
    print("      materials: " + "  ".join(
        f"{n.split('_')[1]} {c}" for n, c in _cls_counts.items()))

    cverts, cfaces, n_canopy = city_model.build_canopy(
        seg_labels, dsm, ground, gsd, gsd, min_area_px=120)
    wverts, wfaces, n_water = city_model.build_water(seg_labels, ground, gsd, gsd)
    vverts, vfaces, n_veh = city_model.detect_vehicles(
        image_np, seg_labels, ground, gsd)
    print(f"      {n_canopy} canopy volumes, {n_water} water bodies, "
          f"{n_veh} vehicle-sized objects (heuristic)")

    export_glb(stem + ".glb", gverts, guvs, gfaces,
               np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
               texture_bytes=tex.getvalue(), building_uvs=None,
               building_colors=None,
               extra_meshes=[
                   (n, v, f, (0.80, 0.82, 0.85, 1.0)) for n, v, f, _ in bclass
               ] + [
                   ("canopy", cverts, cfaces, (0.29, 0.42, 0.24, 1.0)),
                   ("water", wverts, wfaces, (0.20, 0.35, 0.48, 1.0)),
                   ("vehicles", vverts, vfaces, (0.62, 0.63, 0.66, 1.0)),
               ])
    size_mb = os.path.getsize(stem + ".glb") / 1e6
    print(f"[5/5] {len(gfaces)} ground faces + {len(bfaces)} building faces, {size_mb:.1f} MB")

    # Per-building export (sections 48, 49). GeoJSON carries a CRS only because this
    # scene is genuinely georeferenced through the RPC ortho; a relative-only scene
    # must never be given one.
    import json as _json
    tr = o["transform"]
    feats = []
    for rec, prism in zip(disc["instances"], binfo["buildings"]):
        cnt = rec["contour"].reshape(-1, 2)
        ring = [list(tr * (float(px), float(py))) for px, py in cnt]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "id": rec["id"],
                "height_m": round(prism["height_m"], 2),
                "area_m2": rec["area_m2"],
                "perimeter_m": rec["perimeter_m"],
                "size_class": rec["size_class"],
                "confidence": rec["confidence"],
                "provenance": rec["provenance"],
                "evidence": rec["evidence"],
                "height_is_metric": tier[0] in ("A", "B"),
            },
        })
    with open(stem + "_buildings.geojson", "w") as f:
        _json.dump({"type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": o["crs"]}},
                    "features": feats}, f)
    import csv as _csv
    with open(stem + "_buildings.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["id", "height_m", "area_m2", "perimeter_m", "size_class",
                    "confidence", "provenance", "height_is_metric",
                    "ev_height", "ev_edge", "ev_texture", "ev_shadow"])
        for rec, prism in zip(disc["instances"], binfo["buildings"]):
            e = rec["evidence"]
            w.writerow([rec["id"], round(prism["height_m"], 2), rec["area_m2"],
                        rec["perimeter_m"], rec["size_class"], rec["confidence"],
                        rec["provenance"], tier.startswith("B"),
                        e["height"], e["edge"], e["texture"], e["shadow"]])

    meta = {
        "tile": tile, "view": os.path.basename(o["rgb_path"]),
        "off_nadir_deg": o["off_nadir_deg"],
        "sun_elevation_deg": o["sun_elev_deg"], "sun_azimuth_deg": o["sun_azimuth_deg"],
        "gsd_m": round(gsd, 4), "crs": o["crs"], "tier": tier,
        "scale_m_per_unit": round(float(scale), 2),
        "dem_anchor": dem_info,
        "model": "prism city (flat roofs, vertical walls)",
        "buildings_extruded": len(binfo["buildings"]),
        "canopy_volumes": n_canopy,
        "water_bodies": n_water,
        "vehicles": n_veh,
        "extent_m": o["extent_m"],
        "truth_extent_m": o["truth_extent_m"],
        "coverage": round(o["coverage"], 3),
        "median_height_m": round(float(np.median(heights)), 1),
        "max_height_m": round(float(heights.max()), 1),
        "build_seconds": round(time.time() - t_start, 1),
        "discovery": disc["report"],
        "provenance": prov,
        "height_is_metric": tier[0] in ("A", "B"),
        "confidence_median": round(float(np.median(
            [r["confidence"] for r in disc["instances"]])), 3) if disc["instances"] else None,
    }

    if stage:
        os.makedirs(VIEWER_DIR, exist_ok=True)
        shutil.copy2(stem + ".glb", os.path.join(VIEWER_DIR, "terrain.glb"))
        shutil.copy2(stem + "_texture.png", os.path.join(VIEWER_DIR, "terrain_texture.png"))
        # Maps baked for the heightfield describe a surface this model no
        # longer has; leaving them staged would light the flat ground with a
        # previous scene's occlusion.
        for stale in ("terrain_ao.png", "terrain_normal.png",
                      "terrain_roughness.png", "terrain_metalness.png",
                      "terrain_heatmap.png"):
            p = os.path.join(VIEWER_DIR, stale)
            if os.path.exists(p):
                os.remove(p)
        shutil.copy2(stem + "_buildings.geojson",
                     os.path.join(VIEWER_DIR, "buildings.geojson"))
        with open(os.path.join(VIEWER_DIR, "scene.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"      staged to {VIEWER_DIR}  ({meta['build_seconds']}s total)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tile", nargs="?", default="JAX_068")
    ap.add_argument("--px", type=int, default=2048)
    ap.add_argument("--extent", type=float, default=None,
                    help="ground extent in metres (default: the 256 m truth tile)")
    ap.add_argument("--no-stage", action="store_true")
    ap.add_argument("--no-mvs", action="store_true",
                    help="use monocular depth instead of multi-view stereo")
    ap.add_argument("--dem", action="store_true",
                    help="anchor absolute elevation to Copernicus GLO-30 (Tier A)")
    ap.add_argument("--dem-path", default=None,
                    help="use a local DEM GeoTIFF instead of the global model")
    ap.add_argument("--views", type=int, default=6,
                    help="how many near-nadir views to triangulate from")
    a = ap.parse_args()
    build(a.tile, out_px=a.px, stage=not a.no_stage, extent_m=a.extent,
          no_mvs=a.no_mvs, n_views=a.views, use_dem=a.dem, dem_path=a.dem_path)
