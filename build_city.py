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
          extent_m: float = None) -> dict:
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

    cache_key = f"tiled_e{int(o['extent_m'])}"
    height = height_cache.load(tile, cache_key, out_px)
    if height is None:
        t0 = time.time()
        height = DepthBackbone().predict_tiled(Image.fromarray(image_np))
        height_cache.save(tile, cache_key, out_px, height)
        print(f"[2/5] height field in {time.time()-t0:.0f}s")
    else:
        print("[2/5] height field reused from cache")

    seg_labels, _ = seg.segment(image_np, height=height)
    oc = orientation_check(height, seg_labels)
    if oc.get("checked") and not oc["correct_orientation"]:
        raise SystemExit("height field inverted -- refusing to build")
    refined = dsm_refine.refine_dsm(height, image_np)

    cal = shadow_correction.calibrate_scale(
        image_np, seg_labels, refined, o["sun_elev_deg"], o["sun_azimuth_deg"], gsd, gsd)
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
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    ground = city_model.flatten_ground(terrain, seg_labels, smooth_m=35.0, gsd_m=gsd)

    ndsm = np.maximum(dsm - ground, 0.0)

    # High-recall multi-scale discovery, replacing the single-pass single-threshold
    # extraction. One pass has one effective object size; sheds and city blocks are
    # not found by the same kernel.
    shadow_mask = shadow_correction.detect_shadow_mask(image_np)
    disc = bd.discover(image_np, seg_labels, ndsm, gsd,
                       sun_azimuth_deg=o["sun_azimuth_deg"],
                       shadow_mask=shadow_mask, min_area_m2=6.0)
    print(f"[4/5] building discovery")
    print(bd.format_report(disc["report"]))

    footprints = []
    for rec in disc["instances"]:
        cnt = rec["contour"]
        eps = 0.012 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) >= 3:
            footprints.append(poly)
            rec["n_vertices"] = int(len(poly))

    bverts, bfaces, binfo = city_model.build_prisms(
        footprints, dsm, ground, gsd, gsd, min_height_m=1.5, image_np=image_np)

    # Provenance (section 24): a height is only MEASURED when the scene carried a
    # metric scale. On a Tier C scene it remains INFERRED, whatever it looks like.
    prov = bd.MEASURED if tier.startswith("B") else bd.INFERRED
    for rec in disc["instances"]:
        rec["provenance"] = prov

    heights = np.array([r["height_m"] for r in binfo["buildings"]]) if binfo["buildings"] else np.zeros(1)
    print(f"      {len(binfo['buildings'])} prisms extruded from {len(footprints)} footprints "
          f"({binfo['skipped']} below minimum height)")
    print(f"      heights: median {np.median(heights):.1f} m  p90 {np.percentile(heights,90):.1f} m  "
          f"max {heights.max():.1f} m")

    # Ground mesh: smooth bare earth, so a coarse grid carries it with no loss.
    gsmall, _ = mg._resize_for_mesh(ground, seg_labels, GROUND_GRID)
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
    cverts, cfaces, n_canopy = city_model.build_canopy(
        seg_labels, dsm, ground, gsd, gsd, min_area_px=120)
    wverts, wfaces, n_water = city_model.build_water(seg_labels, ground, gsd, gsd)
    vverts, vfaces, n_veh = city_model.detect_vehicles(
        image_np, seg_labels, ground, gsd)
    print(f"      {n_canopy} canopy volumes, {n_water} water bodies, "
          f"{n_veh} vehicle-sized objects (heuristic)")

    export_glb(stem + ".glb", gverts, guvs, gfaces, bverts, bfaces,
               texture_bytes=tex.getvalue(), building_uvs=None,
               building_colors=binfo.get("colors"),
               extra_meshes=[
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
                "height_is_metric": tier.startswith("B"),
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
        "height_is_metric": tier.startswith("B"),
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
    a = ap.parse_args()
    build(a.tile, out_px=a.px, stage=not a.no_stage, extent_m=a.extent)
