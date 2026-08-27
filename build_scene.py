"""
Build a full 3D scene from one DFC2019 tile and stage it for the viewer.

Exists as a file rather than an ad-hoc command so a rendered scene can be
reproduced exactly: the same tile, the same depth mode, the same mesh
settings. Every earlier render was typed inline and could not be repeated.

  python build_scene.py JAX_167 [--global-depth] [--no-stage]
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import rasterio
from PIL import Image

import ortho
import segmentation as seg
import height_cache
import dsm_refine
import dtm as dtm_mod
import mesh_generation as mg
import terrain_maps
import shadow_correction
from depth_model import DepthBackbone, orientation_check

RGB_DIR = "dfc2019_data/rgb/Track3-RGB-1"
TRUTH_DIR = "dfc2019_data/truth/Track3-Truth"
METADATA_DIR = "dfc2019_data/metadata/Track3-Metadata"
OUT_DIR = "final_out"
VIEWER_DIR = "viewer/output"

# Fallback metres-per-unit, used only when the scene yields no measurable
# shadows at all. It is a guess and the build says so out loud; the normal
# path calibrates against shadow geometry instead.
# Assumed height of the tallest structures in a built-up tile, used ONLY when
# shadow calibration fails. A prior about cities, not a measurement of this
# one; every scene built this way is labelled Tier C in the viewer.
ASSUMED_TALL_M = 40.0


def build(tile: str, use_tiled: bool = True, stage: bool = True) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, tile.lower())

    # Orthorectified through the RPC model, and from the straightest available
    # view. Taking the first file on disk instead means an arbitrary off-nadir
    # angle -- up to 29 deg across the views of one tile -- which leans every
    # roof away from its own footprint by height * tan(angle).
    o = ortho.orthorectify(tile, TRUTH_DIR, RGB_DIR, METADATA_DIR, out_px=2048)
    image_np = o["image"]
    pil = Image.fromarray(image_np)
    gsd = o["gsd_m"]
    print(f"[1/6] {tile}  {image_np.shape[1]}x{image_np.shape[0]} @ {gsd:.3f} m/px  "
          f"view {os.path.basename(o['rgb_path'])} ({o['off_nadir_deg']} deg off-nadir)")

    mode = "tiled" if use_tiled else "global"
    t0 = time.time()
    height = height_cache.load(tile, mode, image_np.shape[0])
    if height is None:
        backbone = DepthBackbone()
        height = (backbone.predict_tiled(pil, verbose=False) if use_tiled
                  else backbone.predict(pil))
        height_cache.save(tile, mode, image_np.shape[0], height)
        print(f"[2/6] height field ({mode}) in {time.time()-t0:.0f}s")
    else:
        print(f"[2/6] height field ({mode}) reused from cache")

    # Segmentation is given the height field: without it, buildings are found
    # by roof-edge appearance alone and the mask covers outlines rather than
    # roofs, which then contaminates the DTM and rejects nearly every building
    # as unmeasurable downstream.
    seg_labels, _ = seg.segment(image_np, height=height)
    oc = orientation_check(height, seg_labels)
    if oc.get("checked") and not oc["correct_orientation"]:
        raise SystemExit(
            f"height field is inverted: buildings {oc['building_mean']:.4f} "
            f"below ground {oc['ground_mean']:.4f} -- refusing to build a scene "
            "with every structure in a pit")
    print(f"[3/6] segmentation  building {(seg_labels==seg.CLASS_IDX['building']).mean()*100:.1f}% "
          f"of frame; orientation OK")

    refined = dsm_refine.refine_dsm(height, image_np)

    # Scale comes from shadow geometry, not a constant: h = L * tan(sun
    # elevation) gives a metric height per building, and the median ratio to
    # the same building's height in field units converts the whole field.
    cal = shadow_correction.calibrate_scale(
        image_np, seg_labels, refined, o["sun_elev_deg"], o["sun_azimuth_deg"],
        gsd, gsd)
    scale = cal["scale_m_per_unit"]
    if scale is None or cal["n"] < 10:
        # No trustworthy metric scale from this scene. Rather than fall back to
        # a bare constant -- this file used 200 m/unit, which made 37 m
        # buildings 127 m tall -- anchor the vertical scale to one stated
        # assumption about the district, and label the result relative.
        #
        # The assumption: in a built-up tile the tallest structures reach about
        # ASSUMED_TALL_M. That is a prior about cities, not a measurement of
        # THIS city, and the viewer says so. It is chosen over a raw constant
        # because it is at least expressed in a quantity that means something
        # and can be argued with.
        rel = np.maximum(refined - dtm_mod.estimate_dtm(refined, seg_labels), 0.0)
        b = seg_labels == seg.CLASS_IDX["building"]
        p99 = float(np.percentile(rel[b], 99)) if b.sum() > 1000 else float(np.percentile(rel, 99))
        scale = ASSUMED_TALL_M / max(p99, 1e-6)
        reason = cal.get("reason", f"only {cal['n']} usable shadows")
        print(f"[3b] scale NOT measured ({reason})")
        print(f"     -> {scale:.0f} m/unit by assuming the tallest structures reach "
              f"{ASSUMED_TALL_M:.0f} m. HEIGHTS ARE RELATIVE, NOT MEASURED.")
        tier = "C (relative, assumed vertical scale)"
    else:
        print(f"[3b] scale {scale:.1f} m/unit from {cal['n']} shadow measurements "
              f"(median building {cal.get('median_building_height_m')} m, "
              f"IQR {cal['spread_ratio']*100:.0f}% of median)")
        tier = "B (shadow-calibrated)"

    dsm = refined * scale
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)

    # Square off the buildings. The depth field ramps into every wall over
    # several metres, and a heightfield mesh renders that ramp as a roof
    # slumping into the street. Flattening each footprint to one roof height
    # puts the step back where the wall actually is.
    dsm, n_prism = dsm_refine.prismify_buildings(dsm, seg_labels, terrain)
    # The terrain estimate was derived from the pre-squared surface, so it has
    # to be recomputed: otherwise a roof that moved down to its footprint's
    # true height can end up below the stale terrain beneath it, which is the
    # buried-building failure this pipeline has already had once.
    terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
    print(f"[3c] {n_prism} building footprints squared off to flat roofs")
    dsm, n_lev, n_rej = terrain_maps.flatten_water(dsm, seg_labels)
    st = dtm_mod.structure_height_stats(dsm, terrain, seg_labels)
    print(f"[4/6] terrain separated  mean height above ground "
          f"{st.get('mean_agl', 0):.1f} m, {st.get('negative_fraction', 0)*100:.2f}% negative; "
          f"water {n_lev} levelled / {n_rej} rejected")

    np.save(stem + ".npy", dsm)
    np.save(stem + "_segmentation.npy", seg_labels)

    stats = mg.generate_mesh(
        dsm, image_np, seg_labels,
        out_path=stem + ".glb",
        texture_path=stem + "_texture.png",
        heatmap_path=stem + "_heatmap.png",
        normal_map_path=stem + "_normal.png",
        ao_path=stem + "_ao.png",
        roughness_path=stem + "_roughness.png",
        metalness_path=stem + "_metalness.png",
        ao_quality="high", adaptive=True, max_dim=2048,
        has_real_scale=(tier.startswith("B")), gsd_x_m=gsd, gsd_y_m=gsd)
    stats["tier"] = tier
    stats["scale_m_per_unit"] = scale
    print(f"[5/6] mesh  {stats['buildings_extruded']} buildings, "
          f"{stats['ground_faces']} ground faces, {stats['output_mb']} MB")
    print(f"       roofs: {stats['roof_types']}")

    if stage:
        os.makedirs(VIEWER_DIR, exist_ok=True)
        for src_suffix, dst in [
            (".glb", "terrain.glb"),
            ("_texture.png", "terrain_texture.png"),
            ("_heatmap.png", "terrain_heatmap.png"),
            ("_normal.png", "terrain_normal.png"),
            ("_ao.png", "terrain_ao.png"),
            ("_roughness.png", "terrain_roughness.png"),
            ("_metalness.png", "terrain_metalness.png"),
        ]:
            src = stem + src_suffix
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(VIEWER_DIR, dst))

        # The viewer lights the scene from these angles. They are the capture's
        # real sun position, so the rendered shadows fall the same way as the
        # ones already baked into the satellite texture; a chosen-by-eye sun
        # crosses the two and the scene reads as a photograph with fake
        # lighting painted over it.
        meta = {
            "tile": tile,
            "view": os.path.basename(o["rgb_path"]),
            "off_nadir_deg": o["off_nadir_deg"],
            "sun_elevation_deg": o["sun_elev_deg"],
            "sun_azimuth_deg": o["sun_azimuth_deg"],
            "gsd_m": round(gsd, 4),
            "crs": o["crs"],
            "tier": tier,
            "scale_m_per_unit": round(float(scale), 2),
            "shadow_samples": cal.get("n", 0),
            "buildings_extruded": stats["buildings_extruded"],
            "roof_types": stats["roof_types"],
            "height_range_m": stats["height_range_m"],
        }
        with open(os.path.join(VIEWER_DIR, "scene.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[6/6] staged to {VIEWER_DIR} (sun {o['sun_elev_deg']}deg elev / "
              f"{o['sun_azimuth_deg']}deg az from capture metadata)")
    else:
        print("[6/6] not staged (--no-stage)")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tile", nargs="?", default="JAX_167")
    ap.add_argument("--global-depth", action="store_true",
                    help="whole-image inference instead of tiled crops")
    ap.add_argument("--no-stage", action="store_true")
    a = ap.parse_args()
    build(a.tile, use_tiled=not a.global_depth, stage=not a.no_stage)
