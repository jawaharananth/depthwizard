import argparse
import numpy as np
from PIL import Image

from depth_model import DepthBackbone
from calibration.tiered import calibrate
from calibration.georeferenced import estimate_gsd_meters
import segmentation
import shadow_correction
import dsm_export
import mesh_generation


def load_image_as_rgb(image_path: str):
    """
    PIL can't open multiband/uint16 GeoTIFFs (raises UnidentifiedImageError),
    so satellite GeoTIFF inputs must go through rasterio and get percentile-
    stretched to uint8 RGB for the depth model / texture pipeline.
    """
    if not image_path.lower().endswith((".tif", ".tiff")):
        pil_img = Image.open(image_path).convert("RGB")
        return pil_img, np.array(pil_img)

    import rasterio
    with rasterio.open(image_path) as src:
        band_count = src.count
        n = min(band_count, 3)
        bands = [src.read(i + 1).astype(np.float32) for i in range(n)]

    def _to_uint8(band):
        lo, hi = np.percentile(band, [2, 98])
        return np.clip((band - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    if n == 1:
        gray = _to_uint8(bands[0])
        image_np = np.stack([gray, gray, gray], axis=-1)
    else:
        image_np = np.stack([_to_uint8(b) for b in bands[:3]], axis=-1)

    return Image.fromarray(image_np), image_np


def get_geo_info(image_path: str):
    if not image_path.lower().endswith((".tif", ".tiff")):
        return None
    try:
        import rasterio
        with rasterio.open(image_path) as src:
            if src.crs is None:
                return None
            bounds = src.bounds
            return {"bounds_wgs84": (bounds.left, bounds.bottom, bounds.right, bounds.top)}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--out", default="dsm_output.npy")
    parser.add_argument("--lat", type=float, default=None, help="capture latitude, for shadow sun-angle calc")
    parser.add_argument("--lon", type=float, default=None, help="capture longitude, for shadow sun-angle calc")
    parser.add_argument("--capture-datetime-utc", default=None,
                         help="ISO format e.g. 2026-03-15T06:30:00, for shadow sun-angle calc")
    parser.add_argument("--sun-elevation", type=float, default=None, help="override: skip astro calc entirely")
    parser.add_argument("--sun-azimuth", type=float, default=None, help="override: skip astro calc entirely")
    parser.add_argument("--mesh-resolution", type=int, default=1024,
                         help="Max vertex-grid dimension. Textures are always full source "
                              "resolution regardless. Cost scales quadratically: 800 -> ~36MB, "
                              "1024 -> ~59MB, 1600 -> ~143MB, 2048 -> ~235MB. Above ~1600 the "
                              "GLB gets slow to download and parse in a browser.")
    parser.add_argument("--ao-quality", default="high", choices=["fast", "high", "ultra"],
                         help="Sky-view-factor occlusion sampling density.")
    parser.add_argument("--vertex-budget", type=int, default=400_000,
                         help="Adaptive mesh vertex budget. Vertices go where detail is "
                              "(roof edges, building outlines) instead of spreading evenly, "
                              "so this buys far more visible quality than the same count on "
                              "a uniform grid. ~400k is high quality; 800k+ for maximum.")
    parser.add_argument("--uniform-mesh", action="store_true",
                         help="Disable adaptive tessellation and use a plain uniform grid.")
    args = parser.parse_args()

    print("[1/6] Loading image + running depth model...")
    pil_img, image_np = load_image_as_rgb(args.image)

    model = DepthBackbone()
    relative_depth = model.predict(pil_img)

    print("[2/6] Running semantic segmentation...")
    seg_labels = None
    if args.image.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.open(args.image) as src:
                band_count = src.count
            if band_count >= 4:
                seg_labels, seg_stats = segmentation.segment_from_geotiff(args.image)
            else:
                seg_labels, seg_stats = segmentation.segment(image_np)
        except Exception as e:
            print(f"[segmentation] GeoTIFF multispectral path failed ({e}), falling back to RGB heuristic")
            seg_labels, seg_stats = segmentation.segment(image_np)
    else:
        seg_labels, seg_stats = segmentation.segment(image_np)

    print("Segmentation class distribution:")
    for name in segmentation.CLASSES:
        print(f"  {name:12s} {seg_stats.get(name, 0)*100:5.1f}%")

    print("[3/6] Running calibration...")
    geo_info = get_geo_info(args.image)
    dsm, meta = calibrate(relative_depth, image_np, geo_info=geo_info, dem_path=args.dem,
                           seg_labels=seg_labels)
    print("Tier used:", meta["tier"])

    print("[4/6] Shadow-based height cross-validation...")
    gsd_x_m, gsd_y_m = None, None
    if meta["tier"].startswith("A_"):
        if "gsd_x_m" in meta:
            gsd_x_m, gsd_y_m = meta["gsd_x_m"], meta["gsd_y_m"]
        else:
            gsd_x_m, gsd_y_m = estimate_gsd_meters(geo_info["bounds_wgs84"], relative_depth.shape)
    elif meta["tier"] == "B_object_scale" and "m_per_px_estimate" in meta:
        gsd_x_m = gsd_y_m = meta["m_per_px_estimate"]

    sun_elevation, sun_azimuth = args.sun_elevation, args.sun_azimuth
    if sun_elevation is None or sun_azimuth is None:
        lat, lon, dt_utc = args.lat, args.lon, None
        if args.capture_datetime_utc:
            import datetime as _dt
            dt_utc = _dt.datetime.fromisoformat(args.capture_datetime_utc)
        if (lat is None or lon is None or dt_utc is None) and args.image.lower().endswith((".tif", ".tiff")):
            cap = shadow_correction.extract_capture_metadata(args.image)
            lat = lat if lat is not None else cap["lat"]
            lon = lon if lon is not None else cap["lon"]
            dt_utc = dt_utc if dt_utc is not None else cap["datetime_utc"]
        if lat is not None and lon is not None and dt_utc is not None:
            sun_elevation, sun_azimuth = shadow_correction.sun_position(lat, lon, dt_utc)
        else:
            print("  Skipped: no sun angle available (pass --sun-elevation/--sun-azimuth, or "
                  "--lat/--lon/--capture-datetime-utc, or use a GeoTIFF with capture-time metadata)")

    shadow_results = None
    if gsd_x_m is None:
        print("  Skipped: no real-world scale available (Tier C has no metric reference)")
    elif sun_elevation is not None and sun_azimuth is not None:
        if sun_elevation <= 0:
            print(f"  Skipped: sun below horizon (elevation={sun_elevation:.1f}deg)")
        else:
            print(f"  Sun position: elevation={sun_elevation:.1f}deg azimuth={sun_azimuth:.1f}deg")
            shadow_results = shadow_correction.cross_validate_heights(
                image_np, seg_labels, dsm, sun_elevation, sun_azimuth, gsd_x_m, gsd_y_m)
            checked = [r for r in shadow_results if r["shadow_height_m"] is not None]
            print(f"  {len(shadow_results)} building(s) detected, {len(checked)} with measurable shadow")
            for r in checked[:10]:
                print(f"    building #{r['building_id']}: AI={r['ai_height_m']}m  "
                      f"shadow={r['shadow_height_m']}m  diff={r['diff_m']}m  "
                      f"error={r['relative_error_pct']}%  confidence={r['confidence']}")

    print("[5/6] Exporting DSM GeoTIFF...")
    bounds_wgs84 = geo_info["bounds_wgs84"] if geo_info is not None else None
    export_tags = {"tier": meta["tier"]}
    if "a" in meta:
        export_tags["scale_a"] = meta["a"]
        export_tags["shift_b"] = meta["b"]
    if "curves" in meta:
        export_tags["per_terrain_curves"] = meta["curves"]
    geotiff_out = args.out.replace(".npy", ".tif")
    dsm_export.export_dsm_geotiff(dsm, geotiff_out, bounds_wgs84=bounds_wgs84, tags=export_tags)
    print("  Saved DSM GeoTIFF to", geotiff_out,
          "(georeferenced)" if bounds_wgs84 is not None else "(relative heights, no CRS -- no ground truth available)")

    print("[6/6] Generating 3D mesh (ground + buildings)...")
    mesh_out = args.out.replace(".npy", ".glb")  # binary glTF: ~5x smaller and far faster to load than OBJ
    texture_out = args.out.replace(".npy", "_texture.png")
    heatmap_out = args.out.replace(".npy", "_heatmap.png")
    normal_out = args.out.replace(".npy", "_normal.png")
    ao_out = args.out.replace(".npy", "_ao.png")
    rough_out = args.out.replace(".npy", "_roughness.png")
    metal_out = args.out.replace(".npy", "_metalness.png")
    mesh_gsd_x = gsd_x_m if gsd_x_m is not None else 1.0
    mesh_gsd_y = gsd_y_m if gsd_y_m is not None else 1.0

    # Edge-aware refinement: the depth backbone produces soft, ramped walls;
    # the co-registered image has the sharp edge in the right place. Guided
    # filtering transfers that edge structure onto the elevation, so buildings
    # get crisp boundaries instead of blobs (measured 1.3x edge sharpness).
    #
    # Applied to the MESH ONLY. The scientific DSM exported above is untouched
    # until this is benchmarked against DFC2019 LiDAR -- if it measurably
    # improves RMSE it can be promoted to the exported product too.
    import dsm_refine
    import terrain_maps
    dsm_for_mesh = dsm_refine.refine_dsm(dsm, image_np)
    print(f"  Edge-aware refinement: sharpness "
          f"{dsm_refine.edge_sharpness(dsm):.2f} -> "
          f"{dsm_refine.edge_sharpness(dsm_for_mesh):.2f} "
          f"(mesh only; exported DSM.tif unchanged)")

    # Standing water is flat; the depth model has no such prior and returns a
    # rippled surface, which both reads wrong and destroys the specular
    # reflection that makes water look like water.
    dsm_for_mesh, n_water, n_rej = terrain_maps.flatten_water(dsm_for_mesh, seg_labels)
    print(f"  Water levelling: {n_water} body(s) levelled, {n_rej} candidate(s) rejected "
          f"(too small, too much relief, or shadow-shaped)")

    if gsd_x_m is None:
        # Tier C output is unitless relative depth in [0,1] -- fed to the mesh
        # as-is, an 850m-wide scene gets ~1m of visible relief, effectively
        # flat. This scales it to a legible visualization range ONLY for the
        # mesh (the already-exported DSM.tif above is untouched, still raw
        # 0-1, still honestly labeled non-metric). This is display exaggeration,
        # the same technique any terrain renderer uses for low-relief data --
        # not a metric claim. The mesh/viewer still reports no real GSD.
        exaggeration = 40.0
        dsm_for_mesh = dsm_for_mesh * exaggeration  # chain onto the refined DSM, not raw
        print(f"  Applying {exaggeration:.0f}x visualization-only height exaggeration "
              f"(Tier C shape is correct, scale is not; DSM.tif above is unaffected)")

    mesh_stats = mesh_generation.generate_mesh(
        dsm_for_mesh, image_np, seg_labels, mesh_out, texture_out, heatmap_path=heatmap_out,
        normal_map_path=normal_out, ao_path=ao_out, roughness_path=rough_out,
        metalness_path=metal_out, ao_quality=args.ao_quality,
        max_dim=args.mesh_resolution,
        adaptive=not args.uniform_mesh, vertex_budget=args.vertex_budget,
        gsd_x_m=mesh_gsd_x, gsd_y_m=mesh_gsd_y, has_real_scale=(gsd_x_m is not None))
    print(f"  Saved mesh to {mesh_out} ({mesh_stats['ground_faces']} ground faces, "
          f"{mesh_stats['buildings_extruded']} buildings extruded, {mesh_stats['output_mb']} MB)")
    print(f"  Textures at full source resolution {mesh_stats['texture_resolution']}, "
          f"mesh grid {mesh_stats['mesh_grid']}")
    rt = mesh_stats.get("roof_types") or {}
    if rt:
        total = sum(rt.values()) or 1
        summary = "  ".join(f"{k} {v} ({v/total*100:.0f}%)"
                            for k, v in sorted(rt.items(), key=lambda kv: -kv[1]))
        print(f"  Roof shapes inferred: {summary}")
    if gsd_x_m is None:
        print("  Note: no real GSD available (Tier C) -- mesh uses 1 unit/pixel, shape is correct but not true-to-scale")

    print("Done.")
    print("Calibration metadata:", meta)

    np.save(args.out, dsm)
    seg_out = args.out.replace(".npy", "_segmentation.npy")
    np.save(seg_out, seg_labels)
    print("Saved DSM array to", args.out)
    print("Saved segmentation labels to", seg_out)

    if shadow_results is not None:
        import json
        shadow_out = args.out.replace(".npy", "_shadow_validation.json")
        with open(shadow_out, "w") as f:
            json.dump(shadow_results, f, indent=2)
        print("Saved shadow cross-validation results to", shadow_out)


if __name__ == "__main__":
    main()
