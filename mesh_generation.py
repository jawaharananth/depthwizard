"""
Turns a calibrated DSM + segmentation into real 3D geometry:
  - a ground heightfield mesh, UV-textured with the original image
  - verticalized building boxes extruded from footprint polygons + height

Exports plain OBJ+MTL (no extra 3D library needed, and Three.js's OBJLoader
reads it natively) rather than a heavier format like glTF.
"""
import numpy as np
import cv2
from PIL import Image

import segmentation as seg
import roof_structure


def _resize_for_mesh(dsm: np.ndarray, seg_labels: np.ndarray, max_dim: int = 800):
    """
    Downsamples ONLY the geometry grid (vertex density), never the texture.

    Texture resolution and mesh resolution are completely independent: a
    400x300 vertex grid can carry a 4096x4096 texture with zero quality loss,
    because texels are interpolated across each triangle. An earlier version
    resized the RGB texture to the mesh grid too, which meant zooming in
    showed a 400px image stretched over the whole terrain -- the single
    biggest cause of the blurry close-up look. Full-res texture is now always
    written by generate_mesh() straight from the source image.
    """
    h, w = dsm.shape
    scale = min(1.0, max_dim / max(h, w))
    new_h, new_w = max(2, int(h * scale)), max(2, int(w * scale))

    dsm_small = cv2.resize(dsm.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    seg_small = cv2.resize(seg_labels.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return dsm_small, seg_small


def build_ground_mesh(dsm: np.ndarray, gsd_x_m: float, gsd_y_m: float):
    """
    Grid heightfield: one vertex per pixel, two triangles per quad.
    Returns (vertices Nx3, uvs Nx2, faces Mx3 1-indexed for OBJ).
    """
    h, w = dsm.shape
    xs = np.arange(w) * gsd_x_m
    ys = np.arange(h) * gsd_y_m
    grid_x, grid_y = np.meshgrid(xs, ys)

    vertices = np.stack([grid_x.ravel(), dsm.ravel(), -grid_y.ravel()], axis=-1)  # Y-up, -Z forward
    uvs = np.stack([(grid_x / max(xs.max(), 1e-6)).ravel(), (1 - grid_y / max(ys.max(), 1e-6)).ravel()], axis=-1)

    # Vectorized quad -> 2 triangles. A Python double loop here is ~0.5M
    # iterations at an 800px grid and dominates total runtime; numpy builds
    # the same index arrays in one shot.
    rows = np.arange(h - 1)[:, None]
    cols = np.arange(w - 1)[None, :]
    v0 = (rows * w + cols).ravel()
    v1 = v0 + 1
    v2 = v0 + w
    v3 = v2 + 1

    # CCW winding as seen from above (+Y): (v0,v1,v2) and (v1,v3,v2) both give
    # +Y-facing normals -- reversed order back-face-culls the entire ground
    # mesh invisible from any top-down camera.
    faces = np.empty((v0.size * 2, 3), dtype=np.int64)
    faces[0::2] = np.stack([v0, v1, v2], axis=-1)
    faces[1::2] = np.stack([v1, v3, v2], axis=-1)

    return vertices, uvs, faces


def _building_footprints(seg_labels: np.ndarray, min_area_px: int = 15):
    """
    Connected building regions -> oriented bounding rectangles (4-point,
    always convex), not the raw contour polygon.

    Earlier version used cv2.approxPolyDP's variable-vertex-count polygon
    directly, then fan-triangulated the roof from vertex 0. That's only valid
    for convex polygons -- real building footprints from segmentation are
    often concave/notched/noisy, and fan-triangulating a concave polygon
    produces self-intersecting "crown"/shard geometry (confirmed visually:
    zoomed screenshots showed exactly this on real building extrusions).
    A minAreaRect is always a valid convex quad, so wall+roof extrusion can
    never self-intersect, and it reads as an actual boxy building instead of
    a jagged flag shape -- a fair trade since the footprint mask is noisy
    pixel-classification output anyway, not survey-grade polygon data.
    """
    building_mask = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8) * 255
    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  # 4x2, always a valid convex quad
        polygons.append(box.astype(np.float32))
    return polygons


def build_building_meshes(seg_labels: np.ndarray, dsm: np.ndarray, gsd_x_m: float, gsd_y_m: float,
                           ground_ref: float = None, min_height_m: float = 2.0,
                           infer_roofs: bool = True, dtm: np.ndarray = None):
    """
    Extrudes each building footprint (a 4-point oriented rectangle -- see
    _building_footprints for why) from ground_ref up to its local roof
    height, into a clean box: 4 walls + a 2-triangle flat roof. Always valid
    since a rectangle can't be concave or self-intersecting.

    Also emits planar (top-down) UVs so buildings sample the SAME satellite
    texture as the ground. Roofs then show their real roof imagery instead of
    a flat placeholder color. Walls inherit the UV of the footprint edge they
    rise from, so they show a vertical smear of the ground pixels at their
    base -- the standard trade-off for single-nadir-image city models, since
    a satellite photo contains no facade imagery at all (nothing is visible
    from the side in a top-down capture, so there is nothing truthful to map
    there).
    Returns (vertices Nx3, uvs Nx2, faces Mx3 0-indexed local to this mesh).
    """
    # A single scene-wide base elevation sinks every building that stands on
    # higher-than-average ground -- measured, 30 of 40 buildings on a test
    # scene had their base below the terrain beside them. When a DTM is
    # supplied each building is based on the terrain under its own footprint
    # instead; ground_ref is only a fallback for callers without one.
    if ground_ref is None:
        ground_ref = float(np.percentile(dtm if dtm is not None else dsm, 10))

    img_h, img_w = dsm.shape
    polygons = _building_footprints(seg_labels)

    # Roof shape is classified on a DETRENDED copy: monocular depth imposes a
    # scene-wide gradient that a per-building slope test reports as a shed
    # roof on almost everything (measured 65.5% shed before detrending vs
    # 37.7% after). Heights themselves still come from the untouched DSM.
    roof_lookup = {}
    if infer_roofs:
        flat_dsm = roof_structure.detrend(dsm)
        for i, p in enumerate(polygons):
            roof_lookup[i] = roof_structure.classify_roof(p, flat_dsm)

    all_vertices = []
    all_uvs = []
    all_faces = []
    vcount = 0
    roof_counts = {}
    # Ordered per-building records. roof_counts alone groups by TYPE, which is
    # useless for walking the vertex stream -- buildings are emitted in polygon
    # order with types interleaved. Verification code that slices by type order
    # mis-attributes vertices and reports phantom defects.
    building_index = []
    unmeasurable = 0

    for poly_index, poly_px in enumerate(polygons):
        # Crop to bounding box before rasterizing the fill mask -- allocating a
        # full-image mask per building (thousands of them at full resolution)
        # is thousands of times more work than needed for a typically small footprint.
        poly_px_int = np.round(poly_px).astype(np.int32)
        x0, y0, w, h = cv2.boundingRect(poly_px_int)
        w, h = max(w, 1), max(h, 1)
        local_poly = poly_px_int - [x0, y0]
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(local_mask, [local_poly], 1)
        # Clamp to valid array bounds -- a footprint's oriented rect can
        # extend past the image edge (buildings partially in-frame at a tile
        # boundary), and dsm[...] silently truncates while local_mask doesn't,
        # producing a shape mismatch when boolean-indexing.
        y1, x1 = min(y0 + h, dsm.shape[0]), min(x0 + w, dsm.shape[1])
        y0c, x0c = max(y0, 0), max(x0, 0)
        dsm_crop = dsm[y0c:y1, x0c:x1]
        mask_crop = local_mask[y0c - y0:y1 - y0, x0c - x0:x1 - x0]
        region = dsm_crop[mask_crop.astype(bool)]
        if region.size == 0:
            continue
        roof_h = float(np.percentile(region, 90))

        # Base this building on its OWN terrain, not a global constant.
        #
        # Two failure modes have to be excluded, and neither is guaranteed by
        # the elevation data alone. Measured on a dense downtown tile where the
        # depth signal barely separates buildings from ground (building mean
        # 0.2718 vs ground 0.2736): of 506 buildings, 93 had their ROOF below
        # local terrain -- entirely invisible -- and 92 had their base floating
        # above it, leaving a gap. Only 63% were placed correctly.
        #
        # So the invariant is enforced geometrically rather than inherited:
        #   base at the LOW end of terrain under the footprint  -> no gap; the
        #     wall simply continues below ground, which is hidden.
        #   roof at least min_height above that base            -> always visible.
        if dtm is not None:
            import dtm as dtm_mod
            pts = np.round(poly_px).astype(np.int32)
            bx0, by0, bw, bh = cv2.boundingRect(pts)
            by1, bx1 = min(by0 + bh, dtm.shape[0]), min(bx0 + bw, dtm.shape[1])
            by0c, bx0c = max(by0, 0), max(bx0, 0)
            if by1 > by0c and bx1 > bx0c:
                local_terrain = dtm[by0c:by1, bx0c:bx1]
                base_h = float(np.percentile(local_terrain, 5))
                # Sample terrain at the EXACT footprint corners, then take the
                # highest. A percentile summary of the bounding box cannot
                # guarantee the invariant, because visibility is decided at the
                # corner vertices themselves -- successive percentile tweaks
                # (95th, then 99th) each still let a handful of invisible
                # buildings through. Enforcing on the same points that are
                # checked makes the guarantee exact rather than statistical.
                cx_i = np.clip(poly_px[:, 0].astype(int), 0, dtm.shape[1] - 1)
                cy_i = np.clip(poly_px[:, 1].astype(int), 0, dtm.shape[0] - 1)
                corner_terrain = dtm[cy_i, cx_i]
                terrain_top = float(max(corner_terrain.max(),
                                        float(np.percentile(local_terrain, 95))))
            else:
                base_h = ground_ref
                terrain_top = ground_ref
        else:
            base_h = ground_ref
            terrain_top = ground_ref

        measured_height = roof_h - base_h
        if measured_height < min_height_m:
            continue  # too short to bother extruding as a distinct volume

        # If the measured roof does not clear the terrain it stands on, the
        # elevation data contains no usable height for this structure. Raising
        # it to a minimum would make it visible, but the height would then be
        # asserted rather than measured -- on a dense downtown tile that path
        # fabricated 753 of 808 heights. Omitting the building is the honest
        # outcome: a missing structure is a visible, countable gap, whereas an
        # invented one silently pollutes every measurement taken from the model.
        if roof_h < terrain_top + min_height_m:
            unmeasurable += 1
            continue

        n = len(poly_px)

        def _uv(px, py):
            # same planar UV convention as build_ground_mesh, so buildings and
            # ground index the same satellite texture consistently
            return (px / max(img_w - 1, 1), 1.0 - py / max(img_h - 1, 1))

        # Per-corner eave heights. A flat roof has all four equal; a shed roof
        # ramps them along its slope axis, which tilts the roof plane without
        # needing extra geometry.
        roof_info = roof_lookup.get(poly_index) if roof_lookup else None
        rtype = (roof_info or {}).get("type", "flat")
        roof_counts[rtype] = roof_counts.get(rtype, 0) + 1

        # Heights come from the REAL dsm region, never from classify_roof's
        # return values -- those were measured on the detrended copy, where
        # absolute elevation is meaningless. Only the roof TYPE and slope
        # direction carry over from classification.
        eave_h = [roof_h] * n
        ridge = None

        if roof_info and rtype == roof_structure.SHED:
            hi = float(np.percentile(region, 92))
            lo = float(np.percentile(region, 8))
            centre, long_ax, short_ax, half_long, half_short = \
                roof_structure._oriented_frame(poly_px)
            axis = short_ax if roof_info.get("slope_axis") == "short" else long_ax
            half = half_short if roof_info.get("slope_axis") == "short" else half_long
            sign = float(roof_info.get("slope_sign", 1.0))
            for i, (px, py) in enumerate(poly_px):
                t = float(np.dot(np.array([px, py], np.float32) - centre, axis)) / max(half, 1e-6)
                t = np.clip(t * sign, -1.0, 1.0)
                eave_h[i] = lo + (hi - lo) * (t * 0.5 + 0.5)

        elif roof_info and rtype in (roof_structure.GABLE, roof_structure.HIP):
            eh = float(np.percentile(region, 12))
            rh = float(np.percentile(region, 95))
            eave_h = [eh] * n
            centre, long_ax, short_ax, half_long, half_short = \
                roof_structure._oriented_frame(poly_px)
            # Ridge runs along the building's long axis, as real ridges do.
            # A hip roof insets the ridge from both gable ends; a gable roof
            # carries it the full length.
            inset = 0.35 if rtype == roof_structure.HIP else 0.0
            r_half = half_long * (1.0 - inset)
            r0 = centre - long_ax * r_half
            r1 = centre + long_ax * r_half
            ridge = [(float(r0[0]), rh, float(r0[1])), (float(r1[0]), rh, float(r1[1]))]

        base = [(px * gsd_x_m, base_h, -py * gsd_y_m) for px, py in poly_px]
        eave = [(px * gsd_x_m, eave_h[i], -py * gsd_y_m)
                for i, (px, py) in enumerate(poly_px)]
        uv = [_uv(px, py) for px, py in poly_px]

        verts = base + eave          # 0..n-1 base, n..2n-1 eave
        uvs = uv + uv

        # walls, base ring to eave ring
        for i in range(n):
            j = (i + 1) % n
            b0, b1, e0, e1 = i, j, n + i, n + j
            all_faces.append((vcount + b0, vcount + e0, vcount + b1))
            all_faces.append((vcount + b1, vcount + e0, vcount + e1))

        if ridge is None:
            # flat or shed: the eave ring itself is the roof surface
            for i in range(1, n - 1):
                all_faces.append((vcount + n, vcount + n + i, vcount + n + i + 1))
        else:
            # pitched: two ridge vertices, two sloping roof planes, and closed
            # gable ends. Footprint corner order from boxPoints is consistent,
            # so pairing edges to ridge ends by projection keeps winding sane.
            ridge_idx = 2 * n
            for rx, ry, rz in ridge:
                verts.append((rx * gsd_x_m, ry, -rz * gsd_y_m))
                uvs.append(_uv(rx, rz))

            centre, long_ax, _, half_long, _ = roof_structure._oriented_frame(poly_px)
            # which ridge end each corner belongs to
            side = []
            for (px, py) in poly_px:
                t = float(np.dot(np.array([px, py], np.float32) - centre, long_ax))
                side.append(0 if t < 0 else 1)

            for i in range(n):
                j = (i + 1) % n
                e0, e1 = n + i, n + j
                if side[i] == side[j]:
                    # edge lies at one end of the building -> gable/hip end
                    all_faces.append((vcount + e0, vcount + ridge_idx + side[i], vcount + e1))
                else:
                    # edge runs along the building -> sloping roof plane
                    a = ridge_idx + side[i]
                    b = ridge_idx + side[j]
                    all_faces.append((vcount + e0, vcount + a, vcount + e1))
                    all_faces.append((vcount + e1, vcount + a, vcount + b))

        building_index.append({
            "vertex_offset": vcount,
            "vertex_count": len(verts),
            "roof_type": rtype,
            "base_h": float(base_h),
            "roof_h": float(roof_h),
            "centroid_px": (float(np.mean(poly_px[:, 0])), float(np.mean(poly_px[:, 1]))),
        })
        all_vertices.extend(verts)
        all_uvs.extend(uvs)
        vcount += len(verts)

    if not all_vertices:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int64),
                {"counts": roof_counts, "buildings": building_index,
                 "unmeasurable": unmeasurable})
    return (np.array(all_vertices, dtype=np.float32),
            np.array(all_uvs, dtype=np.float32),
            np.array(all_faces, dtype=np.int64),
            {"counts": roof_counts, "buildings": building_index,
                 "unmeasurable": unmeasurable})


def export_obj(out_path: str, ground_verts, ground_uvs, ground_faces,
               building_verts, building_faces, texture_path: str):
    mtl_path = out_path.replace(".obj", ".mtl")
    mat_name = "terrain_material"

    with open(mtl_path, "w") as f:
        f.write(f"newmtl {mat_name}\nKd 1 1 1\nmap_Kd {texture_path}\n")
        f.write("newmtl building_material\nKd 0.75 0.72 0.68\n")

    with open(out_path, "w") as f:
        f.write(f"mtllib {mtl_path.split('/')[-1].split(chr(92))[-1]}\n")

        f.write("o ground\n")
        for v in ground_verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for uv in ground_uvs:
            f.write(f"vt {uv[0]:.5f} {uv[1]:.5f}\n")
        f.write(f"usemtl {mat_name}\n")
        for face in ground_faces:
            i0, i1, i2 = face + 1  # OBJ is 1-indexed
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

        if len(building_verts) > 0:
            f.write("o buildings\n")
            offset = len(ground_verts)
            for v in building_verts:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            f.write("usemtl building_material\n")
            for face in building_faces:
                i0, i1, i2 = face + offset + 1
                f.write(f"f {i0} {i1} {i2}\n")

    return out_path


def _save_normal_map(dsm: np.ndarray, gsd_x_m: float, gsd_y_m: float, out_path: str,
                      strength: float = 3.0):
    """
    Tangent-space normal map from the DSM's own surface gradient (standard
    Sobel-based technique) -- every roof edge, curb, and elevation ripple gets
    real per-pixel lighting response instead of the flat draped-photo look a
    plain diffuse texture gives, without needing any new source imagery.
    strength exaggerates the gradient (real vertical relief here is often
    just meters over hundreds of meters horizontally -- essentially flat in
    raw XYZ terms -- so a literal-scale normal map would look almost perfectly
    flat; this is the same "exaggeration factor" every terrain renderer uses
    for the same reason, not a fabrication of fake detail).
    """
    dzdy, dzdx = np.gradient(dsm.astype(np.float32), gsd_y_m, gsd_x_m)
    nx = -dzdx * strength
    ny = -dzdy * strength
    nz = np.ones_like(dsm, dtype=np.float32)

    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx, ny, nz = nx / length, ny / length, nz / length

    # standard tangent-space normal map encoding: [-1,1] -> [0,255], Y-down
    # convention matches most glTF/three.js expectations (OpenGL-style, +Y up in tangent space)
    r = ((nx * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((ny * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((nz * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_rgb = np.stack([r, g, b], axis=-1)
    # Flat tangent-space normal (128,128,255) on walls. The gradient there is
    # enormous and gets stretched down the whole surface by the planar UVs,
    # producing vertical banding rather than detail.
    import terrain_maps as _tm
    normal_rgb = _tm.calm_steep_map(normal_rgb, dsm, gsd_x_m, (128, 128, 255))
    Image.fromarray(normal_rgb).save(out_path)


def _save_heatmap_texture(dsm: np.ndarray, out_path: str):
    """Blue (low) -> red (high) elevation colormap, same pixel grid as the ground UVs."""
    import matplotlib
    lo, hi = np.percentile(dsm, [2, 98])
    norm = np.clip((dsm - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = (matplotlib.colormaps["jet"](norm) * 255).astype(np.uint8)
    Image.fromarray(rgba[:, :, :3]).save(out_path)


def generate_mesh(dsm: np.ndarray, image_np: np.ndarray, seg_labels: np.ndarray,
                   out_path: str, texture_path: str, heatmap_path: str = None,
                   normal_map_path: str = None, ao_path: str = None,
                   roughness_path: str = None, metalness_path: str = None,
                   ao_quality: str = "high", adaptive: bool = True,
                   vertex_budget: int = 400_000, separate_terrain: bool = True,
                   gsd_x_m: float = 1.0, gsd_y_m: float = 1.0, max_dim: int = 800,
                   has_real_scale: bool = True):
    """
    max_dim caps only the VERTEX GRID, not texture quality -- see
    _resize_for_mesh. Textures (RGB, heatmap, normal) are always written at
    the source image's full resolution.

    has_real_scale: False for Tier C (relative-only) DSMs, whose values aren't
    meters at all. The building-height filter (min_height_m, default 2.0m)
    would then silently discard every building since a 0-1 normalized DSM
    never reaches "2 meters" -- so with has_real_scale=False the threshold is
    instead computed as a fraction of the DSM's own value range.
    """
    dsm_small, seg_small = _resize_for_mesh(dsm, seg_labels, max_dim)
    # gsd scales up proportionally to the downsample so real-world size is preserved
    scale = dsm.shape[0] / dsm_small.shape[0]
    gsd_x_eff, gsd_y_eff = gsd_x_m * scale, gsd_y_m * scale

    # All three textures at FULL source resolution, independent of mesh density.
    #
    # Steep surfaces get a neutral facade tone first. Planar UVs stretch the
    # texels on a roof edge down the entire wall beneath it, and a nadir
    # capture holds no facade imagery to put there instead -- untreated, every
    # building reads as melting wax from any low camera angle.
    import terrain_maps as _tm
    textured, facade_frac = _tm.neutralise_facades(image_np, dsm, gsd_x_m)
    Image.fromarray(textured).save(texture_path)
    if heatmap_path:
        _save_heatmap_texture(dsm, heatmap_path)
    if normal_map_path:
        # normal maps add apparent detail beyond the mesh's actual vertex
        # density -- using the downsampled grid would throw away exactly the
        # fine relief (roof edges, curbs) this technique exists to preserve.
        _save_normal_map(dsm, gsd_x_m, gsd_y_m, normal_map_path)

    ao_stats = None
    if ao_path or roughness_path or metalness_path:
        # Occlusion baked from the reconstructed geometry itself (sky-view
        # factor), not a screen-space approximation. This is what makes
        # buildings sit in the scene rather than appear pasted on a plane.
        import terrain_maps
        ao_stats = terrain_maps.bake_terrain_maps(
            dsm, seg_labels, gsd_m=(gsd_x_m + gsd_y_m) / 2.0,
            ao_path=ao_path, roughness_path=roughness_path,
            metalness_path=metalness_path, quality=ao_quality)

    # THE GROUND SURFACE MUST BE TERRAIN, NOT THE SURFACE MODEL.
    #
    # A DSM already contains building heights. Building the ground mesh from it
    # and then ALSO extruding building boxes renders every structure twice --
    # once as a bump in the "ground", once as a box occupying the same space,
    # with the box typically buried inside the bump it duplicates. Separating
    # bare-earth terrain from structures is what makes buildings sit ON the
    # ground rather than in it.
    if separate_terrain:
        import dtm as dtm_mod
        terrain = dtm_mod.estimate_dtm(dsm, seg_labels)
        terrain_small = cv2.resize(terrain, (dsm_small.shape[1], dsm_small.shape[0]),
                                    interpolation=cv2.INTER_AREA)
    else:
        terrain = dsm
        terrain_small = dsm_small

    if adaptive:
        # Vertices placed by information content rather than spread evenly:
        # roof edges and building outlines sampled at full resolution, flat
        # ground left to a coarse lattice. Measured on the reference tile at a
        # 150k budget: 28% of the uniform face count for 0.41m RMSE against
        # the true height field, on a 41.9m scene -- about 1% geometric error.
        import adaptive_mesh
        ground_v, ground_uv, ground_f = adaptive_mesh.build_adaptive_ground_mesh(
            terrain_small, gsd_x_eff, gsd_y_eff, seg_small, budget_verts=vertex_budget)
    else:
        ground_v, ground_uv, ground_f = build_ground_mesh(terrain_small, gsd_x_eff, gsd_y_eff)

    min_height = 2.0 if has_real_scale else 0.02 * (float(dsm.max()) - float(dsm.min()))

    # Buildings extruded from FULL-RESOLUTION seg_labels/dsm, not the downsampled
    # ground grid: downsampling for a multi-thousand-pixel tile shrinks small
    # residential footprints below the min-area filter and loses them entirely
    # -- verified on real data, 2416 detected buildings became just 7 extruded
    # volumes at 400px. Ground mesh stays downsampled (per-pixel vertex
    # fidelity matters far less there); buildings need full resolution.
    building_v, building_uv, building_f, building_info = build_building_meshes(
        seg_labels, dsm, gsd_x_m, gsd_y_m, None, min_height_m=min_height,
        dtm=terrain if separate_terrain else None)
    roof_counts = building_info["counts"]

    if out_path.lower().endswith(".glb"):
        import io
        from glb_export import export_glb
        tex_buf = io.BytesIO()
        Image.fromarray(image_np).save(tex_buf, format="PNG")
        export_glb(out_path, ground_v, ground_uv, ground_f, building_v, building_f,
                   texture_bytes=tex_buf.getvalue(), building_uvs=building_uv)
    else:
        export_obj(out_path, ground_v, ground_uv, ground_f, building_v, building_f,
                   texture_path.split("/")[-1].split("\\")[-1])

    import os
    return {
        "ao_stats": ao_stats,
        "roof_types": roof_counts,
        "ground_vertices": len(ground_v), "ground_faces": len(ground_f),
        "building_vertices": len(building_v), "building_faces": len(building_f),
        # Counted directly from roof classification, not derived from vertex
        # count: pitched roofs add ridge vertices, so buildings no longer have
        # a fixed vertex count. Dividing by 8 was correct only while every
        # building was a flat-top box, and silently drifts otherwise.
        "buildings_extruded": sum(roof_counts.values()),
        "mesh_grid": dsm_small.shape,
        "texture_resolution": image_np.shape[:2],
        "height_range_m": [float(dsm.min()), float(dsm.max())],
        "output_mb": round(os.path.getsize(out_path) / 1e6, 1),
        "facade_fraction": round(facade_frac, 4),
    }


if __name__ == "__main__":
    import sys
    from depth_model import DepthBackbone

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.array(pil_img)

    model = DepthBackbone()
    relative_depth = model.predict(pil_img)
    seg_labels, _ = seg.segment(image_np)

    stats = generate_mesh(relative_depth * 50, image_np, seg_labels,
                           "mesh_test.obj", "mesh_test_texture.png")
    print(stats)
