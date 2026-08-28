"""
Clean prism city model: flat ground, buildings as extruded volumes.

WHY THIS REPLACES THE HEIGHTFIELD APPROACH

Until now the whole scene was one heightfield: a grid of vertices lifted by
the DSM, with the satellite image projected straight down onto it. Buildings
were therefore *bumps in the ground*, not objects. That has three consequences
no amount of texturing or shading can fix:

  * a wall is a steep part of the terrain, so it is a ramp, not a plane -- the
    depth model smooths across every roof edge and the ground slopes up into
    each building;
  * planar UVs give a near-vertical face almost no texture area, so a handful
    of roof texels get stretched down the entire wall -- the "melting wax"
    look;
  * the roof is whatever the depth model produced, which is a lumpy surface
    rather than a flat roof.

Modelling buildings as separate prisms removes all three at once, because a
prism has genuine vertical wall faces and a genuinely flat roof by
construction. The ground underneath becomes bare earth, which is smooth, so
the satellite texture sits on it correctly.

The trade is explicit: a prism is a simplification. A building with a pitched
or stepped roof becomes a flat-topped block. That is a deliberate abstraction
-- the same one made by every vector city model -- and it is honest in a way
the lumpy version was not, because it does not pretend to roof detail the
input cannot support.
"""
import numpy as np
import cv2

import segmentation as seg


def _ear_clip(poly: np.ndarray) -> list:
    """
    Triangulate a simple polygon by ear clipping.

    Needed because real building footprints are concave (L-shapes, courtyards,
    notches) and a fan triangulation from vertex 0 -- what this project used
    before -- produces self-intersecting shards on anything non-convex. The
    earlier workaround was to replace every footprint with its minimum-area
    rectangle, which is always convex but throws the actual building shape
    away and turns a city block into a domino.

    Returns a list of index triples into `poly`.
    """
    n = len(poly)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # Work counter-clockwise so an "ear" is a convex corner with positive area.
    idx = list(range(n))
    signed = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                 for i in range(n))
    if signed < 0:
        idx.reverse()

    def point_in_tri(p, a, b, c):
        d1, d2, d3 = area2(p, a, b), area2(p, b, c), area2(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if area2(a, b, c) <= 0:
                continue                     # reflex corner, not an ear
            # An ear must contain no other vertex of the polygon.
            if any(point_in_tri(poly[j], a, b, c)
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break                            # degenerate ring; keep what we have
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def extract_footprints(seg_labels: np.ndarray, ndsm: np.ndarray = None,
                       min_area_px: int = 300, simplify_frac: float = 0.012,
                       band_m: float = 3.0) -> list:
    """
    Building footprints as simplified polygons, split by roof height.

    Connected components of the building mask alone do not give buildings.
    Once segmentation covers a realistic ~40% of a dense tile, neighbouring
    structures touch and the whole city collapses into a handful of giant
    blobs -- measured on JAX_068, 99 components for the entire scene, of which
    only 20 survived as prisms. Worse, a percentile roof height over a blob
    that spans several buildings and the street between them is not any of
    their heights.

    Adjacent buildings are almost always different heights, so quantising the
    normalised surface into bands (one band ~ a couple of storeys) and taking
    components *within* a band separates them, and gives each resulting region
    a roof at a consistent height by construction.

    approxPolyDP tolerance scales with each contour's own perimeter, so a
    large block and a small shed simplify proportionally.
    """
    mask = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if ndsm is None:
        regions = [mask]
    else:
        h = np.where(mask > 0, ndsm, np.nan)
        finite = np.isfinite(h)
        if not finite.any():
            return []
        top = float(np.nanpercentile(h, 99.5))
        n_bands = max(1, int(np.ceil(top / max(band_m, 0.5))))
        regions = []
        for b in range(n_bands):
            lo, hi = b * band_m, (b + 1) * band_m
            band = ((h >= lo) & (h < hi)).astype(np.uint8)
            if band.sum() < min_area_px:
                continue
            # Open to drop the one-pixel fringe where a band boundary cuts
            # across a sloped roof, which would otherwise spawn slivers.
            band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k)
            regions.append(band)

    polys = []
    for band in regions:
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area_px:
                continue
            eps = simplify_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
            if len(approx) < 3:
                continue
            polys.append(approx)
    return polys


def build_prisms(footprints: list, dsm: np.ndarray, dtm: np.ndarray,
                 gsd_x_m: float, gsd_y_m: float, min_height_m: float = 2.5,
                 roof_percentile: float = 80.0, image_np: np.ndarray = None):
    """
    One flat-roofed prism per footprint: vertical walls plus a flat roof.

    Height comes from the DSM inside the footprint against the terrain beneath
    it. The roof uses a percentile rather than the maximum so a single noisy
    pixel, or an aerial on the roof, does not set the whole building's height.

    The base is deliberately sunk to the LOW end of the terrain under the
    footprint. A building based at the average terrain leaves a visible gap on
    the downhill side; sinking it means the wall simply continues below ground
    where nothing can see it.
    """
    verts, faces, index, colors = [], [], [], []
    skipped = 0

    H, W = dsm.shape
    for poly in footprints:
        xs = np.clip(poly[:, 0].astype(int), 0, W - 1)
        ys = np.clip(poly[:, 1].astype(int), 0, H - 1)

        m = np.zeros((H, W), np.uint8)
        x0, y0, w, h = cv2.boundingRect(poly.astype(np.int32))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x0 + w, W), min(y0 + h, H)
        if x1 <= x0 or y1 <= y0:
            skipped += 1
            continue
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, [poly.astype(np.int32) - [x0, y0]], 1)
        inside = sub.astype(bool)
        if inside.sum() < 20:
            skipped += 1
            continue

        roof_vals = dsm[y0:y1, x0:x1][inside]
        roof_h = float(np.percentile(roof_vals, roof_percentile))

        # Base on the LOW end of the terrain under the footprint: a building
        # based at average terrain leaves a visible gap on the downhill side,
        # whereas sinking it means the wall continues below ground where
        # nothing can see it.
        base_h = float(np.percentile(dtm[y0:y1, x0:x1][inside], 5))
        if roof_h - base_h < min_height_m:
            skipped += 1
            continue

        # ROOF PLANE, fitted to the measured surface.
        #
        # Every roof used to be one flat value -- the 80th percentile of the DSM
        # inside the footprint -- so a pitched, shed or stepped roof came out as
        # a flat lid at an arbitrary height, and a whole town of them reads as
        # packing crates. The DSM does carry the roof's slope; it was simply
        # being collapsed to a scalar before anything could use it.
        #
        # A least-squares plane z = ax + by + c over the footprint's own pixels
        # recovers that slope. It is a measurement, not a style: a flat roof
        # fits a plane with a ~ b ~ 0 and is unchanged, while a shed or a tilted
        # roof gets its real pitch. It also needs no extra topology -- each roof
        # vertex simply moves to its own height -- so it costs one small solve
        # per building and nothing downstream changes.
        #
        # Fitted on the middle of the height distribution so a rooftop plant
        # room, an aerial or a parapet does not tilt the whole roof.
        ys_i, xs_i = np.nonzero(inside)
        plane = None
        if ys_i.size >= 12:
            lo, hi = np.percentile(roof_vals, [15, 90])
            keep = (roof_vals >= lo) & (roof_vals <= hi)
            if keep.sum() >= 8:
                A = np.column_stack([
                    (xs_i[keep] + x0).astype(np.float32),
                    (ys_i[keep] + y0).astype(np.float32),
                    np.ones(int(keep.sum()), np.float32)])
                try:
                    coef, *_ = np.linalg.lstsq(A, roof_vals[keep].astype(np.float32),
                                               rcond=None)
                    fitted = A @ coef
                    resid = float(np.std(roof_vals[keep] - fitted))
                    # Reject a fit that explains less than the flat value does --
                    # a noisy or occluded roof should stay flat rather than be
                    # given a slope the data does not support.
                    if resid < float(np.std(roof_vals[keep])) * 0.98:
                        plane = coef
                except np.linalg.LinAlgError:
                    plane = None

        # Roof colour, measured from the image inside this footprint.
        #
        # A uniform grey city is legible but wrong: real building stock is
        # brick, white render, dark membrane roofing, painted metal. The median
        # is used rather than the mean so a bright HVAC unit or a skylight does
        # not shift the whole roof, and it is lifted slightly toward white
        # because a roof seen in a lit 3D scene should not be as dark as its
        # nadir photograph, where it is partly self-shadowed.
        if image_np is not None:
            patch = image_np[y0:y1, x0:x1][inside]
            med = np.median(patch, axis=0).astype(np.float32) / 255.0
            # Keep the roof's real VALUE -- a dark membrane roof should stay
            # darker than white render, and that variation is most of what the
            # eye reads as "different buildings".
            #
            # Then expand chroma about the patch's own luminance. WorldView
            # imagery here is near-neutral (R, G and B within a couple of
            # levels of each other at every percentile), so the genuine colour
            # differences between brick, painted metal and tar are real but
            # tiny. Scaling the distance from grey makes them visible without
            # inventing a hue: the direction is measured, only the magnitude is
            # amplified, and it is a display choice like the ground grade.
            lum = float(med @ np.array([0.299, 0.587, 0.114], np.float32))
            rgb = np.clip(lum + (med - lum) * 2.2, 0.0, 1.0)

            # Steer toward a cool blue-grey, but do not replace the measurement.
            #
            # The scene is graded cool overall, and a neutral-white building
            # stock under a cool sky reads as unfinished CAD rather than a city.
            # Blending toward a blue-grey anchor keeps each roof's own value and
            # its own hue direction -- a brick roof stays warmer than a metal one
            # -- while placing the whole palette in one family. This is a display
            # grade on the colour channel only; no height, footprint or extent is
            # touched by it.
            anchor = np.array([0.62, 0.71, 0.82], np.float32)
            tint = 0.42
            rgb = np.clip(rgb * (1.0 - tint) + anchor * tint * (0.55 + 0.9 * lum),
                          0.0, 1.0)
            # Lift only the deepest shadows so a roof photographed in shade does
            # not become a black hole under scene lighting.
            rgb = np.clip(rgb * 0.90 + 0.13, 0.0, 1.0)
        else:
            rgb = np.array([0.82, 0.80, 0.76], np.float32)

        n = len(poly)
        off = len(verts)
        # World coordinates: X east, Y up, -Z south, matching build_ground_mesh.
        for px, py in poly:
            verts.append((px * gsd_x_m, base_h, -py * gsd_y_m))
        for px, py in poly:
            if plane is not None:
                zh = float(plane[0] * px + plane[1] * py + plane[2])
                # Keep the fitted roof inside the range the footprint actually
                # spans, so an extrapolated corner cannot spike above anything
                # measured, and never let it fall below a visible height.
                zh = min(max(zh, base_h + min_height_m), float(roof_vals.max()))
            else:
                zh = roof_h
            verts.append((px * gsd_x_m, zh, -py * gsd_y_m))
        for _ in range(2 * n):
            colors.append(rgb)

        # Walls: one quad per footprint edge, emitted as two triangles. These
        # are exactly vertical because both rings share X and Z.
        for i in range(n):
            j = (i + 1) % n
            b0, b1 = off + i, off + j
            t0, t1 = off + n + i, off + n + j
            faces.append((b0, t0, t1))
            faces.append((b0, t1, b1))

        for (a, b, c) in _ear_clip(poly):
            faces.append((off + n + a, off + n + b, off + n + c))

        index.append({
            "vertex_offset": off, "vertex_count": 2 * n, "n_sides": n,
            "base_h": base_h, "roof_h": roof_h,
            "height_m": roof_h - base_h,
            "centroid_px": (float(poly[:, 0].mean()), float(poly[:, 1].mean())),
        })

    if not verts:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
                {"buildings": [], "skipped": skipped,
                 "colors": np.zeros((0, 3), np.float32)})

    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)

    # Wall winding: a quad wound the wrong way is back-face culled and the
    # building looks hollow from outside. Rather than reason about polygon
    # orientation per footprint, fix it empirically -- point each wall normal
    # away from its own building centre.
    f = _orient_outward(v, f, index)
    return v, f, {"buildings": index, "skipped": skipped,
                  "colors": np.asarray(colors, dtype=np.float32)}


def _orient_outward(verts: np.ndarray, faces: np.ndarray, index: list) -> np.ndarray:
    """
    Flip any face whose normal points into its own building.

    Contour orientation from findContours depends on whether a region is an
    outer boundary or a hole, so assuming one winding leaves some buildings
    inside-out. Testing the normal against the direction to the building's own
    centre decides it per face and cannot be fooled by that.
    """
    faces = faces.copy()
    for rec in index:
        o, c = rec["vertex_offset"], rec["vertex_count"]
        block = verts[o:o + c]
        centre = block.mean(axis=0)
        sel = np.where((faces[:, 0] >= o) & (faces[:, 0] < o + c))[0]
        for fi in sel:
            a, b, cc = verts[faces[fi, 0]], verts[faces[fi, 1]], verts[faces[fi, 2]]
            n = np.cross(b - a, cc - a)
            if abs(n[1]) > max(abs(n[0]), abs(n[2])):
                # Roof face: must point up.
                if n[1] < 0:
                    faces[fi] = faces[fi][::-1]
            else:
                outward = ((a + b + cc) / 3.0) - centre
                outward[1] = 0.0
                if float(np.dot(n[[0, 2]], outward[[0, 2]])) < 0:
                    faces[fi] = faces[fi][::-1]
    return faces


def flatten_ground(dtm: np.ndarray, seg_labels: np.ndarray,
                   smooth_m: float = 6.0, gsd_m: float = 1.0) -> np.ndarray:
    """
    Bare-earth surface for the ground mesh.

    The DTM still carries the residue of whatever the depth model did around
    each building, and any residual bump there is now pure error: the building
    standing on that spot is a separate prism. Smoothing at a scale wider than
    a building removes the residue while keeping real terrain, which varies
    over much longer distances.
    """
    sigma_px = max(1.0, smooth_m / max(gsd_m, 1e-6))
    return cv2.GaussianBlur(dtm.astype(np.float32), (0, 0), sigmaX=sigma_px)


def build_canopy(seg_labels: np.ndarray, dsm: np.ndarray, ground: np.ndarray,
                 gsd_x_m: float, gsd_y_m: float, min_area_px: int = 250,
                 min_height_m: float = 2.0, band_m: float = 4.0):
    """
    Tree canopy as low rounded volumes.

    A prism city with no vegetation reads as a model of a car park: the
    reference imagery this is aimed at is full of green, and every real tile
    here has substantial tree cover that the pipeline already segments and
    then throws away. Canopy is emitted as its own mesh so it can take a green
    material rather than the buildings' stone.

    Canopy is extruded like a building but with the top ring inset toward the
    patch centre, which gives a tapered crown instead of a green box. Trees are
    not boxes, and at this scale a taper is enough to read as foliage.
    """
    veg = (seg_labels == seg.CLASS_IDX["vegetation"]).astype(np.uint8)
    if not veg.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, k)

    ndsm = np.maximum(dsm - ground, 0.0)
    h = np.where(veg > 0, ndsm, np.nan)
    if not np.isfinite(h).any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    top = float(np.nanpercentile(h, 99.0))
    n_bands = max(1, int(np.ceil(top / max(band_m, 0.5))))

    verts, faces = [], []
    count = 0
    for b in range(n_bands):
        lo, hi = b * band_m, (b + 1) * band_m
        band = ((h >= lo) & (h < hi)).astype(np.uint8)
        if band.sum() < min_area_px:
            continue
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area_px:
                continue
            eps = 0.02 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
            if len(poly) < 3:
                continue

            x0, y0, w, hh = cv2.boundingRect(poly.astype(np.int32))
            x1, y1 = min(x0 + w, dsm.shape[1]), min(y0 + hh, dsm.shape[0])
            x0, y0 = max(x0, 0), max(y0, 0)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(sub, [poly.astype(np.int32) - [x0, y0]], 1)
            inside = sub.astype(bool)
            if inside.sum() < 20:
                continue

            crown = float(np.percentile(dsm[y0:y1, x0:x1][inside], 75))
            base = float(np.percentile(ground[y0:y1, x0:x1][inside], 20))
            if crown - base < min_height_m:
                continue

            centre = poly.mean(axis=0)
            n = len(poly)
            off = len(verts)
            for px, py in poly:
                verts.append((px * gsd_x_m, base, -py * gsd_y_m))
            for px, py in poly:
                # Inset the crown so the volume tapers upward.
                ix = centre[0] + (px - centre[0]) * 0.62
                iy = centre[1] + (py - centre[1]) * 0.62
                verts.append((ix * gsd_x_m, crown, -iy * gsd_y_m))

            for i in range(n):
                j = (i + 1) % n
                b0, b1 = off + i, off + j
                t0, t1 = off + n + i, off + n + j
                faces.append((b0, t0, t1))
                faces.append((b0, t1, b1))
            for (a, bb, c) in _ear_clip(poly):
                faces.append((off + n + a, off + n + bb, off + n + c))
            count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int64)
    return v, f, count


def build_water(seg_labels: np.ndarray, ground: np.ndarray,
                gsd_x_m: float, gsd_y_m: float, min_area_px: int = 4000):
    """
    Water as a single flat surface per body, slightly below its banks.

    Standing water is flat by definition, so it is the one part of the scene
    that should NOT follow the estimated terrain. Sinking it a little under
    the surrounding ground stops the banks z-fighting with the water plane.
    """
    water = (seg_labels == seg.CLASS_IDX["water"]).astype(np.uint8)
    if not water.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    verts, faces = [], []
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            continue
        xs = np.clip(poly[:, 0].astype(int), 0, ground.shape[1] - 1)
        ys = np.clip(poly[:, 1].astype(int), 0, ground.shape[0] - 1)
        level = float(np.percentile(ground[ys, xs], 25)) - 0.3

        off = len(verts)
        for px, py in poly:
            verts.append((px * gsd_x_m, level, -py * gsd_y_m))
        for (a, b, c) in _ear_clip(poly):
            faces.append((off + a, off + b, off + c))
        count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64), count


def detect_vehicles(image_np: np.ndarray, seg_labels: np.ndarray,
                    ground: np.ndarray, gsd_m: float,
                    min_area_m2: float = 4.0, max_area_m2: float = 24.0,
                    height_m: float = 1.5):
    """
    Cars on open ground, as small boxes.

    A car is roughly 4.5 x 1.8 m. At this ortho's 0.25 m/px that is about 130
    pixels -- large enough to find, small enough that the only usable cues are
    size, shape and local contrast. There is no vehicle class in the
    segmentation, so this is a heuristic detector, and it is gated hard:

      * only on open ground -- never on a roof, in vegetation or on water,
        which removes the large majority of look-alike blobs;
      * area within a real vehicle's footprint, in SQUARE METRES rather than
        pixels, so the gate does not silently change with resolution;
      * elongation between 1.5 and 4.5, since cars are oblong and most
        confusers (manhole covers, patches, bins) are not;
      * fill ratio against the oriented box, so ragged shadow slivers fail.

    Expect false positives on roof vents, skips and shadow edges, and misses on
    cars whose colour matches the tarmac. This is decoration, not a measured
    inventory, and the count is returned so it can be reported as such rather
    than implied to be exact.
    """
    open_ground = ((seg_labels == seg.CLASS_IDX["road"]) |
                   (seg_labels == seg.CLASS_IDX["bare_earth"]))
    if not open_ground.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Local contrast: a vehicle is brighter or darker than the surface it sits
    # on, whichever way round, so both tails are taken.
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, 2.5 / gsd_m))
    diff = gray - local
    sigma = float(np.std(diff[open_ground])) or 1.0
    blobs = ((np.abs(diff) > 1.6 * sigma) & open_ground).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, k)
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, k)

    px_area = gsd_m * gsd_m
    min_px = max(6, int(min_area_m2 / px_area))
    max_px = int(max_area_m2 / px_area)

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    verts, faces = [], []
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_px or area > max_px:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        if w < 1e-3 or h < 1e-3:
            continue
        long_side, short_side = max(w, h), min(w, h)
        elong = long_side / short_side
        if elong < 1.5 or elong > 4.5:
            continue
        if area / (w * h) < 0.55:
            continue
        # Reject anything longer than a bus or narrower than a bicycle.
        if long_side * gsd_m > 14.0 or short_side * gsd_m < 1.2:
            continue

        box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
        ys = np.clip(box[:, 1].astype(int), 0, ground.shape[0] - 1)
        xs = np.clip(box[:, 0].astype(int), 0, ground.shape[1] - 1)
        base = float(np.median(ground[ys, xs]))

        off = len(verts)
        for px, py in box:
            verts.append((px * gsd_m, base, -py * gsd_m))
        for px, py in box:
            verts.append((px * gsd_m, base + height_m, -py * gsd_m))
        for i in range(4):
            j = (i + 1) % 4
            b0, b1, t0, t1 = off + i, off + j, off + 4 + i, off + 4 + j
            faces.append((b0, t0, t1)); faces.append((b0, t1, b1))
        faces.append((off + 4, off + 5, off + 6))
        faces.append((off + 4, off + 6, off + 7))
        count += 1

    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64), count
