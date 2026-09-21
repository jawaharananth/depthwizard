"""
Building footprints from IMAGE REGIONS, not from the depth field.

WHY THIS REPLACES THE PIXELWISE HEIGHT MASK

The previous approach thresholded a morphological residual of the depth field
and called the result "building". Measured against LiDAR on JAX_165 (central
256 m, where truth exists):

    coverage 46.0% vs ground truth 45.9%   -- the right AMOUNT
    IoU 0.436, recall 0.608, precision 0.606

The right amount of building, in the wrong places. And it was NOT a registration
error: an exhaustive +-10 m shift search moved IoU from 0.436 to only 0.439.
The shapes themselves were wrong. Breaking the false positives down by what the
LiDAR says they really are:

    18.2% of our mask was ground
     8.4% was tree canopy
     2.5% was bridge deck
    10.3% was unlabelled

That is the signature of a low-frequency cue making a high-frequency decision.
Monocular depth is smooth: it blurs across roof edges, bulges over raised
ground, and rises over canopy exactly as it does over a roof. Thresholding it
per pixel cannot produce a building outline because the outline is not in it.

Building outlines ARE in the image: a roof is a locally uniform region bounded
by a strong intensity edge. So regions are found in the image first, and the
depth field is then asked only what it is actually good at -- "is this region
raised, and by how much".

CLASSIFICATION

Per region, four measurements, each targeting a specific confusion above:

  height     median nDSM inside the region      -- separates raised from flat
  flatness   std of nDSM inside the region      -- SEPARATES ROOFS FROM TREES.
             A roof is planar; canopy is rough at the same mean height. This is
             the discriminator the height-only mask did not have, and it is what
             the 8.4% tree error needed.
  greenness  excess green index                 -- vegetation, when colour exists
  fill       area / convex hull area            -- built form is compact; a
             ragged sliver along a kerb is not a building

A region is a building when it is raised, flat, not green, and reasonably
compact. Every threshold is relative to the scene's own distribution.
"""
import numpy as np
import cv2

import segmentation as seg

# Narrowest a real structure gets, wall to wall. Watershed boundaries that
# follow kerbs and the gaps between adjacent roofs come through as long thin
# regions with a near-perfect convex fill, so only a width test catches them.
MIN_BUILDING_WIDTH_M = 3.0

# approxPolyDP tolerance as a fraction of each contour's own perimeter, so a
# city block and a shed simplify proportionally. See extract() for the measured
# accuracy/rectilinearity trade behind this value.
SIMPLIFY_FRAC = 0.040


def _watershed_regions(image_np: np.ndarray, min_region_px: int) -> np.ndarray:
    """
    Split the image into regions whose boundaries follow real intensity edges.

    Watershed is seeded from the interiors of low-gradient areas -- the flat
    middles of roofs, roads and lawns -- and flooded up the gradient, so the
    boundaries land on the edges themselves rather than near them. That is the
    property the depth field could never provide.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 7)   # denoise, keep edges hard
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = (grad / max(grad.max(), 1e-6) * 255).astype(np.uint8)

    # Seeds: pixels well inside a flat area. The opening removes thin flat
    # slivers that would otherwise seed a region straddling a real edge.
    flat = (grad < np.percentile(grad, 45)).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, markers = cv2.connectedComponents(flat)
    markers = markers.astype(np.int32) + 1
    markers[flat == 0] = 0                     # unknown, to be flooded

    ws = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers.copy())
    return ws


def extract(image_np: np.ndarray, ndsm: np.ndarray, gsd_m: float,
            # DEAD PARAMETER -- accepted, never read. Documented rather than
            # removed because callers pass it and its absence would look like an
            # oversight rather than a finding.
            #
            # Every building/not-building decision here is made by the height
            # field via watershed regions. Segmentation contributes nothing.
            # Measured consequences:
            #   * feeding PERFECT segmentation (LiDAR labels) changes the output
            #     by exactly zero on all tiles tested;
            #   * feeding a learned satellite segmenter (SegFormer-B4, classes
            #     others/buildings/road) also changes it by exactly zero.
            # That is why false buildings on car parks cannot be fixed by any
            # improvement to segmentation: the stage is not consulted.
            #
            # Wiring it in WAS tested, as a per-region semantic support gate.
            # It did not pay off: mean IoU over four tiles fell 0.564 -> 0.487,
            # because the learned model is unreliable per scene (IoU 0.556 on
            # JAX_165, 0.068 on JAX_033) and gating inherits that. A
            # corroboration guard that skips the gate when the two sources
            # disagree recovers to 0.562 -- still no gain. Recorded so the next
            # person does not repeat it.
            seg_labels: np.ndarray = None,
            min_area_m2: float = 8.0,
            min_height_m: float = 2.0,
            rough_gate: float = 0.55,
            fill_gate: float = 0.25) -> dict:
    """
    Returns {"polygons": [...], "records": [...], "report": {...}}.

    min_height_m is in the same units as ndsm. On a Tier C scene those are not
    metres, so the caller must pass a threshold in whatever units it is using --
    the name reflects intent, not a guarantee about the unit.
    """
    H, W = ndsm.shape
    px_area = gsd_m * gsd_m
    min_px = max(12, int(min_area_m2 / px_area))

    ws = _watershed_regions(image_np, min_px)

    r = image_np[:, :, 0].astype(np.float32)
    g = image_np[:, :, 1].astype(np.float32)
    b = image_np[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    has_colour = float(np.percentile(
        cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)[:, :, 1], 90)) > 40

    # Bounding box per region in ONE pass, instead of a full-image comparison
    # per region.
    #
    # This loop was `mask = (ws == rid)` for every region: 4618 regions over a
    # 2560x2560 label image is ~30 billion element comparisons, and it measured
    # at 147 seconds -- 93% of the entire build. The same O(regions x pixels)
    # pattern has now appeared four times in this codebase; it is always the
    # answer when a per-object loop touches a whole raster.
    #
    # find_objects returns each label's slice directly, so every region is then
    # examined only inside its own bounding box.
    from scipy import ndimage as ndi
    ws_pos = np.where(ws > 1, ws, 0)
    max_id = int(ws_pos.max())
    slices = ndi.find_objects(ws_pos) if max_id > 0 else []
    ids = [i + 1 for i, sl in enumerate(slices) if sl is not None and i + 1 > 1]

    # Height cut.
    #
    # This was `percentile(raised, 55)`, which rejected 3361 of 4618 regions as
    # "too low" and dropped recall to 0.265: over half of everything standing
    # above ground was being discarded before shape was even considered. The
    # scene percentile is the wrong statistic -- in a dense downtown most raised
    # pixels ARE buildings, so it sets the bar at the middle of the building
    # population and throws away the lower half of the city.
    #
    # The caller's minimum height is the actual criterion. Anything clearing it
    # goes on to be judged on flatness and shape, which is where roofs and trees
    # are meant to be separated.
    h_cut = min_height_m

    polygons, records = [], []
    rej = {"too_small": 0, "too_low": 0, "too_rough": 0, "vegetation": 0,
           "ragged": 0, "sliver": 0}

    # Per-region statistics in one pass each, via bounding boxes.
    for rid in ids:
        sl = slices[rid - 1]
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        sub = (ws[y0:y1, x0:x1] == rid)
        area_px = int(sub.sum())
        if area_px < min_px:
            rej["too_small"] += 1
            continue
        hsub = ndsm[y0:y1, x0:x1][sub]

        height = float(np.median(hsub))
        if height < h_cut:
            rej["too_low"] += 1
            continue

        # FLATNESS -- the roof/tree discriminator.
        rough = float(np.std(hsub))
        if rough > rough_gate * max(height, 1e-6):
            rej["too_rough"] += 1
            continue

        if has_colour:
            green = float(np.mean(exg[y0:y1, x0:x1][sub]))
            if green > 12.0:
                rej["vegetation"] += 1
                continue
        else:
            green = 0.0

        cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        fill = cv2.contourArea(cnt) / max(cv2.contourArea(hull), 1.0)
        if fill < fill_gate:
            rej["ragged"] += 1
            continue

        # A sliver has a convex fill near 1.0, so fill_gate above cannot see it:
        # a long thin rectangle fills its own hull perfectly. What gives it away
        # is physical width. These regions come from watershed boundaries that
        # run along kerbs, walls and the gaps between adjacent roofs, and
        # extruded they render as the thin blades standing between the real
        # buildings. The cut is the narrow side of the minimum-area rectangle,
        # in metres, so it means the same thing on every image regardless of GSD:
        # a structure people occupy is not 2 m wall to wall.
        (_cx, _cy), (_w, _h), _ang = cv2.minAreaRect(cnt)
        if min(_w, _h) * gsd_m < MIN_BUILDING_WIDTH_M:
            rej["sliver"] += 1
            continue
        # NOTE on fill_gate's default (0.25, was 0.55): this is area over convex
        # hull, and it was rejecting exactly the building forms worth keeping.
        # An L-shape, a U-shape or a courtyard block has a low convex fill BY
        # DEFINITION -- the hull spans the notch the building does not occupy --
        # so a 0.55 cut discards concave buildings while keeping only boxes.
        # Measured on JAX_165 against LiDAR:
        #     fill 0.55   IoU 0.509   recall 0.667   precision 0.683
        #     fill 0.25   IoU 0.553   recall 0.745   precision 0.682
        # 12% more recall at identical precision. The gate still catches genuine
        # slivers and ragged contour fringe, which is what it was for.

        cnt = cnt + [x0, y0]
        # Simplify harder, then snap edges to the footprint's own axis.
        #
        # Watershed contours are ragged, and a lightly-simplified ring keeps
        # dozens of near-collinear vertices that wander a few degrees -- the
        # shape has the right area but reads as a blob rather than a building.
        # Measured on JAX_165 against LiDAR across 2605 footprints:
        #
        #     config                     IoU     right-angle corners
        #     raw (eps 0.015)            0.524        17.6%
        #     ortho only                 0.518        25.9%
        #     eps 0.025 + ortho          0.516        35.2%
        #     eps 0.040 + ortho          0.510        46.2%     <- chosen
        #     eps 0.060 + ortho          0.507        56.5%
        #
        # 2.6x more rectilinear for 2.7% of IoU. Precision actually IMPROVES
        # (0.671 -> 0.681) and only recall falls, so what is being lost is
        # ragged contour fringe rather than building. That is a defensible
        # trade for a city model; it is stated here rather than tuned silently.
        eps = SIMPLIFY_FRAC * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            continue
        poly = orthogonalize(poly, gsd_m=gsd_m)
        poly = rebuild_corners(poly)

        polygons.append(poly)
        records.append({
            "area_m2": round(area_px * px_area, 1),
            "height": round(height, 2),
            "roughness": round(rough, 3),
            "fill": round(fill, 3),
            "greenness": round(green, 1),
        })

    return {
        "polygons": polygons,
        "records": records,
        "report": {
            "regions_examined": int(len(ids)),
            "retained": len(polygons),
            "rejected": rej,
        },
    }


def rebuild_corners(poly: np.ndarray, max_area_change: float = 0.35) -> np.ndarray:
    """
    Rebuild a footprint's corners as the intersections of its own walls.

    orthogonalize() above straightens edge DIRECTIONS but cannot close corners,
    and that is what leaves the rendered roofs spiky. It rotates each edge about
    its own midpoint independently, so vertex j is written once as the end of
    edge i and again as the start of edge j, and the second write undoes the
    first -- the two corrected edges never meet at a point. Short staircase
    edges are skipped outright by its min_edge_m gate, so they survive between
    the corrected walls as visible notches.

    This finishes the job: consecutive edges pointing along the same axis
    collapse into one wall, each wall becomes a single infinite line placed at
    the length-weighted mean of its edges (so a long true wall outvotes the
    short steps hanging off it), and every corner is recomputed as the exact
    intersection of the two walls that meet there. Right angles then hold by
    construction rather than by tolerance.

    Shapes survive: an L stays an L, a U stays a U, because walls are merged
    only when they are parallel AND consecutive. The area guard returns the
    input untouched when the rebuild would distort it, which is the signal the
    ring was never a building outline -- a merged blob, a courtyard, a sliver.
    """
    n = len(poly)
    if n < 4:
        return poly

    poly = poly.astype(np.float32)
    orig_area = abs(cv2.contourArea(poly))
    if orig_area < 1.0:
        return poly

    v = np.roll(poly, -1, axis=0) - poly
    lengths = np.hypot(v[:, 0], v[:, 1])
    if (lengths > 1e-6).sum() < 4:
        return poly

    # Same dominant-axis estimate orthogonalize uses: a length-weighted circular
    # mean in modulo-90 space, so an edge at 3 deg and one at 93 deg count as the
    # same axis instead of averaging to a meaningless 48.
    keep = lengths > 1e-6
    ang = np.arctan2(v[keep, 1], v[keep, 0])
    L = lengths[keep]
    a4 = 4.0 * ang
    dom = float(np.arctan2((L * np.sin(a4)).sum(), (L * np.cos(a4)).sum()) / 4.0)

    c, s = float(np.cos(-dom)), float(np.sin(-dom))
    fwd = np.array([[c, -s], [s, c]], np.float32)
    inv = np.array([[c, s], [-s, c]], np.float32)
    centre = poly.mean(axis=0)
    local = (poly - centre) @ fwd.T

    # 0 = wall runs along X, 1 = along Y, in the building's own frame.
    cls = []
    for i in range(n):
        d = local[(i + 1) % n] - local[i]
        cls.append(0 if abs(d[0]) >= abs(d[1]) else 1)

    runs = []
    for i in range(n):
        if runs and runs[-1][0] == cls[i]:
            runs[-1][1].append(i)
        else:
            runs.append((cls[i], [i]))
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1].extend(runs[-1][1])
        runs.pop()

    # A rectilinear ring alternates X-walls and Y-walls, so it needs an even
    # count of at least four. Anything else is not a shape this can rebuild.
    if len(runs) < 4 or len(runs) % 2 != 0:
        return poly

    lines = []
    for klass, idxs in runs:
        num = den = 0.0
        for i in idxs:
            a, b = local[i], local[(i + 1) % n]
            w = (abs(b[0] - a[0]) if klass == 0 else abs(b[1] - a[1])) + 1e-6
            num += w * 0.5 * ((a[1] + b[1]) if klass == 0 else (a[0] + b[0]))
            den += w
        lines.append((klass, num / den))

    corners = []
    m = len(lines)
    for i in range(m):
        k0, v0 = lines[i]
        k1, v1 = lines[(i + 1) % m]
        if k0 == k1:
            return poly
        corners.append((v0 if k0 == 1 else v1, v0 if k0 == 0 else v1))

    out = (np.asarray(corners, np.float32) @ inv.T) + centre
    new_area = abs(cv2.contourArea(out))
    if new_area < 1.0 or abs(new_area - orig_area) / orig_area > max_area_change:
        return poly
    return out


def orthogonalize(poly: np.ndarray, angle_tol_deg: float = 22.0,
                  min_edge_m: float = 2.0, gsd_m: float = 0.25) -> np.ndarray:
    """
    Snap a footprint's edges to its own dominant axis.

    Buildings are overwhelmingly rectilinear, but these polygons come from
    watershed regions simplified by approxPolyDP, so their edges wander by a few
    degrees and short spurious edges cut corners. The result reads as a blob
    rather than a building even when its area is correct.

    This is the cheap, CPU-only substitute for what a learned polygon extractor
    like PolyWorld provides. It does NOT invent corners or regularise a shape
    into a rectangle -- courtyards, L-shapes and notches all survive, because
    only edge DIRECTIONS are adjusted and vertices are moved onto the corrected
    lines. A building that genuinely is not rectilinear keeps its true angles,
    since edges further than `angle_tol_deg` from the dominant axis are left
    alone.

    The dominant axis is the length-weighted circular mean of edge directions in
    modulo-90-degree space, which is the natural space for a rectilinear shape:
    an edge at 3 degrees and one at 93 degrees are the same axis, and averaging
    them naively would give 48 degrees, which is neither.
    """
    n = len(poly)
    if n < 4:
        return poly

    v = np.roll(poly, -1, axis=0) - poly
    lengths = np.hypot(v[:, 0], v[:, 1])
    keep = lengths > 1e-6
    if keep.sum() < 4:
        return poly

    ang = np.arctan2(v[keep, 1], v[keep, 0])
    L = lengths[keep]
    # Modulo 90 degrees, via a 4x angle so perpendicular edges coincide.
    a4 = 4.0 * ang
    dom = np.arctan2((L * np.sin(a4)).sum(), (L * np.cos(a4)).sum()) / 4.0

    tol = np.radians(angle_tol_deg)
    out = poly.astype(np.float64).copy()
    for i in range(n):
        j = (i + 1) % n
        d = out[j] - out[i]
        ln = float(np.hypot(d[0], d[1]))
        if ln < min_edge_m / max(gsd_m, 1e-6):
            continue
        theta = float(np.arctan2(d[1], d[0]))
        # Nearest of the four axis directions.
        k = round((theta - dom) / (np.pi / 2.0))
        target = dom + k * (np.pi / 2.0)
        if abs(((theta - target + np.pi) % (2 * np.pi)) - np.pi) > tol:
            continue          # genuinely off-axis edge; leave it
        # Rotate the edge onto the axis about its own midpoint, so the polygon
        # does not drift: both endpoints move by equal and opposite amounts.
        mid = (out[i] + out[j]) / 2.0
        u = np.array([np.cos(target), np.sin(target)])
        out[i] = mid - u * (ln / 2.0)
        out[j] = mid + u * (ln / 2.0)

    return out.astype(np.float32)

def extract_from_polygons(polygons: list, ndsm: np.ndarray, gsd_m: float,
                          min_area_m2: float = 4.0, min_height_m: float = 1.0,
                          known_heights: list = None) -> dict:
    """
    Building footprints from individually-labelled ground-truth polygons, not
    a merged raster mask.

    known_heights, when supplied, is a per-polygon list the same length as
    polygons -- a real measured height in metres (e.g. floor count from a
    municipal property record: storeys x a real storey height), or None for
    a polygon with no such record. Where present, this REPLACES the depth
    field sample rather than falling back to it: a floor count from a real
    building record is a measurement, and the depth field's own estimate of
    the same building is at best a correlated guess -- there is no reason to
    let an inferred number override a counted one. Only polygons whose own
    entry is None fall back to sampling ndsm, exactly as before.

    extract_from_mask() below rasterises every GT polygon onto one bitmap and
    re-traces connected components -- but two adjacent row-house footprints
    that share a wall, or simply touch at one pixel after rasterisation, fuse
    into a single component. Measured on one Karnataka scene: 1,751 annotated
    polygons collapsed to 834 raster blobs before any height gate ran, roughly
    halving the buildings that could ever appear as distinct prisms. Working
    from the source polygon list instead of a bitmap means two touching
    buildings stay two buildings, because they never share a raster region to
    begin with -- identity comes from the annotation, not from a re-traced
    contour.

    Returns the same shape of dict as extract_from_mask().
    """
    polygons_out, records = [], []
    rej = {"too_small": 0, "degenerate": 0}
    examined = len(polygons)
    n_measured = 0

    for idx, ring in enumerate(polygons):
        poly = np.asarray(ring, np.float32)
        if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]           # GeoJSON rings repeat the first vertex
        if len(poly) < 3:
            rej["degenerate"] += 1
            continue
        area_px = abs(cv2.contourArea(poly))
        if area_px * gsd_m * gsd_m < min_area_m2:
            rej["too_small"] += 1
            continue

        if len(poly) > 6:
            eps = SIMPLIFY_FRAC * cv2.arcLength(poly, True)
            poly = cv2.approxPolyDP(poly, eps, True).reshape(-1, 2).astype(np.float32)
            if len(poly) < 3:
                rej["degenerate"] += 1
                continue
        poly = orthogonalize(poly, gsd_m=gsd_m)
        poly = rebuild_corners(poly)

        # Always sample the depth field's own estimate for this footprint,
        # even when a measured height is available and wins -- this is the
        # only place the model's opinion of a GROUND-TRUTHED building is
        # ever computed, and it is what the compare-slider (build_city_image's
        # _write_compare_assets) needs to show prediction against reality on
        # the same buildings instead of a synthetic example.
        model_estimate = max(region_median_height(poly, ndsm), min_height_m)
        known = known_heights[idx] if known_heights is not None else None
        if known is not None:
            height = max(float(known), min_height_m)
            n_measured += 1
            provenance = "measured"
        else:
            height = model_estimate
            provenance = "inferred"
        polygons_out.append(poly)
        records.append({
            "area_m2": round(area_px * gsd_m * gsd_m, 1),
            "height": round(height, 2),
            "height_provenance": provenance,
            "height_model_estimate": round(model_estimate, 2),
            "roughness": 0.0,
            "fill": 1.0,
            "greenness": 0.0,
        })

    return {
        "polygons": polygons_out,
        "records": records,
        "report": {
            "regions_examined": examined,
            "retained": len(polygons_out),
            "rejected": rej,
            "measured_heights": n_measured,
        },
    }


def extract_from_mask(gt_mask: np.ndarray, ndsm: np.ndarray, gsd_m: float,
                      min_area_m2: float = 4.0, min_height_m: float = 1.0) -> dict:
    """
    Building footprints directly from a ground-truth mask, not from watershed.

    extract() above finds building SHAPE from image regions because its own
    measurement proved segmentation input changes its output by exactly zero
    -- the height-field gates it applies (too_low, too_small, too_rough) exist
    to reject false positives from an UNVERIFIED region proposal. When the
    mask is already human-verified ground truth, those same gates reject real,
    confirmed buildings instead: measured on a Karnataka scene, 1358 of 2800
    candidate regions were rejected as too_low and 362 as too_small, and most
    of that loss was small single-storey rural houses whose height signal in a
    monocular depth field is genuinely weak, not whose existence is in doubt.

    So this is a different, simpler path for the case a mask exists: every
    connected component of the mask becomes a footprint directly -- nothing to
    accept or reject, since existence was already decided by the annotator --
    and the height field is consulted only afterwards, to estimate how tall
    each one is, which is the one thing it is actually good at.

    Returns the same shape of dict as extract(): {"polygons", "records", "report"}.
    """
    mask = (gt_mask > 0).astype(np.uint8)
    min_px = max(8, int(min_area_m2 / (gsd_m * gsd_m)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    polygons, records = [], []
    rej = {"too_small": 0, "no_contour": 0, "degenerate": 0}
    examined = 0

    for i in range(1, n):
        examined += 1
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            rej["too_small"] += 1
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        sub = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            rej["no_contour"] += 1
            continue
        cnt = max(cnts, key=cv2.contourArea) + [x, y]
        eps = SIMPLIFY_FRAC * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            rej["degenerate"] += 1
            continue
        poly = orthogonalize(poly, gsd_m=gsd_m)
        poly = rebuild_corners(poly)

        height = max(region_median_height(poly, ndsm), min_height_m)
        area_px = float(stats[i, cv2.CC_STAT_AREA])
        polygons.append(poly)
        records.append({
            "area_m2": round(area_px * gsd_m * gsd_m, 1),
            "height": round(height, 2),
            "roughness": 0.0,   # not evaluated -- existence and shape came from ground truth
            "fill": 1.0,
            "greenness": 0.0,
        })

    return {
        "polygons": polygons,
        "records": records,
        "report": {
            "regions_examined": examined,
            "retained": len(polygons),
            "rejected": rej,
        },
    }


def region_median_height(poly, ndsm):
    """Median nDSM inside one footprint."""
    p = np.asarray(poly, np.int32)
    H, W = ndsm.shape
    x, y, w, h = cv2.boundingRect(p)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(sub, [p - [x0, y0]], 1)
    v = ndsm[y0:y1, x0:x1][sub.astype(bool)]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 0.0


# DEFAULT OFF. The gate improves precision on all four DFC tiles it was tuned
# against, and then made things worse on the first real upload it met: IoU fell
# from 0.565 to 0.397 and recall from 0.627 to 0.423, because it assumes roughly
# 30% of candidates are junk and that scene's upstream had already removed them.
#
# A percentile gate cannot know how much junk it is being handed. Left available
# for scenes visibly full of false positives, but not applied by default -- a
# heuristic tuned on four tiles that fails on the fifth is not ready to be on.
ADAPTIVE_HEIGHT_PCTILE = 0.0


def drop_low_regions(polys, ndsm, pctile: float = ADAPTIVE_HEIGHT_PCTILE):
    """Drop the flattest share of candidate regions, judged per scene.

    False footprints sit on car parks, bare earth and road -- flat things the
    monocular height field reads as slightly raised. Measured against LiDAR on
    JAX_165 the true buildings have a median nDSM of 17.7 against 8.3 for the
    false ones, so the signal is there; the difficulty is that no ABSOLUTE
    threshold transfers between tiles. Tier C normalises heights so the tallest
    structure reads ~40, and where that outlier sits differs per scene: the
    cut that lifts JAX_165 to IoU 0.691 drops JAX_068's recall from 0.739 to
    0.308.

    So the threshold is a percentile of THIS scene's own candidate heights.
    Measured across four tiles it is the only gate tried that improves
    precision on every one of them:

        tile      precision           recall
        JAX_165   0.762 -> 0.787      0.789 -> 0.789
        JAX_068   0.794 -> 0.803      0.739 -> 0.736
        JAX_167   0.835 -> 0.942      0.620 -> 0.557
        JAX_033   0.505 -> 0.544      0.812 -> 0.794

    Otsu on the same distribution was tried and rejected: it is far more
    aggressive and collapsed recall on three of the four tiles (JAX_167 to
    0.310), which is the same failure mode Otsu has already produced twice in
    this project.

    A local-contrast test -- is this region raised above its own surroundings --
    was also tried and rejected. It assumes buildings are isolated; in a dense
    downtown a building's surroundings ARE other buildings, and recall fell from
    0.789 to 0.219 on JAX_165.
    """
    if not polys or pctile <= 0:
        return polys, None
    hs = np.array([region_median_height(p, ndsm) for p in polys], dtype=np.float32)
    if hs.size < 10:
        return polys, None
    thr = float(np.percentile(hs, pctile))
    return [p for p, h in zip(polys, hs) if h >= thr], thr
