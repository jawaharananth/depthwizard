"""
Recall-oriented building discovery.

Replaces the previous approach, which was a single-scale connected-component
pass followed by `if contourArea < 15: continue`. Measured on sample.jpg that
discarded 2,101 of 2,515 candidates -- 83.5% -- with no record of what was
dropped or why, and a further 550 structured regions never even reached the
building class because segmentation missed them.

Three changes address that:

1. CANDIDATES COME FROM MORE THAN SEGMENTATION. The semantic building mask is
   one source; structured regions that segmentation missed are a second. A
   low-contrast roof that the class heuristic skipped still gets a hearing.

2. DETECTION RUNS AT MULTIPLE SCALES. Downsampling merges neighbouring
   structures and erases small ones; upsampled/native passes recover them.
   Detections are fused across scales with overlap suppression.

3. NOTHING IS DROPPED ON SIZE ALONE. Every candidate is scored on six
   independent evidence signals and retained unless evidence is effectively
   zero. Every rejection records a reason. A 5x8 pixel structure with a cast
   shadow and a crisp edge is a building; a 5x8 pixel blob with no edge, no
   shadow, no elevation and vegetation-like texture is noise. The difference
   is decided by evidence, not by area.
"""
import numpy as np
import cv2

import segmentation as seg
from buildings.records import Building, Evidence, Rejection, DetectionReport, Provenance

# A candidate must clear this to be kept. Deliberately low: the design bias is
# toward false positives that later stages can weed out, over silent misses.
MIN_EVIDENCE = 0.08
MIN_AREA_ABSOLUTE = 4  # below ~2x2 px there is no measurable evidence of anything


def _edge_density(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    return cv2.blur(edges, (ksize, ksize))


def _local_variance(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (ksize, ksize))
    sq = cv2.blur(g * g, (ksize, ksize))
    return np.clip(sq - mean * mean, 0, None)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (courtyards excepted later) via border flood fill."""
    h, w = mask.shape
    ff = mask.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return (mask | (1 - ff)).astype(np.uint8)


def _candidate_mask(image_np: np.ndarray, seg_labels: np.ndarray) -> np.ndarray:
    """
    Coherent roof REGIONS, not edge pixels.

    An earlier version unioned the segmentation mask with a raw edge-density
    threshold. That was wrong in a way worth recording: high edge density
    marks boundaries, so it yields thin wiggly fragments rather than filled
    roof areas. Visual inspection showed the resulting "buildings" were edge
    scribbles scattered over roads and open ground -- the candidate count rose
    but almost none of it was a building.

    Measurement also disproved the assumption behind that design. Only 19% of
    sub-15px components sit near a large building, so they are mostly not
    fragments of big roofs; and a simple morphological close consolidates
    2,515 raw components into 715 coherent regions. Fragmentation is real,
    but it is fixed by merging, not by adding an edge-derived source.

    So: start from the semantic building class, close gaps that split one roof
    (chimney shadows, ridge lines), remove isolated specks, and fill interior
    holes. Recovery of segmentation misses is handled separately by
    _roof_like_regions, which looks for filled roof-like areas rather than edges.
    """
    from_seg = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    merged = cv2.morphologyEx(from_seg, cv2.MORPH_CLOSE, close_k)

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(merged, cv2.MORPH_OPEN, open_k)

    recovered = _roof_like_regions(image_np, seg_labels)
    # NOTE: no global hole filling here. Building regions across a city form a
    # connected network that touches the image border, so a border flood fill
    # collapses the entire scene into one blob (measured: 2,515 candidates ->
    # 1). Any hole filling must be per-component and size-limited; it is not
    # needed for detection and is left to polygon reconstruction.
    return ((cleaned | recovered) > 0).astype(np.uint8)


def _roof_like_regions(image_np: np.ndarray, seg_labels: np.ndarray) -> np.ndarray:
    """
    Recover buildings segmentation missed, as filled regions.

    A roof that the class heuristic skipped is still typically a compact,
    internally smooth area that is distinctly brighter or darker than the
    ground around it, and is not vegetation or water. That is what is looked
    for here -- deliberately region-shaped, never an edge map.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    var = _local_variance(gray, ksize=5)

    smooth = var < np.percentile(var, 55)          # roof surfaces are internally uniform
    lo, hi = np.percentile(gray, [35, 92])
    distinct = (gray >= lo) & (gray <= hi)         # not deep shadow, not specular blowout

    excluded = (
        (seg_labels == seg.CLASS_IDX["water"]) |
        (seg_labels == seg.CLASS_IDX["vegetation"]) |
        (seg_labels == seg.CLASS_IDX["building"])
    )

    cand = (smooth & distinct & ~excluded).astype(np.uint8)
    # keep only compact blobs: erode then dilate discards thin road-like strips
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)


def _split_instances(mask: np.ndarray, min_area: int = MIN_AREA_ABSOLUTE):
    """
    Watershed on the distance transform, so that buildings sharing a wall
    become separate instances instead of one merged blob. Previously any two
    touching structures collapsed into a single footprint.

    Returns (labels, n_labels, n_splits) where n_splits counts blobs that
    yielded more than one instance.
    """
    n_cc, cc = cv2.connectedComponents(mask, connectivity=8)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # peaks of the distance transform are building centres; the threshold is
    # relative to each blob so large and small buildings are treated alike
    peaks = np.zeros_like(mask, dtype=np.uint8)
    for lbl in range(1, n_cc):
        comp = cc == lbl
        if comp.sum() < min_area:
            continue
        d = dist * comp
        peak_thresh = 0.55 * d.max()
        if peak_thresh <= 0:
            continue
        peaks[(d >= peak_thresh) & comp] = 1

    n_seeds, seeds = cv2.connectedComponents(peaks, connectivity=8)
    if n_seeds <= 1:
        return cc, n_cc, 0

    markers = seeds.astype(np.int32) + 1
    markers[mask == 0] = 1  # background
    bgr = cv2.cvtColor((mask * 255), cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)

    out = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    out[mask == 0] = 0

    # count blobs that actually split
    splits = 0
    for lbl in range(1, n_cc):
        comp = cc == lbl
        if comp.sum() < min_area:
            continue
        sub = np.unique(out[comp])
        sub = sub[sub > 0]
        if len(sub) > 1:
            splits += len(sub) - 1

    n_out = int(out.max()) + 1
    return out, n_out, splits


def _score_candidate(comp_mask, bbox, gray, dens, var, dsm, ndvi,
                     shadow_mask, sun_vec, contour) -> Evidence:
    """Six independent signals; each normalised to [0, 1]."""
    x0, y0, w, h = bbox
    ev = Evidence()
    area = float(comp_mask.sum())
    if area <= 0:
        return ev

    sl = (slice(y0, y0 + h), slice(x0, x0 + w))

    # 1. edge -- BOUNDARY CONTRAST, not interior edge density.
    #
    # Interior edge density is circular here: candidates are selected using
    # edge density, so every candidate scores ~1.0 on it and the signal does
    # no discriminative work (measured: mean 0.982, 100% non-zero). What
    # actually separates a building from a noise blob is a real intensity
    # step across its outline -- a roof is materially different from the
    # ground beside it, whereas a texture artefact fades into its
    # surroundings. Dilate/erode gives the boundary ring and the interior.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cm_u8 = comp_mask.astype(np.uint8)
    inner_ring = cv2.erode(cm_u8, k, iterations=1).astype(bool)
    outer_ring = cv2.dilate(cm_u8, k, iterations=2).astype(bool) & ~comp_mask

    patch = gray[sl].astype(np.float32)
    pad_y0, pad_x0 = max(0, y0 - 3), max(0, x0 - 3)
    pad_y1, pad_x1 = min(gray.shape[0], y0 + h + 3), min(gray.shape[1], x0 + w + 3)
    surround = gray[pad_y0:pad_y1, pad_x0:pad_x1].astype(np.float32)

    if inner_ring.any() and patch.size:
        interior_val = float(patch[inner_ring].mean())
    elif patch.size:
        interior_val = float(patch[comp_mask].mean())
    else:
        interior_val = 0.0

    if outer_ring.any():
        exterior_val = float(patch[outer_ring].mean())
    elif surround.size:
        exterior_val = float(surround.mean())
    else:
        exterior_val = interior_val

    # normalise the step against the scene's own contrast range
    scene_spread = float(np.percentile(gray, 95) - np.percentile(gray, 5)) or 1.0
    ev.edge = float(np.clip(abs(interior_val - exterior_val) / (0.35 * scene_spread), 0, 1))

    # 2. shape -- rectilinearity: contour fills its own minimum rotated rect
    rect = cv2.minAreaRect(contour)
    rect_area = max(rect[1][0] * rect[1][1], 1e-6)
    ev.shape = float(np.clip(cv2.contourArea(contour) / rect_area, 0, 1))

    # 3. texture -- roofs are smoother than canopy at fine scale, but not flat
    v = var[sl][comp_mask].mean()
    ev.texture = float(np.clip(1.0 - abs(np.log10(max(v, 1e-3)) - 1.6) / 2.2, 0, 1))

    # 4. height -- elevated relative to a ring around the footprint
    if dsm is not None:
        pad = max(3, int(0.4 * max(w, h)))
        ry0, ry1 = max(0, y0 - pad), min(dsm.shape[0], y0 + h + pad)
        rx0, rx1 = max(0, x0 - pad), min(dsm.shape[1], x0 + w + pad)
        ring = dsm[ry0:ry1, rx0:rx1]
        inner = dsm[sl][comp_mask]
        if ring.size and inner.size:
            rise = float(np.percentile(inner, 75) - np.percentile(ring, 25))
            spread = float(np.percentile(dsm, 95) - np.percentile(dsm, 5)) or 1.0
            ev.height = float(np.clip(rise / (0.08 * spread), 0, 1))

    # 5. shadow -- cast shadow in the anti-sun direction is strong evidence
    if shadow_mask is not None and sun_vec is not None:
        cy, cx = y0 + h / 2.0, x0 + w / 2.0
        dy, dx = sun_vec
        step = max(2, int(0.6 * max(w, h)))
        py, px = int(cy + dy * step), int(cx + dx * step)
        if 0 <= py < shadow_mask.shape[0] and 0 <= px < shadow_mask.shape[1]:
            patch = shadow_mask[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
            if patch.size:
                ev.shadow = float(patch.mean())

    # 6. spectral -- low NDVI means not vegetation
    if ndvi is not None:
        nv = ndvi[sl][comp_mask].mean()
        ev.spectral = float(np.clip((0.2 - nv) / 0.4, 0, 1))

    return ev


def detect_buildings(image_np: np.ndarray, seg_labels: np.ndarray,
                      dsm: np.ndarray = None, ndvi: np.ndarray = None,
                      sun_elevation_deg: float = None, sun_azimuth_deg: float = None,
                      scales=(1, 2, 4)) -> DetectionReport:
    """
    Multi-scale, evidence-scored building discovery.

    dsm / ndvi / sun angles are all optional -- each simply contributes one
    more evidence signal when present. Detection degrades in confidence
    without them, not in coverage.
    """
    import math

    report = DetectionReport(scales_run=list(scales))
    gray_full = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    shadow_mask = None
    sun_vec = None
    if sun_elevation_deg is not None and sun_azimuth_deg is not None and sun_elevation_deg > 0:
        import shadow_correction
        shadow_mask = shadow_correction.detect_shadow_mask(image_np).astype(np.float32)
        rad = math.radians((sun_azimuth_deg + 180.0) % 360.0)  # shadows point away from sun
        sun_vec = (-math.cos(rad), math.sin(rad))  # (dy, dx) in image coords

    accepted = []  # (polygon_full_res, evidence, scale, area)

    for scale in scales:
        if scale == 1:
            img_s, seg_s = image_np, seg_labels
            dsm_s = dsm
            ndvi_s = ndvi
            shadow_s = shadow_mask
        else:
            h, w = image_np.shape[:2]
            nh, nw = max(8, h // scale), max(8, w // scale)
            img_s = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
            seg_s = cv2.resize(seg_labels.astype(np.uint8), (nw, nh),
                                interpolation=cv2.INTER_NEAREST).astype(seg_labels.dtype)
            dsm_s = None if dsm is None else cv2.resize(dsm.astype(np.float32), (nw, nh),
                                                          interpolation=cv2.INTER_AREA)
            ndvi_s = None if ndvi is None else cv2.resize(ndvi.astype(np.float32), (nw, nh),
                                                            interpolation=cv2.INTER_AREA)
            shadow_s = None if shadow_mask is None else cv2.resize(shadow_mask, (nw, nh),
                                                                     interpolation=cv2.INTER_AREA)

        gray_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)
        dens_s = _edge_density(gray_s)
        var_s = _local_variance(gray_s)

        mask = _candidate_mask(img_s, seg_s)
        inst, n_inst, splits = _split_instances(mask)
        report.instances_split += splits

        for lbl in range(1, n_inst):
            comp_full = inst == lbl
            area = int(comp_full.sum())
            if area == 0:
                continue
            report.raw_candidates += 1

            ys, xs = np.where(comp_full)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            bbox = (x0, y0, x1 - x0, y1 - y0)
            comp_local = comp_full[y0:y1, x0:x1]

            cnts, _ = cv2.findContours(comp_local.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                report.rejections.append(Rejection(area, (float(x0), float(y0)),
                                                    "no_contour_extracted", 0.0, scale))
                continue
            contour = max(cnts, key=cv2.contourArea)

            if area < MIN_AREA_ABSOLUTE:
                report.rejections.append(Rejection(area, (float(x0), float(y0)),
                                                    "below_minimum_measurable_area", 0.0, scale))
                continue

            ev = _score_candidate(comp_local, bbox, gray_s, dens_s, var_s,
                                   dsm_s, ndvi_s, shadow_s, sun_vec, contour)
            total = ev.total()

            if total < MIN_EVIDENCE:
                report.rejections.append(Rejection(area, (float(x0), float(y0)),
                                                    "insufficient_evidence", total, scale))
                continue

            poly = (contour.reshape(-1, 2) + [x0, y0]).astype(np.float32) * scale
            accepted.append((poly, ev, scale, area * scale * scale))

    # ---- fuse across scales: suppress duplicates by centroid proximity ----
    kept = []
    claimed = []  # (cx, cy, radius)
    accepted.sort(key=lambda t: -t[1].total())  # strongest evidence wins a contested area

    for poly, ev, scale, area in accepted:
        cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        r = max(2.0, np.sqrt(area) * 0.5)
        dup = False
        for (ox, oy, orad) in claimed:
            if (cx - ox) ** 2 + (cy - oy) ** 2 < (0.7 * max(r, orad)) ** 2:
                dup = True
                break
        if dup:
            report.merged_duplicates += 1
            continue
        claimed.append((cx, cy, r))
        kept.append((poly, ev, scale, area))

    for i, (poly, ev, scale, area) in enumerate(kept):
        m = cv2.moments(poly.astype(np.float32))
        if abs(m["m00"]) > 1e-6:
            centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        else:
            centroid = (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
        rect = cv2.minAreaRect(poly.astype(np.float32))
        report.buildings.append(Building(
            id=i + 1,
            polygon=poly,
            area_px=float(cv2.contourArea(poly.astype(np.float32))),
            perimeter_px=float(cv2.arcLength(poly.astype(np.float32), True)),
            centroid=centroid,
            orientation_deg=float(rect[2]),
            evidence=ev,
            detection_scale=scale,
            provenance=Provenance.OBSERVED,
        ))

    return report
