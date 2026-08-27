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
            seg_labels: np.ndarray = None,
            min_area_m2: float = 8.0,
            min_height_m: float = 2.0) -> dict:
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

    ids = np.unique(ws)
    ids = ids[ids > 1]                          # 1 = background seed, -1 = ridges

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
    rej = {"too_small": 0, "too_low": 0, "too_rough": 0, "vegetation": 0, "ragged": 0}

    # Per-region statistics in one pass each, via bounding boxes.
    for rid in ids:
        mask = (ws == rid)
        area_px = int(mask.sum())
        if area_px < min_px:
            rej["too_small"] += 1
            continue

        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = mask[y0:y1, x0:x1]
        hsub = ndsm[y0:y1, x0:x1][sub]

        height = float(np.median(hsub))
        if height < h_cut:
            rej["too_low"] += 1
            continue

        # FLATNESS -- the roof/tree discriminator.
        rough = float(np.std(hsub))
        if rough > 0.55 * max(height, 1e-6):
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
        if fill < 0.55:
            rej["ragged"] += 1
            continue

        cnt = cnt + [x0, y0]
        eps = 0.015 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            continue

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
