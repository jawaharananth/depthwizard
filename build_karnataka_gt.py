"""
Build a Karnataka (ramp/DevGlobal) demo scene from a contiguous block of real
256px Maxar chips (30 cm/px), mosaicked from a public, unauthenticated bucket
at data.source.coop, with human-verified building footprints (CC-BY-NC-4.0)
rasterised onto the mosaic's own pixel grid.

Building EXISTENCE and SHAPE come directly from the ground-truth mask via
build_city_image.build(gt_building_mask=...) -- see region_footprints.
extract_from_mask for why: the normal watershed path was measured to ignore
segmentation input entirely, so a GT mask has to drive footprint discovery
itself, not just relabel a channel nothing reads. Vegetation/road/water still
come from our own segmentation, since ramp only labels buildings.

    python build_karnataka_gt.py <tag> [scene_name]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

MOSAIC_DIR = "ramp_karnataka/mosaics"
GSD_M = 0.3  # ramp chips are 30 cm/px


def scene_tallest_assumption_m(gt_mask: np.ndarray) -> float:
    """
    A scene-informed "tallest structure" height, instead of a flat 40 m
    applied regardless of what the ground truth says the scene actually is.

    The pipeline's default TARGET_TALL_M=40 assumes an urban skyline. Applied
    to a low-rise scene it inflates every building proportionally: measured on
    kn_a, median footprint area is 119.5 sq m -- a single-family residential
    plot -- and only 10.2% of footprints exceed 300 sq m (the institutional/
    apartment scale where a genuinely taller structure is plausible), yet the
    flat assumption put the MEDIAN building at 18 m, three-plus storeys for a
    scene that is overwhelmingly single-family homes by footprint.

    This is still an ASSERTION, not a measurement -- Tier C stays Tier C,
    identically to the existing anchor_height_m path this reuses -- but it is
    an assertion grounded in the footprint evidence actually in hand, not a
    constant borrowed from a downtown scene. A single-storey Indian house
    runs 3-3.5 m eave to 4.5-5 m ridge; a scene with a real institutional/
    apartment-scale minority (the >300 sq m 10%) plausibly tops out around
    3-4 storeys, so the assumed tallest is anchored there rather than at 40.
    """
    import cv2
    mask = (gt_mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas_px = stats[1:, cv2.CC_STAT_AREA]
    areas_m2 = areas_px[areas_px >= 8] * GSD_M * GSD_M
    if areas_m2.size < 10:
        return 40.0  # not enough evidence to depart from the default
    frac_large = float((areas_m2 > 300).mean())
    if frac_large < 0.02:
        return 6.0    # almost nothing above single-storey footprint scale
    if frac_large < 0.15:
        return 14.0   # a real but small institutional/apartment minority
    return 25.0        # a substantial denser/taller minority present


if __name__ == "__main__":
    tag = sys.argv[1]
    scene_name = sys.argv[2] if len(sys.argv) > 2 else tag

    mosaic_path = f"{MOSAIC_DIR}/{tag}.png"
    mask_path = f"{MOSAIC_DIR}/{tag}_mask.png"

    gt = np.array(Image.open(mask_path).convert("L")) > 127
    tallest = scene_tallest_assumption_m(gt)
    print(f"      scene-informed tallest assumption: {tallest:.0f} m "
          f"(footprint evidence, not a flat default)")

    poly_path = f"{MOSAIC_DIR}/{tag}_polygons.json"
    polygons = json.load(open(poly_path)) if os.path.exists(poly_path) else None
    if polygons is not None:
        print(f"      using {len(polygons)} individually-labelled polygons "
              f"(preserves touching-building identity)")

    chm_path = f"{MOSAIC_DIR}/{tag}_chm.npy"
    canopy_height_map = None
    if os.path.exists(chm_path):
        canopy_height_map = np.load(chm_path)
        print(f"      real CHM raster found ({canopy_height_map.shape}), "
              f"using it for canopy instead of skipping")

    import build_city_image as bci
    bci.build(mosaic_path, scene_name, max_px=1536, use_dem=False,
              gt_building_mask=gt, gt_building_polygons=polygons,
              anchor_height_m=tallest,
              skip_canopy=(canopy_height_map is None),
              canopy_height_map=canopy_height_map)
