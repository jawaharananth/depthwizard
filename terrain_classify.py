"""
Maps land-cover segmentation classes (building/road/water/vegetation/bare_earth)
plus depth-derived slope into the 4 grading terrain types used for calibration
curves and the accuracy dashboard: urban / hilly / forest / sparse.

Land cover alone can't tell "hilly" from "flat" -- that's a landform property,
not an appearance property -- so slope comes from the relative depth map
itself (local gradient magnitude), independent of segmentation.

Priority, per pixel: urban (structure) > hilly (steep, regardless of cover)
> forest (vegetation) > sparse (default: bare earth, water, low-slope flat).
"""
import numpy as np
import cv2

from calibration.terrain_curves import TERRAIN_CLASSES  # ["urban","hilly","forest","sparse"]
import segmentation as seg

TERRAIN_IDX = {name: i for i, name in enumerate(TERRAIN_CLASSES)}


def compute_grade_map(dem_meters: np.ndarray, gsd_x_m: float, gsd_y_m: float,
                       blur_ksize: int = 15) -> np.ndarray:
    """
    True physical slope (rise/run, i.e. percent grade) from a metric elevation
    array + real ground sample distance. Only meaningful with real units on
    both axes -- see classify_terrain() for why relative depth can't do this.
    """
    dzdy, dzdx = np.gradient(dem_meters, gsd_y_m, gsd_x_m)
    grade = np.sqrt(dzdx ** 2 + dzdy ** 2)
    return cv2.blur(grade.astype(np.float32), (blur_ksize, blur_ksize))


def classify_terrain(seg_labels: np.ndarray, dem_meters: np.ndarray = None,
                      gsd_x_m: float = None, gsd_y_m: float = None,
                      hilly_grade_threshold: float = 0.15,
                      structure_edge_buffer_m: float = 3.0) -> tuple[np.ndarray, dict]:
    """
    seg_labels: HxW int array from segmentation.segment()/segment_from_geotiff().

    dem_meters + gsd_x_m/gsd_y_m: OPTIONAL real metric elevation + ground
    sample distance (meters/pixel). When given, "hilly" is a true physical
    grade (>15% by default -- the common rolling/hilly-terrain cutoff).

    Without a metric DEM (Tier B/C, no ground truth reference), slope CANNOT
    be measured reliably: relative depth is min-max normalized per image, so
    a dead-flat scene and a mountain range both get stretched to fill 0-1 --
    any relative-depth slope threshold (percentile or fixed) would just be
    guessing and mislabel flat scenes as hilly. So without dem_meters this
    function does NOT attempt hilly detection at all -- it's land-cover only
    (urban/forest/sparse) and reports the limitation in stats rather than
    faking a number.

    structure_edge_buffer_m: a building or tree edge produces a near-vertical
    height jump (e.g. a 15m roof over 1 pixel = 1500% grade) that swamps the
    blur radius and gets misread as "hilly" ground right next to the
    structure -- verified on real DFC2019 data where >80% of naive "hilly"
    pixels sat within a few meters of a building/tree. Excluding a buffer
    around building/vegetation footprints from hilly consideration removes
    this false signal without needing a much larger (and slope-detail-losing)
    blur kernel.
    """
    have_metric_slope = dem_meters is not None and gsd_x_m is not None and gsd_y_m is not None
    if have_metric_slope and seg_labels.shape != dem_meters.shape:
        raise ValueError(f"seg_labels shape {seg_labels.shape} != dem_meters shape {dem_meters.shape}")

    urban_mask = (seg_labels == seg.CLASS_IDX["building"]) | (seg_labels == seg.CLASS_IDX["road"])

    if have_metric_slope:
        grade = compute_grade_map(dem_meters, gsd_x_m, gsd_y_m)

        structure_mask = ((seg_labels == seg.CLASS_IDX["building"]) |
                           (seg_labels == seg.CLASS_IDX["vegetation"])).astype(np.uint8)
        buffer_px = max(1, int(round(structure_edge_buffer_m / gsd_x_m)))
        kernel = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), np.uint8)
        structure_buffer = cv2.dilate(structure_mask, kernel).astype(bool)

        hilly_mask = (~urban_mask) & (~structure_buffer) & (grade > hilly_grade_threshold)
    else:
        hilly_mask = np.zeros_like(urban_mask)

    forest_mask = (~urban_mask) & (~hilly_mask) & (seg_labels == seg.CLASS_IDX["vegetation"])
    # everything else (water, bare_earth, low-slope leftovers) -> sparse

    terrain_mask = np.full(seg_labels.shape, TERRAIN_IDX["sparse"], dtype=np.int32)
    terrain_mask[forest_mask] = TERRAIN_IDX["forest"]
    terrain_mask[hilly_mask] = TERRAIN_IDX["hilly"]
    terrain_mask[urban_mask] = TERRAIN_IDX["urban"]

    total = terrain_mask.size
    stats = {name: float((terrain_mask == idx).sum()) / total for name, idx in TERRAIN_IDX.items()}
    stats["_hilly_detection"] = "metric_grade" if have_metric_slope else "unavailable_no_metric_reference"
    return terrain_mask, stats


if __name__ == "__main__":
    import sys
    from PIL import Image
    from depth_model import DepthBackbone

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.array(pil_img)

    seg_labels, seg_stats = seg.segment(image_np)
    # No metric DEM for a plain JPG demo -> land-cover-only classification,
    # hilly detection reported as unavailable (see classify_terrain docstring).
    terrain_mask, terrain_stats = classify_terrain(seg_labels)

    print("Terrain class distribution:")
    for name, frac in terrain_stats.items():
        if name.startswith("_"):
            print(f"  {name}: {frac}")
        else:
            print(f"  {name:8s} {frac*100:5.1f}%")

    colors = {
        "urban": (200, 40, 40), "hilly": (150, 100, 40),
        "forest": (40, 160, 60), "sparse": (170, 140, 90),
    }
    overlay = np.zeros((*terrain_mask.shape, 3), dtype=np.uint8)
    for name, idx in TERRAIN_IDX.items():
        overlay[terrain_mask == idx] = colors[name]
    blended = (0.55 * image_np + 0.45 * overlay).astype(np.uint8)
    Image.fromarray(blended).save("terrain_preview.png")
    print("Saved terrain_preview.png")
