"""
Semantic segmentation for satellite/aerial RGB imagery.

Classifies every pixel into one of: building, road, water, vegetation, bare_earth.

No pretrained satellite-segmentation model is reliably loadable via a plain
transformers pipeline (ADE20k/COCO models don't know satellite classes), so
this uses spectral + texture heuristics on RGB:
  - vegetation: Excess Green Index (ExG) high
  - water:      dark, low-saturation, low local texture variance
  - building:   bright, high local edge density (roof edges/shadows)
  - road:       gray (low saturation), low texture, elongated/connected
  - bare_earth: everything else (default fallback)

This is intentionally classical CV, not a trained net: deterministic, fast on
CPU, needs no labeled data. Swap in a trained model later without touching
callers (same output contract: label array + class names).
"""
import numpy as np
import cv2

CLASSES = ["building", "road", "water", "vegetation", "bare_earth"]
CLASS_IDX = {name: i for i, name in enumerate(CLASSES)}

# BGR-ish display colors for overlay preview (R,G,B)
CLASS_COLORS = {
    "building": (200, 40, 40),
    "road": (120, 120, 120),
    "water": (40, 90, 200),
    "vegetation": (40, 160, 60),
    "bare_earth": (170, 140, 90),
}


def _local_texture(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
    sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (ksize, ksize))
    var = np.clip(sq_mean - mean ** 2, 0, None)
    return var


def _local_edge_density(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    density = cv2.blur(edges, (ksize, ksize))
    return density


def _flat_region_shape_labels(flat_mask: np.ndarray, min_area: int = 40):
    """
    Split a binary 'smooth/flat' mask into road-like (elongated) vs
    water/plaza-like (compact/blobby) components using connected-component
    shape stats. Returns (road_mask, blob_mask).
    """
    flat_u8 = (flat_mask.astype(np.uint8)) * 255
    n, cc_labels, stats, _ = cv2.connectedComponentsWithStats(flat_u8, connectivity=8)

    if n <= 1:
        z = np.zeros_like(flat_mask, dtype=bool)
        return z, z.copy()

    # Classify every component at once and index the label image through the
    # result. The obvious loop -- `comp = cc_labels == i` per component --
    # touches the whole image once per component; on a 2048x2048 mask with tens
    # of thousands of components that is tens of billions of operations and the
    # call simply never returns. Same shape rules, same output, one pass.
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    w = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    h = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    bbox_area = np.maximum(w * h, 1.0)
    extent = area / bbox_area              # near 1 = fills its box (compact/blobby)
    elongation = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1.0)

    keep = area >= min_area
    is_road = keep & (elongation > 4) & (extent < 0.5)
    is_blob = keep & ~is_road

    road_lut = np.concatenate([[False], is_road])
    blob_lut = np.concatenate([[False], is_blob])
    return road_lut[cc_labels], blob_lut[cc_labels]


def _fill_edge_regions(strong_edges: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Turn an edge ribbon into the region it encloses.

    A roof reads as a bright closed outline with a smooth interior, so
    thresholding edge density labels the roof's BOUNDARY and leaves its middle
    unlabelled. Measured effect: building recall 0.10 -- the mask was tracing
    outlines, not covering roofs. Closing the outline and filling what it
    encloses recovers the interior.
    """
    h, w = strong_edges.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(strong_edges.astype(np.uint8), cv2.MORPH_CLOSE, k)

    # Flood from outside a zero-padded border, so a structure touching the
    # image edge is still filled (flooding from (0,0) fails whenever that
    # pixel happens to be foreground).
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = closed
    ffmask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ffmask, (0, 0), 1)
    interior = (padded[1:-1, 1:-1] == 0)

    filled = closed.astype(bool) | interior
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), 8)
    # Select via a label lookup table, not a loop of `keep |= (lab == i)`. A
    # 2048x2048 edge map yields tens of thousands of components, and that loop
    # allocates a full-image boolean per component -- it does not finish.
    big = np.zeros(n, dtype=bool)
    if n > 1:
        big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return big[lab]


def _elevated_mask(height: np.ndarray, ground_kernel: int = 121) -> np.ndarray:
    """
    Regions standing above their local surroundings, from the height field
    alone -- the cue that actually distinguishes a building from the pavement
    beside it. Colour and texture cannot: a flat grey roof and a flat grey car
    park are the same pixels.

    A grey-scale morphological opening with a kernel wider than any building
    removes objects while following terrain, giving an approximate bare earth
    to subtract. Threshold is a percentile of that residual, so it adapts to
    scenes with different relief instead of assuming a fixed metre value on a
    unitless field.
    """
    # Run the opening on a downsampled copy. An elliptical structuring element
    # is not separable, so a 121px kernel over a 2048^2 field costs minutes;
    # at 1/4 scale the kernel shrinks with the image and the same operation is
    # ~250x cheaper. Nothing is lost: the result is an approximate bare earth,
    # which is smooth by definition, and it is resampled back before use.
    h, w = height.shape
    scale = max(1, int(round(max(h, w) / 512)))
    small = cv2.resize(height.astype(np.float32), (w // scale, h // scale),
                       interpolation=cv2.INTER_AREA) if scale > 1 else height.astype(np.float32)
    ks = max(3, (ground_kernel // scale) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    ground_small = cv2.morphologyEx(small, cv2.MORPH_OPEN, k)
    ground = (cv2.resize(ground_small, (w, h), interpolation=cv2.INTER_LINEAR)
              if scale > 1 else ground_small)
    residual = height - ground
    # Threshold at the residual's own noise floor.
    #
    # After a morphological opening the residual is zero on true ground and
    # positive on anything standing above it, so the only question is how much
    # of the near-zero band is noise. The negative residuals are pure noise by
    # construction -- an opening cannot sit above its input, so anything below
    # zero comes from resampling the downscaled ground estimate back up. Their
    # robust sigma therefore measures the noise directly, and three sigma is
    # the cut.
    #
    # Otsu was used here first and was badly wrong: the ground mode dominates
    # the histogram so heavily that Otsu put the split at 0.0614 against a
    # noise sigma of 0.00026 -- 236 times too high. Measured on JAX_068
    # (43.1% buildings by LiDAR label): Otsu marked 5.2% of the frame at recall
    # 0.067, while this rule marks 46.3% at recall 0.599 with the SAME
    # precision, ~0.6. Precision is flat across two orders of magnitude of
    # threshold, so the high cut bought nothing and cost almost every building.
    #
    # That mattered far beyond segmentation: the DTM removes structures using
    # this mask, so an under-detected building stays in the "terrain" and gets
    # rendered as a smeared bump in the ground surface instead of a clean
    # extruded volume.
    if float(np.ptp(residual)) < 1e-9:
        return np.zeros(residual.shape, dtype=bool)
    neg = residual[residual < 0]
    if neg.size > 100:
        sigma = 1.4826 * float(np.median(np.abs(neg - np.median(neg))))
    else:
        sigma = 0.0
    # Floor at a small fraction of scene relief so a genuinely flat scene with
    # near-zero noise does not threshold at zero and mark everything.
    thresh = max(3.0 * sigma, 0.002 * float(np.ptp(height)))
    return residual > thresh


def segment(image_np: np.ndarray, height: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """
    image_np: HxWx3 uint8 RGB array.
    height:   optional HxW relative height field in [0,1]. When supplied,
              buildings are identified as regions standing above local ground
              rather than by appearance alone, which is the difference between
              recall ~0.1 and a usable mask.
    Returns (label_map HxW int array with values in CLASS_IDX, stats dict).
    """
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    r = image_np[:, :, 0].astype(np.float32)
    g = image_np[:, :, 1].astype(np.float32)
    b = image_np[:, :, 2].astype(np.float32)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    has_color = float(np.percentile(sat, 90)) > 40  # is there real chroma, or is this ~grayscale?

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    fine_texture = _local_texture(gray, ksize=5)
    coarse_texture = _local_texture(gray, ksize=15)
    edge_density = _local_edge_density(gray, ksize=9)

    exg = 2 * g - r - b  # Excess Green Index, only meaningful when has_color

    labels = np.full(gray.shape, CLASS_IDX["bare_earth"], dtype=np.int32)

    # All thresholds below are relative to THIS image's own edge/texture
    # distribution (percentiles), not fixed constants -- fixed cutoffs picked
    # on one image silently miscalibrate on another with different contrast.
    edge_hi = np.percentile(edge_density, 85)   # sparse, only real structure edges
    edge_mid = np.percentile(edge_density, 45)  # below-median edges
    fine_hi = np.percentile(fine_texture, 75)
    coarse_lo = np.percentile(coarse_texture, 45)
    fine_lo = np.percentile(fine_texture, 45)
    fine_very_lo = np.percentile(fine_texture, 20)  # near-zero micro-texture: true flat surfaces

    # Vegetation: true color cue when available, else fine speckly texture
    # (canopy noise) with moderate-not-extreme edges (leaf clusters, not roof corners).
    if has_color:
        vegetation_mask = exg > 15
    else:
        vegetation_mask = (fine_texture > fine_hi) & (edge_density < edge_mid)

    # Buildings: the region a roof outline encloses, not the outline itself.
    building_mask = _fill_edge_regions((edge_density > edge_hi) & (val > 60))
    building_mask &= ~vegetation_mask

    # With a height field available, prefer the geometric cue: anything
    # standing above its local surroundings and not vegetation. Appearance
    # alone cannot separate a flat roof from the car park next to it; relief
    # can. The appearance mask is kept as a union so structures the height
    # field smooths over (small sheds, low walls) are not lost.
    if height is not None:
        hf = height.astype(np.float32)
        if hf.shape != gray.shape:
            hf = cv2.resize(hf, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        elevated = _elevated_mask(hf) & (~vegetation_mask)
        # Drop speckle: a real structure is a contiguous region, not a pixel.
        elevated = cv2.morphologyEx(
            elevated.astype(np.uint8), cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
        building_mask = building_mask | elevated

    # Flat/smooth regions (candidates for road, water, plaza, bare earth):
    # low texture at both scales, not already claimed.
    flat_mask = (~vegetation_mask) & (~building_mask) & (coarse_texture < coarse_lo) & (fine_texture < fine_lo)

    road_mask, blob_mask = _flat_region_shape_labels(flat_mask)

    # Among compact/blobby flat regions: genuinely dark+color-blue -> water.
    # Require near-zero MICRO-texture too (real water/pavement is smoother than
    # tree-shadow, which still shows faint canopy noise even when dark) --
    # brightness alone can't tell water from shadow.
    if has_color:
        blue_dominant = (b > r + 10) & (b > g + 5)
        water_mask = blob_mask & (fine_texture < fine_very_lo) & (blue_dominant | (val < 90))
    else:
        water_mask = blob_mask & (fine_texture < fine_very_lo) & (val < np.percentile(val, 35))
    bare_blob_mask = blob_mask & (~water_mask)

    labels[vegetation_mask] = CLASS_IDX["vegetation"]
    labels[building_mask] = CLASS_IDX["building"]
    labels[road_mask] = CLASS_IDX["road"]
    labels[water_mask] = CLASS_IDX["water"]
    labels[bare_blob_mask] = CLASS_IDX["bare_earth"]
    # anything left unclassified (not vegetation/building/flat) defaults to bare_earth already

    total = labels.size
    stats = {name: float((labels == idx).sum()) / total for name, idx in CLASS_IDX.items()}
    stats["_has_color"] = has_color

    return labels, stats


def segment_from_geotiff(tif_path: str, nir_band: int = 4, red_band: int = 1,
                          green_band: int = 2, blue_band: int = 3) -> tuple[np.ndarray, dict]:
    """
    Multispectral GeoTIFF path: use NDVI/NDWI spectral indices for
    vegetation/water instead of the RGB texture heuristic, since a real NIR
    band makes those calls far more reliable than color/texture guessing.
    Building/road still come from the RGB texture pipeline (spectral indices
    don't help there). Band indices are 1-indexed (rasterio convention);
    default assumes R,G,B,NIR band order -- pass explicit indices if your
    source uses a different order (e.g. Sentinel-2 B,G,R,NIR).
    """
    import rasterio

    with rasterio.open(tif_path) as src:
        if src.count < 4:
            raise ValueError(
                f"GeoTIFF has {src.count} band(s); need >=4 (R,G,B,NIR) for NDVI/NDWI. "
                "Falling back to plain RGB segment() is the caller's job.")
        red = src.read(red_band).astype(np.float32)
        green = src.read(green_band).astype(np.float32)
        blue = src.read(blue_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)

    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)  # McFeeters NDWI

    def _to_uint8(band):
        lo, hi = np.percentile(band, [2, 98])
        return np.clip((band - lo) / max(hi - lo, eps) * 255, 0, 255).astype(np.uint8)

    rgb_uint8 = np.stack([_to_uint8(red), _to_uint8(green), _to_uint8(blue)], axis=-1)

    labels, stats = segment(rgb_uint8)

    veg_mask = ndvi > 0.2
    water_mask = ndwi > 0.0

    labels[veg_mask & (labels != CLASS_IDX["building"])] = CLASS_IDX["vegetation"]
    labels[water_mask] = CLASS_IDX["water"]  # strong physical cue, overrides texture guess

    total = labels.size
    stats = {name: float((labels == idx).sum()) / total for name, idx in CLASS_IDX.items()}
    stats["_source"] = "ndvi_ndwi"
    stats["ndvi_mean"] = float(ndvi.mean())
    stats["ndwi_mean"] = float(ndwi.mean())

    return labels, stats


def colorize(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for name, idx in CLASS_IDX.items():
        out[labels == idx] = CLASS_COLORS[name]
    return out


if __name__ == "__main__":
    import sys
    from PIL import Image

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    img = np.array(Image.open(img_path).convert("RGB"))

    labels, stats = segment(img)
    print("Class distribution:")
    for name, frac in stats.items():
        print(f"  {name:12s} {frac*100:5.1f}%")

    overlay = colorize(labels)
    blended = (0.55 * img + 0.45 * overlay).astype(np.uint8)
    Image.fromarray(blended).save("segmentation_preview.png")
    Image.fromarray(overlay).save("segmentation_raw.png")
    print("Saved segmentation_preview.png (blended) and segmentation_raw.png (raw classes)")
