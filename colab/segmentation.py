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

    road_mask = np.zeros_like(flat_mask, dtype=bool)
    blob_mask = np.zeros_like(flat_mask, dtype=bool)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bbox_area = max(w * h, 1)
        extent = area / bbox_area          # near 1 = fills its box (compact/blobby)
        elongation = max(w, h) / max(min(w, h), 1)

        comp = cc_labels == i
        if elongation > 4 and extent < 0.5:
            road_mask |= comp
        else:
            blob_mask |= comp

    return road_mask, blob_mask


def segment(image_np: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    image_np: HxWx3 uint8 RGB array.
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

    # Buildings: strong local edge density (roof/wall boundaries), not vegetation.
    building_mask = (~vegetation_mask) & (edge_density > edge_hi) & (val > 60)

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
