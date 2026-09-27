"""Which bands are red, green and blue, and which pixels carry no image.

Two things a satellite GeoTIFF does that a photo does not:

* **Band order.** Cartosat MX, PlanetScope, Sentinel-2 stacks and most 4-band
  multispectral products store Blue, Green, Red, NIR. Reading bands 1-3 as RGB
  swaps red and blue in what the model sees and what the viewer drapes. A file
  that tags its colour bands is read by its tags; an untagged 4+ band 16-bit
  stack is read as B,G,R,NIR; everything else as R,G,B.
* **No-data.** Scenes arrive with a nodata value, an alpha band, an internal
  mask, or a black collar where the footprint does not fill the rectangle. None
  of that is ground. Heights estimated there are invented, and a mesh drawn over
  it shows up as a flat slab.

Kept free of torch so the server and tests can import it cheaply.
"""
from __future__ import annotations

import numpy as np

# Smallest black-border region treated as a collar, as a fraction of the image.
# Below this a black edge region is more likely a shadow than a footprint edge.
MIN_COLLAR_FRACTION = 0.001


def rgb_band_plan(ds) -> tuple[list[int], str]:
    """(1-based red, green, blue band indexes, how that was decided).

    `how` is "tagged", "grey", "assumed-bgr" or "assumed-rgb".
    """
    from rasterio.enums import ColorInterp

    n = ds.count
    if n < 3:
        return [1, 1, 1], "grey"
    ci = list(ds.colorinterp or [])
    if all(c in ci for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return ([ci.index(ColorInterp.red) + 1, ci.index(ColorInterp.green) + 1,
                 ci.index(ColorInterp.blue) + 1], "tagged")
    if n >= 4 and ds.dtypes[0] != "uint8":
        return [3, 2, 1], "assumed-bgr"
    return [1, 2, 3], "assumed-rgb"


def dataset_valid(ds, bands: list[int]) -> np.ndarray:
    """True where the file itself says there is image: nodata value, alpha band
    and internal masks all honoured (rasterio folds them into read_masks)."""
    m = np.ones((ds.height, ds.width), bool)
    for b in sorted(set(bands)):
        m &= ds.read_masks(b) > 0
    return m


def border_collar(raw: np.ndarray) -> np.ndarray:
    """True on black (all bands exactly 0) regions connected to the image edge.

    `raw` is (bands, H, W) in the file's own values. Exactly zero in every band
    is not something a sensor records over real ground; a dark shadow still has
    a dark-current floor. Only regions touching the edge count, so a black
    object inside the scene is never cut out.
    """
    import cv2

    zero = np.all(raw == 0, axis=0).astype(np.uint8)
    if not zero.any():
        return np.zeros(zero.shape, bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(zero, connectivity=4)
    H, W = zero.shape
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    edge.discard(0)
    floor = MIN_COLLAR_FRACTION * H * W
    keep = [i for i in edge if stats[i, cv2.CC_STAT_AREA] >= floor]
    if not keep:
        return np.zeros(zero.shape, bool)
    return np.isin(lab, keep)


def fill_invalid(rgb8: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Paint invalid pixels with the scene's median colour.

    The model still sees a full rectangle, but a flat neutral one instead of a
    black edge whose hard boundary reads as a wall. Outputs are masked anyway.
    """
    if valid.all() or not valid.any():
        return rgb8
    out = rgb8.copy()
    med = np.median(rgb8[valid], axis=0).astype(rgb8.dtype)
    out[~valid] = med
    return out
