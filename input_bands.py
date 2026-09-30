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


def nir_band(ds, rgb: list[int], how: str) -> int | None:
    """1-based index of a near-infrared band, or None.

    Named in the band description ("nir", "near infrared", "b8"), else the
    4th band of an untagged B,G,R,NIR stack, else the one remaining band of a
    4-band file whose red, green and blue are tagged (NAIP, most orthophotos).
    An alpha band is never taken for NIR.
    """
    from rasterio.enums import ColorInterp

    n = ds.count
    if n < 4 or how == "grey":
        return None
    ci = list(ds.colorinterp or [])
    desc = [(d or "").lower() for d in (ds.descriptions or [None] * n)]
    for i, d in enumerate(desc):
        if "nir" in d or "near" in d or d in ("b8", "b08"):
            return i + 1
    if how == "assumed-bgr":
        return 4
    rest = [i + 1 for i in range(n) if i + 1 not in rgb
            and (i >= len(ci) or ci[i] != ColorInterp.alpha or not _is_mask(ds, i + 1))]
    if how == "tagged" and n == 4 and len(rest) == 1:
        return rest[0]
    return None


def _is_mask(ds, band: int) -> bool:
    """A real alpha band is (almost) only fully off / fully on. GDAL writes the
    4th band of any tagged-RGB 4-band GeoTIFF as 'alpha' by default, so an RGB
    + NIR file arrives with its NIR labelled alpha; its many in-between values
    are what give it away."""
    try:
        f = max(1, max(ds.width, ds.height) // 512)
        v = ds.read(band, out_shape=(max(1, ds.height // f), max(1, ds.width // f)))
        lo, hi = v.min(), v.max()
        return float(((v != lo) & (v != hi)).mean()) < 0.05
    except Exception:
        return True


def dataset_valid(ds, bands: list[int]) -> np.ndarray:
    """True where the file itself says there is image: nodata value, alpha band
    and internal masks all honoured (rasterio folds them into read_masks).

    A pixel is no-data only when EVERY colour band says so (GDAL's own
    dataset-mask rule): with nodata=255, a bright roof saturated in one band
    is still image, not a hole."""
    from rasterio.enums import ColorInterp
    ci = list(ds.colorinterp or [])
    fake_alpha = any(c == ColorInterp.alpha and not _is_mask(ds, i + 1) for i, c in enumerate(ci))
    if fake_alpha and ds.nodata is None:
        return np.ones((ds.height, ds.width), bool)
    m = np.zeros((ds.height, ds.width), bool)
    for b in sorted(set(bands)):
        if fake_alpha:
            # the 'alpha' band is data (NIR, see _is_mask): dark water in the
            # near-infrared must not become no-data, so only nodata values count
            m |= ds.read(b) != ds.nodata
        else:
            m |= ds.read_masks(b) > 0
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
