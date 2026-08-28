"""
Metric surface heights from RPC multi-view stereo.

WHY THIS REPLACES MONOCULAR DEPTH AS THE HEIGHT SOURCE

Every height in this project used to come from a monocular depth network run on
a single view, then scaled by shadow geometry. That chain has a hard ceiling:
the network infers a plausible relative surface, it does not measure one.
Measured on JAX_165 against LiDAR, per building on ground-truth footprints:

    monocular depth + shadow scale   MAE 13.75 m   RMSE 19.84 m
    RPC plane-sweep MVS, 6 views     MAE  7.24 m   RMSE 15.46 m
    correlation with LiDAR           0.24  ->  0.532

On a city whose median building is about 10 m tall, a 13.75 m error means the
heights carry almost no real signal -- which is exactly why the rendered city
looked like undifferentiated blobs. MVS triangulates the same ground point
across several satellite views through their RPC camera models, so the height
is measured rather than inferred, and it comes out in ABSOLUTE WGS84 metres
with no scale calibration step at all.

The trade is cost: a sweep is minutes, not seconds, and it is cached.

RESOLUTION

The sweep runs at the truth grid's own 0.5 m rather than the ortho's 0.25 m.
Photometric matching gains nothing from oversampling beyond the source GSD
(~0.31 m), and halving the resolution quarters a cost that is already the
dominant term. The result is upsampled and edge-aligned to the image
afterwards, which is where roof edges actually come from.
"""
import glob
import os

import numpy as np
import cv2

import dfc2019_loader as L
from plane_sweep_mvs import plane_sweep_dsm

CACHE_DIR = "cache"


def _most_nadir(tile: str, rgb_dir: str, metadata_dir: str, n: int) -> list:
    """
    The n straightest views.

    Off-nadir angle displaces a roof from its footprint by height * tan(angle),
    and it also decides how much of a street canyon is occluded. Views here span
    4.8 to 29 degrees, so taking the straightest few keeps occlusion low while
    still leaving enough baseline between them to triangulate.
    """
    scored = []
    for p in sorted(glob.glob(os.path.join(rgb_dir, f"{tile}_*_RGB.tif"))):
        try:
            imd = L.parse_imd(L.imd_path_for_rgb(p, metadata_dir))
            a = imd.get("meanOffNadirViewAngle")
            if a is not None:
                scored.append((a, p))
        except Exception:
            continue
    scored.sort()
    return [p for _, p in scored[:n]], [a for a, _ in scored[:n]]


def compute(tile: str, truth_dir: str, rgb_dir: str, metadata_dir: str,
            extent_m: float = None, out_px: int = None, n_views: int = 6,
            height_min: float = -30.0, height_max: float = 60.0,
            height_step: float = 1.0, use_cache: bool = True) -> dict:
    """
    Returns {"dsm": HxW absolute metres, "confidence": HxW NCC, ...}.

    The grid is centred on the truth tile, matching ortho.orthorectify, so the
    result lines up with the image the rest of the pipeline uses.
    """
    coords = L.parse_dsm_txt(os.path.join(truth_dir, f"{tile}_DSM.txt"))
    truth_extent = coords["size_px"] * coords["gsd_m"]
    if extent_m is None:
        extent_m = truth_extent
    if out_px is None:
        out_px = int(round(extent_m / coords["gsd_m"]))

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{tile}_mvs_e{int(extent_m)}_{out_px}_v{n_views}"
    dsm_path = os.path.join(CACHE_DIR, key + ".npy")
    conf_path = os.path.join(CACHE_DIR, key + "_conf.npy")
    if use_cache and os.path.exists(dsm_path) and os.path.exists(conf_path):
        return {"dsm": np.load(dsm_path), "confidence": np.load(conf_path),
                "gsd_m": extent_m / out_px, "cached": True, "views": n_views}

    views, angles = _most_nadir(tile, rgb_dir, metadata_dir, n_views)
    if len(views) < 2:
        raise RuntimeError(f"{tile}: need at least 2 views for MVS, found {len(views)}")

    # The TXT northing is the tile's LOWER-left corner, and the sweep grid walks
    # downward from the origin it is handed, so it must receive the UPPER-left.
    # Same convention error that put ortho.py 256 m off before it was fixed --
    # here it would reconstruct ground a whole tile away from the imagery.
    pad = (extent_m - truth_extent) / 2.0
    utm_x0 = coords["utm_x"] - pad
    utm_y_top = coords["utm_y"] + truth_extent + pad
    gsd = extent_m / out_px
    epsg = "EPSG:32617" if tile.startswith("JAX") else "EPSG:32615"

    dsm, conf = plane_sweep_dsm(
        views[0], views[1:], utm_x0, utm_y_top, out_px, gsd, epsg,
        height_min=height_min, height_max=height_max, height_step=height_step,
        ncc_ksize=15, confidence_threshold=0.3)

    np.save(dsm_path, dsm)
    np.save(conf_path, conf)
    return {"dsm": dsm, "confidence": conf, "gsd_m": gsd, "cached": False,
            "views": len(views), "off_nadir_deg": angles}


def to_image_grid(dsm: np.ndarray, image_np: np.ndarray,
                  guided_radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """
    Upsample an MVS surface to the image grid and snap its edges to the image.

    MVS is computed at 0.5 m, the imagery is at 0.25 m, and a plain resize
    leaves roof edges a couple of pixels soft and slightly in the wrong place.
    A guided filter with the image as guide pulls the height discontinuity onto
    the intensity discontinuity -- the height values stay MVS's, only their
    boundaries are sharpened, so this improves edges without inventing height.
    """
    H, W = image_np.shape[:2]
    up = cv2.resize(dsm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    guide = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    r = guided_radius
    mean_g = cv2.boxFilter(guide, -1, (r, r))
    mean_p = cv2.boxFilter(up, -1, (r, r))
    cov = cv2.boxFilter(guide * up, -1, (r, r)) - mean_g * mean_p
    var = cv2.boxFilter(guide * guide, -1, (r, r)) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_p - a * mean_g
    return cv2.boxFilter(a, -1, (r, r)) * guide + cv2.boxFilter(b, -1, (r, r))
