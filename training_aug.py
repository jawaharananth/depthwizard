"""Sensor-gap augmentation for height training (run 03 onwards).

ISRO evaluates on Cartosat-2S at 0.6 m; the training imagery is 0.3 m aerial
(GAMUS) and WorldView-3 (DFC2019). Two gaps, both applied to the RGB only --
heights stay on their own grid, because the ground has not moved:

* resolution: area-average to a coarser ground sample (what a coarser sensor
  records, 1x-3.3x, i.e. 0.3 m -> up to 1.0 m) and resample back to the crop
  grid, which is exactly what the pipeline's auto-zoom hands the model;
* radiometry: independent per-channel gain (white balance), gamma, saturation,
  atmospheric haze, and sensor noise -- the ways a different satellite's colour
  differs from an aerial camera's.

Every parameter is drawn per sample; `p_*` are the probabilities of applying
each. Pure numpy + OpenCV, so it runs inside DataLoader workers.
"""
from __future__ import annotations

import cv2
import numpy as np

P_RES = 0.5
P_COLOUR = 0.8
MAX_DEGRADE = 3.3


def degrade_resolution(rgb: np.ndarray, rng=np.random) -> np.ndarray:
    """rgb float32 HxWx3 in 0-255. Coarsen by a random factor and resample back."""
    h, w = rgb.shape[:2]
    f = rng.uniform(1.0, MAX_DEGRADE)
    if f < 1.1:
        return rgb
    sw, sh = max(8, int(round(w / f))), max(8, int(round(h / f)))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 255)


def shift_colour(rgb: np.ndarray, rng=np.random) -> np.ndarray:
    """rgb float32 HxWx3 in 0-255."""
    x = rgb / 255.0
    x = x * rng.uniform(0.85, 1.15, size=3).astype(np.float32)          # white balance
    x = np.clip(x, 0, 1) ** np.float32(rng.uniform(0.8, 1.25))           # gamma
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * np.float32(rng.uniform(0.6, 1.3))            # saturation
    if rng.rand() < 0.3:                                                 # haze
        a = np.float32(rng.uniform(0.02, 0.15))
        x = x * (1 - a) + a * np.float32(rng.uniform(0.7, 0.95))
    x = x * np.float32(rng.uniform(0.9, 1.1)) + np.float32(rng.uniform(-0.05, 0.05))
    if rng.rand() < 0.4:                                                 # sensor noise
        x = x + rng.normal(0, rng.uniform(0.003, 0.015), x.shape).astype(np.float32)
    return np.clip(x * 255.0, 0, 255).astype(np.float32)


def cartosat_look(rgb: np.ndarray, rng=np.random) -> np.ndarray:
    """How a Cartosat-2S product differs from an aerial frame, beyond resolution
    and colour (run 07 onwards, DW_CARTOSAT_AUG=1):
    * pan-sharpening: colour comes from the 1.6 m multispectral bands fused onto
      0.65 m panchromatic detail, so chroma is ~2.5x blurrier than luminance;
    * optics: a soft system MTF (Gaussian blur);
    * radiometry: 11-bit counts stretched to 8 bits on percentiles, which clips
      bright roofs and dark shadows;
    * delivery: JPEG compression of the distributed product.
    rgb float32 HxWx3 in 0-255."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    if rng.rand() < 0.6:                                   # pan-sharpening chroma blur
        ycc = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        f = rng.uniform(2.0, 3.0)
        for c in (1, 2):
            small = cv2.resize(ycc[:, :, c], (max(4, int(w / f)), max(4, int(h / f))), interpolation=cv2.INTER_AREA)
            ycc[:, :, c] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        x = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    if rng.rand() < 0.5:                                   # system MTF
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.0))
    if rng.rand() < 0.5:                                   # 11-bit -> 8-bit percentile stretch
        lo, hi = np.percentile(x, [rng.uniform(0.5, 3.0), rng.uniform(97.0, 99.5)])
        x = np.clip((x - lo) / max(hi - lo, 1.0) * 255.0, 0, 255)
    if rng.rand() < 0.5:                                   # product JPEG
        q = int(rng.uniform(55, 92))
        ok, buf = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.float32)


def sensor_gap(rgb: np.ndarray, rng=np.random) -> np.ndarray:
    """Apply both, each with its own probability. Input/output float32 0-255.
    With DW_CARTOSAT_AUG=1 also the Cartosat-2S product look, and resolution
    degradation reaching 5x (0.3 m -> 1.5 m) on a quarter of the coarsened
    samples, for ISRO's 0.35-10 m input range."""
    import os
    rgb = np.asarray(rgb, np.float32)
    carto = os.environ.get("DW_CARTOSAT_AUG") == "1"
    if rng.rand() < P_RES:
        if carto and rng.rand() < 0.25:
            global MAX_DEGRADE
            keep, MAX_DEGRADE = MAX_DEGRADE, 5.0
            try:
                rgb = degrade_resolution(rgb, rng)
            finally:
                MAX_DEGRADE = keep
        else:
            rgb = degrade_resolution(rgb, rng)
    if carto and rng.rand() < 0.6:
        rgb = cartosat_look(rgb, rng)
    if rng.rand() < P_COLOUR:
        rgb = shift_colour(rgb, rng)
    return rgb
