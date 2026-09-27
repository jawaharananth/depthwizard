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


def sensor_gap(rgb: np.ndarray, rng=np.random) -> np.ndarray:
    """Apply both, each with its own probability. Input/output float32 0-255."""
    rgb = np.asarray(rgb, np.float32)
    if rng.rand() < P_RES:
        rgb = degrade_resolution(rgb, rng)
    if rng.rand() < P_COLOUR:
        rgb = shift_colour(rgb, rng)
    return rgb
