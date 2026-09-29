"""Is a traced building candidate actually a tree crown?

    import tree_filter
    p = tree_filter.p_tree(image_rgb, region_mask, chm)   # 0..1, or None

A logistic model on region features (scripts/train_tree_filter.py fits it on
DFC2019 LiDAR classes and reports its held-out score in the same file):
Meta/WRI CHMv2 canopy height where the image is georeferenced, otherwise
colour, brightness, saturation and texture only.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

FEATURES = ["chm_med", "chm_frac2", "gr_chroma", "exg", "bright", "sat", "tex", "dark_frac"]
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "tree_filter.json")
_M = None


def features(img: np.ndarray, m: np.ndarray, chm: np.ndarray | None) -> list:
    px = img[m].astype(np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    s = r + g + b + 1e-3
    mx, mn = px.max(1), px.min(1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = gray[y0:y1, x0:x1]
    lap = np.abs(cv2.Laplacian(sub, cv2.CV_32F))[m[y0:y1, x0:x1]]
    if chm is not None:
        c = chm[m]
        chm_med, chm_frac = float(np.median(c)), float((c > 2.0).mean())
    else:
        chm_med = chm_frac = float("nan")
    return [chm_med, chm_frac, float(np.mean((g - r) / s)), float(np.mean(2 * g - r - b)),
            float(np.mean(s / 3)), float(np.mean((mx - mn) / (mx + 1e-3))),
            float(np.mean(lap)), float(np.mean(s / 3 < 60))]


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    idx = [FEATURES.index(c) for c in model["cols"]]
    Z = (np.asarray(X, np.float64)[:, idx] - model["mean"]) / model["std"]
    return 1 / (1 + np.exp(-(Z @ np.asarray(model["w"]) + model["b"])))


def model(with_chm: bool):
    global _M
    if _M is None:
        _M = json.load(open(_PATH))["models"] if os.path.exists(_PATH) else {}
    return _M.get("with_chm" if with_chm else "no_chm")


def p_tree(img: np.ndarray, m: np.ndarray, chm: np.ndarray | None):
    mdl = model(chm is not None)
    if mdl is None or not m.any():
        return None
    return float(predict(mdl, np.array([features(img, m, chm)]))[0])
