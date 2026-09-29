"""Learned building instances: Mask R-CNN trained on 0.6 m NAIP imagery.

    import building_detector as bd
    polys, scores = bd.detect(rgb_uint8, gsd_m, nir=None)   # pixel polygons

The model is giswqs/geoai `building_footprints_usa(_rgbn).pth` (MIT; torchvision
Mask R-CNN ResNet-50-FPN, trained from scratch on USDA NAIP at 0.6 m -- the
ground sample of Cartosat-2S). The image is brought to 0.6 m, cut into
overlapping 512 px chips, and every instance mask above the score threshold
becomes a polygon on the caller's grid. The RGB+NIR model is used when the
file has a near-infrared band.

This replaces height-map watershed as the source of building outlines where
no real outline exists: a network that has seen millions of roofs separates a
roof from a tree crown or a yard far better than a height threshold does.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = {3: os.path.join(ROOT, "checkpoints", "building_maskrcnn", "building_footprints_usa.pth"),
           4: os.path.join(ROOT, "checkpoints", "building_maskrcnn", "building_footprints_usa_rgbn.pth")}
TRAIN_GSD_M = 0.6
CHIP, STRIDE = 512, 384
SCORE_MIN = float(os.environ.get("DW_BLDG_SCORE", "0.5"))
_models = {}
_lock = threading.Lock()


def available(channels: int = 3) -> bool:
    return os.path.exists(WEIGHTS[channels])


def _model(channels: int):
    with _lock:
        if channels in _models:
            return _models[channels]
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        kw = {}
        if channels == 4:
            kw = dict(image_mean=[0.485, 0.456, 0.406, 0.5], image_std=[0.229, 0.224, 0.225, 0.25])
        m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2, **kw)
        if channels == 4:
            c1 = m.backbone.body.conv1
            m.backbone.body.conv1 = torch.nn.Conv2d(4, c1.out_channels, 7, 2, 3, bias=False)
        sd = torch.load(WEIGHTS[channels], map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd)
        m.eval()
        _models[channels] = m
        return m


def detect(rgb: np.ndarray, gsd_m: float, nir: np.ndarray | None = None,
           score_min: float = SCORE_MIN):
    """(polygons in the input's pixel grid, scores). rgb uint8 HxWx3."""
    import torch
    H, W = rgb.shape[:2]
    f = gsd_m / TRAIN_GSD_M                      # to the model's 0.6 m
    w6, h6 = max(64, int(round(W * f))), max(64, int(round(H * f)))
    img = cv2.resize(rgb, (w6, h6), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    chans = [img[..., k] for k in range(3)]
    if nir is not None and available(4):
        n = cv2.resize(nir, (w6, h6), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        if n.dtype != np.uint8:
            lo, hi = np.percentile(n[n > 0], [1, 99]) if (n > 0).any() else (0, 1)
            n = np.clip((n.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        chans.append(n)
    x = np.stack(chans, 0).astype(np.float32) / 255.0
    m = _model(x.shape[0])
    prob = np.zeros((h6, w6), np.float32)      # best instance score covering each pixel
    inst = np.zeros((h6, w6), np.int32)
    polys, scores = [], []
    ys = list(range(0, max(1, h6 - CHIP), STRIDE)) + [max(0, h6 - CHIP)]
    xs = list(range(0, max(1, w6 - CHIP), STRIDE)) + [max(0, w6 - CHIP)]
    with torch.no_grad():
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                chip = x[:, y0:y0 + CHIP, x0:x0 + CHIP]
                out = m([torch.from_numpy(np.ascontiguousarray(chip))])[0]
                for k in range(len(out["scores"])):
                    s = float(out["scores"][k])
                    if s < score_min:
                        continue
                    mk = out["masks"][k, 0].numpy() > 0.5
                    if mk.sum() < 8:
                        continue
                    ch, cw = mk.shape
                    win = prob[y0:y0 + ch, x0:x0 + cw]
                    # an instance already claimed by a better-scoring detection
                    # from an overlapping chip is not added twice
                    if (win[mk] >= s).mean() > 0.5:
                        continue
                    win[mk] = np.maximum(win[mk], s)
                    cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        continue
                    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                    c += [x0, y0]
                    polys.append(c / f)                  # back to the caller's grid
                    scores.append(s)
    return polys, scores
