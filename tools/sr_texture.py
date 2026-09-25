"""
Display-only super-resolution of a scene texture with Real-ESRGAN (x2).

This synthesises plausible high-frequency detail; it does not recover real
detail the sensor never captured. It is therefore used ONLY for the rendered
texture -- every measurement (footprints, heights, colours, audit) keeps
reading the original pixels -- and the scene is labelled as enhanced.

RRDBNet architecture as published with Real-ESRGAN / BasicSR (BSD-3-Clause),
reproduced here so the tool has no dependency on basicsr (which fails to
import against current torchvision).

    python tools/sr_texture.py in.png out.png [--tile 192]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WEIGHTS = "checkpoints/realesrgan/RealESRGAN_x2plus.pth"


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale=2, nf=64, nb=23, gc=32):
        super().__init__()
        self.scale = scale
        in_ch = 3 * (4 if scale == 2 else 16 if scale == 1 else 1)
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def upscale(img: np.ndarray, tile: int = 192, pad: int = 16) -> np.ndarray:
    net = RRDBNet(scale=2)
    sd = torch.load(WEIGHTS, map_location="cpu")
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)
    net.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    H0, W0 = img.shape[:2]
    img = np.pad(img, ((0, H0 % 2), (0, W0 % 2), (0, 0)), mode="edge")  # unshuffle needs even
    H, W = img.shape[:2]
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    out = np.zeros((2 * H, 2 * W, 3), np.float32)
    t0 = time.time()
    tiles = [(y, x0) for y in range(0, H, tile) for x0 in range(0, W, tile)]
    with torch.no_grad():
        for n, (y, x0) in enumerate(tiles):
            y1, x1 = min(y + tile, H), min(x0 + tile, W)
            ya, xa = max(y - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, H), min(x1 + pad, W)
            o = net(x[:, :, ya:yb, xa:xb])[0].permute(1, 2, 0).numpy()
            oy, ox = 2 * (y - ya), 2 * (x0 - xa)
            out[2 * y:2 * y1, 2 * x0:2 * x1] = o[oy:oy + 2 * (y1 - y), ox:ox + 2 * (x1 - x0)]
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  tile {n + 1}/{len(tiles)}  {el:.0f}s elapsed, "
                      f"~{el / (n + 1) * (len(tiles) - n - 1):.0f}s left", flush=True)
    return (np.clip(out[:2 * H0, :2 * W0], 0, 1) * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--tile", type=int, default=192)
    a = ap.parse_args()
    src = np.array(Image.open(a.inp).convert("RGB"))
    Image.fromarray(upscale(src, a.tile)).save(a.out)
    print("wrote", a.out)
