"""Score the run 06 (DINOv3 SAT-493M) height model on the same held-out
protocol as run 03 (scripts/compare_ckpt.py: 18 DFC tiles degraded to 0.6 m,
pixel and per-building errors).

    python scripts/eval_run06.py CKPT.pt [--swiss]

CKPT.pt is run06_best.pt from the Kaggle kernel (fp16 state dict of the whole
network). Needs the DINOv3 model code in third_party/dinov3 (DINOv3 License).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "training", "kaggle_run06_dinov3"))
REPO = os.path.join(ROOT, "third_party", "dinov3")
MEAN = np.array([0.430, 0.411, 0.296], np.float32)
STD = np.array([0.213, 0.156, 0.143], np.float32)


def load(ckpt, device):
    import kaggle_kernel_run06 as k
    real = torch.hub.load

    sys.path.insert(0, REPO)
    from dinov3.hub.backbones import dinov3_vitl16   # not hubconf: that pulls in eval extras

    def no_download(repo, name, source=None, **kw):
        kw["pretrained"] = False           # weights come from the checkpoint
        return dinov3_vitl16(**kw)
    torch.hub.load = no_download
    try:
        # the file name only tells the loader which ViT-L layout (SAT-493M)
        m = k.build_model(REPO, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    finally:
        torch.hub.load = real
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict({kk: v.float() for kk, v in sd.items()}, strict=True)
    return m.to(device).eval()


class Run06:
    """predict_tiled() with the interface compare_ckpt expects."""

    def __init__(self, ckpt, device=None, tile=512, overlap=96):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = load(ckpt, self.device)
        self.tile, self.ov = tile, overlap

    @torch.no_grad()
    def predict_tiled(self, img, raw_metric=True, **_):
        a = (np.asarray(img.convert("RGB"), np.float32) / 255.0 - MEAN) / STD
        H, W = a.shape[:2]
        T, step = self.tile, self.tile - self.ov
        ph, pw = max(T, -(-H // 16) * 16), max(T, -(-W // 16) * 16)
        pad = np.pad(a, ((0, ph - H), (0, pw - W), (0, 0)), mode="reflect")
        acc = np.zeros((ph, pw), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        r = np.minimum(np.arange(T) + 1, T - np.arange(T)).astype(np.float32)
        win = np.minimum.outer(r, r) ** 1.0
        ys = list(range(0, ph - T + 1, step)) or [0]
        xs = list(range(0, pw - T + 1, step)) or [0]
        if ys[-1] != ph - T:
            ys.append(ph - T)
        if xs[-1] != pw - T:
            xs.append(pw - T)
        for y in ys:
            for x in xs:
                t = torch.from_numpy(pad[y:y + T, x:x + T].transpose(2, 0, 1).copy())[None].to(self.device)
                p = self.m(t).float().cpu().numpy()[0, 0]
                acc[y:y + T, x:x + T] += p * win
                wsum[y:y + T, x:x + T] += win
        return (acc / np.maximum(wsum, 1e-6))[:H, :W]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--swiss", action="store_true")
    a = ap.parse_args()
    import compare_ckpt as cc
    bb = Run06(a.ckpt)
    t = time.time()
    r = cc.dfc(bb)
    print(f"run06 DFC held-out: pixel RMSE {r['rmse']:.3f} MAE {r['mae']:.3f} r {r['corr']:.3f} | "
          f"per-building RMSE {r['bldg_rmse']:.3f} bias {r['bldg_bias']:+.3f} n {r['n_buildings']} | "
          f"{time.time() - t:.0f}s on {bb.device}", flush=True)
    out = {"dfc": r}
    if a.swiss:
        out["swiss"] = cc.swiss(bb, "run06")
        print(json.dumps(out["swiss"], indent=1, default=str)[:1500])
    json.dump(out, open(os.path.join(ROOT, "final_out", "eval_run06.json"), "w"), indent=1, default=str)
