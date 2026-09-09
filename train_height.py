"""Fine-tune Depth Anything V2 to predict absolute height in metres.

WHY THIS EXISTS

The shipped pipeline uses Depth Anything V2 out of domain. It is trained on
ground-level photography and applied to nadir satellite imagery, and the cost
was measured on GAMUS tiles from cities it has never seen:

    height correlation with true nDSM   0.299     (one tile NEGATIVE: -0.248)
    MAE after a best-fit affine         4.28 m
    building precision                  0.343

A negative correlation is not a calibration problem. It means the model predicts
high where the ground is low, which no downstream scaling can repair. Five
separate heuristic approaches were measured against LiDAR before this and all
plateaued near IoU 0.56-0.63; adapting the backbone is the first change with
headroom left.

WHAT CHANGES IF THIS WORKS

The backbone currently emits a scale-free relative field, so the pipeline has to
invent a scale -- "assume the tallest structure is about 40 m" (Tier C), or fit
one parameter per tile against the reference, which is what all 18 benchmark
tiles actually did. A model trained on AGL predicts METRES directly. Tier C
disappears, and so does the scale-alignment caveat on the headline number.

HOW SUCCESS IS JUDGED

Against the pretrained baseline on the same held-out tiles, in metres, with no
affine fit allowed for the fine-tuned model -- it is supposed to be absolute.
The baseline is still scored WITH its best-fit affine, which is the most
generous possible treatment. If the fine-tuned model cannot beat a baseline that
is handed the optimal scale, it has not earned the swap.

    python train_height.py --smoke                 # 2 tiles, CPU, ~2 min
    python train_height.py --epochs 12 --batch 4   # the real run, on CUDA
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gamus

MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
CKPT_DIR = "checkpoints"

# Pixels above this are structure; below is ground plus DSM/DTM noise. Used to
# weight the loss and to report separately, because a model that nails 51% of
# the frame at 0 m and gets every building wrong would otherwise look good.
STRUCTURE_M = 2.0
STRUCTURE_WEIGHT = 3.0

# Weight on the gradient-matching term. Height edges are the whole point -- a
# blurred roofline extrudes into a ramp -- but it is an auxiliary signal, not
# the objective.
GRAD_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def height_loss(pred: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor) -> Dict[str, torch.Tensor]:
    """L1 in metres, weighted toward structure, plus gradient matching.

    Plain L1 is deliberate. Scale-invariant log loss is the usual choice for
    monocular depth and is exactly wrong here: it discards the absolute scale,
    which is the only reason for doing this at all.

    Structure is up-weighted because half the frame is ground. Unweighted, the
    cheapest way to cut the loss is to predict the ground everywhere, which is
    the degenerate solution this whole exercise is trying to escape.
    """
    w = torch.where(target > STRUCTURE_M,
                    torch.as_tensor(STRUCTURE_WEIGHT, device=pred.device),
                    torch.as_tensor(1.0, device=pred.device)) * valid
    denom = w.sum().clamp_min(1.0)
    l1 = ((pred - target).abs() * w).sum() / denom

    # Gradient matching on both axes: penalise a smooth prediction across a real
    # height discontinuity.
    def dx(t):
        return t[:, :, 1:] - t[:, :, :-1]

    def dy(t):
        return t[:, 1:, :] - t[:, :-1, :]

    vx = (valid[:, :, 1:] * valid[:, :, :-1])
    vy = (valid[:, 1:, :] * valid[:, :-1, :])
    gx = ((dx(pred) - dx(target)).abs() * vx).sum() / vx.sum().clamp_min(1.0)
    gy = ((dy(pred) - dy(target)).abs() * vy).sum() / vy.sum().clamp_min(1.0)
    grad = gx + gy

    return {"loss": l1 + GRAD_WEIGHT * grad, "l1": l1, "grad": grad}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, allow_affine: bool) -> Dict[str, float]:
    """Metrics in metres on held-out tiles.

    ``allow_affine`` fits the best possible scale+shift before scoring. It is
    granted to the pretrained baseline, whose output is scale-free and would
    otherwise be meaningless, and refused to the fine-tuned model, which is
    supposed to produce metres unaided. That asymmetry is deliberate and in the
    baseline's favour.
    """
    model.eval()
    ae, se, n = 0.0, 0.0, 0
    ae_s, n_s = 0.0, 0
    xs, ys = [], []

    for rgb, agl, valid in loader:
        rgb, agl, valid = rgb.to(device), agl.to(device), valid.to(device)
        pred = model(pixel_values=rgb).predicted_depth
        if pred.shape[-2:] != agl.shape[-2:]:
            pred = F.interpolate(pred.unsqueeze(1), size=agl.shape[-2:],
                                 mode="bilinear", align_corners=False).squeeze(1)
        m = valid > 0
        if not m.any():
            continue
        p, t = pred[m].float(), agl[m].float()

        if allow_affine:
            A = torch.stack([p, torch.ones_like(p)], 1)
            sol = torch.linalg.lstsq(A, t.unsqueeze(1)).solution.squeeze(1)
            p = A @ sol

        err = (p - t).abs()
        ae += err.sum().item()
        se += ((p - t) ** 2).sum().item()
        n += t.numel()

        s = t > STRUCTURE_M
        if s.any():
            ae_s += (p[s] - t[s]).abs().sum().item()
            n_s += int(s.sum().item())

        # Subsample for correlation: full-resolution stacking blows memory.
        step = max(1, p.numel() // 20000)
        xs.append(p[::step].cpu())
        ys.append(t[::step].cpu())

    if n == 0:
        return {}
    x = torch.cat(xs).numpy().astype(np.float64)
    y = torch.cat(ys).numpy().astype(np.float64)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
    return {"mae_m": ae / n, "rmse_m": (se / n) ** 0.5,
            "mae_structure_m": (ae_s / n_s) if n_s else float("nan"),
            "corr": corr, "pixels": n}


def _fmt(tag: str, m: Dict[str, float]) -> str:
    if not m:
        return f"{tag}: no data"
    return (f"{tag}: MAE {m['mae_m']:.2f} m   RMSE {m['rmse_m']:.2f} m   "
            f"MAE>2m {m['mae_structure_m']:.2f} m   corr {m['corr']:.3f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune depth to metric height")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2,
                    help="gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--crops-per-tile", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=CKPT_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the loop works before committing a GPU")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.batch, a.accum, a.crop = 1, 1, 1, 224
        a.crops_per_tile, a.workers = 1, 0
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train = gamus.GamusHeight(split="train", crop=a.crop, augment=True,
                              crops_per_tile=a.crops_per_tile)
    try:
        val = gamus.GamusHeight(split="val", crop=a.crop, augment=False,
                                crops_per_tile=1)
    except FileNotFoundError:
        print("no val split downloaded -- run: python gamus.py download --split val --limit 40")
        raise SystemExit(1)

    if a.smoke:
        train.stems = train.stems[:2]
        val.stems = val.stems[:2]
    print(f"train {len(train.stems)} tiles ({len(train)} crops)   "
          f"val {len(val.stems)} tiles")

    dl_train = DataLoader(train, batch_size=a.batch, shuffle=True,
                          num_workers=a.workers, pin_memory=(device == "cuda"),
                          drop_last=True)
    dl_val = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)

    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device)

    # Baseline first, on the same held-out tiles, WITH its best-fit affine.
    # Establishing the bar before training means the comparison cannot be
    # rationalised afterwards.
    print("\nbaseline (pretrained, best-fit affine granted):")
    base = evaluate(model, dl_val, device, allow_affine=True)
    print("  " + _fmt("baseline", base))

    # Two learning rates: the DPT head is being repurposed from relative depth
    # to metres and needs to move, while the pretrained encoder holds the visual
    # features worth keeping and is only nudged.
    head, backbone = [], []
    for name, p in model.named_parameters():
        (head if ("head" in name or "neck" in name) else backbone).append(p)
    opt = torch.optim.AdamW(
        [{"params": head, "lr": a.lr_head},
         {"params": backbone, "lr": a.lr_backbone}], weight_decay=1e-4)
    steps = max(1, (len(dl_train) // a.accum) * a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr_head, a.lr_backbone], total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    os.makedirs(a.out, exist_ok=True)
    best = float("inf")
    history = []

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, (rgb, agl, valid) in enumerate(dl_train):
            rgb, agl, valid = (rgb.to(device, non_blocking=True),
                               agl.to(device, non_blocking=True),
                               valid.to(device, non_blocking=True))
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                pred = model(pixel_values=rgb).predicted_depth
                if pred.shape[-2:] != agl.shape[-2:]:
                    pred = F.interpolate(pred.unsqueeze(1), size=agl.shape[-2:],
                                         mode="bilinear",
                                         align_corners=False).squeeze(1)
                parts = height_loss(pred.float(), agl, valid)
            scaler.scale(parts["loss"] / a.accum).backward()

            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()

            run += parts["loss"].item() * rgb.size(0)
            seen += rgb.size(0)
            if i % 20 == 0:
                print(f"  ep{ep} {i}/{len(dl_train)}  loss {run/max(seen,1):.3f}"
                      f"  (l1 {parts['l1'].item():.2f}  grad {parts['grad'].item():.2f})",
                      flush=True)

        # No affine for the fine-tuned model: it must produce metres unaided.
        m = evaluate(model, dl_val, device, allow_affine=False)
        history.append({"epoch": ep, "train_loss": run / max(seen, 1), **m})
        print(f"ep{ep} [{time.time()-t0:.0f}s] train {run/max(seen,1):.3f}   "
              + _fmt("val", m))

        if m and m["mae_m"] < best:
            best = m["mae_m"]
            model.save_pretrained(os.path.join(a.out, "height_best"))
            print(f"    saved (best MAE {best:.2f} m)")

    report = {"model": MODEL, "device": device, "epochs": a.epochs,
              "baseline_with_affine": base, "history": history,
              "best_mae_m": best}
    with open(os.path.join(a.out, "training_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 66)
    print("  " + _fmt("baseline (affine granted)", base))
    if history:
        fin = history[-1]
        print("  " + _fmt("fine-tuned (no affine)  ", fin))
        if base and fin.get("mae_m"):
            d = base["mae_m"] - fin["mae_m"]
            print(f"\n  MAE {'improved' if d > 0 else 'WORSE'} by {abs(d):.2f} m"
                  f"   correlation {base['corr']:.3f} -> {fin['corr']:.3f}")
            if d <= 0:
                print("  The fine-tuned model did not beat a baseline that was"
                      "\n  handed the optimal scale. Do not swap it in.")
    print("=" * 66)


if __name__ == "__main__":
    main()
