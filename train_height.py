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

--USE-DFC2019 AND THE DFC2019 BENCHMARK

rescore_baseline.py's 18 DFC2019 tiles are this whole module's yardstick --
"HOW SUCCESS IS JUDGED" above is scored against them, and every RMSE/MAE this
project has quoted from that script assumes DFC2019 is a dataset the backbone
has never seen. --use-dfc2019 (dfc2019_height.py) trains on DFC2019 tiles
directly, which is a real, useful cross-dataset signal for training (GAMUS
and DFC2019 are different sensors and different ground-truth pipelines, so
learning from both should generalise better than either alone).

dfc2019_height.split_tiles() excludes rescore_baseline.py's 18 benchmark
tiles from both its train and val splits by default (exclude_benchmark=True)
-- measured once, 14 of those 18 landed in what would otherwise be the train
split and 4 in val, which would have meant training directly on the tiles
this project's headline accuracy numbers are scored against. Do not call
split_tiles(exclude_benchmark=False) for anything that might become a
licensing candidate; it exists only for deliberately measuring how much of
an HTC run's apparent gain was real generalisation versus memorising tiles
rescore_baseline.py also uses.
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

import depth_model as _dm
import gamus
import htc_bins

# Small is Apache-2.0; Base and Large weights are CC-BY-NC-4.0 and cannot ship
# in a deliverable. --model overrides it for ablations.
MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
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

def _predict_height(model, rgb: torch.Tensor, htc: bool) -> torch.Tensor:
    """Height prediction in metres, (B, H, W), regardless of which head is active."""
    if htc:
        return model(rgb)["height"].squeeze(1)
    return model(pixel_values=rgb).predicted_depth


@torch.no_grad()
def evaluate(model, loader, device, allow_affine: bool, htc: bool = False) -> Dict[str, float]:
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
        pred = _predict_height(model, rgb, htc)
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
    ap.add_argument("--htc", action="store_true",
                    help="use the HTC-AdaBins head (htc_bins.py) instead of plain "
                         "scalar regression -- see htc_bins.py for why")
    ap.add_argument("--n-bins", type=int, default=64,
                    help="adaptive bin count for --htc (ignored otherwise)")
    ap.add_argument("--use-dfc2019", action="store_true",
                    help="also train on DFC2019 (dfc2019_height.py) -- a second, "
                         "independently-collected dataset (different sensor, "
                         "different ground-truth pipeline), not a bigger sample "
                         "of GAMUS's own distribution")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HF id of the depth backbone to fine-tune")
    ap.add_argument("--data-root", default=gamus.DATA_DIR,
                    help="GAMUS root holding train/ and val/ tile folders")
    ap.add_argument("--time-budget-h", type=float, default=0.0,
                    help="stop starting new epochs once another would overrun this "
                         "many hours (Kaggle sessions end at 12 h); 0 = no limit")
    ap.add_argument("--extra-root", default=None,
                    help="second dataset in GAMUS layout (e.g. training/dfc_tiles from "
                         "training/prepare_dfc.py): satellite imagery, the target domain")
    ap.add_argument("--extra-crops-per-tile", type=int, default=8,
                    help="oversampling for the (small) extra dataset")
    ap.add_argument("--sensor-aug", action="store_true",
                    help="run 03+: coarser-sensor (0.3-1.0 m) and satellite-colour "
                         "augmentation (training_aug.py), and select the best epoch on "
                         "the mean of native and 0.6 m-equivalent validation")
    ap.add_argument("--init-from", default=None,
                    help="start from this fine-tuned checkpoint directory instead of "
                         "the pretrained --model weights")
    a = ap.parse_args()

    global MODEL
    MODEL = a.model
    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.batch, a.accum, a.crop = 1, 1, 1, 224
        a.crops_per_tile, a.workers = 1, 0
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    gamus_train = gamus.GamusHeight(root=a.data_root, split="train", crop=a.crop,
                                    augment=True, crops_per_tile=a.crops_per_tile,
                                    sensor_aug=a.sensor_aug)
    try:
        gamus_val = gamus.GamusHeight(root=a.data_root, split="val", crop=a.crop,
                                      augment=False, crops_per_tile=1)
    except FileNotFoundError:
        print("no val split downloaded -- run: python gamus.py download --split val --limit 40")
        raise SystemExit(1)

    if a.smoke:
        gamus_train.stems = gamus_train.stems[:2]
        gamus_val.stems = gamus_val.stems[:2]

    train_sets, val_sets = [gamus_train], [gamus_val]
    n_gamus_train, n_gamus_val = len(gamus_train.stems), len(gamus_val.stems)
    n_dfc_train, n_dfc_val = 0, 0

    if a.use_dfc2019:
        import dfc2019_height as dfc
        dfc_train_stems, dfc_val_stems = dfc.split_tiles(dfc.available_tiles())
        if a.smoke:
            dfc_train_stems, dfc_val_stems = dfc_train_stems[:2], dfc_val_stems[:2]
        dfc_train = dfc.DFC2019Height(dfc_train_stems, crop=a.crop, augment=True,
                                      crops_per_tile=a.crops_per_tile)
        dfc_val = dfc.DFC2019Height(dfc_val_stems, crop=a.crop, augment=False,
                                    crops_per_tile=1)
        train_sets.append(dfc_train)
        val_sets.append(dfc_val)
        n_dfc_train, n_dfc_val = len(dfc_train_stems), len(dfc_val_stems)

    extra_val = None
    if a.extra_root:
        extra_train = gamus.GamusHeight(root=a.extra_root, split="train", crop=a.crop,
                                        augment=True, crops_per_tile=a.extra_crops_per_tile,
                                        sensor_aug=a.sensor_aug)
        extra_val = gamus.GamusHeight(root=a.extra_root, split="val", crop=a.crop,
                                      augment=False, crops_per_tile=1)
        if a.smoke:
            extra_train.stems, extra_val.stems = extra_train.stems[:2], extra_val.stems[:2]
        train_sets.append(extra_train)
        print(f"extra: {len(extra_train.stems)} train / {len(extra_val.stems)} val tile-views "
              f"from {a.extra_root} (x{a.extra_crops_per_tile} crops)")

    from torch.utils.data import ConcatDataset
    train = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    print(f"train: {n_gamus_train} GAMUS tiles"
          + (f" + {n_dfc_train} DFC2019 tiles" if a.use_dfc2019 else "")
          + f"  ({len(train)} crops)")
    print(f"val:   {n_gamus_val} GAMUS tiles"
          + (f" + {n_dfc_val} DFC2019 tiles" if a.use_dfc2019 else ""))

    dl_train = DataLoader(train, batch_size=a.batch, shuffle=True,
                          num_workers=a.workers, pin_memory=(device == "cuda"),
                          drop_last=True)
    dl_val = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    dl_extra_val = (DataLoader(extra_val, batch_size=1, shuffle=False, num_workers=0)
                    if extra_val is not None else None)
    dl_coarse_val = None
    if a.sensor_aug and extra_val is not None:
        # ISRO evaluates on Cartosat-2S at 0.6 m: the same satellite val views,
        # area-averaged 2x (0.3 -> 0.6 m) and resampled back.
        coarse_val = gamus.GamusHeight(root=a.extra_root, split="val", crop=a.crop,
                                       augment=False, crops_per_tile=1, fixed_degrade=2.0)
        coarse_val.stems = list(extra_val.stems)
        dl_coarse_val = DataLoader(coarse_val, batch_size=1, shuffle=False, num_workers=0)

    if a.htc:
        model = htc_bins.HTCDepthModel(model_id=MODEL, n_bins=a.n_bins).to(device)
    else:
        from transformers import AutoModelForDepthEstimation
        model = AutoModelForDepthEstimation.from_pretrained(a.init_from or MODEL).to(device)
        if a.init_from:
            print(f"initialised from {a.init_from}")

    # Baseline first, on the same held-out tiles, WITH its best-fit affine.
    # Establishing the bar before training means the comparison cannot be
    # rationalised afterwards. For --htc the "baseline" is the same untrained
    # HTC head (random bins), which is not a meaningful number on its own --
    # it exists so the printed comparison at the end has the same shape as
    # the plain-regression run, not because an untrained bin head deserves
    # an affine fit.
    print("\nbaseline (pretrained, best-fit affine granted):")
    base = evaluate(model, dl_val, device, allow_affine=True, htc=a.htc)
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
    # OneCycleLR divides the schedule into an up-phase of
    # round(pct_start * total_steps) steps and a down-phase of the rest; at
    # total_steps=4, pct_start=0.25 that up-phase rounds to exactly 0 steps,
    # and OneCycleLR's own get_lr() divides by that phase's (zero) width on
    # the very first step. Only reachable with very few steps -- a smoke run
    # or a tiny --use-dfc2019 subset -- so a real training run never hits it,
    # but --smoke exists specifically to be run before every real one and
    # must not crash. Below that floor, skip the schedule; a few dozen steps
    # of a fixed head/backbone LR does not need annealing anyway.
    sched = None
    if steps >= 8:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[a.lr_head, a.lr_backbone], total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    os.makedirs(a.out, exist_ok=True)
    best = float("inf")
    history = []

    last_epoch_s = 0.0
    for ep in range(1, a.epochs + 1):
        if a.time_budget_h and ep > 1:
            used = time.time() - t_start
            if used + last_epoch_s > a.time_budget_h * 3600:
                print(f"time budget: {used/3600:.2f} h used, next epoch needs "
                      f"~{last_epoch_s/3600:.2f} h -- stopping before epoch {ep}")
                break
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, (rgb, agl, valid) in enumerate(dl_train):
            rgb, agl, valid = (rgb.to(device, non_blocking=True),
                               agl.to(device, non_blocking=True),
                               valid.to(device, non_blocking=True))
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                if a.htc:
                    out = model(rgb)
                    if out["height"].shape[-2:] != agl.shape[-2:]:
                        for k in ("height", "probs", "gate_logit"):
                            out[k] = F.interpolate(
                                out[k].float(), size=agl.shape[-2:],
                                mode="bilinear", align_corners=False)
                    parts_loss, parts = htc_bins.htc_dc_loss(out, agl, valid)
                    parts["loss"] = parts_loss
                else:
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
                if sched is not None and sched.last_epoch < steps - 1:
                    sched.step()

            run += parts["loss"].item() * rgb.size(0)
            seen += rgb.size(0)
            if i % 20 == 0:
                if a.htc:
                    print(f"  ep{ep} {i}/{len(dl_train)}  loss {run/max(seen,1):.3f}"
                          f"  (l1 {parts['l1']:.2f}  chamfer {parts['chamfer']:.2f}"
                          f"  htc {parts['htc']:.2f}  dist {parts['dist']:.2f})",
                          flush=True)
                else:
                    print(f"  ep{ep} {i}/{len(dl_train)}  loss {run/max(seen,1):.3f}"
                          f"  (l1 {parts['l1'].item():.2f}  grad {parts['grad'].item():.2f})",
                          flush=True)

        # No affine for the fine-tuned model: it must produce metres unaided.
        m = evaluate(model, dl_val, device, allow_affine=False, htc=a.htc)
        last_epoch_s = time.time() - t0
        history.append({"epoch": ep, "train_loss": run / max(seen, 1), **m})
        print(f"ep{ep} [{time.time()-t0:.0f}s] train {run/max(seen,1):.3f}   "
              + _fmt("val", m))

        sel = m
        if dl_extra_val is not None:
            mx = evaluate(model, dl_extra_val, device, allow_affine=False, htc=a.htc)
            history[-1]["extra_val"] = mx
            print("        " + _fmt("extra val (selects best)", mx))
            sel = mx
            if dl_coarse_val is not None:
                mc = evaluate(model, dl_coarse_val, device, allow_affine=False, htc=a.htc)
                history[-1]["extra_val_0p6m"] = mc
                print("        " + _fmt("extra val at 0.6 m", mc))
                if mx and mc:
                    sel = dict(mx, mae_m=(mx["mae_m"] + mc["mae_m"]) / 2)
        if sel and sel["mae_m"] < best:
            best = sel["mae_m"]
            if a.htc:
                # Not a transformers.PreTrainedModel -- it's our own wrapper
                # (backbone/neck/conv1/conv2 borrowed from one, HTCAdaBinsHead
                # bolted on), so it has no save_pretrained. A plain state_dict
                # plus the constructor args are enough to reload it exactly
                # (see htc_bins.HTCDepthModel.__init__).
                ckpt = os.path.join(a.out, "height_htc_best")
                os.makedirs(ckpt, exist_ok=True)
                torch.save({"state_dict": model.state_dict(),
                           "model_id": MODEL, "n_bins": a.n_bins,
                           "h_min": model.htc_head.h_min,
                           "h_max": model.htc_head.h_max},
                          os.path.join(ckpt, "htc_model.pt"))
            else:
                ckpt = os.path.join(a.out, "height_best")
                model.save_pretrained(ckpt)
                # The image processor travels with the weights. Without it the
                # checkpoint cannot be loaded through transformers' pipeline at
                # all -- save_pretrained on the model alone writes no
                # preprocessor_config.json, and the load fails at inference
                # time rather than at save time.
                try:
                    from transformers import AutoImageProcessor
                    AutoImageProcessor.from_pretrained(MODEL).save_pretrained(ckpt)
                except Exception as e:
                    print(f"    WARNING: could not save the image processor ({e})")
            print(f"    saved (best MAE {best:.2f} m)")

    report = {"model": MODEL, "device": device, "epochs": a.epochs,
              "baseline_with_affine": base, "history": history,
              "best_mae_m": best}
    with open(os.path.join(a.out, "training_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # LICENSE THE CHECKPOINT, OR DO NOT.
    #
    # Two numeric, stated gates -- not "did MAE go down at all". A single
    # MAE-improved check promotes a run whose gain is noise-level, and
    # separately can't tell a real structural fix from a checkpoint that
    # merely rescaled its errors: MAE can improve by a hair while the model's
    # actual agreement with the ground truth (correlation) stays flat or
    # drops. Correlation is measured alongside RMSE because RMSE alone can
    # hide whether a fix is real. Both bars below:
    #
    #   1. MAE must beat baseline by a real margin, not an infinitesimal one.
    #   2. Correlation must not have quietly regressed while MAE improved.
    #
    # The marker is what depth_model.py reads before it will use a
    # checkpoint. Weights from a rejected run stay on disk for inspection and
    # are never picked up, because a run that failed either gate should not
    # quietly change every build that follows it.
    MAE_MIN_IMPROVEMENT_FRAC = 0.02   # must beat baseline MAE by >=2%, not just >0
    CORR_MAX_REGRESSION = 0.02        # correlation may not drop by more than this

    gate_report = []
    won = False
    if history and base and history[-1].get("mae_m") is not None:
        fin = history[-1]
        mae_improve_frac = ((base["mae_m"] - fin["mae_m"]) / base["mae_m"]
                             if base["mae_m"] > 0 else 0.0)
        corr_delta = fin.get("corr", 0.0) - base.get("corr", 0.0)
        mae_gate_pass = mae_improve_frac >= MAE_MIN_IMPROVEMENT_FRAC
        corr_gate_pass = corr_delta >= -CORR_MAX_REGRESSION
        gate_report = [
            (f"MAE improvement", f"{mae_improve_frac*100:+.1f}%",
             f">= {MAE_MIN_IMPROVEMENT_FRAC*100:.0f}%", mae_gate_pass),
            (f"correlation change", f"{corr_delta:+.3f}",
             f">= -{CORR_MAX_REGRESSION:.2f}", corr_gate_pass),
        ]
        won = mae_gate_pass and corr_gate_pass

    if gate_report:
        print("\n  promotion gates (numeric, both required):")
        for name, value, bar, passed in gate_report:
            print(f"    [{'PASS' if passed else 'FAIL'}] {name}: {value} (bar: {bar})")
        print(f"  -> {'PROMOTED' if won else 'REJECTED'}")

    if a.htc:
        # A different marker name and directory from the plain-regression
        # path (metric_ok.json / height_best) -- the two checkpoints are
        # loaded through completely different code (HTCDepthModel's own
        # state_dict vs. AutoModelForDepthEstimation.from_pretrained) and
        # depth_model.py must not confuse a plain-regression run for an HTC
        # one or vice versa.
        htc_ckpt = os.path.join(a.out, "height_htc_best")
        htc_marker = os.path.join(htc_ckpt, "htc_ok.json")
        if os.path.exists(htc_marker):
            os.remove(htc_marker)
        if won and os.path.isdir(htc_ckpt):
            with open(htc_marker, "w", encoding="utf-8") as f:
                json.dump({
                    "licensed": True,
                    "units": "metres (AGL) predicted directly",
                    "baseline_mae_m_with_affine": round(base["mae_m"], 3),
                    "finetuned_mae_m_no_affine": round(history[-1]["mae_m"], 3),
                    "baseline_corr": round(base["corr"], 4),
                    "finetuned_corr": round(history[-1]["corr"], 4),
                    "promotion_gates": [
                        {"name": n, "value": v, "bar": b, "passed": p}
                        for n, v, b, p in gate_report
                    ],
                    "epochs": a.epochs, "device": device, "n_bins": a.n_bins,
                    "trained_tiles": n_gamus_train + n_dfc_train,
                    "used_dfc2019": a.use_dfc2019,
                    "note": "depth_model.py reads this file to decide whether "
                            "to load the HTC-AdaBins checkpoint instead of the "
                            "stock or plain-regression backbone. Written only "
                            "because this run beat a baseline given the "
                            "optimal affine scale.",
                }, f, indent=2)
            print("")
            print(f"  HTC CHECKPOINT LICENSED -> {htc_marker}")
            try:
                _bb = _dm.DepthBackbone(htc=True)
                _fp = _dm.license_checkpoint(htc_marker, _bb)
                print(f"  behavioural fingerprint recorded ({_fp['hash']}) -- "
                      f"depth_model.py will refuse this checkpoint later if its "
                      f"output on a fixed canary input ever drifts past "
                      f"{_dm.CANARY_REL_TOL*100:.0f}%")
            except Exception as _e:
                print(f"  WARNING: could not record a behavioural fingerprint "
                      f"({_e}) -- checkpoint is licensed but unverifiable at load time")
        else:
            print("")
            print(f"  --htc run finished (did not beat baseline on MAE). Weights "
                  f"are in {htc_ckpt}/htc_model.pt but NOT licensed -- "
                  "no htc_ok.json written.")

    ckpt = os.path.join(a.out, "height_best")
    marker = os.path.join(ckpt, "metric_ok.json")
    if not a.htc and os.path.exists(marker):
        os.remove(marker)
    if not a.htc and won and os.path.isdir(ckpt):
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({
                "licensed": True,
                "units": "metres (AGL) predicted directly",
                "baseline_mae_m_with_affine": round(base["mae_m"], 3),
                "finetuned_mae_m_no_affine": round(history[-1]["mae_m"], 3),
                "baseline_corr": round(base["corr"], 4),
                "finetuned_corr": round(history[-1]["corr"], 4),
                "promotion_gates": [
                    {"name": n, "value": v, "bar": b, "passed": p}
                    for n, v, b, p in gate_report
                ],
                "epochs": a.epochs, "device": device,
                "trained_tiles": n_gamus_train + n_dfc_train,
                "note": "The pipeline reads this file to decide whether to skip "
                        "its Tier C scale assumption. Written only because the "
                        "fine-tuned model beat a baseline given the optimal scale.",
            }, f, indent=2)
        print("")
        print(f"  CHECKPOINT LICENSED -> {marker}")
        print("  Builds will now predict metres directly and skip the "
              "'tallest ~ 40 m' assumption.")
        try:
            _bb = _dm.DepthBackbone(metric=True)
            _fp = _dm.license_checkpoint(marker, _bb)
            print(f"  behavioural fingerprint recorded ({_fp['hash']}) -- "
                  f"depth_model.py will refuse this checkpoint later if its "
                  f"output on a fixed canary input ever drifts past "
                  f"{_dm.CANARY_REL_TOL*100:.0f}%")
        except Exception as _e:
            print(f"  WARNING: could not record a behavioural fingerprint "
                  f"({_e}) -- checkpoint is licensed but unverifiable at load time")
    elif not a.htc and os.path.isdir(ckpt):
        print("")
        print("  Checkpoint NOT licensed: no metric_ok.json written, so the "
              "pipeline will keep using the stock backbone.")

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
