"""One command: download GAMUS, fine-tune, license the checkpoint, re-benchmark.

Run this on the GPU machine and nothing else needs deciding:

    python scripts/run_gamus.py

It will download a training subset, train, judge the result against the stock
backbone, write the licence marker only if the fine-tuned model won, and then
re-run the blind benchmark so the effect on the headline number is measured
rather than assumed.

WHAT IS ACTUALLY BEING FIXED

Blind, with no fitting to ground truth anywhere, the pipeline currently measures
RMSE 16.07 m / MAE 10.33 m. The shape is not the problem -- correlation runs
0.36-0.53 and a truth-fitted scale brings the same predictions to about 3.9 m.
The problem is the scale: the pipeline emits 71-90 m/unit on every tile while
each tile actually needs 17-57, because a relative depth field forces it to
assume one ("tallest structure ~ 40 m") and a DEM offset moves the datum without
touching the scale.

Two other routes to scale were measured and are dead on this data. Shadow
calibration finds 0-6 usable shadows per tile against the ~10 it needs. A
two-parameter DEM fit cannot work on flat terrain: SRTM relief over these tiles
is 1-6 m and SRTM is quantised to 1 m, so fitted slopes came out 0.083-0.572
where they should be ~1.0.

That leaves a model that predicts metres directly. Nothing downstream has to
guess a scale, so the dominant error source is removed rather than reduced.

WHAT SUCCESS LOOKS LIKE

Two gates, in order.

  1. The fine-tuned model must beat the stock backbone on held-out GAMUS tiles
     while the BASELINE is handed a best-fit affine and the fine-tuned model is
     not. Only then is the checkpoint licensed and only then will any build use
     it. Baseline to beat: MAE 4.28 m, correlation 0.299.

  2. The blind benchmark must improve. This is the number that matters and it is
     measured here, not predicted. Current: RMSE 16.07 m / MAE 10.33 m.

A run that passes gate 1 and fails gate 2 has produced a better depth model that
did not help the product, and that is worth knowing before it goes on a slide.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(cmd, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    print(f"[{label}: exit {r.returncode} in {dt/60:.1f} min]", flush=True)
    return r.returncode == 0


def read_blind():
    """Current headline from the blind benchmark, for the before/after."""
    path = os.path.join(ROOT, "BENCHMARK_BLIND.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("**RMSE"):
                return line.strip()
    return "no metric tier reached"


def main() -> None:
    ap = argparse.ArgumentParser(description="GAMUS fine-tune, end to end")
    ap.add_argument("--train-tiles", type=int, default=400,
                    help="GAMUS training tiles to download (~10 MB each)")
    ap.add_argument("--val-tiles", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--bench-tiles", type=int, default=10,
                    help="tiles for the before/after blind benchmark")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the plumbing without training: tiny download, "
                         "one smoke epoch, no benchmark")
    a = ap.parse_args()

    import torch
    cuda = torch.cuda.is_available()
    print(f"device: {'cuda -- ' + torch.cuda.get_device_name(0) if cuda else 'CPU'}")
    if not cuda and not a.dry_run:
        print("\nNo CUDA. A real run on CPU would take many hours; 400 tiles at\n"
              "12 epochs is a GPU job. Use --dry-run here to verify the\n"
              "plumbing, then run this unchanged on the GPU machine.")
        return

    before = read_blind()
    print(f"\nblind benchmark before: {before}")

    if a.dry_run:
        a.train_tiles, a.val_tiles = 8, 6

    if not a.skip_download:
        ok = run([PY, "gamus.py", "download", "--split", "train",
                  "--limit", str(a.train_tiles)], "1/4  download train split")
        ok = run([PY, "gamus.py", "download", "--split", "val",
                  "--limit", str(a.val_tiles)], "2/4  download val split") and ok
        if not ok:
            print("download failed -- stopping"); return

    train_cmd = [PY, "train_height.py"]
    train_cmd += ["--smoke"] if a.dry_run else [
        "--epochs", str(a.epochs), "--batch", str(a.batch), "--accum", str(a.accum)]
    if not run(train_cmd, "3/4  fine-tune"):
        print("training failed -- stopping"); return

    # Gate 1: did the run license itself?
    sys.path.insert(0, ROOT)
    from depth_model import DepthBackbone
    if not DepthBackbone.metric_available():
        print("\n" + "=" * 70)
        print("CHECKPOINT NOT LICENSED.")
        print("The fine-tuned model did not beat a baseline that was handed the")
        print("optimal scale, so no marker was written and every build will keep")
        print("using the stock backbone. See checkpoints/training_report.json.")
        print("Nothing downstream has changed -- that is the intended outcome of")
        print("a losing run, not a failure of this script.")
        print("=" * 70)
        return

    info = DepthBackbone.metric_info()
    print("\n" + "=" * 70)
    print("CHECKPOINT LICENSED")
    print(f"  fine-tuned  MAE {info.get('finetuned_mae_m_no_affine')} m "
          f"(no affine)   corr {info.get('finetuned_corr')}")
    print(f"  baseline    MAE {info.get('baseline_mae_m_with_affine')} m "
          f"(affine granted)   corr {info.get('baseline_corr')}")
    print("  Builds now predict metres directly and skip the scale assumption.")
    print("=" * 70)

    if a.dry_run:
        print("\ndry run: skipping the blind benchmark.")
        return

    # Gate 2: the number that actually matters.
    if not run([PY, "scripts/benchmark_blind.py", str(a.bench_tiles)],
               "4/4  blind benchmark (the headline)"):
        print("benchmark failed -- the checkpoint is still licensed"); return

    after = read_blind()
    print("\n" + "=" * 70)
    print("BLIND BENCHMARK -- no fitting to ground truth at any point")
    print(f"  before: {before}")
    print(f"  after : {after}")
    print("=" * 70)
    print("\nIf the after figure is not better, the model improved and the")
    print("product did not. Say so rather than quoting gate 1.")


if __name__ == "__main__":
    main()
