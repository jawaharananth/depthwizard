"""Export the production depth backbone to a single portable ONNX file.

Distinct from onnx_backend.py's role inside the live pipeline (that module
exists so build_city_image.py *could* use ONNX Runtime at inference time,
mirroring ov_backend.py's OpenVINO path). This script's job is narrower and
more useful in practice: hand someone one file plus `pip install
onnxruntime` and they can run depth inference without PyTorch, transformers,
or any of this project's other dependencies -- something to attach to a
submission or a judge-facing demo, not a speed optimisation for our own
runtime (measured: ONNX Runtime CPU and PyTorch CPU run at comparable speed
on this machine; the OpenVINO iGPU path in ov_backend.py is the actual
speedup, and stays PyTorch-based).

Usage:
    python export_onnx.py                              # stock backbone
    python export_onnx.py --metric                      # the licensed fine-tuned checkpoint, if one exists
    python export_onnx.py --out my_model.onnx --size 768
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

from onnx_backend import ONNXDepth
from depth_model import DepthBackbone


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="HF model id or local checkpoint dir; default: stock "
                         "backbone, or the licensed metric checkpoint with --metric")
    ap.add_argument("--metric", action="store_true",
                    help="use the licensed fine-tuned metric checkpoint "
                         "(DepthBackbone.METRIC_CKPT); fails loudly if none is licensed")
    ap.add_argument("--size", type=int, default=518,
                    help="export/trace resolution; the DPT patch size (14) must divide it")
    ap.add_argument("--out", default=None,
                    help="output .onnx path; default: cache/<tag>_fp32.onnx")
    ap.add_argument("--quantize", action="store_true",
                    help="also attempt int8 dynamic quantization -- expected to FAIL "
                         "verification on this model family, see onnx_backend.py's "
                         "measured 21.65%% deviation; kept as an option, not a default")
    a = ap.parse_args()

    if a.size % 14 != 0:
        raise SystemExit(f"--size {a.size} is not a multiple of 14 (the DPT patch size)")

    if a.metric:
        if not DepthBackbone.metric_available():
            raise SystemExit("no licensed metric checkpoint -- run train_height.py first, "
                             "and it must beat baseline")
        model_id = DepthBackbone.METRIC_CKPT
        tag = "metric"
    else:
        model_id = a.model or "depth-anything/Depth-Anything-V2-Base-hf"
        tag = model_id.split("/")[-1].replace("-hf", "").lower()

    print(f"loading {model_id} ...")
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    proc = AutoImageProcessor.from_pretrained(model_id)

    # A real photo, not synthetic noise -- the tracer's shape-dependent
    # branches (see onnx_backend.py's comment on this) are more likely to
    # exercise their real path on real image statistics than on noise.
    sample_path = "sample.jpg" if os.path.exists("sample.jpg") else None
    if sample_path:
        img = Image.open(sample_path).convert("RGB").resize((a.size, a.size))
    else:
        img = Image.fromarray((np.random.rand(a.size, a.size, 3) * 255).astype(np.uint8))
    px = proc(images=img, return_tensors="pt")["pixel_values"]

    model.eval()
    with torch.no_grad():
        ref = model(pixel_values=px).predicted_depth.numpy()

    backend = ONNXDepth(model, px, tag, quantize=a.quantize)
    if not backend.ok:
        raise SystemExit("ONNX export failed -- see the message above")

    verified = backend.verify(ref, px.numpy())
    if not verified:
        raise SystemExit(
            "Exported, but the ONNX model does NOT reproduce PyTorch within "
            "tolerance -- refusing to hand this out. See onnx_backend.py's "
            "MAX_REL_ERROR and its docstring for the measured int8 finding.")

    out_path = a.out or backend.path
    if a.out and a.out != backend.path:
        import shutil
        shutil.copy(backend.path, a.out)
        out_path = a.out

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nVERIFIED and ready: {out_path} ({size_mb:.1f} MB)")
    print(f"  input: 'pixel_values', float32, shape (1, 3, {a.size}, {a.size})")
    print(f"  output: 'predicted_depth', shape (1, {a.size}, {a.size})")
    print(f"  run anywhere with: pip install onnxruntime pillow numpy")


if __name__ == "__main__":
    main()
