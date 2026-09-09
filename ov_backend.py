"""Optional OpenVINO backend for depth inference on Intel integrated graphics.

There is no NVIDIA GPU on the target machine -- it is a 15 W Core 5 120U with
Intel integrated graphics -- so CUDA is not available and PyTorch runs on CPU.
OpenVINO can target that iGPU, which is the only real acceleration available.

MEASURED, per 518x518 tile:

    PyTorch CPU          4.67 s
    iGPU fp16 (default)  0.97 s   4.8x   but 9.14% max relative error
    iGPU fp32            1.42 s   3.3x   0.20% max relative error, corr 0.999999

fp16 is the default OpenVINO GPU precision and it is NOT used here. A 9%
relative error on the depth field is not a rounding detail: heights, footprint
extraction and the terrain surface are all derived from it, and the pipeline
already reports building heights to 0.1 m. The 3.3x that keeps the numbers
intact is worth more than the 4.8x that does not.

Everything degrades gracefully: no OpenVINO, no iGPU, or a conversion failure
all fall back to PyTorch, because a missing accelerator must never be the
difference between a build and no build.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

# Conversion is slow (~2 min) and the result is reusable, so it is cached beside
# the height fields rather than redone per run.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Anything above this much relative deviation from the PyTorch reference means
# the accelerated path is not computing the same thing, and it is rejected.
MAX_REL_ERROR = 0.01


class OpenVINODepth:
    """iGPU-accelerated wrapper around a HuggingFace depth model."""

    def __init__(self, torch_model, example_input, model_tag: str,
                 device: str = "GPU", verbose: bool = True):
        self.ok = False
        self.device = device
        self._compiled = None
        self._out = None
        try:
            import openvino as ov
        except ImportError:
            if verbose:
                print("      openvino not installed; using PyTorch")
            return

        core = ov.Core()
        if device not in core.available_devices:
            if verbose:
                print(f"      no OpenVINO {device}; using PyTorch")
            return

        os.makedirs(CACHE_DIR, exist_ok=True)
        ir = os.path.join(CACHE_DIR, f"{model_tag}_ov.xml")
        try:
            if not os.path.exists(ir):
                if verbose:
                    print(f"      converting depth model to OpenVINO IR "
                          f"(one-off, ~2 min) ...")
                t0 = time.time()
                m = ov.convert_model(torch_model,
                                     example_input={"pixel_values": example_input})
                ov.save_model(m, ir)
                if verbose:
                    print(f"      converted in {time.time()-t0:.0f}s -> {ir}")
            model = core.read_model(ir)
            # f32 explicitly. The GPU plugin defaults to f16, which measured
            # 9.14% max relative error against PyTorch on this model.
            self._compiled = core.compile_model(
                model, device, {"INFERENCE_PRECISION_HINT": "f32"})
            self._out = self._compiled.output(0)
            self.ok = True
        except Exception as e:
            if verbose:
                print(f"      OpenVINO unavailable ({type(e).__name__}); "
                      f"using PyTorch")
            self._compiled = None

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        return np.asarray(self._compiled({"pixel_values": pixel_values})[self._out])

    def verify(self, reference: np.ndarray, pixel_values: np.ndarray,
               verbose: bool = True) -> bool:
        """Refuse the accelerated path if it does not reproduce PyTorch.

        A silent numerical difference is worse than no acceleration: it would
        change every height in the scene while every log line still said the
        build succeeded.
        """
        if not self.ok:
            return False
        try:
            got = self.infer(pixel_values).ravel().astype(np.float64)
        except Exception:
            self.ok = False
            return False
        ref = np.asarray(reference).ravel().astype(np.float64)
        if got.shape != ref.shape:
            self.ok = False
            return False
        spread = float(ref.max() - ref.min()) or 1.0
        rel = float(np.max(np.abs(got - ref))) / spread
        if rel > MAX_REL_ERROR:
            if verbose:
                print(f"      OpenVINO output differs by {rel*100:.2f}% "
                      f"(limit {MAX_REL_ERROR*100:.0f}%); falling back to PyTorch")
            self.ok = False
            return False
        if verbose:
            print(f"      OpenVINO {self.device} verified "
                  f"({rel*100:.3f}% max deviation)")
        return True
