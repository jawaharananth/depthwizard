"""Optional ONNX Runtime backend for depth inference: portable, no PyTorch
needed at inference time, and small enough to hand to someone as one file.

WHY THIS EXISTS ALONGSIDE ov_backend.py

ov_backend.py accelerates the SAME model on THIS machine's Intel iGPU. It
buys speed here, but the artifact it produces (an OpenVINO IR) needs OpenVINO
installed wherever it runs next -- it is not something you email a judge and
they double-click. An ONNX export with int8 dynamic quantization is: it is a
single ~35-45 MB file, ONNX Runtime is a much smaller and more commonly
pre-installed dependency than a full PyTorch+transformers stack, and it runs
identically on any CPU, not just this one. Competitor repos in this
problem-statement's field (zaidnansari2011/sih2026-depthwizard, for one)
ship exactly this for that reason.

Follows ov_backend.py's own rules deliberately, not by coincidence: convert
once and cache the result, verify the exported model reproduces PyTorch
before trusting it for anything, and degrade to plain PyTorch silently on any
failure -- a missing accelerator must never be the difference between a
build and no build.

MEASURED, depth-anything/Depth-Anything-V2-Small-hf, 518x518:

    fp32 ONNX (99.1 MB)     0.000% max deviation from PyTorch -- verified
    int8 dynamic (27.2 MB)  21.65% max deviation -- REJECTED, default is fp32

int8 dynamic quantization is the smaller, more attractive artifact -- it is
what a 36.8 MB competitor build in this problem statement's field ships --
but it fails this project's own 1% verification floor by more than 20x on
this exact model. The export path itself is exact (fp32's 0.000% proves the
tracing and the ONNX graph are correct); the quantization step specifically
is where the error comes from. Shipping the smaller file anyway because a
competitor's is smaller would be exactly the kind of unverified claim this
project's Tier system exists to refuse. `quantize=True` is left in as an
option, not the default, in case a future onnxruntime version or a
different backbone quantizes this model more cleanly -- verify() still
gates it either way.
"""

from __future__ import annotations

import os
import time

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Same floor as ov_backend.py's MAX_REL_ERROR, for the same reason: heights
# are reported to 0.1 m, so a quantized model's error against full-precision
# PyTorch has to stay small relative to the depth field's own dynamic range,
# not just "close enough" in absolute terms.
MAX_REL_ERROR = 0.01


class ONNXDepth:
    """Portable CPU-only wrapper around a HuggingFace depth model, exported
    once to ONNX and dynamically quantized to int8.
    """

    def __init__(self, torch_model, example_input, model_tag: str,
                quantize: bool = False, verbose: bool = True):
        self.ok = False
        self._session = None
        self._input_name = None
        self._output_name = None
        self.quantize = quantize
        self.path = None

        try:
            import onnxruntime as ort
        except ImportError:
            if verbose:
                print("      onnxruntime not installed; using PyTorch")
            return

        os.makedirs(CACHE_DIR, exist_ok=True)
        suffix = "_int8" if quantize else "_fp32"
        onnx_path = os.path.join(CACHE_DIR, f"{model_tag}{suffix}.onnx")

        try:
            if not os.path.exists(onnx_path):
                if verbose:
                    print(f"      exporting depth model to ONNX (one-off) ...")
                t0 = time.time()
                self._export(torch_model, example_input, onnx_path, quantize, verbose)
                if verbose:
                    size_mb = os.path.getsize(onnx_path) / 1e6
                    print(f"      exported in {time.time()-t0:.0f}s -> "
                          f"{onnx_path} ({size_mb:.1f} MB)")

            self._session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self.path = onnx_path
            self.ok = True
        except Exception as e:
            if verbose:
                print(f"      ONNX export/load failed ({type(e).__name__}: {e}); "
                      f"using PyTorch")
            self._session = None

    def _export(self, torch_model, example_input, onnx_path, quantize, verbose):
        import torch

        raw_path = onnx_path
        if quantize:
            raw_path = onnx_path.replace(".onnx", "_pre_quant.onnx")

        torch_model.eval()
        with torch.no_grad():
            # dynamo=False forces the older TorchScript-based tracer.
            # torch.onnx.export's newer torch.export-based path (default since
            # PyTorch 2.9) rejects this HF model outright (KeyError: -1 inside
            # its own args-normalisation, before tracing even starts). The
            # legacy tracer works, at the cost of the usual tracing caveat:
            # DPT's neck has shape-dependent Python control flow (see the
            # TracerWarnings this prints), so `dynamic_axes` declares intent
            # but does not guarantee correctness away from `example_input`'s
            # own resolution. verify() below is what actually catches that,
            # not this declaration -- exactly the same caveat ov_backend.py's
            # OpenVINO conversion already carries and handles the same way.
            torch.onnx.export(
                torch_model, (example_input,), raw_path,
                input_names=["pixel_values"], output_names=["predicted_depth"],
                dynamic_axes={"pixel_values": {0: "batch", 2: "height", 3: "width"},
                              "predicted_depth": {0: "batch", 1: "height", 2: "width"}},
                opset_version=17, dynamo=False,
            )

        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(raw_path, onnx_path, weight_type=QuantType.QInt8)
            os.remove(raw_path)

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        return self._session.run(
            [self._output_name], {self._input_name: pixel_values.astype(np.float32)}
        )[0]

    def verify(self, reference: np.ndarray, pixel_values: np.ndarray,
              verbose: bool = True) -> bool:
        """Refuse the ONNX path if it does not reproduce PyTorch.

        Same reasoning as ov_backend.OpenVINODepth.verify: a silent numerical
        difference from quantization is worse than no speedup, because it
        would change every reported height while the build log still says
        everything succeeded.
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
                print(f"      ONNX output differs by {rel*100:.2f}% "
                      f"(limit {MAX_REL_ERROR*100:.0f}%); falling back to PyTorch")
            self.ok = False
            return False
        if verbose:
            print(f"      ONNX (CPUExecutionProvider) verified "
                  f"({rel*100:.3f}% max deviation)")
        return True
