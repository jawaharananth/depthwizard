"""Export the metric height checkpoint to ONNX for fast CPU inference.

    python tools/export_onnx.py [checkpoints/height_best] [--int8]

Writes model.onnx (fp32) -- and model_int8.onnx with --int8 -- into the
checkpoint directory, plus onnx_manifest.json recording the safetensors hash
it was exported from, the measured agreement with PyTorch and the timing.
depth_model.DepthBackbone uses the ONNX file only when that hash still
matches the weights next to it AND a canary input reproduces PyTorch within
tolerance at load, so a stale or wrong export can never be used silently.

Fixed 518 x 518. A dynamic-shape export traces the patch-grid resize as a
constant (a 518 x 700 input came back 518 x 518), so the export is fixed and
depth_model uses it only for 518 x 518 inputs -- every square tile crop, i.e.
almost all of the work -- and PyTorch for anything else.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ck = os.path.abspath(args[0] if args else os.path.join(ROOT, "checkpoints", "height_best"))
    from transformers import AutoModelForDepthEstimation
    import onnxruntime as ort

    model = AutoModelForDepthEstimation.from_pretrained(ck).eval()
    w = Wrap(model)
    x = torch.randn(1, 3, 518, 518)
    out_fp32 = os.path.join(ck, "model.onnx")
    torch.onnx.export(w, (x,), out_fp32, input_names=["pixel_values"], output_names=["depth"],
                      opset_version=17, dynamo=False)
    files = {"fp32": out_fp32}
    if "--int8" in sys.argv:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        out_i8 = os.path.join(ck, "model_int8.onnx")
        quantize_dynamic(out_fp32, out_i8, weight_type=QuantType.QInt8)
        files["int8"] = out_i8

    man = {"source_safetensors_sha": sha(os.path.join(ck, "model.safetensors")), "variants": {}}
    rng = np.random.default_rng(0)
    for shape in [(518, 518)]:
        xin = torch.from_numpy(rng.standard_normal((1, 3) + shape).astype(np.float32))
        with torch.inference_mode():
            ref = w(xin).numpy()
            t0 = time.time()
            for _ in range(3):
                w(xin)
            t_torch = (time.time() - t0) / 3
        for name, path in files.items():
            so = ort.SessionOptions()
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            y = s.run(None, {"pixel_values": xin.numpy()})[0]
            t0 = time.time()
            for _ in range(3):
                s.run(None, {"pixel_values": xin.numpy()})
            t_ort = (time.time() - t0) / 3
            v = man["variants"].setdefault(name, {"file": os.path.basename(path),
                                                  "size_mb": round(os.path.getsize(path) / 1e6, 1)})
            v[f"{shape[0]}x{shape[1]}"] = {"max_abs_diff_m": float(np.abs(y - ref).max()),
                                           "mean_abs_diff_m": float(np.abs(y - ref).mean()),
                                           "ms_onnx": round(t_ort * 1000), "ms_torch": round(t_torch * 1000)}
    json.dump(man, open(os.path.join(ck, "onnx_manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
