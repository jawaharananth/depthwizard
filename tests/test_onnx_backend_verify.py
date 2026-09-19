"""Tests for onnx_backend.ONNXDepth.verify()'s accept/reject logic itself --
not the real export (that needs a real model and is exercised manually via
export_onnx.py), but the safety check that decides whether an exported
model's output is trustworthy. This is the exact logic that caught the real
21.65% int8 quantization error this session and correctly refused it; these
tests pin that behaviour with synthetic data so it can't silently loosen.
"""
import numpy as np

from onnx_backend import ONNXDepth, MAX_REL_ERROR


def _bare_instance():
    # Bypass __init__ (which needs a real torch model + onnxruntime install
    # for the constructor path) to test verify()'s pure logic in isolation.
    obj = ONNXDepth.__new__(ONNXDepth)
    obj.ok = True
    return obj


def test_identical_output_passes():
    obj = _bare_instance()
    obj.infer = lambda px: px.copy()
    ref = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
    assert obj.verify(ref, ref) is True
    assert obj.ok is True


def test_large_deviation_is_rejected_and_flips_ok_false():
    obj = _bare_instance()
    ref = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
    # 25% of the value range as a max deviation -- comparable in scale to the
    # real 21.65% int8 quantization error this test suite exists to guard
    # against ever silently passing again.
    bad = ref + 25.0
    obj.infer = lambda px: bad
    assert obj.verify(ref, ref) is False
    assert obj.ok is False, "verify() must flip .ok to False on rejection"


def test_deviation_just_under_threshold_passes():
    obj = _bare_instance()
    ref = np.zeros(100, dtype=np.float32)
    ref[0] = 100.0   # spread = 100
    just_under = ref.copy()
    just_under[0] += (MAX_REL_ERROR * 100.0) * 0.9   # 90% of the allowed deviation
    obj.infer = lambda px: just_under
    assert obj.verify(ref, ref) is True


def test_deviation_just_over_threshold_fails():
    obj = _bare_instance()
    ref = np.zeros(100, dtype=np.float32)
    ref[0] = 100.0
    just_over = ref.copy()
    just_over[0] += (MAX_REL_ERROR * 100.0) * 1.5   # 150% of the allowed deviation
    obj.infer = lambda px: just_over
    assert obj.verify(ref, ref) is False


def test_shape_mismatch_is_rejected():
    obj = _bare_instance()
    obj.infer = lambda px: np.zeros((5, 5), dtype=np.float32)
    ref = np.zeros((10, 10), dtype=np.float32)
    assert obj.verify(ref, ref) is False


def test_not_ok_instance_refuses_immediately():
    obj = _bare_instance()
    obj.ok = False
    obj.infer = lambda px: (_ for _ in ()).throw(RuntimeError("should never be called"))
    ref = np.zeros((5, 5), dtype=np.float32)
    assert obj.verify(ref, ref) is False


def test_infer_exception_is_caught_not_propagated():
    obj = _bare_instance()
    def _boom(px):
        raise RuntimeError("session crashed")
    obj.infer = _boom
    ref = np.zeros((5, 5), dtype=np.float32)
    assert obj.verify(ref, ref) is False
    assert obj.ok is False
