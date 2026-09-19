"""Correctness tests for validation.compute_metrics -- the function every
accuracy number this project has ever quoted (rescore_baseline.py, the
per-terrain table in dashboard/validation_results.json, model_audit.py) goes
through. If RMSE/MAE/bias were computed wrong here, every downstream claim
would be wrong the same way, silently.
"""
import numpy as np
import pytest

from validation import compute_metrics


def test_perfect_prediction_gives_zero_error():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = compute_metrics(gt.copy(), gt, terrain_mask=None)
    assert m["overall"]["rmse_m"] == 0.0
    assert m["overall"]["mae_m"] == 0.0
    assert m["overall"]["bias_m"] == 0.0
    assert m["overall"]["n_pixels"] == 4


def test_known_constant_offset():
    gt = np.full((10, 10), 5.0)
    pred = np.full((10, 10), 8.0)   # over-predicting by exactly 3m everywhere
    m = compute_metrics(pred, gt, terrain_mask=None)
    assert m["overall"]["rmse_m"] == pytest.approx(3.0)
    assert m["overall"]["mae_m"] == pytest.approx(3.0)
    # Signed bias must show the DIRECTION of the error: positive = over-predicting.
    assert m["overall"]["bias_m"] == pytest.approx(3.0)


def test_under_prediction_gives_negative_bias():
    gt = np.full((10, 10), 10.0)
    pred = np.full((10, 10), 6.0)   # under-predicting by 4m
    m = compute_metrics(pred, gt, terrain_mask=None)
    assert m["overall"]["bias_m"] == pytest.approx(-4.0)
    # RMSE/MAE are unsigned regardless of direction.
    assert m["overall"]["rmse_m"] == pytest.approx(4.0)


def test_nan_ground_truth_pixels_are_excluded_not_zeroed():
    gt = np.array([[1.0, np.nan], [3.0, 4.0]])
    pred = np.array([[1.0, 99.0], [3.0, 4.0]])   # garbage where GT is NaN
    m = compute_metrics(pred, gt, terrain_mask=None)
    # The NaN pixel's huge error must not leak into the aggregate.
    assert m["overall"]["rmse_m"] == 0.0
    assert m["overall"]["n_pixels"] == 3


def test_per_terrain_breakdown_isolates_each_class():
    gt = np.array([[10.0, 10.0], [10.0, 10.0]])
    pred = np.array([[10.0, 10.0], [15.0, 10.0]])   # only bottom-left is wrong
    terrain_mask = np.array([[0, 0], [1, 1]])   # 0=urban-like, 1=hilly-like
    m = compute_metrics(pred, gt, terrain_mask=terrain_mask)
    # compute_metrics returns named classes (TERRAIN_CLASSES-indexed), not
    # raw integers, so check by which class actually contains the wrong
    # pixel rather than assuming a specific name for mask value 0 or 1.
    nonzero_classes = [k for k, v in m.items()
                       if k != "overall" and v["n_pixels"] and v["rmse_m"] and v["rmse_m"] > 0]
    zero_classes = [k for k, v in m.items()
                    if k != "overall" and v["n_pixels"] and v["rmse_m"] == 0]
    assert len(nonzero_classes) == 1
    assert len(zero_classes) == 1
    assert m["overall"]["n_pixels"] == 4


def test_class_with_no_pixels_reports_null_not_fabricated_zero():
    gt = np.full((4, 4), 5.0)
    pred = np.full((4, 4), 5.0)
    # Every pixel is class 0; class-index 3 ("sparse" in TERRAIN_CLASSES
    # ordering) never appears in this mask at all.
    terrain_mask = np.zeros((4, 4), dtype=np.int32)
    m = compute_metrics(pred, gt, terrain_mask=terrain_mask)
    empty_classes = [k for k, v in m.items() if k != "overall" and v["n_pixels"] == 0]
    assert empty_classes, "expected at least one terrain class with zero pixels"
    for k in empty_classes:
        assert m[k]["rmse_m"] is None, \
            f"{k} has 0 pixels but reports a fabricated rmse_m instead of None"
        assert m[k]["mae_m"] is None


def test_shape_mismatch_raises_rather_than_silently_broadcasting():
    gt = np.zeros((10, 10))
    pred = np.zeros((5, 5))
    try:
        compute_metrics(pred, gt)
        assert False, "expected a ValueError on shape mismatch"
    except ValueError:
        pass


