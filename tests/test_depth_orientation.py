"""Regression test for a real bug this project already found and fixed
(documented in depth_model.py's predict() docstring): an earlier version
flipped the height field (`depth.max() - depth`), which put every rooftop
BELOW the surrounding ground and extruded buildings into holes instead of
towers. orientation_check() is the standing guard against that regressing
silently -- these tests pin its actual comparison logic.
"""
import numpy as np

from depth_model import orientation_check


def test_correct_orientation_when_buildings_are_higher():
    seg_labels = np.zeros((20, 20), dtype=np.int32)
    seg_labels[0:10, :] = 1     # "building" region (see CLASS_IDX in segmentation.py)
    seg_labels[10:20, :] = 0    # "bare_earth" region
    height = np.zeros((20, 20), dtype=np.float32)
    height[0:10, :] = 20.0      # buildings read taller
    height[10:20, :] = 2.0      # ground reads shorter

    import segmentation as seg
    seg_labels = np.where(seg_labels == 1, seg.CLASS_IDX["building"],
                          seg.CLASS_IDX["bare_earth"])
    result = orientation_check(height, seg_labels)
    assert result["checked"] is True
    assert result["correct_orientation"] is True
    assert result["building_mean"] > result["ground_mean"]


def test_inverted_orientation_is_detected():
    import segmentation as seg
    seg_labels = np.full((20, 20), seg.CLASS_IDX["bare_earth"], dtype=np.int32)
    seg_labels[0:10, :] = seg.CLASS_IDX["building"]
    height = np.zeros((20, 20), dtype=np.float32)
    # Inverted: buildings read LOWER than ground, exactly the historical bug.
    height[0:10, :] = 2.0
    height[10:20, :] = 20.0

    result = orientation_check(height, seg_labels)
    assert result["checked"] is True
    assert result["correct_orientation"] is False


def test_too_few_pixels_is_reported_not_guessed():
    import segmentation as seg
    seg_labels = np.full((20, 20), seg.CLASS_IDX["bare_earth"], dtype=np.int32)
    seg_labels[0, 0] = seg.CLASS_IDX["building"]   # 1 pixel: not enough to judge
    height = np.random.rand(20, 20).astype(np.float32)
    result = orientation_check(height, seg_labels)
    assert result["checked"] is False
