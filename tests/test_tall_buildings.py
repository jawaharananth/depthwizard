"""Synthetic-scene checks for physics/tall_buildings.py.

A box building is rendered with exact geometry (shadow = footprint swept
along sun_az+180 by h*cot(sun_el); roof = footprint moved along sat_az+180 by
h*tan(zenith); facade band in between), next to a green tree blob. The model
nDSM is the true height compressed 1.7x -- the failure this module exists for.
"""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics import tall_buildings as TB  # noqa: E402

GSD = 0.25


def _sweep(fp, vec, length_px, n=None):
    out = np.zeros_like(fp)
    n = n or int(math.ceil(length_px)) + 1
    for t in np.linspace(0, length_px, n):
        M = np.float32([[1, 0, vec[0] * t], [0, 1, vec[1] * t]])
        out |= cv2.warpAffine(fp.astype(np.uint8), M, fp.shape[::-1],
                              flags=cv2.INTER_NEAREST) > 0
    return out


def render(h=30.0, sun_el=36.8, sun_az=152.9, sat_el=70.0, sat_az=144.3, size=480):
    geo = TB.view_geometry(sun_el, sun_az, sat_el, sat_az)
    fp = np.zeros((size, size), bool)
    fp[260:340, 260:340] = True                       # 20 x 20 m base
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), 150, np.float32) + rng.normal(0, 4, (size, size, 3))
    shadow = _sweep(fp, geo["s"], h * geo["k_shadow"] / GSD)
    facade = _sweep(fp, geo["lean"], h * geo["k_lean"] / GSD)
    roof = _sweep(fp, geo["lean"], 0, 1)
    M = np.float32([[1, 0, geo["lean"][0] * h * geo["k_lean"] / GSD],
                    [0, 1, geo["lean"][1] * h * geo["k_lean"] / GSD]])
    roof = cv2.warpAffine(fp.astype(np.uint8), M, fp.shape[::-1], flags=cv2.INTER_NEAREST) > 0
    img[shadow] = 45
    img[facade] = 105
    img[roof] = 205
    # a tree: green, tall in the model
    yy, xx = np.mgrid[:size, :size]
    tree = (yy - 80) ** 2 + (xx - 80) ** 2 < 40 ** 2
    img[tree] = (60, 120, 50)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ndsm = np.zeros((size, size), np.float32)
    ndsm[roof] = h / 1.7
    ndsm[tree] = 12.0
    ndsm = cv2.GaussianBlur(ndsm, (0, 0), 1.5)
    return img, ndsm, roof, tree, dict(sun_el_deg=sun_el, sun_az_deg=sun_az,
                                       sat_el_deg=sat_el, sat_az_deg=sat_az)


def test_azimuth_convention():
    assert np.allclose(TB.az_vec(0), [0, -1])      # north = up (y decreases)
    assert np.allclose(TB.az_vec(90), [1, 0])      # east = right
    assert np.allclose(TB.az_vec(180), [0, 1])     # south = down


def test_shadow_recovers_height_and_fusion_raises_it():
    img, ndsm, roof, tree, g = render(h=30.0)
    fused, rep = TB.refine(img, ndsm, GSD, **g)
    live = [c for c in rep["candidates"] if not c["rejected"]]
    assert len(live) == 1
    c = live[0]
    assert c["h_shadow"] == pytest.approx(30.0, rel=0.12)
    assert c["fired"]
    assert np.median(fused[roof]) > np.median(ndsm[roof]) * 1.25


def test_relief_displacement_recovers_height():
    img, ndsm, roof, tree, g = render(h=30.0, sat_el=65.0)
    _, rep = TB.refine(img, ndsm, GSD, **g)
    c = [c for c in rep["candidates"] if not c["rejected"]][0]
    assert c["h_disp"] == pytest.approx(30.0, rel=0.15)


def test_trees_and_ground_untouched():
    img, ndsm, roof, tree, g = render(h=30.0)
    fused, rep = TB.refine(img, ndsm, GSD, **g)
    assert any(c["rejected"] and "vegetated" in c["rejected"] for c in rep["candidates"])
    far = cv2.dilate(roof.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    assert np.array_equal(fused[far], ndsm[far])           # trees + ground bit-identical
    assert np.array_equal(fused[tree], ndsm[tree])


def test_no_fire_when_model_agrees():
    img, ndsm, roof, tree, g = render(h=30.0)
    ndsm = ndsm * 1.7                                       # model now correct
    fused, rep = TB.refine(img, ndsm, GSD, **g)
    assert rep["n_fired"] == 0
    assert np.array_equal(fused, ndsm.astype(np.float32))
