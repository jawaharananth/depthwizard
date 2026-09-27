"""Band order and no-data handling for satellite GeoTIFFs (input_bands.py).

    python tests/test_input_bands.py
"""
import os
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import input_bands as ib  # noqa: E402


def _write(path, arr, dtype, nodata=None, colorinterp=None):
    n, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=n, dtype=dtype,
                       crs="EPSG:32644", transform=from_origin(500000, 2000000, 0.6, 0.6),
                       nodata=nodata) as ds:
        ds.write(arr.astype(dtype))
        if colorinterp:
            ds.colorinterp = colorinterp


def test_untagged_16bit_4band_is_bgr_nir():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mx.tif")
        a = np.stack([np.full((64, 64), v) for v in (100, 200, 300, 400)])
        _write(p, a, "uint16")
        with rasterio.open(p) as ds:
            bands, how = ib.rgb_band_plan(ds)
        assert bands == [3, 2, 1] and how == "assumed-bgr", (bands, how)


def test_tagged_bands_win():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tagged.tif")
        a = np.stack([np.full((64, 64), v) for v in (1, 2, 3, 4)])
        _write(p, a, "uint16", colorinterp=[ColorInterp.blue, ColorInterp.green,
                                            ColorInterp.red, ColorInterp.undefined])
        with rasterio.open(p) as ds:
            bands, how = ib.rgb_band_plan(ds)
        assert bands == [3, 2, 1] and how == "tagged", (bands, how)


def test_plain_8bit_rgb_unchanged():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rgb.tif")
        _write(p, np.zeros((3, 32, 32)) + 5, "uint8")
        with rasterio.open(p) as ds:
            assert ib.rgb_band_plan(ds)[0] == [1, 2, 3]


def test_grey_single_band():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pan.tif")
        _write(p, np.zeros((1, 32, 32)) + 5, "uint16")
        with rasterio.open(p) as ds:
            assert ib.rgb_band_plan(ds)[1] == "grey"


def test_nodata_value_is_masked():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "nd.tif")
        a = np.full((3, 40, 40), 50)
        a[:, :10, :] = 9999
        _write(p, a, "uint16", nodata=9999)
        with rasterio.open(p) as ds:
            v = ib.dataset_valid(ds, [1, 2, 3])
        assert not v[:10].any() and v[10:].all()


def test_border_collar_found_but_inner_black_kept():
    a = np.full((3, 100, 100), 80)
    a[:, :, :15] = 0                      # collar on the left edge
    a[:, 40:50, 40:50] = 0                # black object inside the scene
    c = ib.border_collar(a)
    assert c[:, :15].all()
    assert not c[40:50, 40:50].any()
    assert not c[:, 20:].any()


def test_tiny_edge_black_is_not_a_collar():
    a = np.full((3, 200, 200), 80)
    a[:, 0:2, 0:2] = 0                    # 4 px, below MIN_COLLAR_FRACTION
    assert not ib.border_collar(a).any()


def test_fill_invalid_uses_scene_colour():
    rgb = np.zeros((10, 10, 3), np.uint8)
    rgb[:, 5:] = (10, 20, 30)
    v = np.zeros((10, 10), bool)
    v[:, 5:] = True
    out = ib.fill_invalid(rgb, v)
    assert (out[:, :5] == (10, 20, 30)).all()


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} passed")
