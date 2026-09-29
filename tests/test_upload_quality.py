"""Upload-quality pieces: NIR band detection, real-outline alignment/merge."""
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import input_bands as ib            # noqa: E402
import footprint_sources as fs      # noqa: E402


def _tif(path, count, dtype, ci=None, desc=None, last=None):
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=count, dtype=dtype,
                       crs="EPSG:32644", transform=from_origin(0, 8, 1, 1)) as d:
        a = np.ones((count, 8, 8), dtype)
        if last is not None:
            a[-1] = last
        d.write(a)
        if ci:
            d.colorinterp = ci
        for i, s in enumerate(desc or []):
            if s:
                d.set_band_description(i + 1, s)


def test_nir_band_cartosat_bgrn(tmp_path):
    p = str(tmp_path / "bgrn.tif")
    _tif(p, 4, "uint16")
    with rasterio.open(p) as s:
        rgb, how = ib.rgb_band_plan(s)
        assert (rgb, how) == ([3, 2, 1], "assumed-bgr")
        assert ib.nir_band(s, rgb, how) == 4


def test_nir_band_tagged_rgbn_and_described(tmp_path):
    # GDAL reads this 4th band back as 'alpha'; its NIR-like values say otherwise
    p = str(tmp_path / "rgbn.tif")
    _tif(p, 4, "uint8", ci=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined],
         last=np.arange(64, dtype="uint8").reshape(8, 8) + 40)
    with rasterio.open(p) as s:
        rgb, how = ib.rgb_band_plan(s)
        assert ib.nir_band(s, rgb, how) == 4
    p2 = str(tmp_path / "desc.tif")
    _tif(p2, 5, "uint8", desc=[None, None, None, None, "Near Infrared"])
    with rasterio.open(p2) as s:
        rgb, how = ib.rgb_band_plan(s)
        assert ib.nir_band(s, rgb, how) == 5


def test_no_nir_for_rgb_or_alpha(tmp_path):
    p = str(tmp_path / "rgb.tif")
    _tif(p, 3, "uint8")
    with rasterio.open(p) as s:
        rgb, how = ib.rgb_band_plan(s)
        assert ib.nir_band(s, rgb, how) is None
    p2 = str(tmp_path / "rgba.tif")      # a true alpha mask: only 0 and 255
    _tif(p2, 4, "uint8", ci=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha],
         last=np.where(np.arange(64).reshape(8, 8) < 20, 0, 255).astype("uint8"))
    with rasterio.open(p2) as s:
        rgb, how = ib.rgb_band_plan(s)
        assert ib.nir_band(s, rgb, how) is None


def _square(x, y, s):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


def test_real_outlines_shifted_onto_image_and_merged():
    import build_city_image as bci
    H = W = 400
    ndsm = np.zeros((H, W), np.float32)
    traced = []
    for i in range(6):
        for j in range(6):
            x, y = 20 + 60 * i, 20 + 60 * j
            ndsm[y:y + 30, x:x + 30] = 6.0
            traced.append(np.array(_square(x, y, 30), np.float32))
    # real outlines for half the buildings, offset by (-7, +4) px from the image
    real = [np.array(_square(20 + 60 * i - 7, 20 + 60 * j + 4, 30), np.float32).tolist()
            for i in range(3) for j in range(6)]
    out = bci.merge_real_outlines(traced, real, (H, W), ndsm, 0.3)
    # every real outline kept and moved back onto its roof; the traced twin dropped
    assert len(out) == 36
    first = np.asarray(out[0])
    assert abs(first[:, 0].min() - 20) <= 1 and abs(first[:, 1].min() - 20) <= 1


def test_real_outlines_over_nodata_are_dropped():
    import build_city_image as bci
    H = W = 200
    ndsm = np.zeros((H, W), np.float32)
    ndsm[20:50, 20:50] = 6
    traced = [np.array(_square(20, 20, 30), np.float32)]
    valid = np.ones((H, W), bool)
    valid[:, 120:] = False
    real = [_square(20, 20, 30), _square(150, 40, 30)]   # second one sits in no-data
    out = bci.merge_real_outlines(traced, real, (H, W), ndsm, 0.3, valid=valid)
    assert all(np.asarray(p)[:, 0].max() < 120 for p in out)


def test_footprint_merge_prefers_primary():
    a = [_square(10, 10, 20)]
    b = [_square(12, 12, 10), _square(60, 60, 10)]      # first is inside a, second is new
    out = fs._merge(a, b, 100, 100)
    assert len(out) == 2
