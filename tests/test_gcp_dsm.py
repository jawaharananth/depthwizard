"""GCP recalibration with ABSOLUTE elevations (~600 m), on synthetic scenes
where the true correction is known."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import gcp_dsm as G  # noqa: E402

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

GSD = 1.0


def _scene(tilt=0.0, seed=0):
    """Ground ~600 m (optionally sloped), 12 flat-roofed blocks 4-60 m tall."""
    rng = np.random.default_rng(seed)
    H = W = 400
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = 600.0 + tilt * xx
    nd = np.zeros((H, W), np.float32)
    roofs = []
    for i in range(12):
        r, c = 30 + (i // 4) * 120, 30 + (i % 4) * 95
        h = [4, 8, 12, 18, 25, 32, 40, 48, 55, 60, 6, 15][i]
        nd[r:r + 40, c:c + 40] = h
        roofs.append((c + 20, r + 20, h))
    return ground, nd, roofs, rng


def _write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:32644",
                       transform=from_origin(500000, 2000000, GSD, GSD)) as d:
        d.write(arr.astype(np.float32), 1)
        d.update_tags(PRODUCT="DSM", VERTICAL_DATUM="test")


def _gcps(truth, pts, rng, noise=0.2):
    return [{"col": c + 0.5, "row": r + 0.5,
             "z": float(truth[r, c] + rng.normal(0, noise))} for c, r in pts]


def test_affine_recovers_height_scale_and_datum(tmp_path):
    ground, nd, roofs, rng = _scene()
    s_true, c_true = 1.35, 3.0
    ours = ground + nd                                # model under-calls heights
    truth = ground + s_true * nd + c_true
    p = str(tmp_path / "x_DSM.tif"); _write(p, ours)
    _write(str(tmp_path / "x_ground.tif"), ground)
    pts = [(c, r) for c, r, _ in roofs[:10]] + [(5, 5), (390, 390), (200, 395)]
    rep = G.calibrate_file(p, _gcps(truth, pts, rng), mode="auto")
    assert rep["method"] == "affine", rep["warnings"]
    assert abs(rep["scale"] - s_true) < 0.03
    assert abs(rep["offset_m"] - c_true) < 0.5
    assert rep["loo_rmse"]["affine"] < rep["loo_rmse"]["offset"] < rep["loo_rmse"]["none"]
    with rasterio.open(rep["output_file"] and str(tmp_path / rep["output_file"])) as d:
        out = d.read(1)
        assert "user GCPs" in d.tags()["VERTICAL_DATUM"]
    err = np.abs(out - truth)
    assert np.median(err) < 0.5 and np.percentile(err, 99) < 3.0


def test_pure_datum_shift_picks_offset(tmp_path):
    ground, nd, roofs, rng = _scene(seed=1)
    ours = ground + nd
    truth = ours + 4.2                               # heights right, datum 4.2 m low
    p = str(tmp_path / "y_DSM.tif"); _write(p, ours)
    pts = [(c, r) for c, r, _ in roofs[:6]] + [(5, 5), (390, 390)]
    rep = G.calibrate_file(p, _gcps(truth, pts, rng), mode="auto")
    assert rep["method"] in ("offset", "affine")
    assert abs(rep["offset_m"] - 4.2) < 0.4
    assert abs(rep["scale"] - 1.0) < 0.05


def test_too_few_points_falls_back_to_offset(tmp_path):
    ground, nd, roofs, rng = _scene(seed=2)
    ours = ground + nd
    truth = ground + 1.3 * nd + 2.0
    p = str(tmp_path / "z_DSM.tif"); _write(p, ours)
    pts = [(c, r) for c, r, _ in roofs[:3]]
    rep = G.calibrate_file(p, _gcps(truth, pts, rng), mode="affine")
    assert rep["method"] == "offset"
    assert rep["affine_refused"]
    assert any("fell back to offset" in w for w in rep["warnings"])


def test_sloped_ground_flags_tilt(tmp_path):
    ground, nd, roofs, rng = _scene(tilt=0.05, seed=3)   # 20 m of relief across the tile
    ours = 600.0 + nd                                     # DEM anchor missed the slope
    truth = ground + nd
    p = str(tmp_path / "w_DSM.tif"); _write(p, ours)
    pts = [(10, 10), (390, 10), (10, 390), (390, 390), (200, 200), (100, 300)]
    rep = G.calibrate_file(p, _gcps(truth, pts, rng), mode="offset")
    assert any("do not agree on one datum shift" in w for w in rep["warnings"])
    assert any("trend with position" in w for w in rep["warnings"])


def test_geographic_points_are_reprojected(tmp_path):
    from pyproj import Transformer
    ground, nd, roofs, rng = _scene(seed=4)
    ours = ground + nd
    truth = ours + 2.5
    p = str(tmp_path / "v_DSM.tif"); _write(p, ours)
    t = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    pts = []
    for c, r in [(5, 5), (390, 390), (200, 395), (395, 200)]:
        lon, lat = t.transform(500000 + c + 0.5, 2000000 - r - 0.5)
        pts.append({"x": lon, "y": lat, "crs": "EPSG:4326", "z": float(truth[r, c])})
    rep = G.calibrate_file(p, pts, mode="offset")
    assert rep["n"] == 4 and not rep["dropped"]
    assert abs(rep["offset_m"] - 2.5) < 0.1


def test_building_heights_as_h_agl(tmp_path):
    """The viewer sends known building heights, not elevations."""
    ground, nd, roofs, rng = _scene(seed=5)
    ours = ground + nd
    p = str(tmp_path / "u_DSM.tif"); _write(p, ours)
    _write(str(tmp_path / "u_ground.tif"), ground)
    pts = [{"col": c + 0.5, "row": r + 0.5, "h_agl": 1.3 * h} for c, r, h in roofs[:9]]
    pts += [{"col": 5.5, "row": 5.5, "h_agl": 0.0}, {"col": 390.5, "row": 390.5, "h_agl": 0.0}]
    rep = G.calibrate_file(p, pts, mode="auto")
    assert rep["method"] == "affine", rep["warnings"]
    assert abs(rep["scale"] - 1.3) < 0.02 and abs(rep["offset_m"]) < 0.3
