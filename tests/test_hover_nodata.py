"""No-data in the cursor rasters must survive the building burn-in.

burn_buildings once read hover_agl.u16 back as plain numbers, so a masked
image border (no-data, 65535) came back as 655.35 m and was re-written as a
real height -- the viewer then read "655.3 m +/- 655.3 m" over black collars.
"""
import json

import numpy as np

import hover_export as hx


def test_burn_keeps_nodata(tmp_path):
    h, w = 40, 60
    agl = np.full((h, w), 5.0, np.float32)
    agl[:, :10] = np.nan                         # a masked border strip
    sig = np.full((h, w), 1.5, np.float32)
    sig[:, :10] = np.nan
    hover = hx.write(str(tmp_path), agl, sig)
    json.dump({"hover": hover, "facts": {"pixel_m": 1.0}}, open(tmp_path / "scene.json", "w"))
    # one building over image, one reaching into the border
    feats = [{"type": "Feature", "properties": {"height_m": 20.0, "interval_half_width_m": 3.29},
              "geometry": {"type": "Polygon", "coordinates": [[[20, -10], [30, -10], [30, -20], [20, -20], [20, -10]]]}},
             {"type": "Feature", "properties": {"height_m": 12.0, "interval_half_width_m": 3.29},
              "geometry": {"type": "Polygon", "coordinates": [[[2, -2], [15, -2], [15, -8], [2, -8], [2, -2]]]}}]
    gj = tmp_path / "b.geojson"
    json.dump({"type": "FeatureCollection", "features": feats}, open(gj, "w"))
    assert hx.burn_buildings(str(tmp_path), str(gj), (1, 0, 0, 0, -1, 0), (w, h)) == 2

    a = hx._decode(str(tmp_path / "hover_agl.u16"), h, w)
    s = hx._decode(str(tmp_path / "hover_sigma.u16"), h, w)
    assert np.isnan(a[:, :10]).all() and np.isnan(s[:, :10]).all()      # border stays no-data
    assert np.isclose(a[15, 25], 20.0) and np.isclose(s[15, 25], 2.0, atol=0.01)
    assert np.isclose(a[5, 12], 12.0)                                  # building on image is burned
    raw = np.fromfile(tmp_path / "hover_agl.u16", dtype="<u2")
    assert not (raw == hx.NODATA - 1).any()
    facts = json.load(open(tmp_path / "scene.json"))["facts"]
    assert facts["height_max_m"] == 20.0
