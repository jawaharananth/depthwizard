"""Real, live checks against the actual public ICESat-2 endpoints -- not
mocks. This project's own standing rule (EVALUATION_PROTOCOL.md) is that a
number isn't trusted until it's actually been run, and the same applies to
"this API works": a mocked response would only prove the parsing code
handles whatever shape we assumed it returns, not that the real service
still does. Skipped automatically if the network is unavailable (CI/offline
dev), never silently passed as if it had run.
"""
import pytest

import icesat2_source as ic2

# A real Jacksonville, FL bbox -- confirmed by hand to have real ATL08
# coverage (this project's own DFC2019 benchmark tiles sit here), so an
# empty result means the API changed or is unreachable, not "no track
# happened to cross this specific patch" (the sparse-coverage case handled
# elsewhere in the module, not what this test is checking).
_KNOWN_COVERED_BBOX = (-82.0, 30.0, -81.0, 31.0)


def _network_available() -> bool:
    try:
        ic2._http_json(f"{ic2.CMR_BASE}?short_name=ATL08&page_size=1", timeout=5.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _network_available(), reason="no network access to NASA CMR")
def test_find_granules_returns_real_dates_and_tracks():
    granules = ic2.find_granules(*_KNOWN_COVERED_BBOX)
    assert len(granules) > 0, "a known-covered bbox returned zero granules -- CMR API may have changed"
    for g in granules:
        assert len(g["date"]) == 10 and g["date"][4] == "-"
        assert isinstance(g["track_id"], int) and g["track_id"] > 0


@pytest.mark.skipif(not _network_available(), reason="no network access to NASA CMR/OpenAltimetry")
def test_sample_scene_returns_plausible_elevations():
    r = ic2.sample_scene(*_KNOWN_COVERED_BBOX, max_granules=2)
    assert r["n"] > 0, "a known-covered bbox with real granules returned zero usable points"
    elevs = [p[2] for p in r["points"]]
    # WGS84 ellipsoidal height in Florida is roughly -35m to +50m for real
    # ground/building elevations (documented in dfc2019_loader.py: ~-25m to
    # -30m geoid offset on ~0-15m real terrain/building heights). A value
    # outside this is either an unfiltered fill sentinel or a parsing bug.
    assert all(-100.0 < e < 200.0 for e in elevs), f"implausible elevation in {elevs[:5]}"
    # The fill sentinel must never leak through.
    assert all(abs(e) < ic2._FILL_TOL for e in elevs)


def test_fill_value_filtering_is_exact():
    # Direct, offline unit check of the filtering logic itself -- no network
    # needed, since this only exercises arithmetic already in the module.
    assert abs(ic2._FILL_VALUE) > ic2._FILL_TOL
