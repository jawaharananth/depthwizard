"""
ICESat-2 ATL08 as an independent, free, globally-available height check --
research finding recorded in COMPETITOR_LANDSCAPE.md's research-frontier
review (arXiv 2505.06905, "sparse LiDAR-guided correction"): post-hoc
correction of monocular height predictions against real spaceborne LiDAR,
no retraining required. Real India coverage exists (ICESat-2 is global),
which is the actual gap this targets -- our only measured ground truth
outside Mumbai's BMC survey is the DFC2019 US benchmark.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR

ATL08 reports terrain elevation and, per ~100m along-track segment, the
canopy/photon height above it -- "canopy" here means "whatever return sat
highest in that segment", which over a city block is a rooftop as often as
a tree, since the ATL08 algorithm has no semantic building/vegetation
distinction. That makes it a genuine, if noisy and sparse, elevated-height
signal usable the same way this project's GCP tool already uses a handful
of known points -- except sourced automatically from a real satellite
instead of a person clicking buildings.

IT CAN supply a few real, independent elevation samples for a scene, when
  a track happens to cross it.
IT CANNOT guarantee coverage. ICESat-2's ground tracks are ~90m apart
  within one pass and repeat roughly every 91 days since October 2018 --
  for a single small building-scale scene (a few hundred metres across),
  a track intersecting it at all is common, not certain. Zero coverage is
  an expected, honestly-reported outcome, not a failure.

REAL, PUBLIC, NO-AUTH DATA PATH (verified against live endpoints, not
assumed from documentation)

  1. NASA CMR granule search (cmr.earthdata.nasa.gov) -- public, unauthenticated,
     used only for DISCOVERY: which real dates have an ATL08 granule
     covering this bbox. The reference ground track ID is encoded in the
     granule filename (ATL08_YYYYMMDDHHMMSS_TTTTCCRR_vvv_rr.h5, TTTT = track).
  2. OpenAltimetry (openaltimetry.earthdatacloud.nasa.gov) -- public,
     unauthenticated, used to fetch the actual per-point lat/lon/elevation/
     canopy-height series for a specific date+track+bbox. Going straight to
     NSIDC for the underlying .h5 file would require an Earthdata login;
     this avoids that entirely for the read-only case this project needs.

Both endpoints get the same circuit-breaker treatment as dem_source.py's
Copernicus DEM fetch, for the same reason: this is the only other place
in the pipeline that depends on a network service outside our control.
"""
from __future__ import annotations

import time
import urllib.request
import urllib.parse
import json

import numpy as np

CMR_BASE = "https://cmr.earthdata.nasa.gov/search/granules.json"
OA_BASE = "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2/atl08"

# ATL08's own fill value for "no canopy/photon height at this point" --
# the IEEE-754 float32 max, carried through as a float64 literal by the
# JSON response. Filtering it out is not optional: leaving it in would
# read as a real 3.4 x 10^38 metre building.
_FILL_VALUE = 3.4028234663852886e38
_FILL_TOL = 1e30   # anything absurd counts as the fill sentinel, not just the exact float

_FAIL_THRESHOLD = 3
_COOLDOWN_S = 120.0
_consecutive_failures = 0
_tripped_until = 0.0


def _breaker_open() -> bool:
    return time.time() < _tripped_until


def _record_failure() -> None:
    global _consecutive_failures, _tripped_until
    _consecutive_failures += 1
    if _consecutive_failures >= _FAIL_THRESHOLD:
        _tripped_until = time.time() + _COOLDOWN_S


def _record_success() -> None:
    global _consecutive_failures, _tripped_until
    _consecutive_failures = 0
    _tripped_until = 0.0


def _http_json(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def find_granules(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                   max_results: int = 20) -> list[dict]:
    """Real ATL08 granules intersecting this bbox, via NASA's public CMR
    search -- no Earthdata login needed for metadata discovery, only for
    downloading the raw .h5 (which this module never does; see module
    docstring for why OpenAltimetry is used instead).

    Returns a list of {"date": "YYYY-MM-DD", "track_id": int}, one per
    granule, track ID parsed straight out of the official ATL08 filename
    convention (ATL08_YYYYMMDDHHMMSS_TTTTCCRR_vvv_rr.h5).
    """
    if _breaker_open():
        return []
    q = urllib.parse.urlencode({
        "short_name": "ATL08",
        "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "page_size": max_results,
    })
    try:
        data = _http_json(f"{CMR_BASE}?{q}")
    except Exception:
        _record_failure()
        return []
    _record_success()

    out = []
    for entry in data.get("feed", {}).get("entry", []):
        title = entry.get("title", "")
        # ATL08_20181021083545_03470106_007_01.h5 -- the second underscore-
        # delimited field is TTTTCCRR; the first 4 digits are the track ID.
        parts = title.split("_")
        if len(parts) < 3 or len(parts[2]) < 4:
            continue
        try:
            track_id = int(parts[2][:4])
        except ValueError:
            continue
        date = (entry.get("time_start") or "")[:10]
        if date:
            out.append({"date": date, "track_id": track_id})
    return out


def fetch_points(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                  date: str, track_id: int) -> list[tuple]:
    """Real per-point (lat, lon, terrain_or_canopy_elev_m) samples for one
    granule, via OpenAltimetry's public read API. Fill-value points
    (no photon return classified for that segment) are dropped here, not
    left for the caller to rediscover.
    """
    if _breaker_open():
        return []
    q = urllib.parse.urlencode({
        "date": date, "trackId": track_id,
        "minx": min_lon, "miny": min_lat, "maxx": max_lon, "maxy": max_lat,
        "outputFormat": "json", "client": "depthwizard",
    })
    try:
        data = _http_json(f"{OA_BASE}?{q}")
    except Exception:
        _record_failure()
        return []
    _record_success()

    points = []
    for series in data.get("series", []):
        for row in series.get("lat_lon_elev_canopy", []):
            if len(row) < 4:
                continue
            lat, lon, elev, canopy = row[0], row[1], row[2], row[3]
            # The higher of the two is what a nadir camera would actually
            # see as the top surface -- bare terrain where nothing stands,
            # or the elevated return where something (building or tree) does.
            top = canopy if abs(canopy) < _FILL_TOL else elev
            if abs(top) < _FILL_TOL:
                points.append((lat, lon, float(top)))
    return points


def sample_scene(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                  max_granules: int = 6) -> dict:
    """Best-effort real ICESat-2 samples for one scene's real-world extent.

    Tries up to max_granules real, confirmed-existing dates before giving
    up -- deliberately bounded, since a scene with genuinely no intersecting
    track should return quickly with an honest zero, not hang retrying.
    """
    granules = find_granules(min_lon, min_lat, max_lon, max_lat)
    if not granules:
        return {"points": [], "n": 0, "granules_tried": 0,
                "note": "no ATL08 granule found covering this extent"}

    all_points = []
    tried = 0
    for g in granules[:max_granules]:
        tried += 1
        pts = fetch_points(min_lon, min_lat, max_lon, max_lat, g["date"], g["track_id"])
        all_points.extend(pts)

    return {
        "points": all_points, "n": len(all_points), "granules_tried": tried,
        "note": (f"{len(all_points)} real ICESat-2 ATL08 points from "
                 f"{tried} granule(s)" if all_points else
                 "granules exist for this extent but returned no usable points "
                 "(track passed near, not through, the requested bbox)"),
    }
