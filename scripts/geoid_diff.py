"""Geoid height difference N(EGM96) - N(EGM2008) at each showcase place.

    python scripts/geoid_diff.py

SRTM heights are on EGM96, Copernicus GLO-30 on EGM2008, so the same ground
reads differently in the two DEMs by this much (before any other DEM
disagreement). PROJ, with its online grid access enabled. Writes
results/isro/geoid_diff.json.
"""
import json
import os

import pyproj
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACES = {
    "hitec_2x2": (78.3778, 17.4489), "mumbai_lower_parel": (72.8256, 19.0011),
    "gangtok": (88.613, 27.3315), "shimla": (77.1721, 31.1059), "periyar_forest": (77.1584, 9.5638),
    "kanha": (80.6407, 22.2802), "jaisalmer_village": (70.7094, 26.6106),
}


def main():
    pyproj.network.set_network_enabled(True)
    # ellipsoidal height 0 -> orthometric height on each geoid = -N
    t96 = Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True)
    t08 = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
    out_p = os.path.join(ROOT, "results", "isro", "geoid_diff.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for name, (lon, lat) in PLACES.items():
        n96 = -t96.transform(lon, lat, 0.0)[2]
        n08 = -t08.transform(lon, lat, 0.0)[2]
        out[name] = {"lon": lon, "lat": lat, "N_egm96": round(n96, 2), "N_egm2008": round(n08, 2),
                     "egm96_minus_egm2008_m": round(n96 - n08, 2)}
        print(f"{name:20s} EGM96 {n96:8.2f}  EGM2008 {n08:8.2f}  diff {n96 - n08:+.2f} m")
    json.dump(out, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()
