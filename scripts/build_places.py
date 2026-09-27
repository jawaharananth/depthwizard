"""Build every place in viewer/places.json that is not built yet.

    python scripts/build_places.py

New places are built from their source image (build_city_image.build with
staging), saved as scenes/<id>, and -- for the Swiss places -- given airborne
LiDAR with tools/attach_lidar.py. The two showcase scenes that predate the
cursor readout get it with tools/attach_hover.py.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SW = os.path.join(ROOT, "data", "swiss_lidar")

SOURCES = {
    "zurich": ("final_out/swiss_inputs/zurich_0.6m.tif", "2683-1247"),
    "zermatt": ("final_out/swiss_inputs/zermatt_0.6m.tif", "2624-1096"),
    "sihlwald": ("final_out/swiss_inputs/sihlwald_0.6m.tif", "2684-1234"),
    "kerzers": ("final_out/swiss_inputs/kerzers_0.6m.tif", "2581-1202"),
    "shimla": ("final_out/isro_eval/shimla_ridge.tif", None),
    "sangareddy_farmland": ("final_out/isro_eval/telangana_farmland.tif", None),
}
ATTACH = {"hitec_2x2": "hitec2x2.tif", "periyar_forest": "periyar.tif"}


def run(args):
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT).returncode == 0


def main():
    places = json.load(open(os.path.join(ROOT, "viewer", "places.json")))
    for p in places:
        pid = p["id"]
        sdir = os.path.join(ROOT, "scenes", pid)
        have = os.path.exists(os.path.join(sdir, "hover_agl.u16"))
        if pid in ATTACH:
            if not have:
                run([PY, "tools/attach_hover.py", sdir, ATTACH[pid]])
            continue
        if pid not in SOURCES:
            continue
        src, tile = SOURCES[pid]
        if not have:
            if not run([PY, "build_city_image.py", src, "--name", pid]):
                print(f"build failed: {pid}", flush=True)
                continue
            run([PY, "-c", f"import scenes; scenes.save({pid!r}, {p['name'] + ', ' + p['region']!r})"])
        if tile and not os.path.exists(os.path.join(sdir, "lidar_agl.u16")):
            dsm = glob.glob(os.path.join(SW, f"swisssurface3d-raster_*_{tile}_0.5_2056_5728.tif"))
            dtm = glob.glob(os.path.join(SW, f"swissalti3d_*_{tile}_0.5_2056_5728.tif"))
            if dsm and dtm:
                yr = os.path.basename(dsm[0]).split("_")[1]
                run([PY, "tools/attach_lidar.py", sdir, dsm[0], dtm[0],
                     "--source", f"swisstopo swissSURFACE3D {yr} minus swissALTI3D"])


if __name__ == "__main__":
    main()
