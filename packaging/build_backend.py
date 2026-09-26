"""Assemble the self-contained processing backend the desktop app ships.

    python packaging/build_backend.py            # -> dist_backend/backend/

Layout (what src-tauri/src/main.rs looks for beside the installed exe):

    backend/python/python.exe      official Python embeddable runtime, with the
                                   serving requirements installed into
                                   python/Lib/site-packages (see below)
    backend/server.py + pipeline modules, calibration/, tools/, scripts/,
    backend/viewer/, dashboard.html, checkpoints/height_best/

The runtime is the python.org embeddable package, not a copied venv: a venv
points back at the Python it was created from, so it breaks on any other PC.
Create it once with:

    curl -o py.zip https://www.python.org/ftp/python/3.13.7/python-3.13.7-embed-amd64.zip
    unzip into dist_backend/backend/python, add "Lib\\site-packages", ".." and
    "import site" to python313._pth, run get-pip.py, then
    python -m pip install -r requirements-serve.txt pip-system-certs requests
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dist_backend", "backend")

# The import closure of server.py -> build_city_image.py (same list the
# Dockerfile ships), plus modules added since.
MODULES = """server.py prebuild.py progress.py build_city_image.py adaptive_mesh.py
building_discovery.py city_model.py dem_source.py depth_model.py dfc2019_loader.py
dsm_export.py dsm_refine.py dtm.py float_check.py geometry_validate.py glb_export.py
height_cache.py htc_bins.py image_grade.py mesh_generation.py mesh_repair.py
model_audit.py open_buildings_source.py osm_structures.py ov_backend.py
overlay_rejection.py region_footprints.py roof_structure.py segmentation.py
shadow_correction.py sun_from_shadows.py terrain_classify.py terrain_maps.py
vlm_audit.py dashboard.html LICENSE""".split()

DIRS = {"calibration": "calibration", "tools": "tools", "viewer": "viewer",
        os.path.join("checkpoints", "height_best"): os.path.join("checkpoints", "height_best")}
SCRIPTS = ["measure_image_lean.py"]            # imported lazily by build()
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "output", "*.log")


def main():
    if not os.path.isfile(os.path.join(OUT, "python", "python.exe")):
        sys.exit(f"embedded runtime missing at {OUT}\\python -- see this file's docstring")
    missing = [m for m in MODULES if not os.path.isfile(os.path.join(ROOT, m))]
    if missing:
        sys.exit(f"missing source files: {missing}")
    for m in MODULES:
        shutil.copy2(os.path.join(ROOT, m), os.path.join(OUT, m))
    for src, dst in DIRS.items():
        d = os.path.join(OUT, dst)
        if os.path.isdir(d):
            shutil.rmtree(d)
        shutil.copytree(os.path.join(ROOT, src), d, ignore=SKIP)
    os.makedirs(os.path.join(OUT, "scripts"), exist_ok=True)
    for s in SCRIPTS:
        shutil.copy2(os.path.join(ROOT, "scripts", s), os.path.join(OUT, "scripts", s))
    # Nothing a local test run produced may ship.
    for junk in ("uploads", "final_out", "cache", "__pycache__", os.path.join("viewer", "output")):
        shutil.rmtree(os.path.join(OUT, junk), ignore_errors=True)
    os.makedirs(os.path.join(OUT, "viewer", "output"), exist_ok=True)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(OUT) for f in fs) / 1e9
    print(f"backend assembled at {OUT}  ({size:.2f} GB)")


if __name__ == "__main__":
    main()
