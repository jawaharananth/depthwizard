# DepthWizard -- serving image (Cloud Run / any container host).
# Single stage. Not built on the dev machine (no Docker installed there);
# the COPY list below was checked statically against the tree.
FROM python:3.13-slim

# opencv-python-headless still needs glib at runtime. rasterio and pyproj
# PyPI wheels bundle their own GDAL/PROJ, so no libgdal package is needed
# (the old `libgdal32` does not exist on the trixie-based 3.13 image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Import closure of server.py, found by walking every import statement
# (top-level and lazy, inside functions) recursively from server.py:
# server -> prebuild, progress, build_city_image -> everything below.
# vlm_audit (needs `anthropic`, opt-in) and ov_backend (needs openvino,
# optional) are copied but their deps are not installed; both degrade
# gracefully when the dependency is missing.
COPY server.py prebuild.py progress.py build_city_image.py \
     adaptive_mesh.py building_discovery.py city_model.py dem_source.py \
     depth_model.py dfc2019_loader.py dsm_export.py dsm_refine.py dtm.py \
     float_check.py geometry_validate.py glb_export.py height_cache.py \
     htc_bins.py image_grade.py mesh_generation.py mesh_repair.py \
     model_audit.py open_buildings_source.py osm_structures.py \
     ov_backend.py overlay_rejection.py region_footprints.py \
     roof_structure.py segmentation.py shadow_correction.py \
     sun_from_shadows.py terrain_classify.py terrain_maps.py vlm_audit.py \
     dashboard.html ./

# calibration/ is a package (conformal.py reads conformal_quantile.json).
COPY calibration/ ./calibration/
# tools/build_context_city.py and tools/sr_texture.py are lazily imported
# by build_city_image.py via sys.path.
COPY tools/ ./tools/
COPY viewer/ ./viewer/
COPY checkpoints/height_best/ ./checkpoints/height_best/
# The two demo scenes served by /scenes/ (~520 MB together). Drop these
# two lines for a smaller image; the dashboard then lists no scenes.
COPY scenes/hitec_2x2/ ./scenes/hitec_2x2/
COPY scenes/periyar_forest/ ./scenes/periyar_forest/

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Cloud Run injects $PORT; must bind 0.0.0.0 (not the 127.0.0.1 local-dev
# default) or the health check never reaches the process.
CMD python server.py --host 0.0.0.0 --port ${PORT:-8080}
