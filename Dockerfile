# DepthWizard -- Google Cloud Run serving image.
# Multi-stage: build stage installs deps once, runtime stage stays lean.
FROM python:3.13-slim AS base

# rasterio/opencv need these system libs at runtime, not just build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what the serving path (server.py's real import graph) touches --
# training scripts, notebooks, eval reports, dataset loaders for
# offline-only formats, and the multi-GB dfc2019_data/ directory are
# deliberately not copied into the image.
COPY server.py progress.py build_city_image.py depth_model.py htc_bins.py \
     building_discovery.py calibration.py city_model.py dem_source.py \
     dfc2019_loader.py dsm_export.py dsm_refine.py dtm.py float_check.py \
     geometry_validate.py glb_export.py height_cache.py image_grade.py \
     mesh_generation.py mesh_repair.py model_audit.py \
     overlay_rejection.py prebuild.py region_footprints.py \
     roof_structure.py segmentation.py shadow_correction.py \
     terrain_classify.py terrain_maps.py adaptive_mesh.py ./

COPY viewer/ ./viewer/
COPY checkpoints/height_best/ ./checkpoints/height_best/

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Cloud Run injects $PORT; must bind 0.0.0.0 (not the 127.0.0.1 local-dev
# default) or the health check never reaches the process.
CMD python server.py --host 0.0.0.0 --port ${PORT:-8080}
