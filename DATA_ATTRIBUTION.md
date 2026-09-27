# Data in this repository — sources and licences

Binary data is committed here so the demo runs without a download step. Each
source has a different licence and they are not interchangeable.

## `gamus_data/`, `gamus_probe/` — GAMUS

**Licence: CC-BY-4.0.** Redistributable *with attribution*, which is what this
file provides.

> Xiong, Z., Chen, S., Wang, Y., Mou, L., Zhu, X.X.
> *GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for
> Remote Sensing Data.* arXiv:2305.14914 (2023).
> Dataset: https://huggingface.co/datasets/earthflow/GAMUS
> Code: https://github.com/EarthNets/RSI-MMSegmentation

What is here is a small subset of a 80 GB dataset — the tiles used to measure
the domain gap recorded in `BENCHMARK.md`, plus a training slice. The full set,
or any other subset, is reproducible:

```bash
python gamus.py download --split train --limit 400
python gamus.py download --split val --limit 40
```

Tiles are `*_RGB.h5` (imagery), `*_AGL.h5` (height above ground, in metres,
LiDAR-derived) and `*_CLS.h5` (six semantic classes).

## Model weights

| Weights | Licence | Where used |
|---|---|---|
| depth-anything/Depth-Anything-V2-Small-hf | Apache-2.0 | default backbone for fine-tuning (`train_height.py`) |
| depth-anything/Depth-Anything-V2-Base-hf | CC-BY-NC-4.0 (non-commercial) | base of the current `checkpoints/height_best/`; to be replaced by the Small retrain |
| Meta/WRI Global Canopy Height Map v1/v2 | CC-BY-4.0 | canopy heights (Periyar, HITEC trees) |

## `scenes/` — pre-built reconstructions

Output of this pipeline. `scenes/jax165` is derived from DFC2019 imagery (see
below); `scenes/varanasi` is derived from OpenAerialMap imagery.

## `uploads/` — test input

⚠️ **Check this before publishing.** The image here was uploaded through the
dashboard for testing and appears to be derived from DFC2019 imagery.

**DFC2019 is licensed by IEEE GRSS and is not redistributable.** This
repository's `.gitignore` excludes `dfc2019_data/` for exactly that reason, so
committing a derivative of it is inconsistent with that position. It is kept
here because it was explicitly requested; if this repository is submitted or
made public, remove `uploads/` and the `scenes/jax165` textures, or replace them
with imagery you hold redistribution rights to.

Obtain DFC2019 directly from
[IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019).

## `stage_timings.json` — ETA calibration

Measured stage durations from **one specific machine** (Intel Core 5 120U, 15 W,
no discrete GPU). They are stored per megapixel, so they scale with image size,
but not across hardware: a machine with a CUDA GPU is roughly an order of
magnitude faster on the depth stage, which is ~94% of a build.

Delete the file to recalibrate. `progress.py` shows no ETA until a stage has
been timed rather than displaying a fabricated one, so an empty calibration is
handled correctly and is safer than an inherited wrong one.

## Evaluation and training sources added September 2026

| Source | Used for | Licence / terms |
|---|---|---|
| swisstopo SWISSIMAGE (orthophoto), swissSURFACE3D Raster (LiDAR DSM), swissALTI3D (LiDAR DTM) | four-landscape benchmark (`scripts/lidar_landscapes.py`, 4 tiles) and run-04 training tiles (`training/prepare_swiss.py`, built inside the Kaggle session, benchmark tiles + 1 km buffer excluded) | Swiss open government data, free use with attribution: (c) swisstopo |
| Copernicus DEM GLO-30 | absolute elevation anchor; reference in `scripts/isro_protocol_eval.py` | (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA |
| SRTM 1 arc-second (via the AWS terrain tiles) | alternative anchor and reference | NASA / USGS, public domain |
| ICESat-2 ATL08 (via NASA OpenAltimetry) | independent spaceborne-LiDAR check over India | NASA, public |
| EGM2008 geoid grid (`us_nga_egm08_25.tif`, PROJ CDN) | ellipsoid to orthometric conversion of ICESat-2 heights | NGA, public |
| Esri World Imagery (zoom 18) | the four Indian evaluation scenes (`final_out/isro_eval/`) | Esri and its data providers; used for evaluation only, not redistributed |

## Not committed

`checkpoints/` weights (`model.safetensors`, ~99 MB each; ONNX exports) are
not in git. The production checkpoint is `checkpoints/height_best/` (run 03,
promoted 27 Sep 2026 on held-out evidence recorded in its `metric_ok.json`);
its marker, uncertainty calibration and ONNX manifest are committed.

`dfc2019_data/` — 16 GB, IEEE GRSS licensed.
