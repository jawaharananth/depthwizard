# DepthWizard

Single satellite image → navigable 3D city model.

Built for SIH26175 (ISRO): reconstruct elevation and a 3D terrain model from a
single-view satellite or aerial image, with a visualisation layer.

![whole-tile](https://img.shields.io/badge/blind%20whole--tile%20RMSE-4.97%20m%20(zero--pred%205.14%20m)-orange)
![per-building](https://img.shields.io/badge/per--building%20RMSE-3.13%20m%20(50%20tiles%2C%201%2C063%20bldgs)-blue)
![tier](https://img.shields.io/badge/heights-Tier%20A%20when%20anchorable-green)
![interval](https://img.shields.io/badge/90%25%20interval-%C2%B14.53%20m-informational)

Two different, non-comparable numbers — see `EVALUATION_PROTOCOL.md` before
citing either one; that file exists because this project once conflated them.

---

## What it does

An orthorectified nadir image goes in. Out comes a 3D scene: bare-earth terrain
carrying the satellite texture, buildings as individually extruded prisms with
flat roofs and true vertical walls, plus tree canopy, water surfaces and
vehicle-sized objects — viewable in the browser and exportable as GLB, GeoJSON
and CSV.

```
python build_city.py JAX_165 --extent 640 --px 2560   # DFC2019 tile
python build_city_image.py my_nadir.tif --name city   # any nadir image
python scenes.py show jax165                          # restore a saved scene
```

Then open `http://localhost:8800/viewer/index.html`.

**Upload dashboard:** `python server.py` serves a drag-and-drop dashboard at
`http://127.0.0.1:8000/` (defaults `--host 127.0.0.1 --port 8000`; the
container uses `--host 0.0.0.0 --port 8080`). It is a stdlib
`ThreadingHTTPServer`, runs one build at a time, and lists pre-built scenes.

**Pre-built scenes** (served at `/scenes/<id>/`):

| scene | landscape | height sources |
|---|---|---|
| `scenes/hitec_2x2` | HITEC City, Hyderabad, urban, 2.45 km | real footprints + Google Open Buildings 2.5D heights; OSM flyovers/metro viaducts (deck height from shadow where measurable, else class-typical; piers inferred) |
| `scenes/periyar_forest` | Periyar Tiger Reserve, Kerala, forest + lake | Meta/WRI canopy-height map; terrain from Copernicus GLO-30 DEM, canopy-penetration corrected (x0.6, an assumed literature fraction); absolute DSM 855–1056 m |

Neither scene is scored against LiDAR; the only benchmark is Jacksonville
(DFC2019), see `BENCHMARK.md`.

**DSM export:** every build writes a GeoTIFF. For georeferenced input that
reaches a metric tier with an external datum it is `<name>_DSM.tif`
(absolute elevation, real geotransform, `VERTICAL_DATUM` tag naming the
datum). Otherwise (e.g. PNG/JPG input) it is `<name>_rDSM.tif`, relative
heights, `VERTICAL_DATUM=relative`.

---

## Honest status

This section exists because the project has previously published numbers that
turned out to be wrong, and the correction mattered more than the number —
twice: once for comparing misregistered rasters (withdrawn), and again when a
silent normalisation bug in `depth_model.py` discarded the licensed model's
real metre output for every downstream consumer. Both are fixed now; the
numbers below were re-measured after both fixes, not carried over from before
them. Full history in `EVALUATION_PROTOCOL.md`.

**Accuracy, measured blind** (no scale fitted to the LiDAR at any point;
DFC2019 Jacksonville, airborne LiDAR nDSM as truth):

| Protocol | Scope | RMSE | MAE | Correlation | Sample |
|---|---|---|---|---|---|
| Whole-tile, blind (`scripts/benchmark_blind.py`) | every pixel | **4.97 m** | 2.82 m | median 0.51 per tile | 10 tiles, none dropped |
| &nbsp;&nbsp;floor: predict 0 m everywhere | same pixels | 5.14 m | 2.36 m | -- | same 10 tiles |
| Per-building (`per_building_metric.py`) | median height per labelled building | **3.13 m** | 1.87 m | -- | 50 tiles, 1,063 buildings |
| Upper bound, NOT deployable | whole tile, scale fitted to the LiDAR per tile | 3.74 m | 2.02 m | median 0.68 | 18 tiles |

What these say, plainly: the current checkpoint gets building heights within
~3 m on average, but on whole tiles it only just beats predicting zero,
because it compresses heights to roughly 1/1.7 of the truth (the
`fitted_scale` column in `BENCHMARK_BLIND.md`) and fails on tall buildings
(JAX_068). That compression is the target of the GAMUS retrain on
Depth-Anything-V2-Small (`training/`). The last row is kept only as a ceiling:
it needs the answer to compute the answer.

A licensed model (`checkpoints/height_best/`, GAMUS-fine-tuned) predicts
**metric height directly** when present — no per-tile scale to assume, which
is the majority of blind error on the stock relative-depth backbone. Absent
that checkpoint, the pipeline falls back to Tier C (shape only, vertical scale
assumed and declared, never claimed as metres).

**Height tiers** are labelled in every scene and shown in the viewer HUD:

| Tier | Meaning | Source |
|------|---------|--------|
| **A** | DEM-anchored — real metric elevation | Copernicus GLO-30 (or SRTM) offset fitted over bare-ground pixels; on by default in `build_city_image.py` for projected-CRS GeoTIFFs. Forest mode (`terrain_from_dem`, opt-in): terrain = GLO-30, canopy-corrected, plus a real canopy-height map |
| **B** | Metric heights | shadow calibration `h = L · tan(sun elevation)` (`build_city.py`), a metric checkpoint, or relative depth calibrated to Google Open Buildings heights by robust median log-ratio (`build_city_image.py`, needs >=20 control buildings) |
| **C** | Relative — shape only, vertical scale assumed and declared | monocular depth alone |

A Tier C scene never claims metres, and exports from an ungeoreferenced source
never carry a CRS.

**Known limits, measured:**

- Monocular depth degrades as buildings get taller. Correlation with LiDAR falls
  from 0.522 (JAX_068, 44 m) to 0.086 (JAX_167, 161 m) — street canyons defeat it.
- Buildings are **flat-roofed prisms**. Domes and spires become boxes. This is a
  deliberate abstraction, not an accident.
- Walls are textured only where the image actually photographed them: an
  oblique satellite view shows the sensor-facing side of tall buildings, found
  from the image's measured relief displacement
  (`scripts/measure_image_lean.py`). Walls the sensor never saw reuse the same
  building's photographed facade and are labelled inferred; a near-nadir image
  has no facade pixels at all, and its walls take the building's measured
  colour.
- Vehicle detection is a size/shape heuristic, not an inventory.
- Dense tree canopy is misread as buildings where canopy height matches roof
  height (measured: 40% "building" on a campus that is ~8% built).

---

## Pipeline

```
image → RPC orthorectification → tiled monocular depth → segmentation
      → shadow calibration → terrain/structure separation → roof squaring
      → multi-scale building discovery → prism extrusion → GLB → viewer
```

Key modules:

| file | role |
|------|------|
| `ortho.py` | RPC orthorectification onto the truth grid; near-nadir view selection |
| `depth_model.py` | Depth Anything V2, tiled inference with affine merge |
| `segmentation.py` | building / road / water / vegetation / bare earth |
| `building_discovery.py` | multi-scale high-recall discovery, evidence-based retention |
| `city_model.py` | footprints, ear-clipped prisms, canopy, water, vehicles |
| `shadow_correction.py` | solar geometry, shadow measurement, metric calibration |
| `dtm.py`, `dsm_refine.py` | terrain separation, guided filtering, roof squaring |
| `overlay_rejection.py` | strips map pins/labels from screenshot input |
| `verify_pipeline.py` | 8-stage verification, hard-fails on invariant violation |
| `viewer/index.html` | WebGPU renderer, GTAO → denoise → bloom → FXAA; orbit + pointer-lock walk-mode flythrough; live compare-slider (prediction vs LiDAR ground truth) on benchmark scenes |
| `region_footprints.py` | real per-building ground truth (e.g. BMC municipal floor-count records) overrides the model where available, never blended silently with an estimate |

---

## Setup

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
```

Depth Anything V2 weights download automatically from Hugging Face on first run.

**Dataset is not included** — 16 GB and licensed. Get DFC2019 Track 3 (RGB,
Truth, Metadata) from [IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)
and place it as:

```
dfc2019_data/rgb/Track3-RGB-1/
dfc2019_data/truth/Track3-Truth/
dfc2019_data/metadata/Track3-Metadata/
```

**Inference runs on CPU** — no GPU required to build a scene or serve the
viewer (`Dockerfile`/`requirements-serve.txt` deploy CPU-only). Depth
inference is the bottleneck (~6 min per 2500 px tile on CPU), and is cached,
so rebuilds take ~30 s. **Training** the metric checkpoint (`train_height.py`)
does need a GPU — done on Kaggle/Colab, not locally; see `DEPLOY.md` for
hosting the built app publicly for free.

---

## Verification

```bash
python verify_pipeline.py JAX_068     # 8-stage harness, non-zero exit on failure
```

Checks input registration, height-field orientation, segmentation against LiDAR
labels, edge refinement, terrain separation, water levelling, building geometry,
and accuracy — and refuses to pass a scene whose prediction does not correlate
with the ground it was computed from.

---

## Licence and attribution

Code: MIT. See `LICENSE`.

Data is **not** covered by that licence: DFC2019 is licensed by IEEE GRSS;
OpenAerialMap imagery carries its own per-scene licence (CC-BY / CC-BY-SA /
ODbL). Check before redistributing anything derived from either.

**Model weights.** The checkpoint in `checkpoints/height_best/` was fine-tuned
from Depth-Anything-V2-**Base**, whose weights are CC-BY-NC-4.0
(non-commercial). It is fine for research and evaluation but not for a
commercial deliverable. `train_height.py` now defaults to
Depth-Anything-V2-**Small** (Apache-2.0); the retrain on the full GAMUS set
(`training/kaggle_kernel.py`) replaces the Base checkpoint once it passes the
promotion gates.
