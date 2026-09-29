# DepthWizard

Single satellite image → navigable 3D city model.

Built for SIH26175 (ISRO): reconstruct elevation and a 3D terrain model from a
single-view satellite or aerial image, with a visualisation layer.

![pixel](https://img.shields.io/badge/held--out%20pixel%20RMSE%20at%200.6%20m-2.81%20m%20(corr%200.76)-blue)
![per-building](https://img.shields.io/badge/held--out%20per--building%20RMSE%20at%200.6%20m-1.87%20m%20(18%20tiles%2C%20458%20bldgs)-blue)
![isro](https://img.shields.io/badge/ISRO%20protocol%20vs%20chosen%20DEM-0.66%E2%80%931.07%20m-green)
![interval](https://img.shields.io/badge/per--building%2090%25%20interval-%C2%B13.02%20m%20(90.2%25%20held--out%20coverage)-informational)

Held-out DFC2019 LiDAR, images degraded to ISRO's 0.6 m, production model
(run 03), `scripts/compare_ckpt.py` -> `final_out/tta_eval.json`. Older
figures (2.82 m whole-tile, 2.19 m per-building) are run 02 on other protocols.
Two different, non-comparable numbers — see `EVALUATION_PROTOCOL.md` before
citing either one; that file exists because this project once conflated them.

![HITEC City, Hyderabad, built from one image](docs/deck/s5v5_hitec_2x2.png)

**ISRO's GeoTIFF protocol** (exported DSM vs a public 30 m DEM, block-averaged
to 30 m; `scripts/isro_protocol_eval.py`), four uploads in three CRSs
(Web Mercator, UTM 44N, WGS84 lat/lon): RMSE **0.78–1.07 m**, bias within
±0.02 m against Copernicus GLO-30; **3.3–4.7 m** against SRTM. The DSM's
ground is anchored to GLO-30, so the first figure is partly by construction;
the SRTM figure is the independent one.

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
| Whole-tile, blind (`scripts/benchmark_blind.py`) | every pixel | **2.82 m** | **1.59 m** | pooled **0.787** (median per tile 0.73) | 10 held-out tiles, none dropped |
| &nbsp;&nbsp;floor: predict 0 m everywhere | same pixels | 5.14 m | 2.36 m | -- | same 10 tiles |
| Per-building (`per_building_metric.py 18`) | median height per labelled building | **2.19 m** | 1.58 m | bias -1.19 m | 18 held-out tiles, 461 buildings |
| Upper bound, NOT deployable | whole tile, scale fitted to the LiDAR per tile | 3.74 m | 2.02 m | median 0.68 | 18 tiles (older model) |

Model: Depth-Anything-V2-**Small** (Apache-2.0) fine-tuned on GAMUS (4,972
aerial tiles, DC/Philadelphia/NYC) and then on DFC2019 Track 3 satellite
tile-views from tiles **outside** the benchmark set (`training/`). Held-out
tiles are the 18 `dfc2019_height.benchmark_tiles()`, excluded from training.
History on the same blind protocol: previous Base checkpoint 4.97 m / 2.82 m / 0.51;
run 02 through the old post-processing 4.36 m / 2.36 m / 0.49; run 02 with the
model's own height-above-ground as the product (post-processing now shapes only
the 3D mesh) **2.82 m / 1.59 m / 0.787**. By terrain class: urban 3.83 m, forest
4.41 m, "hilly" 1.53 m, sparse 1.86 m RMSE. ("Hilly" there is a slope class inside
flat Jacksonville, not real hills; the real hilly case is Zermatt, below.)

**At ISRO's evaluation conditions** (FAQ, 22 Sep 2026: Cartosat-2S at **0.6 m**,
GeoTIFFs scored as an **absolute DSM**):

| Test | Result | Where |
|---|---|---|
| 0.6 m input, 18 held-out DFC tiles, per building (run 03) | 1.87 m vs 1.88 m native: **-0.6 %**, flat from 0.25 m to 1.0 m (fed without upsampling: +5.4 %). Run 02 was +2.9 %; the sensor-gap training closed it | `BENCHMARK_RESOLUTION.md` |
| Four landscapes vs airborne LiDAR at 0.6 m (swisstopo; height above ground) | urban 8.45 m, hilly 8.54 m, forest 25.15 m, sparse 3.08 m (zero-prediction floor 12.31 / 12.73 / 31.48 / 4.13 m) | `BENCHMARK_LANDSCAPES.md` |
| Absolute DSM vs the same LiDAR elevations | urban 6.9 m, hilly 9.5 m, forest 8.8 m, sparse 2.9 m (Copernicus GLO-30 alone: 8.4 / 11.1 / 9.6 / 3.6 m) | `BENCHMARK_LANDSCAPES.md` |

Measured weaknesses, stated plainly: tall forest (30-40 m trees) and dense tall
blocks are under-called (bias -23 m in Sihlwald, -4.6 m in Zurich's old town);
the training data (US towns) holds few of either. Run 04 adds Swiss LiDAR tiles
(tall buildings, tall forest, mountains) to training to address exactly that.

Projection accuracy (`scripts/projection_accuracy.py`, HITEC, 5,822 buildings): roof outlines as drawn sit a median 0.6 m (one pixel) from the image's edges, p75 0.84 m, p90 1.25 m with the measured relief-displacement correction (1.02 / 1.49 m without it); the ground drape is exact by construction.

What these say, plainly: whole-tile error is well below the zero-prediction floor on every
tile (5.14 m), building heights are within ~2 m, and the tall-building tile JAX_068 is the
weakest (4.09 m RMSE, corr 0.86) because tall buildings are still under-called. The DFC
tiles are one city; training and test tiles are different areas of it, not different cities.

A licensed model (`checkpoints/height_best/`, GAMUS-fine-tuned) predicts
**metric height directly** when present — no per-tile scale to assume, which
is the majority of blind error on the stock relative-depth backbone. Absent
that checkpoint, the pipeline falls back to Tier C (shape only, vertical scale
assumed and declared, never claimed as metres).

**Height tiers** are labelled in every scene and shown in the viewer HUD:

| Tier | Meaning | Source |
|------|---------|--------|
| **A** | DEM-anchored — real metric elevation | Terrain from Copernicus GLO-30 (or SRTM) at 30 m, minus the measured fraction k of the model's height that the radar DEM already contains, plus the model's height above ground (`anchoring.py`; k = 0.4 GLO-30, 0.0 SRTM, measured on LiDAR). On by default for georeferenced GeoTIFFs. Forest mode (`terrain_from_dem`, opt-in): terrain = GLO-30, canopy-corrected, plus a real canopy-height map |
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
