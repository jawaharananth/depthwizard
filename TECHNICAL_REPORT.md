# DepthWizard — Technical Report

**SIH 2026 · Problem Statement SIH26175 · ISRO / Space Applications Centre**
Single-view height estimation and 3D flythrough from optical remote-sensing imagery.

---

## 1. Summary

DepthWizard turns **one optical RGB image** into (a) an elevation product in a standard
geospatial format and (b) a navigable, measurable 3D model.

| Input | Output |
|---|---|
| PNG / JPG (no coordinates) | **Relative DSM (rDSM)** GeoTIFF, `PRODUCT=rDSM`, `VERTICAL_DATUM=relative` |
| GeoTIFF (CRS + geotransform) | **Absolute DSM** GeoTIFF in metres, same grid as the input, `VERTICAL_DATUM` naming the datum |
| Either | 3D scene (terrain mesh with the original image draped, building solids, water, trees), per-pixel height uncertainty (`_sigma.tif`), per-building GeoJSON/CSV |

**Accuracy, measured blind** (no scaling or fitting to the reference anywhere), IEEE GRSS
DFC2019, airborne LiDAR truth, 10 held-out tiles, none dropped:

| | RMSE | MAE | Correlation |
|---|---|---|---|
| **DepthWizard** | **2.82 m** | **1.59 m** | pooled 0.787 (median per tile 0.73) |
| Predicting 0 m everywhere (floor) | 5.14 m | 2.36 m | — |
| Per building (18 held-out tiles, 461 buildings) | 2.19 m | 1.58 m | bias −1.19 m |

---

## 2. Problem-statement mapping

| Requirement (SIH26175) | Implementation | Evidence |
|---|---|---|
| Non-georeferenced PNG/JPG → rDSM | `build_city_image.build()`, rDSM GeoTIFF with no CRS | upload tests (`scripts/smoke_uploads.py`) |
| Georeferenced GeoTIFF → absolute DSM, metric | DSM = ground + model height above ground; datum from Copernicus GLO-30 (EGM2008) | bounds identical to input; `VERTICAL_DATUM` tag |
| Pre-trained monocular depth backbone | Depth-Anything-V2-Small (Apache-2.0), fine-tuned to metres | `train_height.py`, `training/` |
| Scale calibration: DEM | Copernicus GLO-30 datum over bare ground | `dem_source.py` |
| Scale calibration: minimal GCPs | offset / above-ground affine, leave-one-out selection, absolute elevations or known heights | `calibration/gcp_dsm.py`, `tests/test_gcp_dsm.py`, `POST /api/gcp` |
| Scale calibration: scene statistics / semantic priors | shadow + solar ephemeris, sensor lean, building-height and canopy priors | `shadow_correction.py`, `scripts/measure_image_lean.py` |
| Project the original image onto a 3D mesh | original input draped by default; sharper imagery opt-in and labelled | `viewer/index.html` |
| Rendering engine | Three.js r180, WebGPU with WebGL2 fallback | `viewer/` |
| First-person navigation | on-foot walk (ground-following, walls block), flight mode, orbit, flythrough | `&walktest=1` self-test |
| Height and slope analysis | measure height/area, cross-section, line of sight, slope map, per-building readout | viewer tools |
| Upload imagery | dashboard, `POST /api/upload` | `server.py` |
| Validate against reference data | upload any reference DSM/LiDAR → RMSE, MAE, bias, correlation, error map | `POST /api/reference` |
| DSM in a standard geospatial format | GeoTIFF (CRS, affine, vertical datum, tier) | `dsm_export.py` |
| Stability across urban / sparse / hilly / forested | per-class blind metrics (§6.3) | `BENCHMARK_BLIND.md` |
| Standalone deployment | self-contained backend on an embedded Python runtime (upload-tested); Dockerfile for a one-command deployment (image not yet built); hosted web deployment planned | `packaging/`, `Dockerfile` |

---

## 3. Architecture

```
image (PNG/JPG/GeoTIFF)
  │ ingest: 8/16-bit, 1-3 bands, lat/lon -> local UTM, GSD from the file
  ▼
monocular height model (Depth-Anything-V2-Small, fine-tuned to metres AGL)
  │ tiled 768-px inference, blended seams; 4-view spread -> calibrated sigma
  ▼
scale & datum
  │ metric model -> metres directly; DEM datum (GLO-30); GCPs; shadows/sun; priors
  ▼
products                                   3D scene
  DSM / rDSM GeoTIFF + _sigma.tif           terrain mesh + original image drape
  per-building GeoJSON / CSV                building solids (roof = photographed pixels,
                                            walls = pixels the oblique sensor saw)
                                            water, trees, rooftop structures, flyovers
  ▼
interactive platform (browser): upload · view · walk · measure · validate · download
```

**Why the elevation product and the 3D mesh are built separately.** The model predicts height
above ground directly. The 3D mesh needs flat roofs and a clean ground surface, so it is
post-processed (refinement, roof squaring, ground estimation). Measured blind on the same
10 tiles, deriving the elevation product from the post-processed surface cost accuracy
(RMSE 4.36 m, correlation 0.49) against the model's own field (2.82 m, 0.787), so the
product is `ground + model height`, and post-processing shapes only the mesh.

---

## 4. Elevation estimation

**Backbone.** Depth-Anything-V2-Small (ViT + DPT decoder), Apache-2.0.

**Domain adaptation.** Monocular models trained on ground-level photographs predict relative
depth and suffer a domain gap looking straight down. The model is fine-tuned to regress
**height above ground in metres**:

| Run | Data | Result |
|---|---|---|
| 01 | GAMUS (aerial, 0.33 m): 4,972 train / 859 val tiles, Washington DC, Philadelphia, New York | GAMUS val MAE 1.40 m, corr 0.915 (baseline with optimal affine: 2.65 m, 0.766) |
| 02 | + DFC2019 Track 3 satellite tile-views from tiles **outside** the benchmark set, several acquisition dates each; initialised from run 01 | best epoch chosen on satellite validation: MAE 2.59 m, corr 0.706 |

Loss: structure-weighted L1 in metres plus gradient matching (sharp roof edges). Training on
Kaggle T4 GPUs (`training/kaggle_kernel.py`, resumable). Held-out benchmark tiles
(`dfc2019_height.benchmark_tiles()`) are excluded from training and validation.

**Inference.** 768-px tiles with overlap, each crop's prediction aligned to a whole-image pass
before blending (no seams). Coarse input is resampled to the model's scale (auto-zoom).

**Per-pixel uncertainty.** Spread of the prediction over four rotated views, mapped to metres
by `sigma = sqrt((3.0 * spread)^2 + 1.5^2)`, fitted on 14 DFC validation views and checked on
14 held-out ones: expected calibration error 0.085, 64% of errors within ±1σ (68% expected),
rank correlation between σ and the actual error +0.72. Written as `_sigma.tif`.

---

## 5. Scale calibration and absolute elevation

| Tier | Source | Applies when |
|---|---|---|
| A — absolute | Copernicus GLO-30 DEM sets the vertical datum (EGM2008 geoid) from bare-ground pixels; GCPs refine it | GeoTIFF with a CRS and a reachable DEM |
| B — metric | the fine-tuned model (metres directly); or shadow length × tan(sun elevation) with the sun from the capture time; or calibration to reference building heights | metric model or physical evidence |
| C — relative | shape only, vertical scale declared as assumed | plain image, no reference |

**Ground control points** (`calibration/gcp_dsm.py`). A GCP carries an absolute elevation, or a
known height above ground. Two corrections are fitted and chosen by leave-one-out error:
offset (`DSM + c`) and above-ground affine (`DSM + (s−1)·max(DSM−ground, 0) + c`), the latter
only with ≥ 5 points spanning ≥ 5 m of height. Tests on synthetic scenes with ~600 m
elevations recover a 1.35× height error and +3 m datum shift; on the HITEC DSM a synthetic
+3.00 m shift is recovered as +3.06 m (leave-one-out RMSE 0.28 m vs 3.10 m uncorrected).

**Physics from the image.** The solar position at the capture time (ephemeris, checked against
the DFC IMD metadata: 108.5° vs 108.0°) drives shadow analysis; the sensor lean (relief
displacement) is measured from the image's own tall buildings (HITEC: 0.25 m/m toward 324°
from 233 buildings) and places roof and wall textures where the image photographed them.

---

## 6. Evaluation

### 6.1 Protocol
IEEE GRSS DFC2019 Track 3 (Jacksonville, WorldView-3), airborne LiDAR nDSM as truth. Each test
tile's most-nadir view is orthorectified through its own RPC model with a terrain prior from a
global DEM (no read of the tile's truth), passed through the production pipeline, and scored
only afterwards. The predicted array is hashed before scoring to prove it was not touched.
Every tile is scored; none is dropped.

### 6.2 Headline (10 held-out tiles)

| | RMSE | MAE | Pooled corr |
|---|---|---|---|
| DepthWizard | **2.82 m** | **1.59 m** | **0.787** |
| Zero floor | 5.14 m | 2.36 m | — |

Per tile, RMSE 1.55–4.09 m, every tile below its zero floor. Per building (18 held-out tiles,
461 LiDAR building instances, median heights): RMSE 2.19 m, MAE 1.58 m, bias −1.19 m.

### 6.3 By landscape (blind, whole tile)

| Class | RMSE | MAE |
|---|---|---|
| Urban | 3.83 m | 2.57 m |
| Forest | 4.41 m | 3.43 m |
| "Hilly" (relief-based class; Jacksonville has no real hills) | 1.53 m | 0.70 m |
| Sparse | 1.86 m | 0.89 m |

### 6.4 Robustness to input resolution (raw model, held-out tiles)

| Input GSD | RMSE | MAE | Pooled corr |
|---|---|---|---|
| 0.25 m | 2.85 m | 1.59 m | 0.747 |
| 0.50 m | 2.83 m | 1.58 m | 0.751 |
| 0.75 m | 2.85 m | 1.60 m | 0.748 |
| 1.00 m | 2.97 m | 1.69 m | 0.725 |

Accuracy holds for Cartosat-class (0.6–1 m) imagery.

### 6.5 Projection accuracy
Roof outlines as the viewer draws them, against the image's edges (HITEC City, 5,822 buildings):
median 0.6 m (one pixel), p75 0.84 m, p90 1.25 m with the relief-displacement correction
(p75 1.02 m, p90 1.49 m without). The ground drape is exact by construction.

---

## 7. Visualization platform

Three.js r180, WebGPU with automatic WebGL2 fallback.

- **Rendering:** physically based materials; sky-based lighting with the sun at the image's
  capture time; ambient occlusion applied to indirect light only; temporal anti-aliasing.
- **Model content:** the original image draped on the terrain; building solids with
  photographed roofs and walls; OSM flyovers and metro decks with shadow-measured heights;
  rooftop structures measured from their shadows; water surfaces; canopy.
- **Navigation:** orbit; on-foot walk (eye 1.7 m, 1.2 m steps, walls block); flight; flythrough.
- **Analysis:** height/area measurement, cross-section, line of sight, slope map, height and
  uncertainty views, flood simulation, per-building readout.
- **Validation:** upload a reference DSM → RMSE / MAE / bias / correlation and an error map.
- **Calibration:** ground control points written back to the DSM file.
- **Usability:** getting-started guide, tooltips on every tool, frame-rate readout (F).

---

## 8. Showcase scenes (India)

| Scene | Content |
|---|---|
| **HITEC City, Hyderabad** (2.45 × 2.45 km, urban) | 6,781 buildings; heights from Open Buildings fused with the model and calibrated to 3,693 real building heights; 32 elevated structures (metro, flyovers); 3,033 rooftop structures; absolute DSM 577–699 m |
| **Periyar Tiger Reserve, Kerala** (2.45 × 2.45 km, forest + lake) | continuous canopy from the Meta/WRI canopy-height map; terrain from GLO-30 with canopy-penetration correction; lake water surface; 855–1,056 m |

No public airborne LiDAR exists for these sites, so they demonstrate the product rather than
serve as accuracy tests.

---

## 9. Limitations

- A single image cannot give metres without a reference: plain PNG/JPG gives relative heights.
- Tall buildings are still under-called (bias −1.19 m per building; the tall-building tile
  JAX_068 is the weakest at 4.09 m RMSE).
- The accuracy benchmark is one US city (DFC2019 Jacksonville). Training and test tiles are
  different areas of it, not different cities.
- Copernicus GLO-30 is a 30 m surface model; it sets the datum, not fine terrain.
- Detail is bounded by input resolution.

## 10. Data and licences

| Data | Use | Licence |
|---|---|---|
| Depth-Anything-V2-Small | backbone | Apache-2.0 |
| GAMUS | fine-tuning | research dataset |
| IEEE GRSS DFC2019 | fine-tuning (non-benchmark tiles) and evaluation | IEEE DataPort, open access |
| Copernicus GLO-30 DEM | vertical datum | Copernicus licence |
| Google Open Buildings v3 / 2.5D | footprints, heights (showcase) | CC BY 4.0 / ODbL |
| Microsoft Global ML Building Footprints | footprints (showcase) | ODbL |
| OpenStreetMap | flyovers, water | ODbL |
| Meta/WRI Global Canopy Height | canopy (showcase) | CC BY 4.0 |
| Esri World Imagery | showcase imagery | Esri terms |

## 11. Reproducing

```bash
python scripts/benchmark_blind.py 10        # headline blind accuracy
python per_building_metric.py 18            # per-building accuracy
python scripts/resolution_test.py           # resolution robustness
python scripts/projection_accuracy.py scenes/hitec_2x2 archive/hitec_city_2x2/source/hitec2x2.tif
python -m pytest tests/test_gcp_dsm.py      # GCP calibration
python server.py                            # platform at http://127.0.0.1:8000/
```
