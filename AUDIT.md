# DepthWizard — Full Project Audit

**Date:** 2026-08-27
**Repository:** https://github.com/jawaharananth/depthwizard (`cdafc92`)
**Scope:** every module, every data path, every measured number, every known defect.

Written to be checkable. Where a number appears, it was measured on this machine
and the tile it was measured on is named. Where something is unknown or
unverified, it says so rather than estimating.

---

## 1. What the system does

One nadir satellite or aerial image goes in. Out comes a navigable 3D city:
bare-earth terrain carrying the source imagery, buildings as individually
extruded prisms with flat roofs and true vertical walls, tree canopy, water
surfaces, vehicle-sized objects, and per-building attributes exported as GeoJSON
and CSV.

Built for **SIH26175 (ISRO)**: single-view satellite image → high-precision
elevation and a navigable 3D terrain model, with a visualisation layer.

---

## 2. Hardware and environment (measured)

```
python        3.13.7
torch         2.13.0+cpu
cuda          False          device count 0
nvidia-smi    absent -- no NVIDIA GPU present
onnxruntime   MISSING     tensorrt   MISSING
gsplat        MISSING     diffusers  MISSING
detectron2    MISSING
segment_anything  package present, NO checkpoint on disk
rasterio 1.5.1   GDAL 3.12.4   opencv 5.0.0.93   numpy 2.5.2
transformers 5.15.1
```

**This is a CPU-only machine.** It is the single most consequential fact in this
audit: it determines what can and cannot be implemented, and it makes depth
inference the dominant cost of every run.

---

## 3. Codebase inventory

**8,171 lines of Python across 45 modules**, plus 1,120 lines of HTML/JS.
63 files tracked in git (0.5 MB); ~18.5 GB deliberately excluded.

### 3.1 Core pipeline modules

| LOC | Module | Responsibility |
|----:|--------|----------------|
| 569 | `mesh_generation.py` | ground heightfield, legacy building extrusion, texture/normal/AO writing, GLB+OBJ dispatch |
| 556 | `city_model.py` | **prism city**: footprints, ear clipping, prisms, canopy, water, vehicles |
| 437 | `shadow_correction.py` | NOAA solar position, shadow detection, shadow-length measurement, metric calibration |
| 409 | `terrain_maps.py` | sky-view factor, curvature, water levelling, facade neutralisation, steep-surface calming |
| 397 | `build_city_image.py` | entry point for plain/GeoTIFF images (no DFC truth) |
| 396 | `buildings/detection.py` | earlier detection experiment (superseded by `building_discovery.py`) |
| 350 | `segmentation.py` | 5-class semantic segmentation, height-aware |
| 313 | `verify_pipeline.py` | 8-stage verification harness |
| 277 | `build_city.py` | entry point for DFC2019 tiles |
| 271 | `roof_structure.py` | roof-type classification with detrending |
| 259 | `run_pipeline.py` | original end-to-end runner (legacy) |
| 258 | `building_discovery.py` | multi-scale high-recall discovery with evidence retention |
| 244 | `glb_export.py` | hand-written glTF 2.0 binary writer |
| 224 | `adaptive_mesh.py` | detail-driven vertex placement, Delaunay |
| 213 | `build_scene.py` | heightfield entry point (superseded by `build_city.py`) |
| 212 | `buildings/records.py` | per-building record dataclasses |
| 205 | `rescore_baseline.py` | accuracy re-scoring on orthorectified input |
| 200 | `dsm_refine.py` | guided filter, edge sharpening, roof squaring |
| 187 | `ortho.py` | **RPC orthorectification**, near-nadir view selection |
| 173 | `dtm.py` | DSM → DTM bare-earth separation |
| 168 | `plane_sweep_mvs.py` | multi-view plane-sweep (implemented, not in the active path) |
| 164 | `overlay_rejection.py` | strips map pins/labels from screenshot input |
| 154 | `dfc2019_benchmark.py` | benchmark driver |
| 149 | `depth_model.py` | Depth Anything V2, tiled inference, orientation check |
| 148 | `rpc_model.py` | RPC00B rational polynomial camera |
| 138 | `scenes.py` | save/restore finished scenes |
| 129 | `dfc2019_loader.py` | DFC2019 DSM/CLS/TXT/IMD parsing |
| 126 | `terrain_classify.py` | urban/hilly/forest/sparse classification by real slope |
| 106 | `validation.py` | RMSE/MAE per terrain class |
| 71 | `dsm_export.py` | GeoTIFF DSM export |
| 66 | `fetch_crop.py` | windowed read of remote cloud-optimised imagery |
| 34 | `height_cache.py` | depth-field disk cache |

`calibration/` holds five tier strategies (`tiered`, `georeferenced`,
`shadow_based`, `shadow_hybrid`, `shadow_primary`, `object_scale`,
`terrain_curves`).

### 3.2 Frontend

| LOC | File | Role |
|----:|------|------|
| 743 | `viewer/index.html` | WebGPU renderer, materials, post chain, measurement, HUD |
| 206 | `webapp/index.html` | landing page |
| 171 | `dashboard/index.html` | accuracy dashboard |

---

## 4. Architecture and data flow

### 4.1 Active path (DFC2019 tile)

```
build_city.py
  │
  ├─ ortho.orthorectify ────────────── most_nadir_view (IMD off-nadir angle)
  │     └─ rasterio.warp.reproject with RPC model → north-up UTM grid
  │        output: image_np, truth arrays, gsd_m, sun angles, transform
  │
  ├─ height_cache.load ─── miss ──► depth_model.predict_tiled
  │                                   ├─ predict()  whole-image reference
  │                                   ├─ 25 crops @768px, 50% overlap
  │                                   ├─ np.polyfit affine-align each to reference
  │                                   └─ Hann-window blend
  │
  ├─ segmentation.segment(image, height)
  │     ├─ _fill_edge_regions   roof outline → filled region
  │     └─ _elevated_mask       morphological opening → residual > 3σ noise
  │
  ├─ orientation_check ─── fails ──► ABORT (buildings below ground)
  │
  ├─ dsm_refine.refine_dsm         guided filter, edge-aware
  ├─ shadow_correction.calibrate_scale
  │     ├─ detect_shadow_mask      local contrast, not Otsu
  │     ├─ measure_shadow_lengths  many rays from anti-sun edge
  │     └─ ratio of medians        → m/unit, or refuse
  │     └─ tier = B if measured, else C with declared assumption
  │
  ├─ dtm.estimate_dtm              distanceTransform nearest-terrain fill
  ├─ dsm_refine.prismify_buildings flat roof per footprint
  ├─ dtm.estimate_dtm  (again)     terrain re-derived after squaring
  ├─ city_model.flatten_ground     35 m smoothing → bare earth
  │
  ├─ building_discovery.discover   scales 1×/2×/4×, height bands, 4 evidence channels
  ├─ city_model.build_prisms       ear-clipped roofs, vertical walls, roof colour
  ├─ city_model.build_canopy       tapered crowns
  ├─ city_model.build_water        flat planes
  ├─ city_model.detect_vehicles    gated small-object boxes
  ├─ mesh_generation.build_ground_mesh
  │
  ├─ glb_export.export_glb         ground + buildings + canopy + water + vehicles
  ├─ GeoJSON + CSV per building
  └─ stage → viewer/output/ + scene.json
```

### 4.2 Plain-image path

`build_city_image.py` — same core, minus RPC/truth, plus:
`check_nadir` (refuses oblique), `overlay_rejection.clean` (strips map UI),
`estimate_sun_azimuth` (from the image's own shadows), GeoTIFF CRS/GSD reading.

### 4.3 Internal dependency graph

```
build_city ──┬─► ortho ──► dfc2019_loader ──► terrain_classify, segmentation
             ├─► depth_model ──► segmentation
             ├─► segmentation
             ├─► shadow_correction ──► segmentation
             ├─► dsm_refine ──► depth_model
             ├─► dtm ──► segmentation
             ├─► building_discovery ──► segmentation
             ├─► city_model ──► segmentation
             ├─► mesh_generation ──► adaptive_mesh, roof_structure, terrain_maps, glb_export
             └─► glb_export
```

`segmentation.py` is the hub — nine modules depend on it. **It is therefore the
project's single largest correctness risk:** its recall directly determines DTM
quality, building count, and canopy/building confusion.

---

## 5. Data

### 5.1 On disk

```
dfc2019_data/rgb/Track3-RGB-1/        1015 files   raw NITF-derived RGB + RPC
dfc2019_data/truth/Track3-Truth/       330 files   DSM/CLS/TXT, 512² @ 0.5 m
dfc2019_data/metadata/Track3-Metadata/   2 dirs    IMD/RPB per view
                                        16 GB total
```

53 tiles usable (RGB + truth), Jacksonville FL and Omaha NE.

### 5.2 Formats

| Artefact | Format | Notes |
|---|---|---|
| Input | GeoTIFF/NITF, RPC00B | not orthorectified, not north-up |
| Truth DSM | float32 GeoTIFF | WGS84 ellipsoidal metres |
| Truth CLS | uint8 | LAS codes: 2 ground, 5 tree, 6 building, 9 water, 17 bridge, 65 unlabelled |
| Tile origin | `_DSM.txt` | easting, **lower-left** northing, size, gsd |
| Mesh | GLB (glTF 2.0) | hand-written; POSITION/NORMAL/TEXCOORD_0/COLOR_0 |
| Buildings | GeoJSON + CSV | CRS only when genuinely georeferenced |
| Scene meta | `scene.json` | tier, provenance, sun, discovery report |

---

## 6. Calibration tiers

| Tier | Source | Claim |
|------|--------|-------|
| **A** | DEM-anchored | true metric elevation |
| **B** | `h = L·tan(sun elevation)` from measured shadows | metric heights |
| **C** | relative depth | shape only; vertical scale assumed and declared |

Enforced, not decorative: a Tier C scene never claims metres, provenance stays
`INFERRED`, and exports from an ungeoreferenced source carry `crs: null`.

**Achieved:** Tier B on JAX_165 (150.9 m/unit from 40 shadow measurements).
Tier C elsewhere. **Tier A has never been exercised** — no DEM ingested.

---

## 7. Measured results

### 7.1 Accuracy — WITHDRAWN

All previously published accuracy figures are withdrawn. Reason in §9.1.
`rescore_baseline.py` is rewritten for orthorectified input but **has not been
run**. The project currently has **no accuracy figure**, which is correct but
must not persist.

### 7.2 Depth quality vs building height (measured, 8 tiles)

Correlation of predicted nDSM with LiDAR nDSM, whole-image inference:

| tile | corr | tallest |
|---|---|---|
| JAX_122 | +0.528 | 13 m |
| JAX_068 | +0.522 | 37 m |
| JAX_260 | +0.486 | 21 m |
| JAX_168 | +0.339 | 18 m |
| JAX_166 | +0.243 | 42 m |
| JAX_165 | +0.241 | 53 m |
| JAX_175 | +0.168 | 22 m |
| JAX_214 | +0.113 | 71 m |
| JAX_167 | +0.086 | 161 m |

**The taller the buildings, the worse monocular depth performs.** Street canyons
defeat it. This is a property of the approach, not a tuning failure.

### 7.3 Segmentation (JAX_068, vs LiDAR labels)

| | precision | recall | coverage |
|---|---|---|---|
| edge-only (original) | 0.09 | 0.10 | 16% |
| + region fill + height cue | 0.34 | 0.20 | — |
| + 3σ noise threshold | ~0.60 | **0.60** | 46.3% (GT 43.1%) |

### 7.4 Building discovery (JAX_165, 640 m)

```
candidates examined 3545  →  retained 1287  →  910 prisms extruded
by size: tiny 642   small 373   medium 218   large 54
rejected: no_evidence 8   duplicate 2250
```
Only **8** dropped for lack of evidence; 2,250 "duplicates" are correct
cross-scale fusion.

### 7.5 Height distribution vs LiDAR

| | model | ground truth |
|---|---|---|
| JAX_068 median | 11.1 m | 11.5 m |
| p90 | 31.9 m | 36.4 m |
| max | 46.2 m | 43.9 m |

Distribution agreement, **not** per-building validation.

### 7.6 Performance (CPU)

| stage | time |
|---|---|
| tiled depth, 2048² | ~430 s |
| tiled depth, 2560² | ~580 s |
| full build, cached depth | **12–32 s** |
| ortho window from remote COG | 5–20 s |

---

## 8. Verification

`verify_pipeline.py` — 8 stages, non-zero exit on hard failure:

1. input resolution / channels / ortho coverage / near-nadir angle
2. height field shape, range, NaN, **orientation** (buildings above ground)
3. segmentation vs LiDAR labels
4. edge refinement sharpening; 4b metric scale stability
5. DTM never above DSM; no negative AGL
6. water levelling bounded
7. building geometry: NaN, UV count, index bounds, degenerate faces, none
   invisible, none floating
8. accuracy + **correlation guard** — refuses to pass a scene whose prediction
   does not correlate with the ground it came from

`test_overlay.py` — two-sided test of overlay rejection (negative control is the
important half).

**Gap: there is no unit-test suite.** Verification is integration-level only.
Directive §51/§52 test cases (one tiny building, courtyard, image edge, 100-small-
buildings scene) are **not** implemented.

---

## 9. Defects found and fixed

### 9.1 RPC misregistration — invalidated every published number

Track 3 RGB frames carry an RPC model and **no geotransform**. The pipeline
aligned them to north-up LiDAR truth with `cv2.resize`, comparing two different
pieces of ground. Verified visually: on JAX_033 the storage tanks sit
bottom-left in the image and top-right in the truth.

Fixed by `ortho.py`. **Consequence: 4.64 m RMSE and the RS3DAda comparison are
withdrawn, not adjusted.**

### 9.2 Truth-grid origin convention

`_DSM.txt` northing is the **lower**-left corner. Read as upper-left, JAX_167
lands 256 m off — the whole frame in the river beside the skyline. Determined by
testing five conventions and a ±40 m offset search; peaks at exactly zero on two
independent tiles.

### 9.3 View selection

Views span 4.8°–29° off-nadir; roof displacement is `height · tan(angle)` — 55 m
for a 161 m tower at 18.9°. "First file in the folder" was an 18.9° view.

### 9.4 Otsu misuse (twice)

- **Shadow detection**: Otsu labelled **53.8%** of a sunlit scene "dark".
  Downstream shadow runs measured 2.2 m against a true median building height of
  11.5 m — 7× short. Replaced with local-contrast; now 6.5%.
- **Elevated mask**: Otsu put the split at 0.0614 against a noise sigma of
  0.00026 — **236× too high**, marking 5.2% of frame at recall 0.067.

### 9.5 Median-of-ratios instability

Dividing a solid numerator by a near-zero denominator per building gave scale
spreads of 143% then 686% of median, and made 37 m buildings **127 m** tall.
Replaced with ratio of medians.

### 9.6 O(n × pixels) component loops

`comp = cc_labels == i` per component allocates a full-image boolean each time;
on 2048² with tens of thousands of components the call never returns. Found in
three places. Vectorised via label LUTs.

### 9.7 Geometry defects

Sign-inverted height field (every building in a pit); ground mesh
backface-culled invisible; fan-triangulated concave footprints producing
self-intersecting shards; bounding-rectangle fallback turning city blocks into
dominoes; buildings buried by a stale DTM after roof squaring; height re-fit
applied only above ground, leaving **234 m of relief across a flat campus**.

### 9.8 Viewer defects

`UnrealBloomPass` blacked out half the viewport; `Sky.js` incompatible with the
WebGPU build; texture downsampled to mesh resolution (the main blur cause);
`ao` imported from `three/tsl` instead of the addon path, silently `undefined`;
`three/tsl` aliased to the webgpu bundle (missing `Fn`); **a static TSL import
cannot be guarded** — one missing symbol renders nothing (8 KB blank page).

### 9.9 My own verification errors

Walking the vertex stream by roof type instead of polygon order (phantom "132
buried"); reading the eave ring instead of the apex on pitched roofs; adding
height clamping that **fabricated 753 of 808 heights** (reverted); a
`git check-ignore` test that misread negation matches as exclusions.

---

## 10. Known limitations

| Limitation | Evidence |
|---|---|
| Depth degrades with building height | corr 0.522 → 0.086 (§7.2) |
| Flat-roof prisms only | domes/spires become boxes — deliberate abstraction |
| No facade data exists in nadir imagery | wall appearance is synthetic, labelled |
| Canopy misread as buildings | IIT-BHU: 40% "building" on a ~8% built campus |
| Vehicles are heuristic | no vehicle class; size/shape gating only |
| Overlay rejection partial | 6/9 pins in test; negative control clean (0.003%) |
| Tier A never exercised | no DEM ingested |
| MVS implemented but unused | `plane_sweep_mvs.py` not in the active path |
| No unit tests | integration-level verification only |
| Truth covers only central 256 m | of a 640 m rendered scene |

---

## 11. Directive compliance (63-section spec)

Detailed matrix in `ARCHITECTURE_AUDIT_2.md`. Summary:

- **Done:** repo audit, tier preservation, no fabricated performance, polygon
  footprints, vegetation/water layers, WebGPU viewer, multi-scale recall,
  small-object retention with reasons, provenance, GeoJSON/CSV export
- **Blocked (no GPU):** Gaussian splatting (§5/20/31), LoRA domain adaptation
  (§14), TRELLIS/Hunyuan completion (§23), GPU-first/TensorRT (§33–35, 55),
  per-building neural refinement (§15), SAM instance segmentation (no checkpoint)
- **Not started:** building inspection mode (§40), confidence visualisation
  (§43), LOD (§26), synchronised image/3D views (§42), formal geometry
  validator (§28)

---

## 12. Highest-priority next actions

1. **Run `rescore_baseline.py`** — restore an accuracy figure. Nothing else is
   more important; the project currently publishes none.
2. **Fix canopy/building confusion** — the largest remaining quality defect;
   blocks vegetated scenes entirely.
3. **Ingest a DEM (CartoDEM 30 m is free)** — unlocks Tier A, the only tier
   never exercised.
4. **Unit tests** for §51/§52 cases.
5. **Building inspection mode** — surfaces the per-building evidence already
   computed and currently only reachable via CSV.

---

## 13. Licensing

Code MIT (`LICENSE`). **Data is not:** DFC2019 is IEEE GRSS-licensed and
non-redistributable; OpenAerialMap scenes carry per-scene CC-BY / CC-BY-SA /
ODbL terms. `.gitignore` excludes both, deliberately.
