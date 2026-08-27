# DepthWizard — Architecture Audit

**Phase 1 deliverable. No implementation performed.**
Date: 2026-08-25 · Codebase: 3,112 lines across 21 Python modules + 2 HTML apps

---

## 0. Executive summary

Three findings dominate everything else in this audit:

1. **85.8% of building candidates are silently discarded.** Measured, not estimated:
   2,515 connected components → 414 survive the area filter → 357 survive the height
   filter. This is the direct opposite of the spec's highest-priority requirement
   (§7, §10, §52) and is the single highest-value thing to fix.

2. **There is no NVIDIA GPU on this machine.** Intel Core 5 120U with Intel integrated
   graphics, `torch.cuda.is_available() == False`, torch is a CPU-only build. This
   hard-gates roughly a third of the specification.

3. **There are zero tests.** Not one test file exists. Spec §51/§52 require an extensive
   test matrix, and the "critical failure test" in §52 is precisely the small-building
   recall problem identified in finding 1.

---

## 1. Hardware reality (the gating constraint)

Measured on this machine:

| Property | Value |
|---|---|
| CPU | Intel Core 5 120U (10 threads, low-power mobile U-series) |
| RAM | 15.6 GB total, **1.0 GB free at audit time** |
| GPU | Intel(R) Graphics, integrated, 2 GB shared |
| CUDA | Not available. `torch 2.13.0+cpu`, `torch.version.cuda = None` |
| Disk free | 160 GB |
| Build toolchain | No `make`, no `gcc`/`g++`, no WSL |
| HuggingFace cache | 5.1 GB (DA-V2 Large + UniDepth v2 already downloaded) |

The free-RAM figure matters as much as the GPU absence. The machine is running several
Chrome instances, multiple VS Code windows, and Avast concurrently; an earlier benchmark
run took over 18 minutes on a single tile that had previously completed in 170 seconds,
purely from CPU contention.

This is a thin-and-light laptop, not a workstation. Any plan that assumes otherwise will
produce code that cannot be run or validated here.

### What this hard-blocks

| Spec section | Requirement | Blocker |
|---|---|---|
| §5, §20, §21, §31 | 3D Gaussian Splatting, SkySplat, RPC-aware 3DGS | Every practical 3DGS rasterizer (`gsplat`, `diff-gaussian-rasterization`) is a CUDA extension. No CUDA, and no compiler to build one. |
| §23 | TRELLIS, Hunyuan3D, Stable Fast 3D, SPAR3D | Need 8–16 GB VRAM. Available: 2 GB shared integrated. |
| §33 | TensorRT | NVIDIA-only runtime. |
| §14 | LoRA / adapter fine-tuning | Technically runnable on CPU, but a meaningful run over DFC2019 would take days on this CPU, with 1 GB free RAM. Not validatable within any reasonable iteration loop. |

These are environment limits, not judgements about merit. §5 and §14 in particular are
genuinely the right ideas — 3DGS is the correct technique for photoreal reconstruction of
captured scenes, and domain adaptation is the correct fix for the nadir-bias problem
documented in §4 below. Both need hardware this machine does not have.

Per spec §60: adapter interfaces and install instructions should be written for these, and
they must not be faked or stubbed as if running.

### What still works well here

Everything CPU-bound and geometric: building detection and instance separation, polygon
reconstruction, multi-signal height fusion, shadow geometry, terrain reconstruction, mesh
topology and LOD, geometry validation, provenance tracking, exports, and the entire viewer
(WebGPU on Intel iGPU is supported by current Chrome — only this headless test environment
lacks a backend).

---

## 2. Current architecture

```
run_pipeline.py  (6-stage orchestrator, CLI)
  │
  ├─1 depth_model.py            Depth Anything V2 Large → relative depth
  ├─2 segmentation.py           5-class land cover (NDVI/NDWI | texture heuristic)
  ├─3 calibration/tiered.py     Tier A / B / C selection + fallback
  │     ├── georeferenced.py    DEM fit, per-terrain curves, GSD estimation
  │     ├── shadow_hybrid.py    SHIPPED Tier B: direct + regression
  │     ├── shadow_based.py       └ regression sub-tier
  │     ├── shadow_primary.py     └ direct-measurement sub-tier
  │     ├── object_scale.py     GSD estimation only (height formula retired)
  │     └── terrain_curves.py   per-class curve fitting
  ├─4 shadow_correction.py      NOAA solar position, shadow→height
  ├─5 dsm_export.py             GeoTIFF DSM (CRS omitted when non-metric)
  └─6 mesh_generation.py        heightfield + building extrusion + normal map
        └ glb_export.py         binary glTF writer

Support:  terrain_classify.py · validation.py · dfc2019_loader.py
          dfc2019_benchmark.py · rpc_model.py · plane_sweep_mvs.py

Frontend: viewer/index.html (WebGPU + WebGL2 fallback)
          dashboard/index.html (accuracy dashboard)
```

---

## 3. Measured weaknesses

### 3.1 Building recall — the critical defect

Measured on `sample.jpg` (850×637):

```
2,515  raw connected components
  414  survive  min_area_px = 15        (-2,101 lost, 83.5%)
  357  survive  min_height_m = 2.0      (-57 lost)
─────
14.2%  end-to-end retention
```

Component size distribution before filtering:

| Bucket | Count |
|---|---|
| tiny (<15 px) | 1,919 |
| small (15–50 px) | 327 |
| medium (50–200 px) | 206 |
| large (>200 px) | 63 |

The `< 15 px` bucket is discarded wholesale by a bare threshold with no evidence path —
exactly the anti-pattern spec §10 names. Some of those 1,919 are segmentation noise, but
at this image's scale 15 px is roughly a 15 m² footprint, which is a real shed or garage.
The system currently cannot distinguish "noise" from "small building" because it never
looks — it filters on area alone before any evidence is gathered.

**Secondary defect:** `_building_footprints` returns `cv2.minAreaRect` boxes. This was a
deliberate fix for a real bug (fan-triangulation of concave polygons produced
self-intersecting geometry), but it means every L-shaped building, courtyard, and notch is
destroyed. Spec §11 correctly calls for real polygon reconstruction with topology
validation instead.

**Tertiary defect:** adjacent buildings sharing a wall merge into one connected component
and become a single box. No instance separation exists (spec §9).

### 3.2 Depth signal is near-useless on nadir imagery

Established earlier by measurement, restated because it drives the architecture:

- DA-V2 imposes a top-to-bottom frame gradient on nadir imagery (mean relative depth
  0.68 at frame top → 0.43 at bottom), learned from ground-level photography.
- Correlation between model depth and true shadow-derived building height: **r ≈ 0.08**.
- UniDepth v2 was installed and benchmarked as the proposed fix: **r = 0.077**. No
  improvement — its camera-agnostic design targets varying focal length in perspective
  imagery, not orthographic nadir geometry.

This is why building height currently comes from shadow geometry rather than the network,
and why spec §14 (domain adaptation) is the correct long-term fix.

### 3.3 No instance model

`segmentation.py` produces a semantic mask only. There is no building ID, no per-building
confidence, no provenance, and no per-object record — so §9, §24, §40, and §43 have nothing
to attach to. This is a structural gap, not a tuning issue.

### 3.4 Single-signal height

`shadow_hybrid.py` fuses exactly two signals (shadow-direct for buildings, depth regression
for terrain) with a hard-coded split, not a confidence-weighted fusion. Spec §12 calls for
seven signals with data-driven weights. DEM and stereo signals exist in the codebase
(`georeferenced.py`, `plane_sweep_mvs.py`) but are not wired into a fusion layer.

### 3.5 Flat-box roofs

Every building is extruded to a single flat height (`percentile(region, 90)`). No roof-type
inference exists (§16).

### 3.6 Mesh is a uniform grid

`build_ground_mesh` emits one vertex per pixel of a uniformly downsampled grid. No adaptive
tessellation, no curvature-aware subdivision, no LOD chain (§25, §26). At `max_dim=800`
this is 479k vertices / 956k faces / 27.5 MB regardless of where detail actually exists.

### 3.7 No geometry validation

No manifold check, winding check, degenerate-triangle check, or self-intersection check
exists (§28). The concave-polygon bug shipped precisely because nothing validated output.

### 3.8 No tests, no provenance, no run report

Zero test files (§51, §52). No per-object provenance or confidence state (§24). No
structured run report (§56).

---

## 4. Scientifically validated — must not be damaged

Per spec §58, these are empirically grounded and must be preserved through any refactor:

| Component | Why it is trustworthy |
|---|---|
| `shadow_correction.py` solar position | NOAA algorithm validated against 3 independent astronomical reference points (Tropic of Cancer solstice noon 89.58°, equator equinox noon 89.85°, polar night −22.44°) |
| Shadow→height geometry | Verified against synthetic ground truth at two sun angles, recovering known heights within edge-effect margin |
| `validation.py` RMSE/MAE/bias | Verified exact against known injected error (urban 2.0/2.0, hilly 3.0/3.0, forest 0.0/0.0) |
| Tier A/B/C structure and honesty guarantees | Relative-only output is deliberately written without CRS; every non-metric path is labelled in metadata, console, and dashboard |
| `rpc_model.py` | Forward/inverse self-consistency verified to sub-millimetre over 200 random points |
| DFC2019 benchmark harness | Runs against real airborne LiDAR. Current result **4.58 m RMSE / 2.25 m MAE** (nDSM vs nDSM). A previously reported 5.91 m came from comparing predicted object height against *absolute* DSM, which charged the system for terrain it never claimed to predict -- see the metric note below. |
| Datum handling in benchmark | Ground-references both sides before comparison, avoiding the ~23 m geoid-offset error |

`plane_sweep_mvs.py` is honest negative-result code (9.98 m RMSE, worse than baseline). It
should be kept and clearly marked as such, not deleted — spec §45 and §47 both depend on
keeping measured negative results visible.

---

## 5. Where buildings can disappear (spec §18–§20 requirement)

Exhaustive list of loss points found in the current code:

| # | Location | Mechanism | Measured loss |
|---|---|---|---|
| 1 | `segmentation.segment` | Edge-density threshold; low-contrast roofs never enter the building class | not yet quantified |
| 2 | `_building_footprints` | `if contourArea < min_area_px: continue` — bare area filter, no evidence path | **2,101 of 2,515** |
| 3 | `_building_footprints` | `minAreaRect` collapses concave shapes; adjacent buildings merge into one box | shape loss, not count loss |
| 4 | `build_building_meshes` | `if roof_h - ground_ref < min_height_m: continue` | 57 |
| 5 | `build_building_meshes` | `if region.size == 0: continue` — silent skip | rare |
| 6 | `_resize_for_mesh` | Segmentation downsampled for the ground grid (buildings now correctly read full-res, but this path still exists) | previously catastrophic (2,416 → 7) |
| 7 | `terrain_classify` | Structure buffer excludes building-adjacent pixels from hilly classification | affects class stats, not geometry |

Every one of these is a silent `continue`. None reports a reason. Spec §27 and §52 require
exactly the opposite.

---

## 6. Computational bottlenecks

Profiled during this audit:

- **DA-V2 inference on CPU:** ~30–60 s per image, dominating single-image runs.
- **Per-building loop in `build_building_meshes`:** Python-level, ~2,500 iterations. Already
  optimised with bounding-box cropping; acceptable at current counts, will need vectorising
  if recall rises to ~2,500 retained objects.
- **`build_ground_mesh`:** already vectorised (1.5 s at 956k faces; was a Python double loop).
- **GLB write:** 27.5 MB, ~1 s.
- **Browser load of 27.5 MB GLB:** roughly 13 s in headless software rendering. Will be far
  faster on real hardware, but argues strongly for the LOD/streaming work in §26 and §32.
- **System contention:** the dominant real-world factor. See §1.

---

## 7. Proposed plan

Ordered by measured value per unit of risk, and constrained to what can actually be built
and validated on this hardware.

### Priority 1 — Building recall and instance model (spec §7–§11, §27, §52)

Addresses the 14.2% retention defect directly. Entirely CPU-bound.

- Replace bare area filters with an **evidence-scored candidate pipeline**: every component
  is scored on edge, shadow, spectral, texture, and height evidence, and is retained unless
  evidence is zero. Rejections are recorded with a reason, never silent.
- **Multi-scale detection** at full / 2× / 4× plus high-resolution local crops, with fusion.
- **Instance separation** via watershed or distance-transform splitting so shared-wall
  buildings stop merging.
- **Real polygon reconstruction** — contour refinement, orthogonalisation, concavity and
  courtyard preservation — with a robust triangulator (ear clipping with hole support)
  replacing `minAreaRect`, plus topology validation so the original self-intersection bug
  cannot recur.
- **`SMALL_OBJECT_RECALL_TEST`** as specified in §27, reporting the full funnel.

### Priority 2 — Per-object model with provenance (§9, §24, §43, §48)

The structural prerequisite for confidence visualisation, building inspection, and exports.
A `Building` record carrying ID, polygon, area, orientation, height, per-signal confidences,
and a `provenance` enum (`OBSERVED` / `MEASURED` / `INFERRED` / `AI_COMPLETED`).

### Priority 3 — Multi-signal height fusion (§12)

Wire the signals that already exist — shadow, depth, DEM, stereo, roof-ground discontinuity
— into a confidence-weighted fusion with data-driven weights (shadow weight down under
canopy, DEM weight up when DEM quality is high, depth weight down given the measured r≈0.08).

### Priority 4 — Geometry validation + test suite (§28, §51, §52)

Manifold, winding, degenerate, duplicate, self-intersection, and bounds checks, with safe
auto-repair. The §51 test matrix, including the tiny-building, concave, courtyard,
edge-of-image, and adjacent-building cases.

### Priority 5 — Roof structure (§16)

Planar segmentation of the local height field to infer flat / gable / hip / shed, falling
back to a conservative plane when evidence is insufficient. Explicitly no hallucinated roofs.

### Priority 6 — Mesh quality, LOD, viewer (§25, §26, §30, §40, §41, §42, §43)

Adaptive tessellation, an LOD chain, building inspection mode, expanded measurement tools,
synchronised image/3D views, and confidence overlays.

### Deferred — documented, interfaced, not faked (§60)

3D Gaussian Splatting (§5, §20, §31), image-to-3D visual completion (§22, §23), LoRA domain
adaptation (§14), and TensorRT (§33). Each gets a clean adapter interface, a real non-GPU
fallback, and install instructions — and each is reported as unavailable at runtime rather
than mocked.

---

## 8. Recommended deviation from spec

Spec §60 invites documented deviation. One is proposed:

**Do not pursue Gaussian Splatting as the primary visual upgrade path on this hardware.**
Beyond the CUDA blocker, 3DGS reconstructs *observed* surfaces, and a nadir satellite capture
observes no facades at all. Even with a GPU, a single-image 3DGS reconstruction would have
nothing truthful to place on building sides. The higher-value visual work here is adaptive
mesh detail, real roof geometry, and correct LOD — all of which are also the things the
current output most visibly lacks.

If a CUDA machine becomes available, the ordering changes and 3DGS becomes worth
implementing for multi-view DFC2019 tiles specifically, where genuine multi-view coverage
exists.

---

## 8b. Metric correction (added after RS3DAda benchmarking)

Three separate defects in this project came from the same root error:
**comparing two quantities that are not the same thing.**

1. absolute elevation vs relative height (caught during Tier B work)
2. surface model vs terrain model (caught when buildings rendered buried)
3. absolute DSM vs nDSM (caught when a purpose-built model scored worse than
   its own output images suggested it should)

The third invalidated the headline accuracy figure. DFC2019 ground truth is
absolute DSM -- terrain plus objects. Our prediction, and RS3DAda's, are
object height *above* terrain. Subtracting one offset per tile removes a
constant but not terrain variation *within* the tile, measured at 18.0 m on
JAX_004 against a 16.1 m total prediction range.

Corrected by deriving a ground-truth nDSM from the LiDAR's own LAS class 2
(Ground) returns. Both systems then scored identically:

| System | RMSE | MAE |
|---|---|---|
| **DepthWizard shadow-hybrid** | **4.58 m** | **2.25 m** |
| RS3DAda (NeurIPS 2024, purpose-built RS height model) | 5.28 m | 3.75 m |

Shadow-hybrid won on all five tiles on both metrics. The measured conclusion
is that physical shadow geometry outperforms a learned height prior here --
which is worth stating precisely because the RS3DAda output *looked*
substantially better as an image.

---

## 9. What this audit did not do

No code was modified. No benchmarks were re-run beyond the measurements reported here. The
segmentation-stage loss (loss point 1 in §5) is not yet quantified and should be measured
before Priority 1 work begins, since it may be a larger contributor than the area filter.
