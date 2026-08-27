# DepthWizard

Single satellite image → navigable 3D city model.

Built for SIH26175 (ISRO): reconstruct elevation and a 3D terrain model from a
single-view satellite or aerial image, with a visualisation layer.

![status](https://img.shields.io/badge/heights-Tier%20B%2FC-orange)
![status](https://img.shields.io/badge/accuracy-being%20re--measured-red)

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

---

## Honest status

This section exists because the project has previously published numbers that
turned out to be wrong, and the correction mattered more than the number.

**Accuracy figures are currently withdrawn.** An earlier headline of 4.64 m RMSE
— and a favourable comparison against RS3DAda — were computed on *misregistered*
rasters. DFC2019 Track 3 RGB tiles are raw satellite frames carrying an RPC
camera model and **no geotransform**; the pipeline was aligning them to the
north-up LiDAR truth with `cv2.resize`, which compares two different pieces of
ground. `ortho.py` now projects the truth grid through the RPC model so the two
share a grid by construction, but `rescore_baseline.py` has **not yet been
re-run**. Until it is, this project has no accuracy figure. That is the correct
state; better no number than a wrong one.

**Height tiers** are labelled in every scene and shown in the viewer HUD:

| Tier | Meaning |
|------|---------|
| **A** | DEM-anchored — real metric elevation |
| **B** | Shadow-calibrated — metres from `h = L · tan(sun elevation)` |
| **C** | Relative — shape only, vertical scale assumed and declared |

A Tier C scene never claims metres, and exports from an ungeoreferenced source
never carry a CRS.

**Known limits, measured:**

- Monocular depth degrades as buildings get taller. Correlation with LiDAR falls
  from 0.522 (JAX_068, 44 m) to 0.086 (JAX_167, 161 m) — street canyons defeat it.
- Buildings are **flat-roofed prisms**. Domes and spires become boxes. This is a
  deliberate abstraction, not an accident.
- Nadir imagery contains **no facade pixels**. Wall appearance is synthetic and
  labelled as such.
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
| `viewer/index.html` | WebGPU renderer, GTAO → denoise → bloom → FXAA |

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

No GPU required; it runs on CPU. Depth inference is the bottleneck (~6 min per
2500 px tile), and is cached, so rebuilds take ~30 s.

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
