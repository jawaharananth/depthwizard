# DepthWizard 2.0 — Audit Against the 63-Section Directive

**Date:** 2026-08-27
**Purpose:** Phase 1 deliverable required by §63 ("produce the architecture audit" before implementing).
**Status of the directive as a whole: NOT COMPLETE. Roughly a third is done, a third is
achievable here and in progress, and a third cannot run on this machine at all.**

This document exists because §47 forbids fabricating capability and §60 forbids
placeholder implementations that pretend to be advanced AI. Reporting "done" against
this directive would violate both.

---

## 1. Hardware and dependency reality

Measured, not assumed:

```
python        3.13.7
torch         2.13.0+cpu
cuda avail    False
device count  0
nvidia-smi    absent -- no NVIDIA GPU
onnxruntime   MISSING
tensorrt      MISSING
gsplat        MISSING
diffusers     MISSING
detectron2    MISSING
segment_anything  present (package only -- NO checkpoint on disk)
```

This single fact decides the feasibility of a large part of the directive. **There is no
GPU.** Depth inference over one 2048px tile takes ~7 minutes on CPU.

---

## 2. Section-by-section status

Legend: **DONE** · **PARTIAL** · **DOING** (this session) · **BLOCKED** (hardware/weights) · **NOT STARTED**

### Scientific foundation

| § | Requirement | Status | Evidence |
|---|---|---|---|
| 0 | Measured vs inferred vs synthetic separation | PARTIAL | Tier B/C labelled in `scene.json` + HUD; per-object provenance not yet emitted |
| 1 | Repository audit | **DONE** | this document |
| 2 | Preserve depth model, segmentation, Tier A/B/C | **DONE** | `depth_model.py`, `segmentation.py`, `shadow_correction.calibrate_scale` |
| 47 | Never fabricate performance | **DONE** | every number in this repo is measured; withdrawn figures were retracted, not adjusted |
| 58 | Do not destroy existing work | **DONE** | benchmark, LiDAR validation, shadow geometry all intact |

### What genuinely works today

| Component | Evidence |
|---|---|
| RPC orthorectification onto the truth grid | `ortho.py`; corrected a misregistration that invalidated every earlier RMSE |
| Truth-grid origin convention | determined empirically over 5 candidate conventions, ±40 m offset search peaks at exactly 0 on two independent tiles |
| Near-nadir view selection | 4.8° chosen from ~25 views spanning 4.8–29° |
| Tiled depth inference | 768 px crops, affine-aligned to a whole-image reference |
| Height-band building separation | 20 → 248 prisms on JAX_068 |
| Ear-clipped concave footprints | replaced bounding-rectangle fallback |
| Shadow calibration (Tier B) | JAX_165: 150.9 m/unit from 40 measurements |
| GTAO ambient occlusion | WebGPU-native TSL path, three r180 |
| 8-stage verification harness | `verify_pipeline.py`, hard-fails on invariant violation |

### Blocked by hardware — cannot be honestly implemented here

| § | Requirement | Why blocked |
|---|---|---|
| 5, 20, 21, 31 | Satellite 3D Gaussian Splatting (SkySplat, RPC-aware 3DGS) | `gsplat` absent; training and rendering both require CUDA. A CPU implementation would take days per scene and could not be validated |
| 14, Phase 5 | LoRA / adapter domain adaptation on DFC2019 | fine-tuning a ViT-L depth backbone on CPU is not tractable |
| 23 | TRELLIS / Hunyuan3D / SPAR3D visual completion | `diffusers` absent; these need multi-GB VRAM |
| 33, 34, 35, 55 | GPU-first execution, TensorRT, ONNX GPU, hardware-aware model selection | no CUDA, no TensorRT, no onnxruntime |
| 15 | Per-building neural refinement network | no such trained model exists for this domain; training is blocked as above |
| 9 (neural) | Instance segmentation via SAM | package present, **no checkpoint**; ViT-H on CPU over 2560 px is not viable |

Writing adapter stubs for these and calling them complete is exactly what §60 prohibits.
A clean interface with a documented "requires CUDA" failure is the honest form, and is
worth doing — but it is not the feature.

### Achievable here — highest impact, being implemented now

| § | Requirement | Status |
|---|---|---|
| 7, 8 | High-recall multi-scale building detection | **DOING** |
| 10 | Small-object recovery, no blind area-threshold deletion | **DOING** |
| 27, 52 | Explicit small-object recall reporting with rejection reasons | **DOING** |
| 9, 24 | Per-building instance records with provenance + confidence | **DOING** |
| 28 | Geometry validation (winding, degenerate, self-intersection) | PARTIAL — winding fixed empirically; formal validator pending |
| 40 | Building inspection mode in viewer | NOT STARTED |
| 43 | Confidence/uncertainty visualisation | NOT STARTED |
| 48, 49 | Provenance metadata + GeoJSON/CSV export | NOT STARTED |
| 26 | LOD generation | NOT STARTED |
| 42 | Synchronised original-image / 3D views | NOT STARTED |

### Already substantially done

| § | Requirement | Status |
|---|---|---|
| 11 | Proper polygon footprints, no bounding-rect fallback | **DONE** (ear clipping; orthogonalisation still open) |
| 16 | Roof structure inference | PARTIAL — `roof_structure.py` classifies flat/shed/gable/hip; prism model currently emits flat only |
| 17 | Continuous terrain, edge-aware | PARTIAL |
| 18, 19 | Vegetation / water / roads as distinct layers | **DONE** — canopy and water are separate meshes with own materials |
| 30 | WebGPU-first viewer with WebGL2 fallback, PBR, IBL, soft shadows, AO, fog | **DONE** |
| 41 | Measurement tools | PARTIAL — 2-point distance/height/slope exist; area, volume, profile pending |
| 44 | Scientific dashboard | PARTIAL — exists, currently showing withdrawn-figures notice |
| 45, 46 | Benchmark everything / A/B harness | PARTIAL — `rescore_baseline.py` rewritten for orthorectified input but **not yet re-run** |

---

## 3. The single most important outstanding item

**The accuracy numbers are withdrawn and have not been regenerated.**

`rescore_baseline.py` was rewritten to orthorectify through the RPC model and to check
each tile's registration before scoring, but has not been executed. Until it runs, this
project has *no* published accuracy figure. That is the correct state — better no number
than a wrong one — but it must not persist.

---

## 4. Honest summary

What exists is a **scientifically careful single-view reconstruction pipeline with a
good real-time viewer**. It is not yet a "confidence-aware AI-assisted digital twin",
and on CPU-only hardware it cannot become one in the Gaussian-splatting / domain-adapted
sense the directive describes.

The realistic path on this machine is to push classical-CV recall, per-object provenance,
validation and presentation as far as they go — which is substantial and directly serves
the stated priority that no building should silently disappear — and to keep the
GPU-dependent sections explicitly marked as blocked rather than faked.
