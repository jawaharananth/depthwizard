# DepthWizard — Measured Accuracy

**Dataset:** IEEE GRSS DFC2019 Track 3, Jacksonville FL
**Reference:** airborne LiDAR (the contest's own ground truth)
**Metric:** nDSM vs nDSM — object height above local terrain, not absolute elevation
**Input:** single RPC-orthorectified satellite view per tile, straightest available
**Date:** 2026-09-04

---

## Tile-level accuracy — 20 tiles

| | value |
|---|---|
| **RMSE** | **3.87 m** |
| **MAE** | **2.01 m** |
| tiles scored | 18 of 20 |
| tiles rejected | 2 (registration guard) |

Two tiles were **excluded, not quietly averaged in**: JAX_149 (correlation 0.229)
and JAX_156 (0.208), both below the 0.25 floor at which a prediction is not
describing the ground it came from. That guard exists because this project
previously published figures computed on misregistered rasters; excluding a tile
loudly is the point of it.

A 5-tile subset gives RMSE 3.74 m / MAE 2.05 m, so the number is not an artefact
of which tiles were chosen.

## Per-building accuracy — JAX_165, 259 buildings inside the truth extent

| | value |
|---|---|
| MAE | 2.63 m |
| RMSE | 4.25 m |
| bias | +1.48 m |
| within 2 m | 52.5% |
| within 5 m | 90.7% |
| within 10 m | 97.3% |

Aggregate agreement hides per-object error, so both are reported. Scene median
height is 20.19 m against LiDAR's 19.88 m — a 0.31 m agreement that would look
excellent on its own while individual buildings still differ by 2.63 m on
average.

**Known residual:** +1.48 m of one-directional bias, concentrated in low
buildings (2–10 m: +9.63 m, n=5) where structure height approaches the MVS noise
floor.

## Footprint accuracy — JAX_165

| | value |
|---|---|
| IoU | 0.553 |
| recall | 0.745 |
| precision | 0.682 |
| right-angle corners | 46.2% |

## Height source comparison — same tile, same metric

| method | per-building MAE | correlation |
|---|---|---|
| monocular depth + shadow scale | 8.53 m | 0.413 |
| **RPC plane-sweep MVS, 6 views** | **6.53 m** | 0.187 |

Neither dominates: MVS is more accurate in magnitude, monocular is better at
spatial shape. The shipped pipeline uses monocular to decide *where* buildings
are and MVS to decide *how tall*, which measured better than either alone.

## Confidence

Per-pixel photometric confidence (plane-sweep NCC) on JAX_165: median 0.60,
14.9% of pixels below the 0.30 reject threshold, 35.9% above 0.7.

Per-building reliability: 593 HIGH, 573 MEDIUM, 126 LOW, 52 WEAK.

---

## On comparing with published results

Published figures for related work exist — SECT-Net reports 4.5–6.6 m, and
Depth2Elevation reports a 24% error reduction over its own baseline. **They are
deliberately not placed in a table beside the numbers above.**

An RMSE is only comparable when the dataset, the metric and the split match.
These results are nDSM-vs-nDSM on DFC2019 Track 3 single-view input; a figure
computed on absolute DSM, on a different city, or from multi-view input is a
different quantity that happens to share a unit. Putting them in one table would
imply a comparison that has not been made.

This project has already had to withdraw a headline number — 4.64 m RMSE, and a
favourable comparison against RS3DAda — after discovering the imagery was never
registered to the ground truth. Both sides of that comparison were wrong in the
same way, which made it meaningless rather than fair. That is the specific
mistake this section exists to avoid repeating.

A defensible comparison requires re-running the published method on these tiles,
with this metric. Until that is done, the honest statement is: **RMSE 3.87 m and
MAE 2.01 m on DFC2019 Track 3, measured, with the method and exclusions above.**

## Reproducing

```bash
python benchmark_all.py 20        # tile-level table
python validate_buildings.py JAX_165   # per-building
python verify_pipeline.py JAX_165      # 8-stage invariant checks
```
