# Evaluation protocol — which number means what

Written 2026-09-20, after this project generated four different per-building
RMSE numbers in one session (3.063m, 4.219m, 2.164m, and whatever the expanded
fixed run reports) and it became genuinely easy to cite the wrong one. This
file exists so that never happens silently again — cite a protocol name, not
just a bare number.

## The bug that made this necessary

`depth_model.py`'s `predict()`/`predict_tiled()` silently normalised every
output to `[0,1]`, even when the licensed GAMUS metric checkpoint was active,
and going through transformers' `pipeline()` wrapper made it worse (that
wrapper's `result["depth"]` is an 8-bit visualisation rescale, not the raw
regression output). Fixed 2026-09-19/20 with a `raw_metric=True` option. Every
number measured before that fix landed is now void — not "less accurate," void,
because it was never measuring the trained model's real output at all.

**Rule: any per-building or whole-tile number from before commit `0922f87`
(the `depth_model.py` fix) must not be cited, compared against, or averaged
with anything after it.** They are different measurements of different
things, not two data points on the same scale.

## Three evaluation entry points, all still in the codebase

| # | Script | Scores | Aggregation |
|---|---|---|---|
| 1 | `train_height.py`'s in-training `evaluate()` | GAMUS's own held-out val split | per-pixel, whole-tile |
| 2 | `rescore_baseline.py` | DFC2019, whole-tile pixel RMSE | per-pixel, aggregated across tiles |
| 3 | `per_building_metric.py` | DFC2019, per-building median height | one number per ground-truth building instance |

Protocols 2 and 3 are NOT the same measurement even when run on the identical
tile set. Protocol 2 includes every pixel — roads, bare ground, whatever a
terrain class contains — and a scene that's mostly flat ground can look good
almost by default (measured: "hilly" terrain alone is 1.82m RMSE, pulling the
whole-tile average down). Protocol 3 only ever sees pixels a human labelled as
a building, which is a harder and more specific question. **Never quote
protocol 2's number as if it answers protocol 3's question** ("how good are we
at guessing a building's height") — they answer different questions.

## Sample-size discipline, learned the hard way today

An 18-tile subset (461 buildings) is NOT a validated sample size on its own —
today it produced two contradictory verdicts (RMSE 3.063m one run, then 2.164m
after the bug fix) and a THIRD run on 50 tiles (1,063 buildings, still under
the bug) reversed the direction entirely (4.219m, worse than the comparison
figure). Whether a result holds at 50+ tiles is not optional colour, it is
part of the result. **A number from fewer than ~50 tiles / ~1,000 buildings
is provisional and must be labelled as such until it is checked at that
scale.**

## Headline numbers, dated and protocol-tagged (fill in as they're confirmed)

| Date | Protocol | Tiles | Buildings | RMSE | MAE | Bias | Status |
|---|---|---|---|---|---|---|---|
| 2026-09-04 | 2 (whole-tile) | 18 | — | 3.872m | 2.012m | — | superseded by 09-19 shadow-calib fix |
| 2026-09-19 | 2 (whole-tile) | 18 | — | 3.743m | 2.024m | — | current, post shadow-calib fix |
| 2026-09-19 | 3 (per-building) | 18 | 461 | ~~3.063m~~ | ~~2.306m~~ | ~~+0.900m~~ | **VOID — predates raw_metric fix** |
| 2026-09-19 | 3 (per-building) | 50 | 1,063 | ~~4.219m~~ | ~~2.998m~~ | ~~+1.465m~~ | **VOID — predates raw_metric fix** |
| 2026-09-20 | 3 (per-building) | 18 | 461 | 2.164m | 1.597m | -0.596m | provisional, superseded by 50-tile row below |
| 2026-09-20 | 3 (per-building) | 50 | 1,063 | **3.132m** | **1.870m** | **-0.608m** | **CURRENT, validated at scale** |

**The 3.132m / 50-tile / 1,063-building row is the number to cite.** It
survived the exact test that broke the earlier 3.063m claim: RMSE widened
from 2.164m to 3.132m going from 18 to 50 tiles (expected — more, harder
buildings included), but the direction held — still ahead of the 3.464m
comparison figure, this time on a real, checked sample. Bias also stayed
consistent (-0.596m -> -0.608m) rather than drifting the way the broken
pipeline's bias did (+0.900m -> +1.465m), which is itself evidence this
result is real rather than a sample-size artifact.

## Standing rules for this project

1. A number is not real until it survives being run again at 3x the sample.
2. State which script/protocol produced a number every time it's quoted, not
   just the bare value.
3. A bug that silently changes what's being measured is not a "worse result,"
   it's not a result at all — mark the affected rows VOID, don't average them in.
4. When a comfortable number and a harsher one disagree, investigate the
   discrepancy before trusting either — do not report the comfortable one and
   quietly drop the other.
