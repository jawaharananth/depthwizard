# Pre-registered predictions

Practice borrowed from zaidnansari2011/sih2026-depthwizard (SIH26175 competitor):
write down a falsifiable prediction *before* running an evaluation, so a
result can't be quietly reframed as expected after the fact. Entries are
append-only — a wrong prediction stays on the record next to the right ones.

---

## 2026-09-19 — re-running rescore_baseline.py after the shadow-calibration fix

**Context**: `dashboard/validation_results.json` (per-terrain RMSE/MAE against
DFC2019 LiDAR nDSM, 18 tiles) is dated 2026-09-04. Since then, one fix landed
that touches this exact benchmark: `build_city_image.py`'s shadow calibration
now refuses `anchor_shadow` scaling when the derived sun elevation exceeds
80 degrees (previously, near-overhead sun made `tan(elevation)` numerically
unstable and could blow a tile's scale up by orders of magnitude — this is
the bug that produced kn_d's 170 m median height earlier this session).
`rescore_baseline.py`'s `predict_ndsm()` calls the same `shadow_correction`
path, so it is exposed to the same fix.

Ground-truth-fusion fixes (footprint-vs-wrong-mask, touching-building merge,
BMC/Open Buildings integration) do NOT apply here — this benchmark scores a
continuous depth/height field against LiDAR nDSM, not extracted footprints,
and none of the 18 DFC2019 tiles carry ground-truth building records.

**Prediction, made before re-running**: overall RMSE will not get meaningfully
worse than the recorded 3.87 m, and will improve only if one or more of the
18 tiles happened to hit the >80-degree sun-elevation edge case the fix
guards against. Given that edge case requires near-solar-noon imagery, which
is a minority condition, I expect at most 1-2 of the 18 tiles to be affected
and predict the overall RMSE lands in **3.6-3.9 m**, i.e. flat to slightly
improved, not a large swing either direction.

Baseline being compared against (2026-09-04):

| terrain | rmse_m | mae_m | n_pixels |
|---|---|---|---|
| urban   | 6.487 | 4.794 | 626,263 |
| hilly   | 1.431 | 0.521 | 959,958 |
| forest  | 5.903 | 4.577 | 925,192 |
| sparse  | 2.039 | 0.744 | 2,115,114 |
| overall | 3.872 | 2.012 | 4,626,527 |

**Result (2026-09-19, same day)**: **prediction correct on the aggregate, but understated how much the per-terrain breakdown actually moved.**

| terrain | rmse_m (09-04) | rmse_m (09-19) | delta |
|---|---|---|---|
| urban   | 6.487 | 5.398 | -1.089 (improved) |
| hilly   | 1.431 | 1.819 | +0.388 (worse) |
| forest  | 5.903 | 5.557 | -0.346 (improved) |
| sparse  | 2.039 | 2.646 | +0.607 (worse) |
| overall | 3.872 | 3.743 | -0.129 (improved) |

The overall RMSE landed at 3.74 m, inside the predicted 3.6-3.9 m band, and
did not get worse -- that part of the prediction holds. But calling it "flat
to slightly improved" undersold what happened underneath: urban and forest
both improved by more than 0.3 m while hilly and sparse both got
measurably worse, and those two pairs partly cancel out in the aggregate.
The prediction reasoned only about the shadow-calibration >80-degree fix,
which should affect urban scenes (buildings cast the shadows that
calibration reads) -- consistent with urban's improvement -- but did not
account for whatever is making hilly and sparse worse, which this entry does
not have an explanation for yet. Recorded honestly rather than only
reporting the part that matched.

---

## 2026-09-19 — finding, not a prediction: the DFC2019 rescore was already a cross-domain result

Not pre-registered because it was not a test being designed -- it was
recognizing what an existing measurement already was. Recorded here anyway
because it directly reframes the numbers two entries above.

`checkpoints/height_best/metric_ok.json` shows the depth backbone currently
in production (`DepthBackbone.metric_available() == True`) was fine-tuned on
**400 GAMUS tiles**, 12 epochs, and licensed because it beat a baseline given
the optimal affine scale (MAE 2.439m -> 1.444m, correlation 0.784 -> 0.912,
measured on GAMUS's own held-out validation split).

Every rescore_baseline.py / per_building_metric.py number in this file and
in `dashboard/` is that same GAMUS-trained checkpoint scored against
DFC2019 -- a different sensor (WorldView-3, not GAMUS's source), different
cities (Jacksonville/Omaha vs. GAMUS's PHL/NYC/DC), different ground-truth
pipeline (airborne LiDAR vs. GAMUS's own AGL), never seen during training.
That is a real cross-domain generalisation test, structurally the same claim
zaidnansari2011/sih2026-depthwizard makes with their DFC2019-trained,
held-out-DC-block result (1.788m per-building) -- this project already had
the equivalent, just never stated it as one. The whole-tile RMSE (3.74m,
entry above) and the per-building RMSE (see per_building_metric.py, run
2026-09-19) should both be reported with this framing explicitly: trained on
GAMUS, tested cross-domain on DFC2019, never touched in training.

**Per-building result (2026-09-19)**: 461 buildings, all 18 tiles scored --
RMSE 3.063m, MAE 2.306m, bias +0.900m (systematic over-prediction, not
random scatter). Direct comparison to zaidnansari2011's reported headline
number, same kind of metric (per-building median height vs LiDAR, on a
dataset the model never trained on):

| | RMSE | n buildings |
|---|---|---|
| this project (GAMUS -> DFC2019) | **3.063 m** | 461 |
| zaidnansari2011 (DFC2019+GAMUS -> held-out DC) | 3.464 m | 3,090 |

Ahead on the number itself, but honestly: 461 buildings is a much smaller
sample than 3,090, so the confidence interval on this result is wider than
theirs, and two tiles with small building counts (JAX_020, 7 buildings,
7.66m; JAX_068, 10 buildings, 7.33m) are pulling the aggregate up --
worth investigating what is specifically wrong with those two scenes rather
than averaging past them. The +0.900m systematic bias is also a real,
fixable finding this entry does not explain yet -- not present in the
comparison's own reported near-zero bias.
