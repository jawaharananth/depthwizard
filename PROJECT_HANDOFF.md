# DepthWizard — full project handoff

Read this first, in full, before touching any code. It's written so a
fresh agent (or person) with zero prior context can pick this project up
and know exactly what it is, what state it's in, what's been tried and
rejected, and what to do next. Where this document summarizes something,
the actual authoritative source is named — read that file too before
making a claim this document doesn't cover.

---

## 1. What this project is

**DepthWizard** — built for **SIH26175**, a Smart India Hackathon 2026
problem statement from **ISRO**: take a single nadir (straight-down)
satellite or aerial image and produce (a) a high-precision elevation model
and (b) a navigable 3D city model, with honest uncertainty/provenance
reporting throughout.

Repo: `github.com/jawaharananth/depthwizard` (private).

Core pipeline, one sentence: a monocular depth-estimation neural network
predicts a height field from the flat image; segmentation separates
buildings/roads/water/vegetation/bare-earth; footprints get extracted per
building; each footprint's height comes from the depth field UNLESS a real
measured height is available (municipal records, LiDAR, satellite laser
altimetry), in which case the real number wins; buildings get extruded
into 3D prisms; everything gets assembled into a GLB + GeoJSON + a
browser-based WebGPU viewer.

## 2. The one rule that governs everything else

**Never fake or spin a result. Report honest numbers even when they
contradict an earlier claim.** This isn't a style preference — it's been
enforced the hard way, repeatedly, across this project's history:

- A checkpoint-promotion gate that let a noise-level "improvement" through
  was found and replaced with two real numeric gates (MAE improvement AND
  correlation must both pass a stated bar) — see `train_height.py`,
  search "LICENSE THE CHECKPOINT".
- A silent bug (`depth_model.py`) discarded a licensed model's real metre
  output for months and was only caught by re-measuring at 3x sample
  size when a number looked too good. See `EVALUATION_PROTOCOL.md` — it
  exists specifically because this happened and must not happen silently
  again.
- The single most important bug found in this project's whole history
  (see §6 below) was caught by empirically checking whether real
  municipal building-height records actually reached the exported 3D
  model, rather than trusting that they did because the code looked like
  it should.

**When you find a number that looks wrong, or a claim that seems too
convenient, check it before repeating it.** This project's credibility
depends on that discipline more than on any single feature.

## 3. Real, current accuracy — the one number to cite, and why

Per-building height accuracy: **RMSE 3.132 m, MAE 1.870 m**, measured on
50 DFC2019 benchmark tiles / 1,063 ground-truth buildings, beating a 3.464 m
comparison figure. This number:
- Survived being re-checked at 3x the original sample size (a smaller
  18-tile sample gave a different, non-reproducing number first — see
  `EVALUATION_PROTOCOL.md`'s "Sample-size discipline" section).
- Is DIFFERENT FROM, and not comparable to, the whole-tile pixel RMSE
  (3.74 m) also reported elsewhere — these are two different protocols
  answering two different questions. **Read `EVALUATION_PROTOCOL.md`
  before quoting either number** — citing the wrong one, or averaging
  them, is exactly the mistake that file exists to prevent.

## 4. Architecture — the height-tier honesty system

Every scene is labelled with which tier it actually reached; never claim
a tier the pipeline didn't earn:

| Tier | Meaning |
|------|---------|
| **A** | DEM-anchored — real absolute elevation (Copernicus GLO-30) |
| **B** | Shadow-calibrated — metres from `h = L · tan(sun elevation)` |
| **C** | Relative only — vertical scale assumed and declared, never claimed as real metres |

On top of this, the licensed metric model (`checkpoints/height_best/`,
present only after a real training run beats baseline — see §7) predicts
metres directly, removing the Tier C scale-guessing problem entirely when
present.

**Ground-truth fusion** (the project's sharpest real differentiator): where
a REAL measured height exists for a specific building — municipal
floor-count records (Mumbai BMC), human-verified LiDAR labels (DFC2019),
satellite laser altimetry (ICESat-2), a global ML-derived building-height
raster (Open Buildings 2.5D) — that real number overrides the model's
guess for that one building. The model's own estimate is still computed
and kept for audit (`height_model_estimate` field), never silently
discarded.

**Read `README.md` for the full pipeline stage list and module map.**

## 5. Key files, grouped by what they do

**Core pipeline:**
- `depth_model.py` — the depth backbone wrapper (`DepthBackbone` class).
  Handles stock/metric/HTC checkpoint selection, behavioral checkpoint
  verification (see §8), the `raw_metric=True` bypass that actually
  returns real metres instead of a [0,1]-normalized field.
- `build_city_image.py` — builds a scene from ANY plain image (no ground
  truth needed). The general-purpose entry point.
- `build_city.py` — builds a scene from a DFC2019 benchmark tile
  specifically (has real LiDAR ground truth; produces the compare-slider
  assets and error/confidence maps that `build_city_image.py` never does).
- `region_footprints.py` — footprint extraction + the known-heights
  override logic.
- `mesh_repair.py` / `mesh_generation.py` — turns footprints + a height
  field into a validated 3D mesh. **Important: this is what actually
  determines final building height in the exported geometry — it samples
  the surface model (`dsm`) directly, not `region_footprints`'s own height
  records. See §6.**
- `htc_bins.py` — from-scratch HTC-DC Net (Chen et al., IEEE TGRS 2023)
  implementation: adaptive per-image height bins, head-tail-cut
  foreground/background split, KL-divergence distribution constraint.
  Built independently from the paper, not from any competitor's code.
- `train_height.py` — the training script. Two-gate promotion logic (see
  §3/§7).

**Data source modules** (all real, live-tested against actual public
APIs, not mocked):
- `dem_source.py` — Copernicus GLO-30 DEM, public/no-auth, with a circuit
  breaker (fails fast after repeated network failures rather than hanging).
- `icesat2_source.py` — real spaceborne LiDAR (NASA ICESat-2 ATL08), via
  two public no-auth APIs (NASA CMR for discovery, OpenAltimetry for data).
  Sparse coverage — a real, honestly-reported "sometimes zero points"
  outcome, not a bug.
- `open_buildings_source.py` — Google Open Buildings 2.5D Temporal dataset
  (real per-building height raster, public GCS bucket, no auth). Finding
  the right tile for an arbitrary lat/lon required reverse-engineering an
  undocumented manifest format — see the module's own docstring for the
  exact (non-obvious) URL-concatenation bug that had to be found and fixed
  along the way.

**Frontend:**
- `viewer/index.html` — the entire viewer: WebGPU renderer (Three.js
  r180, vendored locally in `viewer/vendor/`), orbit + pointer-lock
  walk-mode camera, scene-query natural-language-ish chat, compare-slider,
  measurement tools, annotation pins, video export, flood simulation,
  GCP calibration tool with real regression diagnostics (condition number,
  leave-one-out cross-validation, bootstrap confidence interval on the
  fitted exponent).

**Backend:**
- `server.py` — the upload/build server. Stdlib-only (`ThreadingHTTPServer`,
  no Flask/FastAPI, deliberately — see the module's own docstring for why).
  Has health/readiness endpoints, upload integrity checking (rejects
  truncated/corrupt uploads instead of silently building garbage), a
  content-hash result cache, measured-cost-derived rate limiting.

**Process / anti-drift infrastructure** (genuinely more rigorous than
either competitor audited — see §9):
- `scripts/verify_capabilities.py` → regenerates `CAPABILITIES.md`, a
  LIVE, code-derived report of what's actually wired up (not a
  hand-maintained checklist that can silently go stale).
- `scripts/extract_decisions.py` → regenerates `DECISIONS.md` by pulling
  rationale straight out of existing code comments (325+ entries).
- `.github/workflows/ci.yml` — CI that fails the build if either of the
  above two files is stale relative to the actual code, not just if tests
  fail.

**Deployment:**
- `Dockerfile`, `requirements-serve.txt`, `.dockerignore`, `.gcloudignore`
  — Google Cloud Run deployment package (CPU-only, no GPU on the serving
  side ever).
- `DEPLOY.md` — exact deploy commands, with every flag justified against
  this app's own measured numbers.
- **Deployment status: not live yet.** Was waiting on a GitHub
  Education → Azure for Students verification (a card-free option); check
  current status before assuming either GCP or Azure is the live target.

## 6. The most important bug found in this project's history (fixed, verified)

**Symptom that led to finding it:** a rebuilt scene (Agra, India) rendered
with wildly inconsistent, chaotic building heights — neighboring buildings
off by absurd amounts, looking like shattered glass rather than a skyline.

**Investigation:** compared Agra (no ground truth available) against
Mumbai Colaba (which DOES have real BMC municipal floor-count records).
Mumbai rendered clean; Agra didn't. This proved the chaos was a real
model-generalization problem on dense Indian layouts with no ground-truth
anchor — a genuine, hard, unsolved accuracy limitation, not a rendering bug.

**But then a second, much bigger problem was found while investigating
the first one:** tracing exactly how Mumbai's real ground truth reaches
the final model revealed that **it never actually did.**
`region_footprints.extract_from_polygons()` correctly computes each
building's real measured height (from BMC records) into `records[i]["height"]`
— but `build_city_image.py` only ever kept `rres["polygons"]` (the SHAPES)
from that function's return value. `rres["records"]` (containing the real
heights) was computed, its count printed to the console as a nice summary
("X buildings carry a REAL measured height"), and then **discarded**.
`mesh_repair.repair_build()` — which is what actually determines the
exported/rendered height — samples the depth-model-derived surface
(`dsm`) directly, with no known-heights parameter of its own. So every
building's FINAL height was always the noisy model guess, even for
buildings with a real municipal record, for as long as this code has
existed.

**Verified empirically, not just by re-reading the code:** checked the
actual exported heights in `buildings.geojson` for statistical evidence of
real floor-count data (real floor counts × a 3.3m storey height cluster at
exact multiples of 3.3). Before the fix: 9% of heights were near a 3.3m
multiple (consistent with coincidence). After the fix: **87.95%** are.

**The fix** (`build_city_image.py`, search "REAL BUG FIX"): after
extracting footprints with known heights, burn each measured building's
real height directly into the `dsm` array (surface model) BEFORE
`mesh_repair.repair_build()` reads it — the same "splice the real signal
in before anything downstream reads it" technique the codebase already
used for the segmentation mask. This means every later consumer of `dsm`
sees the real number as if it had always been there.

**Action item for whoever picks this up next:** the fix has only been
verified for Mumbai (`mumbai_colaba`, `mumbai_colaba_full`). Any OTHER
scene using `gt_building_polygons`/`gt_building_heights` should be
rebuilt and spot-checked the same way (does the exported height
distribution show the expected real-data signature?) before trusting it.

## 7. Training status — what's actually been run vs. what's still pending

**Be careful here — this project has twice already confused "code is
ready" with "the run happened."** Verify the checkpoint's own marker file
(`checkpoints/height_best/metric_ok.json` or
`checkpoints/height_htc_best/htc_ok.json`) and its `promotion_gates` field
before believing any accuracy claim about a checkpoint.

- **Currently licensed / in production:** `checkpoints/height_best/` — a
  plain-regression fine-tune, DA-V2-Base backbone, GAMUS + DFC2019, 400
  training tiles, real MAE 1.444m (no-affine) beating a 2.439m baseline.
  This IS the checkpoint behind the "3.132m per-building RMSE" headline
  number in §3.
- **HTC-DC Net (the "big swing" architecture upgrade):** implemented
  (`htc_bins.py`), but the only run that's actually completed is an 8-tile
  CPU smoke test (`htc_train_run1.log`) which correctly did NOT beat
  baseline — expected for a smoke test with almost no data, NOT a real
  result. **The real run has never been executed.** A ready-to-go
  notebook exists at `kaggle_htc_train.ipynb` (300+60 GAMUS tiles + DFC2019,
  20 epochs) but was never run to completion (Kaggle GPU quota issues).
  **This is very likely the actual task the user wants done on the new
  GPU laptop.**
- **Full-scale GAMUS availability, confirmed live:** the ENTIRE real GAMUS
  dataset is 6,557 tiles total (train 3,837 + val 859 + test 1,861) —
  verified by querying the live HuggingFace dataset repo directly. This
  matches the competitor's own claimed scale exactly, i.e. this is not a
  subset we're being stingy with, it's the actual ceiling of what's real.
  Do not train on the `test` split — it must stay untouched for honest
  final evaluation.

**To actually run the real HTC-DC Net training on a local GPU** (no Colab,
no Kaggle, no quota):
```bash
python gamus.py download --split train           # pulls the real, full 3,837-tile set (or --limit N for less)
python gamus.py download --split val
python train_height.py --htc --use-dfc2019 --epochs 20 --batch 8 --accum 2 --crops-per-tile 4
```
Batch/accum sizing above was tuned for a T4's ~15GB VRAM — raise `--batch`
on a bigger card, lower it on a smaller one. Watch for the two-gate
promotion report at the end (MAE improvement ≥2% AND correlation
non-regressing) — a run that fails either gate correctly does NOT get
licensed, and that is the correct outcome to report, not a failure to hide.

Also still real and pending, from the same research pass that identified
the HTC-DC Net direction (see `COMPETITOR_LANDSCAPE.md`'s research-frontier
section for full citations):
- **ICESat-2 sparse-LiDAR post-hoc correction** — module built and
  live-verified (`icesat2_source.py`), NOT yet wired into the actual build
  pipeline as a correction step.
- **Weak-label fine-tuning on Open Buildings/GlobalBuildingAtlas for
  Indian cities** — real, cited research recipe (18-23% RMSE reduction
  demonstrated in the literature for an analogous problem), not yet
  implemented. `open_buildings_source.py` (built and live-verified this
  session) is the real data-fetch half of this; the training-loop changes
  are not yet written.
- **Shadow-consistency auxiliary training loss** — cited research idea,
  not yet implemented at all.

## 8. Checkpoint integrity — a real, different, more advanced mechanism than file hashing

`depth_model.py` verifies a checkpoint's actual BEHAVIOR (not just its
file bytes) before trusting it: at load time, it runs a fixed synthetic
"canary" input through the model and compares the output against a
fingerprint recorded when the checkpoint was originally licensed. A
mismatch (tampering, corruption, a wrong-but-similarly-named file) gets
refused with a loud error, not silently accepted. This was built as a
deliberately different, arguably stronger technique than a competitor's
SHA-256 file-hash approach — it catches behavioral drift a hash can't, and
doesn't false-positive on harmless re-saves a hash would wrongly flag. See
`depth_model.py`'s module-level comment block above `CANARY_SEED` for the
full reasoning, and `train_height.py`'s calls to
`depth_model.license_checkpoint()` for where a fresh fingerprint gets
recorded.

## 9. Competitor landscape — read before claiming "we're ahead" of anyone

Three real competitors have been deep-audited, file by file (not just
READMEs), with findings saved so they survive context loss:
- `COMPETITOR_INTEL.md` — zaidnansari2011/sih2026-depthwizard (the
  strongest, most complete competitor found; real HTC-DC Net + GAMUS work,
  live deployment, rigorous evaluation discipline).
- `COMPETITOR_LANDSCAPE.md` — the full ~45-repo ranked sweep, plus deep
  audits of #2 (amogh-hub/depthwizard — Depth Anything 3, strong GCP
  calibration rigor, but their own custom model failed to beat baseline)
  and #3 (abxijth/Lorem-Ipsum-SIH26175 — smaller but honest, a real
  working A/B fine-tune methodology worth the pattern, not the code).

**Do not re-run the discovery sweep from scratch** — it's expensive (GitHub
API rate limits, ~30-90 min of agent time) and the ranked list is already
saved. Pick up from where these files leave off.

**Standing self-assessment** (as of the last time this was computed,
verify it's still current before repeating it): ahead of all three
competitors on frontend feature depth, checkpoint integrity, calibration
rigor, and process/anti-drift infrastructure. Not ahead on: a completed
large-scale architecture-upgrade training run (competitor #1 did this;
we haven't) and live deployment (competitor #1 has one; we don't yet).

## 10. Test suite

`tests/` — pytest, run with `python -m pytest tests/ -q`. As of the last
run: 30 tests, all passing. Includes real (not mocked) live-network checks
against the ICESat-2 APIs (skipped automatically if offline, never
silently passed as if they'd run). Run this before and after any change,
and specifically re-run it after touching `depth_model.py`, `train_height.py`,
`region_footprints.py`, or `build_city_image.py` — these have the most
subtle failure modes in the whole codebase.

## 11. Standing constraints — do not violate these

- **Never discuss SIH judging scores.** Not a technical constraint, a
  standing instruction from the project owner.
- **Keep the GitHub repo private.**
- **Never commit large/licensed data.** `.gitignore` documents exactly
  what's excluded and why (DFC2019 benchmark data, GAMUS subsets, source
  mosaics for the Karnataka/Mumbai ground-truth scenes) — read it before
  adding anything that looks like a data file.
- **Verify before claiming "done."** This document's whole existence is
  downstream of that rule being violated a few times and caught. Don't
  make it a fourth time.

## 12. If you only read one more thing after this

Read `EVALUATION_PROTOCOL.md` in full. It's short, and it is the single
best-encoded example of this project's real culture: exactly which
numbers are safe to cite, which are void, and why, written by someone who
got burned by not having this written down sooner.
