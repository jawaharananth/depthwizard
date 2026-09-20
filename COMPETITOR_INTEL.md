# zaidnansari2011/sih2026-depthwizard — full findings (2026-09-19/20)

Consolidated from a full read of their repo: model.py, losses.py, PLAN.md (1066
lines), literature.md, evidence-pack.md, evaluation-protocol.md, deployment.md,
serve_app.py (825 lines), viewer/main.js (2082 lines), and the GCP calibration
work. This is the single strongest competitor found across all ~90 SIH26175
repos scanned this session. Kept here so later decisions can cite specifics
instead of a vague "they're good."

## Model architecture
- DA-V2 backbone + DPT neck + twin (mean, log-variance) head — heteroscedastic
  Gaussian NLL, not MSE. Deliberately uses DA-V2 **Small** (Apache-2.0) over
  Base/Large (CC-BY-NC), because the larger variants can't ship in a deliverable.
- HTC-DC Net (Chen et al., IEEE TGRS 2023) reimplemented: adaptive per-image
  height bins, head-tail-cut separating foreground/background distributions,
  distribution-based KL constraint with an analytically-solved reference sigma.
- Real root-caused bug history: a 9-GPU-hour run (run05) was voided because
  DA-V1-Large's neck emits residuals ~40x larger than DA-V2-Small's, driving
  their NLL into a dead clamp — found via a dedicated diagnostic script.

## Evaluation discipline
- Three separate evaluation protocols (in-training val / whole-tile val /
  whole-tile test) documented explicitly after two numbers from the same
  checkpoint looked contradictory (8.22 vs 3.42 RMSE) — turned out to be
  different splits, not a bug. `docs/evaluation-protocol.md` exists specifically
  so this doesn't happen again.
- Region-disjoint train/val/test splits, held out from day one, never tuned
  against ("Standing rule" #5).
- Old pre-`--split`-flag artifacts explicitly marked "spent" so nobody
  accidentally uses a contaminated number to pick a checkpoint.

## The India validation (Google Open Buildings, Sikkim)
- Real quantitative result on Indian ground: Maxar Open Data over the Teesta
  valley vs. Google Open Buildings 2.5D Temporal `building_height`.
- Found their own comfortable number (-1.15m bias) was an artifact of a
  dilated/blurry footprint mask. Re-checked by filtering to Open Buildings'
  own high-confidence pixels only, and the honest number got WORSE: -6.94m on
  buildings both models agree are confident. Reported the worse number as
  "the one that actually matters," explicitly refusing to quote the flattering
  one.
- Explicitly tested for terrain-slope leakage (AGL should be terrain-invariant;
  hilly Sikkim tests this, flat Jacksonville/Omaha never could) and proved it
  wasn't happening (r = -0.044 against slope).
- Caveat they state themselves: this is "model versus model" — Open Buildings'
  heights are themselves inferred from Sentinel-2, not surveyed. Weaker
  evidence than real ground truth by their own admission.
- Cross-validated finding: buildings are under-predicted by ~7m, independently,
  on two continents (US DFC2019 + India Sikkim), against two different
  reference datasets.

## GCP (ground control point) calibration
- Found affine correction is the wrong functional form — their height error is
  a RATIO (1.20x at 0-3m, 0.52x above 40m), not a constant offset. A power-law
  fit beats affine: -24.9% RMSE vs affine's -15.1%, with less MAE cost.
- Guard thresholds (minimum control points, minimum span) chosen from measured
  false-positive rates on real data, not picked arbitrarily.
- Unit test recovers a known synthetic exponent (1.3889) to 7 decimal places.
- Fails closed: on tiles where the fit is untrustworthy, falls back to
  offset-only and states why, rather than forcing a bad correction through.

## Backend / deployment (real, live, at project5.zaidansari.tech)
- Hand-rolled `ThreadingHTTPServer` (no FastAPI/Flask) — deliberate minimalism
  for a CPU-only free-tier host.
- Rate limits (upload size, megapixel ceiling, per-IP spacing, queue depth) are
  all DERIVED from a measured cost formula (`FIXED_S + mpx × SEC_PER_MPX`), not
  guessed numbers.
- Checkpoint stripped for serving: 297MB (full training state) → 99.2MB
  (inference-only weights + just enough metadata to name itself).
- Caught and fixed a real bug: a stale deployment was missing a Maxar CC-BY
  image-credit line — a licensing problem, not a cosmetic one — because a
  manual copy-across had gone stale. Built a script specifically so a redeploy
  always refreshes from the repo, not from a possibly-stale staging copy.
- Documents its own gaps honestly: model reloads every request (~8s cost),
  one crashing `--help` flag, DEM anchoring off by default.

## Frontend (viewer/main.js, 2082 lines, hand-rolled, no framework)
- Real georeferencing readout (lon/lat, easting/northing) at any point.
- Live uncertainty (sigma) lookup at any hovered point, not just at a click.
- Custom terrain raycasting for pixel-accurate picking (not a stock Three.js helper).
- Flood simulation with a seed point and a rising-water-level control.
- A compare mode with a draggable screen divider (prediction vs. reference side by side).
- Real cartographic furniture: scale bar, compass, slope shading.
- Pointer-lock mouse capture for a proper first-person flythrough.
- Uncertainty bar is validated, not just displayed: separation-dependent
  coverage table showing the confidence interval is honest at 3-15m
  separation (71.4% coverage, close to nominal) and breaks down outside that
  window (51.7% at 60-300m) — disclosed, not hidden.

## Team process (their own written "Standing rules," 7 items)
1. Viewer must never be the last night's work — it's 50% of the score.
2. Report uncertainty, never bare precision.
3. Follow ISRO's exact output spec, even over a better model.
4. Build only what you can explain live at the finale.
5. Hold out a test set from day one; never tune against it.
6. Preprocess locally, train on Kaggle, deploy on Azure.
7. No code living only in notebooks — everything comes from git.

## Self-diagnosed weakness (their own words, section "7.6 What actually beats us")
> "Not a team with a better model. A team with a mediocre model and a
> beautiful viewer, because they collected the half we neglected. Or a team
> that followed the output spec while we didn't."

## Real ISRO portal findings (Bhoonidhi) — useful intel for us too, not just them
- Tried real Cartosat integration via ISRO's own Bhoonidhi STAC API.
- Hard ceiling found: no sub-metre optical Cartosat imagery available via the
  public API at all. Best optical is LISS4 at 5.8m (too coarse for building
  height — a 19x GSD gap vs their 0.3m training data).
- CartoDEM (30m posting) has 8m LE90 vertical accuracy — worse than their own
  1.92m ground RMSE, and measures absolute elevation, not AGL, so it can't
  even validate building heights directly.
- Relevant to us: if we ever plan to claim "real Cartosat integration" as a
  roadmap item, this is the actual current ceiling, verified by someone who
  tried it for real, not assumed.

## What's genuinely still different about our approach (not superseded by the above)
- Our BMC data is real municipal PROPERTY-SURVEY floor counts — actual
  measured ground truth. Their Open Buildings comparison is against another
  model's inference (Sentinel-2-derived), which they themselves call weaker
  evidence. The "real vs. inferred ground truth" distinction still holds.
- We have real municipal-record + Open-Buildings FUSION (known heights
  override the model per-building). They don't fuse; they compare.
