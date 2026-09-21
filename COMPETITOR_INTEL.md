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

## Re-audit, 2026-09-20 — file-by-file re-read, checking for drift since the above

**Procedural finding first: no commits since 2026-09-12.** Their own
`docs/submission-plan.md` (last touched in that same commit) plans work
through Sep 28 — a tall-building fix, GSD-generalization test, demo video —
none of which is in the public repo yet. Either they're working locally
without pushing (against their own written rule #7) or paused. As of today
their tall-building fix and demo video are still genuinely missing, not
stale intel on our part.

**New detail on the HTC-DC head** (already known at a high level above, but
deeper on read): adaptive bin widths come from a small MLP on global
mean/std features (softmax-normalised, summing to the full height range);
Chamfer loss supervises bin EDGES not centres (a correction from an earlier
failed attempt that copied AdaBins' centre-supervision); the head/background
gate is routed by ground truth during early warmup because a
randomly-initialised 1x1-conv gate was measured sending 0% or 99.4% of
pixels down one branch at init. Also present: heteroscedastic NLL with
optional beta-NLL reweighting (Seitzer et al. 2022) and a gradient-matching
loss on first differences for sharp roof edges. The full loss is a five-term
composite (NLL + gradient-matching + Chamfer + distribution-KL + gate CE).

**GAMUS data fusion — missed entirely in the original audit.** They unioned
DFC2019 with GAMUS (CC-BY-4.0 aerial nDSM/RGB, 6,557 usable tiles) to attack
their diagnosed weak point: DFC2019 alone has only 166 buildings above 30m,
GAMUS gave them ~6,060. Pre-registered pass/fail criteria before training
(>20m bias must beat -13m). **The primary criterion still failed** (-15.6m
vs the -13m bar) and they shipped anyway, with a paired-bootstrap
significance check (20,000 resamples) and an explicit "failed but shipped"
writeup rather than moving the goalposts. Real cross-domain transfer number:
union model 3.497m RMSE / ECE 0.044 on a spatially-disjoint GAMUS holdout vs
DFC2019-only model's 6.873m / ECE 0.231 — careful to label this
domain-adaptation, not generalization. Caught four silent GAMUS data traps
(inverted class codes, a -5.00 void sentinel, leaking official splits,
missing NYC imagery) before a run could be corrupted by them.
**Net: even after this whole campaign, their tall-building problem remains
open** — which is exactly the failure mode our real municipal-record fusion
targets directly, with measured ground truth rather than more inferred data.

**Viewer (`main.js`) — undersold in the original pass.** "Custom terrain
raycasting" is specifically Amanatides & Woo's 1987 DDA voxel-traversal
algorithm, adopted after measuring three.js's default `intersectObject()`
walking every one of ~2.1M triangles per hover frame. Separately, real
demo-hall hardening not previously noted: WebGL context-loss recovery
(`webglcontextlost`/`restored` handlers — a projector swap or driver reset
would otherwise freeze the frame silently), a no-GPU preflight check with
actionable remediation text instead of a stack trace, touch/pinch support
(pointer-lock doesn't exist on mobile), and a HUD layout rewritten after it
measured 216px of overflow on a 1366x768 hall laptop. Their commit messages
literally say "make the viewer survive a demo hall" — this overlaps directly
with our own live-demo-reliability concern and is worth checking our own
viewer against the same failure modes (small/low-res display, GPU driver
hiccup, touch input) before finals.

**`serve_app.py` — a sharper upload bug than previously known.** GDAL/rasterio
was found to silently decode a PNG truncated to a third of its bytes,
padding missing rows with black — a confident height map over fabricated
pixels, no error raised. Fixed by forcing a strict PIL decode first (PIL
raises on truncation where GDAL doesn't). This is a submission-integrity bug,
not a cosmetic one: an interrupted judge upload would silently produce a
fabricated result. Worth checking whether our own upload path in `server.py`
has the same blind spot.

**`infer.py` TTA is more than averaging.** `--fuse-zoom` does a
Laplacian-pyramid-style multi-scale fusion (a zoom-1 pass for absolute
calibration/low frequencies, a zoomed pass for high-frequency detail,
Gaussian-blur frequency-split blend; measured +29% sharpness for +1% RMSE).
D4 TTA's 8-view disagreement is fed into the reported uncertainty as a
genuine epistemic term, not just averaged into the mean.

**`metrics.py`**: Spearman rank correlation between predicted sigma and
realised |error| (0.877 on their shipped run), justified explicitly because
"calibration can look fine on average while sigma is uninformative pixel to
pixel" — a real hole a plain coverage/ECE curve alone misses. Cheap to add
to our own uncertainty reporting if we want the same rigor.

**Balance — real, current weaknesses on their side:**
- Tall-building bias remains unsolved (-15.6m) despite the full HTC-DC +
  GAMUS campaign — a structural, still-open problem for them.
- No training data above 50m exists anywhere they've found (GAMUS tops at
  44.5m); they state this ceiling "survives this work untouched."
- Forest canopy ground error (~2.47m) attributed to optics, called unfixable
  by more data.
- Viewer's `calibration.json` still points at `run02/best.pt` while every
  other artifact is on `run07` — an acknowledged, unresolved inconsistency
  as of their last commit.
- Their Sikkim hilly-terrain co-registration is admitted noise (peak
  correlation 0.006) — hilly-terrain "accuracy" there is visual/label
  agreement, not a measured claim, weaker evidence than urban/sparse.
- GSD-generalization to ISRO's actual ~0.6-1.0m resolution is untested on
  their current (run07) checkpoint; they train at 0.3m and call this
  "Tier 4, first to cut if time runs short."
- No demo video exists yet, by their own words "the single largest hole" in
  their submission.
- Deployment has had real incidents (a docker-build-context mistake caused a
  two-day outage), not just the one credit-line bug previously known.
- Model still reloads per request (~8s cost) — unresolved, as before.

**Bottom line.** Their self-assessment ("not a team with a better model,
because they collected the half we neglected") is now dated — they've
meaningfully upgraded the model side (real HTC-DC Net, GAMUS fusion) while
keeping the viewer/rigor edge already known. But their hardest problem
(tall buildings) is still open even after that whole campaign, and it's
exactly the failure mode our real municipal ground-truth fusion targets
with measured data instead of more inferred data — that distinction still
holds and is arguably sharper now that their answer to the same problem
(more inferred training data) demonstrably wasn't enough on its own. Two
things worth adapting by technique, not code: (1) a head-tail/adaptive-bin
structural approach to the long-tailed height distribution could compound
with our fusion override rather than compete with it; (2) their
falsification-first, pre-registered-criteria, bootstrap-significance
evaluation discipline is a credibility practice worth matching regardless
of what we build next, and costs nothing to adopt.

## Standing gap list vs. zaidnansari2011, as of 2026-09-20 (post frontend/backend sprint)

Frontend and backend-reliability work this session closed most of the gap
list that existed before it (walk-mode, compare-slider, scene-query,
video export, health/readiness checks, upload integrity, DEM circuit
breaker, result caching -- all real-browser or live-tested, not just
written). What's left, honestly:

1. **Model/architecture: they're ahead, not us.** Their HTC-DC Net + GAMUS
   fusion is real and shipped (still didn't solve tall buildings, but it
   ran, at their full ~6,557-tile scale). Our own HTC-AdaBins attempt has
   only run as an 8-tile CPU smoke test -- the real Kaggle-scale run
   (300+60 GAMUS tiles, `kaggle_htc_train.ipynb`) has not actually executed
   yet. Do not claim parity here until that run lands and is honestly
   scored against baseline.
2. **Live public deployment: they have one today, we don't yet.** Blocked
   on GitHub Education / Azure student credit verification (72h wait,
   started ~2026-09-20). Closes on its own once that clears -- see
   `DEPLOY.md`.
3. **Multi-point GCP calibration is a diagnostic on our side, not a
   correction.** Our GCP tool now detects and reports a height-dependent
   error curve (verified against a known synthetic exponent), but the
   "Apply" action still uses one flat ratio for the whole scene. Applying
   a true per-vertex power-law correction to actual geometry is a real,
   larger, riskier change than what's been done -- not attempted.
4. **SIGTERM graceful-shutdown handling is written but unverified.**
   Correct-looking code (signal handler -> drain -> `srv.shutdown()`), but
   could not be tested live on the Windows dev machine (no real POSIX
   signal delivery, no WSL). Needs a real check once actually running on
   the Linux deploy target.
5. Everything else previously tracked here (demo-hall viewer hardening,
   corrupted-upload handling, uncertainty-error rank correlation via the
   scene-query/compare-slider work, cartographic furniture, walk-mode) is
   now closed and verified -- see the Day 1-3 session work for specifics,
   not restated here to avoid this file going stale the way the old
   headline numbers did.
