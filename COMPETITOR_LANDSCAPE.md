# SIH26175 competitor landscape — full sweep (2026-09-20)

Redo of a GitHub discovery sweep whose original ranked list was lost when an
earlier conversation was compacted. This file exists specifically so that
doesn't happen again — it is the durable record. `COMPETITOR_INTEL.md` (in
this repo) holds the deep audit of the #1 competitor,
`zaidnansari2011/sih2026-depthwizard`; this file starts from the assumption
that #1 is already known and finds/ranks everyone else.

## Methodology (and its limits — stated honestly)

- No GitHub token was available (`gh auth status` → not logged in; no
  `GH_TOKEN` in env). Everything below used **unauthenticated** GitHub REST
  calls: `/search/repositories` (10 req/min limit), `/repos/{o}/{r}/git/trees`
  and `/repos/{o}/{r}/commits` (60 req/hour core limit), plus
  `raw.githubusercontent.com` for READMEs/source files (no rate limit
  consumed).
- **`/search/code` requires authentication** and returned 401 unauthenticated,
  so the "search by code content, not just repo name" instruction could only
  be partially honored — covered by searching repo **description** text
  (which GitHub's repo-search indexes) for phrases like "single-view height
  estimation", "digital surface model", "GAMUS", "Cartosat", etc., rather than
  full file-content search across all ~90 repos' code.
- 20 repo-search queries were run across two angles: (a) name/keyword
  variants of "DepthWizard"/"SIH26175"/"SIH 2026", and (b) technical-content
  phrases independent of the team's chosen name (shadow geometry, monocular
  depth + DSM, nadir + DTM/DSM, GAMUS/DFC2019, Cartosat, digital twin).
  Angle (b) returned almost nothing new — nearly every real team named their
  repo some variant of "DepthWizard" or included "SIH26175"/"26175" in the
  name, which is itself informative (strong naming convergence around the
  official PS title).
- Total raw hits across all queries: **169 unique repos**. The overwhelming
  majority (~120) are **false positives**: the bare numeric query `26175`
  pulled in unrelated repos (Colombian SENA/ADSO coursework repos, Konflux
  perf-scale repos, a Sentry fork, Minecraft server configs, Project Gutenberg
  books) that happen to contain that digit string somewhere. Those are
  excluded below entirely — full raw JSON is kept in this session's scratch
  directory if anyone needs to re-derive the list, but they carry zero signal
  for this problem statement.
- After stripping noise: **~45 genuine SIH26175/DepthWizard-named repos**
  plus a handful of independently-named ones (TARA-3D, Elevate3d). Of those,
  the ~17 with a real description, non-trivial size, and recent commits got a
  lightweight pass (file tree via `git/trees?recursive=1`, README, commit
  count via the `Link` header's `rel="last"` page number as a cheap proxy for
  total commits).

## Likely official template repo (not a competitor)

- **`IMG-PROCESS-SAC/SIH-DepthWizard-2026`** — 23 stars, 1 file (README only),
  org name plausibly "Image Processing, Space Applications Centre" (ISRO/SAC).
  Almost certainly the official problem-statement repo/description that many
  teams starred and copy-pasted their own README text from — it explains why
  dozens of unrelated teams' repos share the identical phrase "Single-View
  Height Estimation and 3D Flythrough" verbatim. Not a competitor; noted here
  so it isn't mistaken for one later.

## Ranked list of real competitors (excluding us and zaidnansari2011)

Ranked primarily by commit count (a strong genuineness signal — most SIH
repos here have <10 commits, i.e. a handful of squashed bulk pushes),
secondarily by code depth (real pipeline/eval code vs. boilerplate) and
stated evaluation rigor. Stars are noted but not weighted heavily (most of
this pool has 0-2 stars regardless of quality).

| # | Repo | Commits (approx.) | Stars | What it does |
|---|---|---:|---:|---|
| **1** | **`amogh-hub/depthwizard`** | **~723** | 0 | Tauri desktop app; DA3MONO-LARGE (Depth Anything 3) monocular prior, hash/revision-pinned; DEM+GCP+fusion calibration with leave-one-out CV and fail-closed quality gates; Three.js GLB/LOD terrain with 5 navigation modes; ADR-driven architecture; formal "11/11 completion gate" acceptance-test framework; packaged macOS ARM64 release. **Deep-audited below.** |
| 2 | `abxijth/Lorem-Ipsum-SIH26175` | ~37 | 1 | `depthwizard` Python package (calibrate/dsm/georef/pipeline/srtm/validate); GAMUS-based fine-tuning with an A/B harness (`ab_finetuned_vs_baseline.py`); held-out validation across DC/NYC/Philadelphia tiles plus external landscapes (forest, hilly); documented de-risk tests and a `VALIDATION_LANDSCAPES.md`. Real research discipline, smaller in scope than #1. |
| 3 | `ArnabTechiee/depthwizard` | ~21 | 2 | Shadow-geometry-based height estimation (distinct technical approach — not monocular-depth-model-based) validated against **real USGS LiDAR**; tested on actual disaster imagery (Hurricane Ian, Turkey/Antakya earthquake, Cape Coral) — good disaster-response framing relevant to the ISRO use case. Narrower scope (single `app.py`, no desktop packaging). |
| 4 | `satyam15e-ops/TARA-3D` | ~44 | 1 | Independently-named (not "DepthWizard") — foundation-model depth + "deterministic solar physics"; FastAPI + Celery job queue, ONNX-exported depth engine, ISPRS-style GAMUS DC inputs, a technical-dossier PDF. Decent engineering, more product-shaped than research-shaped. |
| 5 | `vignesh-mahankali/DepthWizard-TrailBlazerz` | ~18 | 0 | Mostly a vendored copy of upstream `Depth-Anything-V2` plus wrapper pipeline scripts. Notable for testing on **real Indian disaster imagery** (Wayanad landslide, Kolkata) with real SRTM DEM data — good local relevance — but the depth model itself is unmodified upstream code, not original engineering. |
| 6 | `devendrakushwah80/DepthWizard-SIH26175` | **~8** (but huge tree) | 0 | On paper the most feature-complete after #1: FastAPI backend + React/Vite frontend with tests, GAMUS domain-stratified splits (forest/hilly/sparse/urban) with leakage checks, staged RDAH-net training (a0/a1/a2), M2/M3 checkpoints with SHA-256, model cards, GCP deploy scripts. **Caveat:** only ~8 commits despite 216 files — almost certainly imported/squashed from elsewhere rather than built incrementally in this repo, so the commit history can't be used to verify the work was actually done as claimed. Worth a second look if commit provenance can be clarified, but ranked below more verifiably-incremental repos for now. |
| 7 | `nandvilkararyan/DepthWizard` | ~7 | 1 | Flask-ish app + Unity integration (WebGL bridge, terrain mesh generator) — the only repo found using **Unity** instead of Three.js/Babylon for the 3D layer. Flood simulator module. Small Python core (`depth_extractor.py`, `scale_calibrator.py`). |
| 8 | `sagarkapoor03/DepthWizard` | ~4 | 1 | TypeScript, small (200KB) — disaster-management framing in description, but too little code to assess depth. |
| 9 | `PrinceKasundra06/DepthWizard` | ~6 | 0 | Built around **TripoSR** (single-object image-to-3D, not a DSM/elevation model) — a technical mismatch with the actual problem statement (object reconstruction vs. terrain elevation). Real FastAPI backend and frontend, but the core model choice suggests the team may not have converged on what the PS is actually asking for. |
| 10 | `yats0x7/DepthWizard` | ~25 | 1 | TypeScript, explicit "SIH 2026 PS 26175 (ISRO)" framing, moderate commit count, but repo is small (1.1MB) — likely frontend-only or early-stage. |

Repos seen but not deep-checked due to near-zero size/no description (almost
certainly empty scaffolds or personal forks, listed for completeness so a
future sweep doesn't have to re-discover them):
`Lucifer7636/DEPTHWIZARD`, `AjayBora002/Depth-Wizard`, `shyam883/depth-wizard`,
`ashumohanty678-beep/DEPTH-WIZARD`, `ParmarManthanrajsinh/DepthWizard`,
`Chethan26code/Depth_Wizard`, `10cpulkitpal-stack/depth-wizard`,
`AayanMohammed/DepthWizard-SIH26175`, `Nyxthron001/DepthWizard`,
`amulyadsouza-tech/DepthWizard` (desc. mentions DA-V2+SRTM but only 299KB),
`madhu-mitha-e/DepthWizard`, `GyaraManasa17/depth-wizard`,
`Subhraneel103/DepthWizard_SIH_26`, `Haresh-kumar28/DepthWizard` (good
description, only 257KB), `Notvishuuuu/SIH26175-DepthWizard-*`,
`CodyRohith7/Depthwizard-SIH26175`, `Bsaikumar-2007/depth-wizard-ai-*`,
`KalparunSarkar/DepthWizard_SIH_2026`, `preetham-reddy-n/Depth-wizard*`,
and dozens more one-person/empty repos from the same naming cluster.

Not competitors, excluded as noise: `CNES/cars` (a real, pre-existing French
space agency open-source photogrammetry tool with 398 stars — legitimate
software but not built for SIH26175, predates it, and is a general DSM-from-
stereo tool rather than monocular single-view); everything matching the bare
`26175` query that isn't SIH-related (SENA/ADSO Colombian coursework repos,
`smartinternz02` Salesforce badges, a Sentry fork, Minecraft configs, etc.).

---

# Deep audit: amogh-hub/depthwizard (2026-09-20)

Read directly from source via `raw.githubusercontent.com` (not just the
README): `src/depthwizard/geometry_prior/da3.py`,
`src/depthwizard/calibration/gcp.py`, `src/depthwizard/calibration/robust.py`,
`src/depthwizard/evaluation/building_height.py`,
`src/depthwizard/evaluation/potsdam.py`,
`src/depthwizard/height_model/confidence.py`, `docs/adr/0002-*.md`,
`docs/adr/0006-*.md`, `docs/potsdam-external-benchmark-v2-result.md`,
`docs/sih26175-problem-statement-traceability.md`,
`docs/final-sih-qualification.md`, `docs/third-party-baselines.md`, and the
full 316-file repo tree. Repo has **~723 commits** (by far the highest commit
count of any candidate found in this sweep, including possibly higher than
zaidnansari2011 — not confirmed, since that repo wasn't re-read per
instructions) — this is not a bulk-imported/squashed repo, it reads as
genuine iterative engineering.

## Model / geometry pipeline

- Production geometry prior is **DA3MONO-LARGE** (Depth Anything **3**, not
  V2) — a newer model generation than zaidnansari2011's DA-V2-Small choice.
  Apache-2.0 licensed. Pinned by exact HF revision hash
  (`f465978e618db8cc79c83b8bbf24964857db1875`) **and** by SHA-256 of the
  checkpoint file itself; `_resolve_production_snapshot()` refuses to load
  the model at all if the downloaded checkpoint's hash doesn't match —
  supply-chain integrity most hackathon teams don't bother with.
- Deliberate choice **not** to per-tile-normalize depth output
  (`depth_to_affine_height_evidence`): the docstring explains that
  percentile-normalizing each inference tile independently would "imprint
  the tile grid into the final surface," so raw affine evidence is preserved
  until an overlap harmonizer runs scene-global normalization. This is the
  same seam-artifact problem zaidnansari2011 also had to solve, solved
  differently (harmonize-then-normalize vs. zaid's per-tile heteroscedastic
  head).
- A custom "V4" learned refiner was built and evaluated for promotion over
  the frozen DA3 baseline — and **failed** its own predeclared promotion
  rule (see Evaluation below). Production still runs on frozen DA3 + affine
  calibration, not a custom-trained height-regression network.

## Calibration (`calibration/gcp.py`, `calibration/robust.py`)

- GCP calibration is affine (scale+offset), robustly fit via
  `robust_affine_calibration` (Huber/IRLS style, `require_positive_scale`).
- Real statistical gates before a calibration is accepted, all fail-closed:
  minimum GCP count (6 recommended, 4 hard floor), **convex-hull spatial
  coverage** check (rejects clustered GCPs), an **SVD-based collinearity/rank
  check** (rejects near-collinear GCP layouts), minimum anchor correlation
  `|r| >= 0.35`, maximum anchor RMSE, and maximum **leave-one-out
  cross-validation RMSE** — computed by literally refitting the affine model
  N times with each GCP held out.
- Orientation-flip detection: if GCP-vs-relative-height correlation comes out
  negative, the code inverts polarity before fitting rather than silently
  producing a negative-scale (physically meaningless) calibration.
- DEM+GCP fusion is architecturally constrained on purpose: re-fitting both
  scale and offset to a handful of GCPs on top of an already-scaled DEM
  surface "can distort every roof, tree, and slope in the scene," so GCPs are
  only allowed to correct a single **global vertical-datum offset** (weighted
  median of residuals) when fused with a DEM — a real methodology decision
  with stated reasoning, not an arbitrary simplification.
- Sparse GCP residuals are turned into a smooth low-frequency bias field via
  Gaussian-filtered impulse/support accumulation (Nadaraya-Watson-style
  kernel smoothing) rather than a single global offset when GCP-only.

## Evaluation discipline

- `evaluation/building_height.py` does **per-building-instance** evaluation
  (not just pixel RMSE): connected-component labeling of a building mask,
  per-instance height/top/ground error, roof dispersion, MAE/RMSE/bias plus
  within-1m / within-2m / "catastrophic >3m error" fractions.
- A formal **model-promotion gate** (`BuildingHeightPromotionThresholds`):
  a candidate model must show ≥15% MAE reduction, ≥10% RMSE reduction, ≥5%
  p90 reduction, zero degradation in within-2m fraction, zero increase in
  catastrophic-error fraction, and bounded top/ground MAE degradation — all
  simultaneously — before it's allowed to replace the frozen baseline. This
  is genuine MLOps promotion-gating discipline, unusual for a hackathon repo.
- **Independent external validation on ISPRS Potsdam** (a real, standard
  airborne LiDAR benchmark dataset — different provenance from their own
  GAMUS-based training/dev data): calibrated DA3 gets **5.432 m RMSE / 4.106
  m MAE** aggregate across 4 frozen tiles. Their own custom "V4" refiner was
  tested against this **frozen, pre-sealed protocol** (protocol hash recorded
  *before* touching the reference values) and came out **worse**: -0.09%
  RMSE, -0.27% MAE — reported honestly as "a valid negative result," with an
  explicit note that "the result must not be reframed as a learned-model
  win." This is a genuinely honest disclosure of a failed research direction,
  in the same spirit as zaidnansari2011 reporting their worse -6.94m Sikkim
  number instead of the flattering -1.15m one.
- Cites a real 2026 peer-reviewed baseline for comparison — RDAH-Net (Jiang
  et al., *Remote Sensing* 18(7):1024, 2026) — reproduced from a pinned
  upstream commit and a Figshare-DOI'd checkpoint, explicitly scoped as
  "not represented as DepthWizard-authored" and flagged as a cross-domain
  comparison (RDAH predicts nDSM-like height, their own benchmark pair is a
  DSM) rather than an apples-to-apples number. Real literature engagement.

## Engineering / release process

- Ships as a **packaged Tauri desktop app** (Rust + React/TypeScript
  frontend, Python sidecar) rather than a web server — a materially different
  deployment model from both us and zaidnansari2011 (both server-based).
- Self-imposed "11/11 completion gates" acceptance framework mapping every
  clause of the official SIH26175 problem statement to required evidence
  (`docs/sih26175-problem-statement-traceability.md`) and a fail-closed CLI
  checker (`scripts/check_sih26175_completion.py --strict`) that refuses to
  print "complete" unless dependency locks, packaged-build reproducibility,
  a 2-hour stability soak, ≥30 FPS sustained rendering evidence (mean AND
  p5, not "one rounded toolbar sample"), and clean-machine packaging all
  independently pass. Architecture Decision Records (7 of them) document
  reasoning for major calls (geospatial truthfulness, UI system, monocular
  mosaicking, terrain decomposition, etc.).
- CI via GitHub Actions (`ci.yml`, `release.yml`); dependency locks for
  Python (`uv.lock`), npm, and Cargo are committed and audited via
  `make reproducibility-audit`.

## Honest weaknesses / gaps

- **Packaged release is macOS ARM64 only** (the README's own badge says
  "standalone: macOS_ARM64"), and the DMG is explicitly **ad-hoc signed, not
  notarized** — a judge on Windows (the SIH-majority platform) cannot run the
  packaged app at all without building from source; Gatekeeper friction even
  on Mac. This is a real, self-disclosed distribution gap.
- **No production uncertainty output**: the README states plainly "no
  uncertainty raster is synthesized for the production path" — this is a
  meaningful capability gap versus zaidnansari2011's validated,
  coverage-tested per-pixel sigma display, which is one of the specific
  things the SIH rubric rewards (report uncertainty, not bare precision).
- The team's own custom learned model (**V4 refiner**) did not beat the
  off-the-shelf frozen DA3 baseline on external validation — unlike
  zaidnansari2011, which built and shipped a working custom architecture
  (twin mean/log-variance head + reimplemented HTC-DC Net) that is the actual
  production model. amogh-hub's production path is frozen DA3 + affine
  calibration; the custom-model research effort is a documented failure, not
  a shipped advantage.
- **No India-specific ground validation found** anywhere in the docs read
  (no Cartosat, Bhoonidhi, Sentinel, or Indian-region ground-truth dataset
  mentioned) — all evaluation is on GAMUS/DFC2019-style US tiles and ISPRS
  Potsdam (German airborne LiDAR). zaidnansari2011's Sikkim/Google-Open-
  Buildings validation is a real point of difference we and they both have
  that amogh-hub does not appear to.
- `devendrakushwah80/DepthWizard-SIH26175` (ranked #6 above) has a more
  fully-realized *web* product (FastAPI+React, deployed-looking) — amogh-hub
  chose a desktop app instead, which is a legitimate but judge-facing-risk
  design choice (nothing to click on the web during a demo; must install a
  DMG).
- 723 commits and heavy process language ("elite-finalization",
  "release-train-5", 11/11 gates) could indicate scope churn/over-engineering
  risk as much as diligence — the repo's own history shows real R&D failures
  being tracked (V4 rejection) rather than hidden, which is a point in its
  favor, but the sheer volume of ADRs/gates/traceability docs is itself a
  time investment that a hackathon team might not be able to replicate at
  their finale and could be over-indexed on process versus a working demo.

## Verdict

**Yes — amogh-hub/depthwizard is more advanced than our current repo in
specific, concrete ways**, and is the clear #2 competitor after
zaidnansari2011:

1. **Model provenance/integrity**: SHA-256-pinned, revision-pinned DA3
   (newer than DA-V2) with a hard refusal to load an unverified checkpoint —
   we should check whether our own model loading has equivalent integrity
   guarantees.
2. **GCP calibration rigor**: convex-hull spatial-coverage + SVD collinearity
   checks + leave-one-out CV RMSE as *gates*, not just diagnostics, is more
   statistically defensible than a simple affine/power-law fit alone.
3. **Model-promotion gating**: a formal, multi-metric, simultaneously-must-
   pass promotion threshold before any new model replaces a frozen baseline
   is a concrete MLOps practice worth adopting.
4. **Desktop packaging**: a signed, standalone Tauri app is a different (and
   for some judges more impressive) delivery mechanism than a web server —
   though its macOS-only + unnotarized status currently limits how much that
   actually helps them.

Where we (and zaidnansari2011) still have things amogh-hub does not:
production-path uncertainty quantification, a working custom-trained height
model that beats the off-the-shelf baseline, and India-specific ground
validation. Those remain genuine, currently-uncontested differentiators.

---

# Deep audit: abxijth/Lorem-Ipsum-SIH26175 (2026-09-20)

Read directly from source via the GitHub API/raw content, not just the
README: `depthwizard/{calibrate,depth,dsm,georef,pipeline,srtm,validate,
config}.py`, `scripts/ab_finetuned_vs_baseline.py`, `scripts/finetune_gamus.py`,
`scripts/_gamus.py`, `results/{DE_RISK_REPORT.md,VALIDATION_LANDSCAPES.md}`,
`results/{ab_finetuned/summary.json,heldout/validation_summary.json,
external/summary.json}`, `models/finetuned/meta.json`, `app/{main.py,jobs.py}`,
and the full 161-item repo tree. **37 commits**, all dated 2026-09-01 to
2026-09-03 (a ~3-day hackathon sprint, not a single squash-import): the
sequence is a genuine narrative — de-risk test → core pipeline → calibration
→ GAMUS fine-tune → external validation → README → web viewer/frontend
polish — with real per-commit content (e.g. `f5d58c04 test(phase0): de-risk
Depth-Anything on nadir imagery`, `ee39841d feat: fine-tune depth backbone on
GAMUS: −22% urban / −62% hilly / −31% forest RMSE`). This is the opposite
failure mode from devendrakushwah80 (216 files / 8 commits): here the file
count (161 items, ~100 MB of real data — 9 held-out GAMUS tiles + 2 external
landscape rasters + a 99 MB fine-tuned checkpoint) is matched by commit
granularity and message specificity. Verdict: **genuinely incremental,
verifiable work**, compressed into a short but real timeline.

## Model / pipeline architecture

- Backbone: **Depth-Anything-V2-Small-hf** (not V3, unlike amogh-hub's DA3).
  Locked in after an explicit Phase-0 de-risk test (`de_risk_test.py` /
  `de_risk_test_tiled.py`): full-tile inference on GAMUS gives **Pearson
  r ≈ 0.009** (no signal) because a single top-down 1024×1024 tile downscaled
  to the model's ~518px native res destroys texture; tiled inference (518×518
  patches, stride 400, cosine-feather stitched) recovers **r ≈ 0.325** across
  18 tiles / 3 cities (DC 0.285, NYC 0.159, PHL 0.533) — a real, quantified,
  city-dependent domain-gap finding, not an assumed win.
- Fine-tune (`scripts/finetune_gamus.py`): freezes all but `encoder.layer.10`,
  `encoder.layer.11`, `neck.*`, `head.*` (last 2 of 12 encoder layers + neck +
  head), trains against GAMUS AGL nDSM (metres, normalized by dividing by
  `NORM=45.0`) with **SmoothL1 loss**, AdamW with a 4x LR multiplier on the
  head vs. backbone (`lr=2e-4` backbone / `8e-4` head), fp16 autocast +
  GradScaler on CUDA, 518×518 random crops (patches-per-tile configurable,
  shipped meta.json shows 6 epochs × 20 patches/tile × 36 train tiles across
  DC/PHL/NYC). Straightforward, correctly-scoped for a laptop 8GB-VRAM GPU —
  not a from-scratch architecture, a targeted partial fine-tune.
- Calibration (`depthwizard/calibrate.py`): RANSAC-robust global linear fit
  (`structural_height = slope·depth + intercept`), with a per-class variant
  (`fit_calibration_classwise`, dormant — see below), a piecewise-linear
  variant (`fit_piecewise` / `fit_piecewise_segments`, quantile-binned,
  evaluated and **explicitly rejected**), and a GCP-refit path
  (`gcp_refit`: least-squares with ≥2 points, single-point intercept-only
  shift with 1). This calibration module alone is more complete than most
  competitors' entire pipelines.
- Absolute DSM = SRTM 30 m terrain baseline (self-hosted fetch from AWS
  `elevation-tiles-prod/skadi`, no GDAL CLI/API key, `.hgt.gz` parsing +
  `rasterio.merge`) **+** calibrated structural height. An `AGL` GeoTIFF tag
  convention correctly skips the SRTM add for tiles that are already
  structural-only (e.g. re-feeding GAMUS nDSM), avoiding a double-count bug
  class other repos might not think to guard against.
- Full FastAPI backend (`app/main.py`, `app/jobs.py`): upload → background
  worker → job polling → live GCP re-fit endpoint → Deck.gl/Three.js asset
  export → GeoTIFF download. Frontend (`static/js/{viewer,deck-view,main}.js`,
  vendored Three.js + Deck.gl, no npm build) genuinely ships **two** 3D views
  (Three.js first-person flythrough with fly/orbit + RGB/elevation/slope
  overlays, and a Deck.gl colorized point-cloud map view), a polygon
  region-select tool with per-pixel region statistics, and a "use region as
  GCP" live recalibration loop — this is a materially richer interactive
  demo surface than the initial lightweight-pass summary suggested.

## The A/B harness (`scripts/ab_finetuned_vs_baseline.py`) — verified real

This is genuinely the most scientifically honest part of the repo. It re-fits
calibration from scratch on the **fine-tuned** depth signal (fresh GAMUS
train tiles, excluding the 9 held-out ids) via the exact same
`fit_calibration` RANSAC routine used for the frozen baseline, then evaluates
on the same 9 held-out tiles with the same `depthwizard.validate.compare`
comparator — an apples-to-apples swap of only the depth signal, calibration
protocol held fixed. Numbers pulled directly from
`results/ab_finetuned/summary.json` (this specific run: `--fit-per-city 8`):

| | RMSE (m) | MAE (m) | Pearson |
|---|---|---|---|
| Frozen baseline (`baseline_ref` in the same file) | 5.63 | 4.51 | 0.255 |
| Fine-tuned + recalibrated (this run) | 4.33 | 3.11 | **0.611** |

The shipped/README numbers (fit with `--per-city 20` via
`fit_calibration_finetuned.py`) are RMSE 4.37 / MAE 3.17 / Pearson 0.61 —
consistent with the A/B script's own smaller-fit-set run, i.e. the two
independent runs agree within noise. The **Pearson jump (0.26 → 0.61) is the
real story**, not the RMSE drop — it means the fine-tune fixed the
structural-signal problem (weak/absent correlation) that no amount of linear
recalibration of the frozen backbone could fix, which is exactly the
question the script's own docstring says it's testing. Per-tile output shows
the improvement is **not uniform**: DC and NYC_00921 stay the weakest
(Pearson 0.42–0.66) while PHL improves most (0.62–0.71) — reported honestly,
not smoothed over.

External landscape numbers (`results/VALIDATION_LANDSCAPES.md`,
`results/external/summary.json`) for the same fine-tuned pipeline vs. frozen
baseline (frozen numbers in parentheses, sourced from the same file):

| Landscape | RMSE (m) | Pearson | Note |
|---|---|---|---|
| Hilly (Asheville, NC) | **3.08** (8.05) | 0.99 | SRTM (terrain), not depth model, sets the floor |
| Forest (Great Smoky Mtns) | **8.11** (11.79) | 0.999 | canopy only *partially* captured (~8 m of ~21 m mean GLO-30 canopy) |

Matches the commit message's claimed −22%/−62%/−31% urban/hilly/forest RMSE
reduction almost exactly (4.37/5.63=−22.4%, 3.08/8.05=−61.7%, 8.11/11.79=
−31.2%) — the headline claim is real and internally consistent across the
README, the commit message, and the raw JSON, not spun.

## Held-out validation — real but narrower than it first reads

`scripts/_gamus.py` shows the actual split mechanics: train tiles are drawn
from GAMUS's own `train` split; the 9 held-out tiles are GAMUS's own
`val`/`test` split (`DC:val, PHL:val, NYC:test` — NYC has no val split). This
is a **genuine tile-disjoint split** (the fine-tune never sees the held-out
pixels), not leakage in the crude sense. But it is **not city-disjoint** —
DC/NYC/Philadelphia appear in both the fine-tune's training data and the
"held-out" evaluation set, just as different tiles of the same city. So the
claimed "held-out across DC/NYC/Philadelphia" generalizes across *tiles*
within the same 3 cities, not to a genuinely new city — the real
out-of-distribution test is the external forest/hilly cases (different
country region, different imagery source, different reference-DEM
provenance), and those are honestly labeled as SRTM-terrain-ceiling-limited
rather than depth-model wins.

## Other honest engineering discipline (decision gates)

The repo runs a small but real "promote or reject" experimental discipline,
documented with numbers rather than assertions:
- **Per-class oracle-ceiling probe** (`scripts/probe_per_class_ceiling.py`,
  `results/probe/ceiling.json`): an oracle upper bound for per-class
  (segmentation-conditioned) calibration gives only a **9.6% RMSE gain** —
  below a stated 10% "promote" bar, so a segmentation head was **not
  trained** (`models/gamus_seg.pt` referenced in config but never shipped;
  per-class calibration code exists in `calibrate.py` but is dormant/unused
  in production). Explicitly reported as a negative result, not hidden.
- **Piecewise calibration** (`fit_piecewise`, `fit_piecewise_segments`,
  `results/piecewise/summary*.json`): tested, found to help some cities
  (−7.4%) and hurt others (DC/NYC regress), and **rejected** in favor of the
  simpler global RANSAC line.
- **GCP refit**: verified end-to-end on held-out PHL, RMSE 3.17 → 1.86 m with
  2 clicked points — a concrete number for what the "minimal GCP" feature
  the problem statement asks for actually buys.

## Honest weaknesses (self-reported and found independently)

1. **Same-city held-out, not cross-city.** As above — the fine-tune's
   generalization claim is real but narrower than "region-disjoint" implies;
   the true out-of-domain test (forest/hilly) is smaller (2 cases, 1 tile
   each, ~1 M pixels) and dominated by the 30 m SRTM terrain floor rather
   than the depth model's own accuracy.
2. **Forest canopy capture is explicitly partial** (README's own words): ~8 m
   recovered vs. ~21 m mean GLO-30 canopy height — the team states this
   plainly rather than rounding it up.
3. **Single-model, single-GPU serialization.** `app/jobs.py` uses one global
   `threading.Lock()` around the whole pipeline (`_worker_lock`) — jobs are
   processed strictly one at a time system-wide, fine for a hackathon demo,
   a real scale bottleneck if judges hit it with concurrent uploads.
4. **Small n throughout.** 9 held-out urban tiles, 2 external landscape
   cases, ~36-tile fine-tune set — every number in the README is honestly
   reported but statistically thin; per-tile variance (Pearson 0.42–0.71
   across just 9 tiles) is wide enough that a different random held-out
   split could move the headline numbers noticeably.
5. **No model-integrity/provenance guardrails.** Unlike amogh-hub's SHA-256
   +revision-pinned model loading, this repo's `depth.py` loads
   `models/finetuned/pytorch_model.bin` with a bare existence check and
   `strict=False` — no hash verification, no promotion gate comparing a new
   checkpoint against the currently-shipped one before it becomes default.
6. **No desktop packaging / no India-specific ground validation** — same gap
   as amogh-hub, and the external landscape cases (Asheville NC, Great Smoky
   Mtns) are US-based like the GAMUS training data itself, not
   India-relevant.
7. Two "first commit" entries and a near-identical early commit series
   suggest a squash/re-init early in the 3-day window — cosmetic, doesn't
   affect the credibility of the later incremental history.

## Verdict

**Partially more advanced than our current repo, in one specific technique
worth adopting — not the whole pipeline.** abxijth is a smaller, more
narrowly-scoped effort than #1 (amogh-hub) and clearly research-first rather
than product-first, but it does one thing better than most competitors
including (as far as this audit can tell without re-reading our own repo in
this pass) what a frozen-backbone approach gets you for free:

1. **The fine-tune-vs-frozen A/B methodology is the adoptable idea**, not
   their specific checkpoint: hold the calibration protocol fixed, swap only
   the depth signal, and report Pearson (not just RMSE) to isolate whether a
   backbone change fixes a *structural signal* problem vs. just rescaling
   noise. If our depth backbone is frozen off-the-shelf, this is a cheap,
   well-specified experiment (partial fine-tune of the last 2 encoder layers
   + neck + head against nDSM, SmoothL1 loss) that this repo shows produces
   a real, honestly-verified Pearson 0.26→0.61 jump on GAMUS — worth
   running ourselves rather than assuming a frozen backbone is sufficient.
2. **Numeric promote/reject gates** (the 10%-RMSE-gain bar for segmentation,
   the piecewise-calibration rejection) is a good honesty/rigor pattern for
   our own experiment log, independent of whether we adopt their specific
   architecture choices.
3. Everything else — DA-V2-Small (not V3), no model-integrity gating, no
   India-specific validation, single-worker web backend, thinner held-out
   set — is weaker than or roughly at parity with what a competitive repo
   needs, and clearly weaker than amogh-hub (#1) on process maturity and
   scope. Not a repo to copy from wholesale; the A/B-harness discipline and
   the decision-gate habit are the two things worth taking.
