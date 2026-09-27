# DepthWizard — 24-hour completion checklist

Legend: [x] done and verified · [~] in progress · [ ] not started · [!] blocked (needs input)

## Accuracy (SIH evaluation 50%)
- [x] Retrained model run 02 (Depth-Anything-V2-Small, Apache-2.0) + product fix: blind DFC RMSE 2.82 m / MAE 1.59 m / pooled corr 0.787 (was 4.97 / 2.82 / 0.51), per-building 2.19 m
- [x] Honest blind benchmark: every tile scored, zero-prediction floor reported
- [x] Per-pixel uncertainty: calibrated (held-out ECE 0.085, rank corr 0.72), `_sigma.tif` + viewer layer, verified in a real build
- [x] GCPs recalibrate the DSM file (tests pass; +3 m shift recovered to 0.06 m)
- [~] Hilly / forest / sparse LiDAR benchmark (NEON / USGS 3DEP) — paused partway (will resume, no agents)
- [~] ICESat-2 automatic GCPs + independent Indian accuracy — paused partway (will resume, no agents)
- [~] Tall-building physics fusion (relief displacement + shadows) — paused partway (will resume, no agents)
- [!] Run 03 training on all DFC2019 (needs Track 1 + Track 3 RGB-2/3 downloads)
- [x] Resolution robustness: 0.25->1.0 m input, RMSE 2.85->2.97 m, corr 0.75->0.73 (raw model, held-out tiles)

## Visualization & platform (SIH evaluation 50%)
- [x] Original input image draped by default (sharper imagery opt-in)
- [x] Lean fix: roofs/walls textured from the correct pixels
- [x] Rooftop structures (3,033), sliver buildings removed (170)
- [x] Frame-rate readout, getting-started guide, tooltips
- [x] Upload -> DSM -> view -> validate -> download (tested)
- [x] Standalone backend (embedded runtime) built + upload-tested; installer DROPPED by decision -- product will be hosted as a website
- [x] Periyar lake water surface (6 confirmed water bodies, translucent reflective layer at measured level)
- [ ] Dark slab artifact (HITEC) — source not yet found
- [ ] Time-machine 3D change detection — not started
- [ ] Analysis suite (solar potential, shadow simulation) — not started
- [x] Projection-accuracy measurement: HITEC roofs median 0.6 m / p75 0.84 m / p90 1.25 m to image edges (lean-corrected)
- [ ] Real-GPU frame-rate number (needs a run on a real browser)

## Documentation & submission
- [x] README / BENCHMARK / EVALUATION_PROTOCOL lead with blind numbers
- [x] Technical report (TECHNICAL_REPORT.md) + user manual (USER_MANUAL.md)
- [!] SIH idea PPT (due 30 Sep) — team

## Also required for "fully finished" (beyond the 12 beast moves)
- [x] Projection-accuracy measurement (see above)
- [!] Real-GPU frame rate (open viewer with &fps=1 in Chrome)
- [x] First-person walk: ground-following at 1.7 m eye height, 1.2 m step limit, walls block and slide, G toggles flight (self-test: 16.7 m in 3 s, eye 1.5-1.9 m)
- [x] Technical report (TECHNICAL_REPORT.md) + user manual (USER_MANUAL.md)
- [ ] Awkward-file smoke test (tiny, huge, 16-bit, lat/lon, corrupt)
- [ ] Browser matrix: Chrome/Edge, WebGPU and WebGL
- [ ] Web deployment (host TBD): public server config, Docker image as the standalone deliverable
- [!] Head-to-head on competitor protocol (DFC Track 1) + run 03 -- downloads in progress
- [ ] Final consistency pass: rebuild scenes, re-run benchmarks, refresh all docs
- [ ] Demo script, 2-minute video, live-upload rehearsal, offline fallback
- [ ] Final deliverables pack (source, docs, evidence, Docker image)

- [x] Pooled correlation like-for-like: 0.787 after the product fix (competitor reports 0.84 on their tiles)
