# DepthWizard — 24-hour completion checklist

Legend: [x] done and verified · [~] in progress · [ ] not started · [!] blocked (needs input)

## Accuracy (SIH evaluation 50%)
- [x] Retrained model run 02 (Depth-Anything-V2-Small, Apache-2.0): blind DFC RMSE 4.36 m (was 4.97), held-out per-building 2.19 m
- [x] Honest blind benchmark: every tile scored, zero-prediction floor reported
- [x] Per-pixel uncertainty: calibrated (held-out ECE 0.085, rank corr 0.72), `_sigma.tif` + viewer layer, verified in a real build
- [x] GCPs recalibrate the DSM file (tests pass; +3 m shift recovered to 0.06 m)
- [~] Hilly / forest / sparse LiDAR benchmark (NEON / USGS 3DEP) — agent running
- [~] ICESat-2 automatic GCPs + independent Indian accuracy — agent running
- [~] Tall-building physics fusion (relief displacement + shadows) — agent running
- [!] Run 03 training on all DFC2019 (needs Track 1 + Track 3 RGB-2/3 downloads)
- [ ] Resolution robustness test (0.3 / 0.6 / 1.0 m)

## Visualization & platform (SIH evaluation 50%)
- [x] Original input image draped by default (sharper imagery opt-in)
- [x] Lean fix: roofs/walls textured from the correct pixels
- [x] Rooftop structures (3,033), sliver buildings removed (170)
- [x] Frame-rate readout, getting-started guide, tooltips
- [x] Upload -> DSM -> view -> validate -> download (tested)
- [x] Standalone backend (embedded runtime) built + upload-tested; installer DROPPED by decision -- product will be hosted as a website
- [x] Periyar lake water surface (6 confirmed water bodies, translucent reflective layer at measured level)
- [ ] Dark slab artifact (HITEC) — source not yet found
- [ ] Time-machine 3D change detection — agent queued
- [ ] Analysis suite (solar potential, shadow simulation) — agent queued
- [x] Projection-accuracy measurement: HITEC roofs median 0.6 m / p75 0.84 m / p90 1.25 m to image edges (lean-corrected)
- [ ] Real-GPU frame-rate number (needs a run on a real browser)

## Documentation & submission
- [x] README / BENCHMARK / EVALUATION_PROTOCOL lead with blind numbers
- [ ] Technical report + user manual
- [!] SIH idea PPT (due 30 Sep) — team

## Also required for "fully finished" (beyond the 12 beast moves)
- [x] Projection-accuracy measurement (see above)
- [!] Real-GPU frame rate (open viewer with &fps=1 in Chrome)
- [ ] First-person walk: collision + smoothness check
- [ ] Technical report + user manual
- [ ] Awkward-file smoke test (tiny, huge, 16-bit, lat/lon, corrupt)
- [ ] Browser matrix: Chrome/Edge, WebGPU and WebGL
- [ ] Web deployment (host TBD): public server config, Docker image as the standalone deliverable
- [!] Accuracy parity with competitor on MAE/correlation (run 03 on DFC Track 1)
- [ ] Final consistency pass: rebuild scenes, re-run benchmarks, refresh all docs
- [ ] Demo script, 2-minute video, live-upload rehearsal, offline fallback
- [!] Final deliverables pack (source, docs, installer, evidence)

- [x] Pooled correlation measured like-for-like: 0.49 (competitor reports 0.84) -- behind; run 03 is the lever
