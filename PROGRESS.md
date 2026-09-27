# DepthWizard — master checklist (to first place)

Legend: [x] done and verified · [~] in progress · [ ] not started · [!] blocked (needs input) · [-] dropped by decision

Ground rules from ISRO's FAQ (22 Sep 2026, github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026/issues/1):
final evaluation on hidden **Cartosat-2S 0.6 m** images; GeoTIFFs scored as an **absolute DSM against
SRTM / Copernicus**; PNG/JPG must still work; ranked against other teams; first submission is the
**PPT only** (problem, idea, architecture + one preliminary-work slide).

## Already done
- [x] Retrained model (run 02) + raw-height product: blind DFC RMSE 2.82 m / MAE 1.59 m / corr 0.787; per-building 2.19 m
- [x] Honest blind benchmark: every tile scored, zero-prediction floor reported
- [x] Per-pixel uncertainty: ECE 0.085, rank corr 0.72, `_sigma.tif` + viewer layer
- [x] Ground control points recalibrate the DSM file (6 tests pass)
- [x] Resolution check 0.25 -> 1.0 m: RMSE 2.85 -> 2.97 m
- [x] Original image draped by default; lean-corrected roof/wall texturing
- [x] Rooftop structures (3,033); sliver buildings removed (170); Periyar water surface
- [x] Frame-rate readout, help guide, tooltips; first-person walk with collision
- [x] Upload -> DSM -> view -> validate -> download, tested
- [x] Projection accuracy: median 0.6 m / p90 1.25 m
- [x] Awkward-file smoke test: 8 file types incl. 16-bit and lat/lon (16-bit fix in server.py)
- [x] README / BENCHMARK / EVALUATION_PROTOCOL / TECHNICAL_REPORT / USER_MANUAL
- [-] Desktop installer (product will be hosted as a website)
- [-] Time-machine change detection (left, by decision; uncommitted code stays in server.py / viewer)

## Today — the whole technical prototype
- [x] 0. BUG FIXED: predict_tiled fitted metric crops to a ~518 px whole-image prediction -> large scenes flattened (HITEC 4096 px p99 0.61 m). A/B on 18 held-out tiles: alignment HELPS at 2048 px (per-building 1.86 vs 2.09 m), so it is kept within ~2048 px blocks and large scenes are blended block-wise; the model now always runs on the SOURCE pixels at the trained 0.3 m (one pass feeds both the full-res DSM file and the 3D scene)
- [x] 0a. ABSOLUTE-DSM FIX: ground = smooth30(DEM) - k x smooth30(model height), surface = ground + model height (anchoring.py); k measured on swisstopo LiDAR: 0.4 GLO-30, 0.0 SRTM. Old DSM RMSE hilly 147.7 / forest 65.7 / urban 16.4 / sparse 10.7 m -> new 9.5 / 8.8 / 6.9 / 2.9 m (GLO-30 alone 11.1 / 9.6 / 8.4 / 3.6). SRTM fetch fixed
- [~] 0b. Run 04 RUNNING on Kaggle: run 03 recipe + ~330 km2 Swiss LiDAR tiles (tall buildings, 30-40 m forest, Alps), benchmark tiles + 1 km buffer held out
- [x] 1. Cartosat input handling: B,G,R,NIR band order, per-band 16-bit stretch, nodata/black-collar mask, very large scenes, tests -- band order + nodata DONE (8 tests pass; synthetic 16-bit B,G,R,NIR Hyderabad file: bands read 3,2,1, 5.7% border masked, 44k mesh faces cut); full-resolution DSM for large scenes DONE (4096 px HITEC: DSM + sigma written at source grid, model at x1.9, 606 s CPU)
- [ ] 2. Dark slab (HITEC): check if it is the unmasked border (fixed by 1)
- [x] 3. 0.6 m test on the same held-out tiles (competitor: +15.4 %) -- 0.6 m upsampled +2.9 % per-building (1.92 vs 1.86 m), native pixels +7.3 %; also fixed: old table used 2 non-held-out tiles
- [~] 4. Run 03 DONE (6 epochs; DFC val MAE 2.537 vs run 02 2.593 m). Held-out comparison (DFC 0.6 m + Swiss) running before any promotion
- [~] 5. ISRO-style self-evaluation: absolute DSM vs Copernicus and SRTM over Indian scenes (urban HITEC, forest Periyar, hilly Shimla, sparse Telangana farmland) -- scenes fetched, script written
- [x] 6. Anchoring study: done as 0a (k swept 0-1 on LiDAR elevations, both DEMs)
- [x] 7. Four-landscape LiDAR benchmark (NEON API now 403 -> swisstopo LiDAR). Block-wise inference: Zurich 8.45, Zermatt 8.54, Sihlwald forest 25.15, farmland 3.08 m AGL RMSE (BENCHMARK_LANDSCAPES.md)
- [!] 8. DFC Track 1 RGB + Reference and Track 3 RGB-2 downloads (Track 1 not on disk yet)
- [ ] 9. Head-to-head on the competitor's exact 80-tile protocol (per building, whole tile, buildings <= 20 m)
- [~] 10. Tall-building physics fusion (relief displacement + shadows)
- [~] 11. ICESat-2 automatic GCPs + independent Indian check
- [ ] 12. No-GPU fallback for the judges' machine; timed upload of a 0.6 m GeoTIFF
- [ ] 13. Browser matrix: Chrome/Edge, WebGPU and WebGL
- [ ] 14. Web deployment + Docker image as the standalone deliverable
- [ ] 15. Final consistency pass: rebuild scenes, re-run benchmarks, refresh all docs
- [ ] 16. Final deliverables pack: source, docs, evidence, Docker image
- [ ] 17. (optional, last) Analysis suite: solar potential, shadow simulation
