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
- [x] 0. BUG FIXED: large scenes were flattened by whole-image crop alignment; now block-wise at the trained 0.3 m scale (one pass feeds the full-res DSM and the 3D scene)
- [x] 0a. ABSOLUTE-DSM FIX (anchoring.py, k measured on LiDAR): hilly 147.7 -> 9.5 m, forest 65.7 -> 8.8, urban 16.4 -> 6.9, sparse 10.7 -> 2.9 m; beats GLO-30 alone everywhere. SRTM fetch fixed
- [x] 1. Cartosat input: B,G,R,NIR order, 16-bit stretch, nodata/border mask, full-res DSM for large scenes (8 tests)
- [x] 2. HITEC slab: not reproduced in a headless render (no flat slab); one tall tower's facade is streaky from its own shadow pixels (texture quality, noted)
- [x] 3. 0.6 m test: +2.9 % per-building (competitor +15.4 %); old table's 2 non-held-out tiles removed
- [x] 4. Run 03 trained and PROMOTED (0.6 m per-building 1.873 vs 1.916 m, bias -0.085 vs -0.350; absolute DSM better in all 4 landscapes; ECE 0.111 vs 0.118)
- [~] 4b. Run 04 (Swiss LiDAR tiles) trained; held-out comparison vs run 03 running -> promote if better
- [~] 5. India, ISRO's protocol: absolute DSM vs Copernicus/SRTM + ICESat-2 (Hyderabad, Periyar, Shimla, Telangana) -- running on run 03
- [x] 6. Anchoring study (k swept on LiDAR, both DEMs)
- [x] 7. Four-landscape LiDAR benchmark (swisstopo; NEON API now 403)
- [!] 8. DFC Track 1 downloads -- USER: RGB images 1/1 (6.59 GB) + Reference (8.17 GB)
- [ ] 9. HEAD-TO-HEAD on the competitor's exact 80-tile protocol (per building, whole tile, <= 20 m) -- needs 8; the one claim not yet proven
- [~] 10. Tall-building physics fusion -- currently hurts small buildings; needs tall towers from 8
- [~] 11. ICESat-2 automatic GCPs (leave-one-track-out) + India check -- in 5
- [x] 12. No-GPU: the whole pipeline runs CPU-only (this machine has no GPU); server job limit 1800 s, uploads 1 GB
- [ ] 13. FAST CPU INFERENCE: ONNX export (+ int8), parity check vs PyTorch, timing (competitor: 536 ms/tile)
- [ ] 14. UNCERTAINTY CALIBRATION GAP: ECE 0.111 vs competitor 0.063 -- better sigma model (e.g. per-height-band scaling), refit, held-out check
- [ ] 15. OFFLINE VIEWER: one self-contained HTML (scene data inlined) that opens from disk with no server (competitor has one)
- [ ] 16. LARGER EVALUATION: more held-out tiles / buildings (competitor: 80 tiles, 3,090 buildings); DFC Track 1 via 8 + all Track 3 held-out views
- [~] 17. Browser matrix: Chrome WebGL (SwiftShader) render verified; Edge headless screenshot not possible on this machine; USER: real-GPU `&fps=1` in Chrome + open once in Edge
- [ ] 18. Web deployment config + Docker image (Dockerfile import closure verified statically; no Docker here) -- USER: hosting choice
- [ ] 19. Documentation page for judges (method, evidence, limits) served by the site
- [ ] 20. Final consistency pass: rebuild HITEC/Periyar on run 03/04 + new anchoring, re-run blind benchmark + per-building, refresh README badges/docs
- [ ] 21. Final deliverables pack: source, docs, evidence, Docker image
- [ ] 22. (optional, last) Analysis suite: solar potential, shadow simulation
