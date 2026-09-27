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
- [~] 1. Cartosat input handling: B,G,R,NIR band order, per-band 16-bit stretch, nodata/black-collar mask, very large scenes, tests -- band order + nodata DONE (8 tests pass; synthetic 16-bit B,G,R,NIR Hyderabad file: bands read 3,2,1, 5.7% border masked, 44k mesh faces cut); full-resolution DSM for large scenes written, test pending
- [ ] 2. Dark slab (HITEC): check if it is the unmasked border (fixed by 1)
- [~] 3. 0.6 m test on the same held-out tiles (competitor: +15.4 %) -- running; also fixed: old table used 2 non-held-out tiles
- [~] 4. Run 03 RUNNING on Kaggle (started 27 Sep ~11:45 IST), init run 02 (long job): GAMUS + DFC on disk + 0.3-1.0 m resolution + Cartosat-like colour augmentation
- [~] 5. ISRO-style self-evaluation: absolute DSM vs Copernicus and SRTM over Indian scenes (urban HITEC, forest Periyar, hilly Shimla, sparse Telangana farmland) -- scenes fetched, script written
- [ ] 6. Anchoring study: DSM that averages back to the reference over 30 m cells vs current anchoring; keep the better
- [ ] 7. Hilly/forest/sparse LiDAR benchmark (NEON) replacing the flat-Jacksonville "hilly" class; four-landscape table
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
