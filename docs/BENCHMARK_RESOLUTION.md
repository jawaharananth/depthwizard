# Accuracy vs input resolution (blind, DFC2019, 18 held-out tiles, 458 buildings)

ISRO evaluates on Cartosat-2S at 0.6 m. Each held-out tile's 0.25 m ortho is area-averaged to the coarser ground sample (what a coarser sensor records), then either resampled back to the working grid (the pipeline's auto-zoom) or fed at its own coarse pixel count. Scored against the LiDAR nDSM with no fitting to truth; per building = median height of each LiDAR building instance on both sides. Model: the raw height field (depth_model.predict_tiled).

| input | per-pixel RMSE | MAE | pooled corr | per-building RMSE | bias | buildings <= 20 m | vs native |
|---|---|---|---|---|---|---|---|
| 0.25 m native | 2.80 m | 1.55 m | 0.758 | 1.88 m | -0.08 m | 1.87 m | +0.0 % |
| 0.50 m, upsampled | 2.80 m | 1.55 m | 0.757 | 1.88 m | -0.08 m | 1.86 m | -0.4 % |
| 0.60 m, upsampled | 2.81 m | 1.56 m | 0.756 | 1.87 m | -0.08 m | 1.86 m | -0.6 % |
| 0.60 m, native pixels | 2.84 m | 1.52 m | 0.749 | 1.99 m | +0.09 m | 1.98 m | +5.4 % |
| 0.75 m, upsampled | 2.81 m | 1.56 m | 0.756 | 1.85 m | -0.08 m | 1.84 m | -1.8 % |
| 1.00 m, upsampled | 2.82 m | 1.57 m | 0.754 | 1.87 m | -0.06 m | 1.86 m | -0.5 % |

Simulated by downsampling WorldView-3, so this measures the resolution gap, not the sensor gap.

Regenerate: `python scripts/resolution_test.py`.
