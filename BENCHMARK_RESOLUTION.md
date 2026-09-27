# Accuracy vs input resolution (blind, DFC2019, 18 held-out tiles, 458 buildings)

ISRO evaluates on Cartosat-2S at 0.6 m. Each held-out tile's 0.25 m ortho is area-averaged to the coarser ground sample (what a coarser sensor records), then either resampled back to the working grid (the pipeline's auto-zoom) or fed at its own coarse pixel count. Scored against the LiDAR nDSM with no fitting to truth; per building = median height of each LiDAR building instance on both sides. Model: the raw height field (depth_model.predict_tiled).

| input | per-pixel RMSE | MAE | pooled corr | per-building RMSE | bias | buildings <= 20 m | vs native |
|---|---|---|---|---|---|---|---|
| 0.25 m native | 2.81 m | 1.58 m | 0.754 | 1.86 m | -0.06 m | 1.86 m | +0.0 % |
| 0.50 m, upsampled | 2.79 m | 1.57 m | 0.758 | 1.88 m | -0.19 m | 1.87 m | +1.0 % |
| 0.60 m, upsampled | 2.79 m | 1.57 m | 0.758 | 1.92 m | -0.35 m | 1.89 m | +2.9 % |
| 0.60 m, native pixels | 2.85 m | 1.52 m | 0.747 | 2.00 m | -0.07 m | 1.99 m | +7.3 % |
| 0.75 m, upsampled | 2.81 m | 1.59 m | 0.756 | 2.00 m | -0.56 m | 1.95 m | +7.6 % |
| 1.00 m, upsampled | 2.93 m | 1.68 m | 0.733 | 2.30 m | -1.00 m | 2.23 m | +23.6 % |

Simulated by downsampling WorldView-3, so this measures the resolution gap, not the sensor gap.

Regenerate: `python scripts/resolution_test.py`.
