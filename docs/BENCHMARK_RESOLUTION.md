# Accuracy vs input resolution, 0.25-10 m (blind, DFC2019, 18 held-out tiles, 458 buildings)

ISRO evaluates on Cartosat-2S at 0.6 m and asks for models that hold from 0.35 m to 10 m. Each held-out tile's 0.25 m ortho is area-averaged to the coarser ground sample (what a coarser sensor records), then resampled back to the 0.25 m working grid, fed at its own coarse pixel count, or (2.5 m and coarser) upsampled at most 3x exactly as the pipeline's auto-zoom does. Scored against the LiDAR nDSM with no fitting to truth; per building = median height of each LiDAR building instance on both sides. Model: the raw height field (depth_model.predict_tiled).

| input | per-pixel RMSE | MAE | pooled corr | per-building RMSE | bias | buildings <= 20 m | vs native |
|---|---|---|---|---|---|---|---|
| 0.25 m native | 2.80 m | 1.55 m | 0.758 | 1.88 m | -0.08 m | 1.87 m | +0.0 % |
| 0.35 m, upsampled | 2.80 m | 1.55 m | 0.757 | 1.88 m | -0.08 m | 1.86 m | -0.1 % |
| 0.50 m, upsampled | 2.80 m | 1.55 m | 0.757 | 1.88 m | -0.08 m | 1.86 m | -0.3 % |
| 0.60 m, upsampled | 2.81 m | 1.56 m | 0.756 | 1.87 m | -0.08 m | 1.86 m | -0.4 % |
| 0.60 m, native pixels | 2.84 m | 1.52 m | 0.749 | 1.99 m | +0.09 m | 1.98 m | +5.6 % |
| 0.75 m, upsampled | 2.81 m | 1.56 m | 0.756 | 1.85 m | -0.08 m | 1.84 m | -1.7 % |
| 1.00 m, upsampled | 2.82 m | 1.57 m | 0.754 | 1.87 m | -0.06 m | 1.86 m | -0.4 % |
| 2.5 m, auto-zoom x3 | 4.63 m | 2.15 m | 0.299 | 3.73 m | -3.07 m | 3.59 m | +98.3 % |
| 5 m, auto-zoom x3 | 4.83 m | 2.23 m | nan | 3.78 m | -3.17 m | 3.63 m | +100.6 % |
| 10 m, auto-zoom x3 | 4.83 m | 2.23 m | nan | 3.78 m | -3.17 m | 3.63 m | +100.6 % |

At 2.5 m and coarser a typical building is a few pixels or less, so building heights blur toward their surroundings and are under-called; the absolute DSM then rests mostly on the terrain model it is anchored to. At 5 m and 10 m the model returns close to zero everywhere: those rows equal the predict-zero floor on these tiles. Upsampling does not rescue it (measured 2026-10-01, same tiles, per-building RMSE at 2.5 m: native pixels 3.55 m, x3 3.73 m, to 0.6 m 3.73 m, to 0.3 m 3.71 m, to 0.25 m 3.74 m): the checkpoint was trained on 0.3-1.0 m ground samples, so coarse input needs multi-resolution training, not resampling. Products from inputs coarser than 2 m carry a RESOLUTION_NOTE tag saying so. Simulated by downsampling WorldView-3, so this measures the resolution gap, not the sensor gap.

Regenerate: `python scripts/resolution_test.py`.
