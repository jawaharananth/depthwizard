# Accuracy vs input resolution (blind, DFC2019, 20 held-out tiles)

Each tile's 0.25 m ortho is area-averaged to the coarser ground sample (what a coarser sensor records), resampled back to the working grid (the pipeline's auto-zoom), and scored against the LiDAR nDSM with no fitting to truth.

Measured on the model's raw height field (depth_model.predict_tiled), i.e. before the pipeline's terrain/building post-processing, over every ortho in final_out/blind_input (the held-out benchmark tiles).

| input GSD | RMSE | MAE | pooled corr |
|---|---|---|---|
| 0.25 m | 2.85 m | 1.59 m | 0.747 |
| 0.50 m | 2.83 m | 1.58 m | 0.751 |
| 0.75 m | 2.85 m | 1.60 m | 0.748 |
| 1.00 m | 2.97 m | 1.69 m | 0.725 |

Regenerate: `python scripts/resolution_test.py`.
