# DepthWizard -- Ablation: DEM vs GAMUS metric checkpoint

**Question:** the headline blind number (RMSE 5.21 m / MAE 2.41 m, 8/10 tiles) was measured with both `--dem` (default True) and the GAMUS-finetuned metric checkpoint present at the same time. `build_city_image.py` attempts Tier A (DEM) before it ever checks the metric backbone, so which one is actually responsible was not known until this ablation ran.

**Method:** same 10 DFC2019 tiles, same `benchmark_blind.py` path, three configurations of the two flags. No ground truth is used before scoring in any run; results are not tuned.

## Aggregate result per configuration

| config | RMSE | MAE | tiles scored |
|---|---|---|---|
| dem_only | 16.08 | 10.34 | 5/10 |
| metric_only | 5.23 | 2.42 | 8/10 |
| dem+metric | 5.23 | 2.42 | 8/10 |

## Per-tile scale_source, all three configs

| tile | dem_only | metric_only | dem+metric (winner) |
|---|---|---|---|
| JAX_004 | shape_only | metric (RMSE 4.03) | dem (RMSE 4.03) |
| JAX_017 | dem (RMSE 15.65) | metric (RMSE 5.25) | dem (RMSE 5.25) |
| JAX_018 | dem (RMSE 17.50) | metric (RMSE 3.82) | dem (RMSE 3.82) |
| JAX_020 | dem (RMSE 16.66) | metric (RMSE 4.31) | dem (RMSE 4.31) |
| JAX_022 | shape_only | metric (RMSE 4.97) | dem (RMSE 4.97) |
| JAX_028 | shape_only | metric (RMSE 5.59) | dem (RMSE 5.59) |
| JAX_031 | rejected | rejected | rejected |
| JAX_033 | dem (RMSE 12.53) | metric (RMSE 1.89) | dem (RMSE 1.89) |
| JAX_068 | dem (RMSE 17.54) | metric (RMSE 9.09) | dem (RMSE 9.09) |

**Reading the last column:** `dem+metric`'s RMSE equals `metric_only`'s RMSE on every single scored tile (JAX_004 through JAX_068, all identical to 2 decimal places). The "dem" label there is a tier-relabeling artifact, not a different result: `build_city_image.py` applies the DEM offset identically to `dsm`, `terrain`, and `ground` (build_city_image.py:680), so it cancels out of nDSM entirely, and nDSM is what `benchmark_blind.py` scores against ground truth (`ground_truth_ndsm`, not the absolute DSM). DEM anchoring only overwrites the tier string from B to A when it succeeds -- it never touches a scored value once the metric model has already produced the height field.
| JAX_070 | rejected | rejected | rejected |

## Reproducing

```bash
python scripts/ablation_blind.py 10
```

