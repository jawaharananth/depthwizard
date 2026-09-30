# Four landscapes vs airborne LiDAR at 0.6 m (Switzerland, swisstopo)

    python scripts/lidar_landscapes.py

The problem statement asks for "performance stability across urban, sparse,
hilly, and forested landscapes". DFC2019 (Jacksonville, Omaha) is flat, so it
cannot carry the hilly case. This benchmark uses Swiss federal open data, one
1 km tile per landscape, chosen by a named place rather than by result:

  * image:  SWISSIMAGE 10 cm orthophoto, area-averaged to 0.6 m (Cartosat-2S,
            per ISRO's FAQ), then run exactly as the pipeline runs a 0.6 m
            source: upsampled to the 0.3 m the model was trained at;
  * truth:  swissSURFACE3D Raster (LiDAR DSM, 0.5 m) minus swissALTI3D (LiDAR
            bare-earth DTM, 0.5 m) = height above ground, the quantity the
            model predicts. Both block-averaged to 1 m before scoring.

Nothing is fitted to the truth. The zero-prediction row per site ("everything
is ground") is the floor any estimate must beat. The photo and the LiDAR are
flown in different years where no same-year pair exists; the gap is reported
per site, and change in between (new buildings, tree growth) counts against us.

## Results (height above ground, 1 m cells)

| landscape | place (tile) | relief | LiDAR / photo year | RMSE | MAE | bias | corr | zero-prediction RMSE |
|---|---|---|---|---|---|---|---|---|
| urban | Zurich old town (2683-1247) | 51 m | 2024 / 2025 | 8.45 m | 6.28 m | -4.59 m | 0.782 | 12.31 m |
| hilly | Zermatt village (2624-1096) | 497 m | 2021 / 2020 | 8.54 m | 6.65 m | -4.02 m | 0.568 | 12.73 m |
| forested | Sihlwald forest (2684-1234) | 278 m | 2024 / 2025 | 25.15 m | 23.31 m | -23.13 m | 0.677 | 31.48 m |
| sparse | Grosses Moos farmland (Kerzers) (2581-1202) | 33 m | 2025 / 2023 | 3.08 m | 1.73 m | -1.08 m | 0.814 | 4.13 m |

## Absolute surface vs the LiDAR DSM (1 m cells; the product ISRO scores)

ground = smooth30(DEM) - k x smooth30(model height); surface = ground + model height. RMSE (bias) per anchor DEM. 'flat plane' is the old pipeline behaviour (one datum offset).

| landscape | DEM | DEM alone | flat plane (old) | k=0.0 | k=0.25 | k=0.5 | k=0.75 | k=1.0 |
|---|---|---|---|---|---|---|---|---|
| urban | glo30 | 8.36 (-2.2) | 16.39 (-1.1) | 7.12 (+1.7) | 6.93 (+0.7) | 6.89 (-0.2) | 7.03 (-1.2) | 7.32 (-2.2) |
| urban | srtm | 9.19 (-3.0) | 16.63 (-3.0) | 7.21 (+0.9) | 7.23 (-0.1) | 7.42 (-1.1) | 7.76 (-2.1) | 8.22 (-3.0) |
| hilly | glo30 | 11.13 (-5.1) | 147.70 (-25.8) | 8.98 (+0.1) | 9.12 (-1.2) | 9.49 (-2.5) | 10.06 (-3.8) | 10.80 (-5.1) |
| hilly | srtm | 9.56 (-1.3) | 146.89 (-20.7) | 9.80 (+3.9) | 9.31 (+2.6) | 9.04 (+1.3) | 8.99 (-0.0) | 9.19 (-1.3) |
| forested | glo30 | 9.61 (-1.9) | 65.72 (+4.8) | 9.75 (+4.3) | 9.16 (+2.8) | 8.84 (+1.2) | 8.81 (-0.4) | 9.08 (-1.9) |
| forested | srtm | 14.85 (-10.2) | 65.81 (-5.9) | 10.71 (-3.9) | 11.42 (-5.5) | 12.32 (-7.0) | 13.35 (-8.6) | 14.50 (-10.2) |
| sparse | glo30 | 3.56 (-0.9) | 10.70 (-0.5) | 2.89 (+0.1) | 2.90 (-0.2) | 2.94 (-0.4) | 3.01 (-0.7) | 3.10 (-0.9) |
| sparse | srtm | 4.13 (-1.8) | 10.76 (-1.2) | 3.30 (-0.8) | 3.38 (-1.1) | 3.48 (-1.3) | 3.60 (-1.6) | 3.74 (-1.8) |

Data (c) swisstopo (open government data). Regenerate: `python scripts/lidar_landscapes.py`.
