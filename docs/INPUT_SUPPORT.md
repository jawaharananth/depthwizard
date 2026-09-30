# Input support

What the upload pipeline (`build_city_image.build`, loader in
`build_city_image.py` and `input_bands.py`) accepts, what it does with each
kind of file, and a measured robustness matrix.

## Products

- **DSM** (`city_<name>_DSM.tif`): written when the input is a GeoTIFF with a
  CRS and a real geotransform. Absolute heights, ground tied to a public 30 m
  DEM (Copernicus GLO-30 by default, SRTM optional), in the file's own CRS and
  at the file's own pixel grid.
- **rDSM** (`city_<name>_rDSM.tif`): written for everything else (PNG, JPEG,
  WebP, a GeoTIFF without a usable geotransform). Relative heights above local
  ground, no CRS; horizontal scale uses the assumed upload GSD (0.6 m,
  `DW_DEFAULT_GSD`).

## GeoTIFF handling

| Situation | Behaviour |
|---|---|
| 8-bit RGB | read as is |
| 11/12/16-bit or float counts | each band stretched to 8 bits over its own 2nd-98th percentile of valid pixels (no clipping to white) |
| Band order tagged (colour interpretation) | bands read by tag, e.g. B,G,R stored in that order |
| Untagged 4+ band 16-bit stack | read as B,G,R,NIR (Cartosat MX / PlanetScope order) |
| Near-infrared band present | detected (band description, B,G,R,NIR order, or the 4th band of tagged RGB); vegetation from NDVI |
| Single band (panchromatic) | replicated to three channels |
| nodata value, alpha band, internal mask | pixel is no-data only when every colour band says so; masked in every output |
| Black (all-zero) collar touching the edge | masked; black objects inside the scene are kept |
| Rotated or sheared geotransform | resampled once to north-up at the file's own pixel size, then processed as a normal DSM case |
| Geographic CRS (lat/lon) | reprojected once to the local UTM zone |
| Conformal projection with scale factor far from 1 (Web Mercator, LCC) | pixel size corrected to ground metres; output keeps the file's CRS |
| CRS present but identity geotransform | treated as not georeferenced: rDSM, no CRS |
| BigTIFF, tiled, JPEG/YCbCr compressed | read through GDAL; DSM written at the full source grid, 3D scene at `DW_MAX_PX` |

## PNG / JPEG / WebP handling

| Situation | Behaviour |
|---|---|
| EXIF orientation tag | applied before processing (image processed upright) |
| CMYK JPEG | converted to RGB |
| Greyscale | replicated to three channels |
| 16-bit greyscale PNG | stretched to 8 bits over its 0.5-99.5 percentiles |
| Transparent pixels (alpha) | masked in every output |
| File extension does not match content (WebP named `.jpg`) | opened by content |

## Robustness matrix

Inputs from `scripts/make_cartosat_like.py` (all cut from the same HITEC City
scene, EPSG:3857, ~0.6 m), built through the upload pipeline with
`DW_MAX_PX=2048` by `scripts/robustness_matrix.py`; results in
`results/robustness/matrix.json`. "vs GLO-30 RMSE" is the DSM scored against
Copernicus GLO-30 at the 30 m grid (`scripts/isro_protocol_eval.py`); the DSM
ground is anchored to GLO-30, so this is a consistency check that the file was
placed and scaled correctly, not an independent accuracy figure. Seconds are
build wall time on a laptop CPU and include network fetches (DEM, building
outlines, OSM).

| Case | Expected | Result | Product | Seconds | vs GLO-30 RMSE (m) |
|---|---|---|---|---|---|
| c1_bgrn_11bit_tagged.tif | DSM | ok | DSM | 83 | 0.81 |
| c2_rgb_12bit.tif | DSM | ok | DSM | 74 | 0.81 |
| c3_bgr_tagged.tif | DSM | ok | DSM | 97 | 0.83 |
| c4_nodata255_collar.tif | DSM | ok | DSM | 78 | 0.87 |
| c5_epsg7755_india_lcc.tif | DSM | ok | DSM (EPSG:7755) | 68 | 0.83 |
| c6_rotated_15deg.tif | DSM or clear refusal | ok | DSM (north-up) | 88 | 0.87 |
| c7_crs_no_transform.tif | rDSM or clear refusal | ok | rDSM | 88 | -- |
| c8_8k_bigtiff_jpeg.tif | DSM or size refusal | ok | DSM (8192 x 8192) | 144 | 0.56 |
| c9_exif_rotated.jpg | rDSM, upright | ok | rDSM (upright, 1800 x 1600) | 53 | -- |
| c10_cmyk.jpg | rDSM | ok | rDSM | 50 | -- |
| c11_gray.jpg | rDSM | ok | rDSM | 54 | -- |
| c12_gray16.png | rDSM | ok | rDSM | 39 | -- |
| c13_webp_named.jpg | rDSM or clear refusal | ok | rDSM | 41 | -- |

Every DSM carries a CRS, and its mean difference from GLO-30 at the 30 m grid
(`bias_m` in `matrix.json`) is within +/-0.02 m. c4's DSM is valid on 88.1% of pixels,
which is exactly the area outside the white collar; c6's is valid on 66.5%,
the rotated footprint inside its north-up bounding box.

Reproduce:

```powershell
.\venv\Scripts\python.exe scripts\make_cartosat_like.py work\cartosat_like
$env:DW_MAX_PX = "2048"
.\venv\Scripts\python.exe -u scripts\robustness_matrix.py work\cartosat_like
```

Cases already in `matrix.json` are skipped; delete a key to rebuild that case.
