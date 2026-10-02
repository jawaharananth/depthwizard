# ISRO protocol: absolute DSM vs public DEMs, per landscape

ISRO's evaluation FAQ states that GeoTIFF outputs are scored as an absolute
DSM against a public DEM such as SRTM or Copernicus. This page reproduces that
check on seven places across four landscape types.

**Method.** Each place is rebuilt through the upload pipeline
(`server.upload_build_kwargs()`, `DW_MAX_PX=2048`, display sharpening and PBR
maps off) twice: once with the DSM ground anchored to Copernicus GLO-30 and
once anchored to SRTM 30 m. Each DSM is then scored against both DEMs with
`scripts/isro_protocol_eval.py`: the DSM is block-averaged to ~30 m (the
DEM's own grid; a 30 m DEM cannot resolve single buildings) and compared cell
by cell. Figures below are the `at_30m` block of `results/isro/landscapes.json`.

**How to read it.** The DSM's ground is anchored to the chosen DEM, so the
score against that same DEM ("matched") is low partly by construction. The
score against the *other* DEM ("cross-DEM") is the independent figure: it
includes the real disagreement between the two DEMs, including their
different vertical datums.

## Results (30 m grid)

Error = our DSM minus the reference DEM. RMSE, MAE and bias in metres; r is
Pearson correlation over the 30 m cells.

| Place | Landscape | Anchored to | vs GLO-30 RMSE | MAE | bias | r | vs SRTM RMSE | MAE | bias | r |
|---|---|---|---|---|---|---|---|---|---|---|
| hitec_2x2 | urban | GLO-30 | 0.76 | 0.54 | +0.00 | 1.00 | 3.06 | 2.28 | -0.60 | 0.97 |
| hitec_2x2 | urban | SRTM | 3.19 | 2.34 | +0.60 | 0.97 | 0.65 | 0.47 | +0.00 | 1.00 |
| mumbai_lower_parel | urban | GLO-30 | 0.92 | 0.63 | -0.00 | 0.99 | 4.84 | 3.39 | +1.73 | 0.64 |
| mumbai_lower_parel | urban | SRTM | 5.02 | 3.48 | -1.73 | 0.62 | 0.75 | 0.50 | +0.00 | 0.99 |
| gangtok | hilly | GLO-30 | 1.23 | 0.93 | -0.02 | 1.00 | 8.32 | 6.13 | +4.25 | 1.00 |
| gangtok | hilly | SRTM | 8.59 | 6.41 | -4.29 | 1.00 | 1.14 | 0.83 | -0.02 | 1.00 |
| shimla | hilly | GLO-30 | 1.50 | 1.13 | -0.02 | 1.00 | 6.77 | 5.09 | -3.22 | 1.00 |
| shimla | hilly | SRTM | 7.30 | 5.57 | +3.19 | 1.00 | 1.44 | 1.05 | -0.02 | 1.00 |
| periyar_forest | forest | GLO-30 | 0.68 | 0.49 | -0.00 | 1.00 | 6.20 | 4.25 | -2.20 | 0.99 |
| periyar_forest | forest | SRTM | 6.49 | 4.63 | +2.19 | 0.98 | 0.58 | 0.42 | -0.00 | 1.00 |
| kanha | forest | GLO-30 | 0.73 | 0.52 | +0.00 | 1.00 | 4.46 | 3.77 | -2.79 | 0.95 |
| kanha | forest | SRTM | 4.67 | 3.97 | +2.79 | 0.94 | 0.48 | 0.36 | -0.00 | 1.00 |
| jaisalmer_village | sparse | GLO-30 | 0.13 | 0.09 | -0.00 | 1.00 | 1.66 | 1.36 | +1.07 | 0.97 |
| jaisalmer_village | sparse | SRTM | 1.60 | 1.32 | -1.07 | 0.97 | 0.21 | 0.16 | -0.00 | 1.00 |

Matched-DEM RMSE ranges from 0.13 m (Jaisalmer, GLO-30) to 1.50 m (Shimla,
GLO-30). Cross-DEM RMSE ranges from 1.60 m (Jaisalmer, SRTM-anchored, vs
GLO-30) to 8.59 m (Gangtok, SRTM-anchored, vs GLO-30). For every place the
cross-DEM bias is close to equal and opposite between the two anchors (for
example Kanha +2.79 / -2.79 m), i.e. it is mostly the offset between the two
DEMs themselves rather than something the DSM adds.

## Vertical datum note

SRTM heights are referenced to the EGM96 geoid and Copernicus GLO-30 heights to
EGM2008. The geoid height difference N(EGM96) - N(EGM2008) at each scene
centre (PROJ, `scripts/geoid_diff.py` -> `results/isro/geoid_diff.json`) is
between -0.82 m and +0.16 m at five of the seven sites, +4.39 m at Gangtok and
-3.77 m at Shimla. For the same surface this makes SRTM heights about 4.4 m lower
than GLO-30 at Gangtok and about 3.8 m higher at Shimla, which accounts for most
of the cross-DEM bias at those two hill towns (Gangtok +4.25 m / -4.29 m, Shimla
-3.22 m / +3.19 m) and a large part of their cross-DEM RMSE. At the other sites
the datum difference is under 1 m, and the rest of the cross-DEM error is the two
DEMs' own disagreement (acquisition epoch, radar penetration into canopy, void
filling).

## Reproduce

From the repository root (Windows PowerShell, project venv):

```powershell
$env:DW_MAX_PX = "2048"
.\venv\Scripts\python.exe -u scripts\isro_landscapes.py            # all seven places
# or a subset; keys already in results\isro\landscapes.json are skipped:
.\venv\Scripts\python.exe -u scripts\isro_landscapes.py mumbai_lower_parel gangtok jaisalmer_village
# score any DSM on its own:
.\venv\Scripts\python.exe scripts\isro_protocol_eval.py final_out\city_isro_gangtok_glo30_DSM.tif --dem srtm
```

Both DEMs are read over the network (GLO-30 from the Copernicus AWS mirror,
SRTM from the AWS elevation-tiles mirror). A transient download failure makes
the build fall back to a relative DSM; the script records that case as an
error instead of scoring a DSM left over from an earlier run. Delete the key
from `landscapes.json` and rerun the place to retry.
