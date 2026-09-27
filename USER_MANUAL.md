# DepthWizard — User Manual

DepthWizard turns one satellite or aerial photo into a height map (DSM) and a 3D model you can
walk through, measure and check against reference data.

## 1. Start

**Local (this repository):**

```bash
python server.py            # then open http://127.0.0.1:8000/
```

**Hosted:** open the site address you were given. Everything below works the same way.

## 2. Upload an image

1. Open the start page and drop an image on it, or click to choose one.
   Accepted: **PNG, JPG, TIFF / GeoTIFF**.
2. Watch the progress bar. A 2048-px image takes a few minutes on a normal laptop.
3. When it finishes you get three buttons:
   - **Open 3D viewer**
   - **Download DSM (GeoTIFF)**
   - **Validate against reference**

**What you get depends on the file:**

| You upload | You get |
|---|---|
| A **GeoTIFF** (has map coordinates) | An **absolute DSM**: real elevations in metres above sea level (EGM2008), on exactly the same map grid as your image |
| A **PNG or JPG** (no coordinates) | A **relative DSM (rDSM)**: heights relative to each other. Without a location there is no way to know the true elevation |

Every DSM also comes with `_sigma.tif`: the estimated uncertainty (± metres) of each pixel.

**If an upload is refused,** the message says why and what to do (for example, an image too
small to analyse, or a file type that is not an image).

## 3. Move around the 3D model

| Action | How |
|---|---|
| Orbit | drag with the left mouse button |
| Pan | drag with the right mouse button |
| Zoom | scroll |
| **Walk on the ground** | click **Walk** (bottom right), then **W A S D** + mouse; walls stop you |
| Fly instead of walk | press **G** while walking; **Space** up, **C** down |
| Run | hold **Shift** |
| Leave walk mode | **Esc** |
| Automatic flight | **Flythrough** |
| Hide all panels | **H** |
| Show frame rate | **F** |
| This help | **?** (top right) |

## 4. Measure and analyse

Open the tool panel with **□** (top left).

| Tool | What it does |
|---|---|
| Click any building | its height and details |
| **Measure Height** | click two points: height difference and distance |
| **Measure Area** | click around an area |
| **Cross-Section** | draw a line: height profile along it |
| **Line of Sight** | can point B be seen from point A? |
| **SLOPE** | colours the ground by steepness |
| **HEATMAP** | colours by height |
| **CONFIDENCE** | colours by uncertainty; hover shows ± metres |
| **Flood Source** | raise a water level and see which buildings flood |
| **Add Pin**, **Export PDF** | annotate and export |

## 5. Check accuracy against your own reference

1. Click **Upload reference DSM** (in the viewer or on the start page).
2. Choose a GeoTIFF of known heights: a LiDAR DSM or a survey DEM, in any map projection.
3. You get **RMSE, MAE, bias and correlation**, and the **VS LiDAR** view shows where the error
   is. For a relative DSM, a scale and offset are fitted first and reported.

## 6. Calibrate with ground control points

1. Click **Ground Control**.
2. Click a building or ground point and type its **known height** in metres.
3. Add at least 2 points; 5 or more spread over different heights allow a scale correction.
4. **Apply scale → metric** adjusts the 3D view; **Write to DSM file** recalibrates the
   downloadable GeoTIFF (the original is kept). The report shows which correction was chosen
   and its leave-one-out error.

## 7. Showcase scenes

Two finished examples open from the start page: **HITEC City, Hyderabad** (urban, metro and
flyovers) and **Periyar Tiger Reserve, Kerala** (forest and lake). Add `&context=1` to the viewer
address to show the surrounding area, and `&tex=hires` for sharper same-area imagery.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Heights are "relative" | the image had no map coordinates; upload a GeoTIFF, or add ground control points |
| Upload slow | large images take longer; about 2048 px on the long side is a good size |
| 3D view blank | use a current Chrome or Edge; the viewer falls back to WebGL automatically |
| Progress stopped updating | the page reconnects on its own; reload if it does not |
