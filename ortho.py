"""
Orthorectify a DFC2019 Track 3 RGB view onto the ground-truth grid.

WHY THIS EXISTS -- and why every accuracy number produced before it was wrong.

Track 3 RGB tiles are raw satellite frames carried in NITF, with an RPC camera
model attached and no geotransform (rasterio reports
`NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs`, then
`src.rpcs` is populated). They are NOT orthorectified and NOT north-up. The
ground truth beside them is a 512x512 north-up UTM raster at 0.5 m.

Those two rasters do not share a pixel grid, so `cv2.resize(truth, rgb.shape)`
-- the alignment this project used everywhere -- lines up two different pieces
of ground. Verified visually: on JAX_033 the storage-tank cluster sits
bottom-left in the RGB and top-right in the truth. On JAX_167, whole-image
depth over the truth's building mask scored buildings BELOW ground
(-0.061 separation) while the height field itself is plainly correct, with
rooftops bright and the river dark. The metric was reading the wrong pixels.

Every RMSE this project has reported was computed on a misregistered pair.

The fix is to project the truth grid through the RPC model and resample the
RGB onto it, so image and truth are aligned by construction.

Two further points that matter for accuracy:

  * View choice. Each tile ships ~25 views from different satellite passes,
    from 4.8 deg to 29 deg off-nadir. Off-nadir angle displaces a building's
    roof radially by height * tan(angle) -- at 18.9 deg, a 161 m tower's roof
    lands 55 m from its footprint. The default "first file in the folder" was
    an 18.9 deg view. `most_nadir_view` picks the straightest one instead.

  * Terrain height. Without a DEM the projection assumes one flat elevation,
    which is standard practice and leaves the roof lean described above.
    Passing the truth DSM as `dem_path` removes it exactly -- but that feeds
    ground truth into the input, so it is available only for visualisation and
    must never be used on a tile that is then scored.
"""
import glob
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

import dfc2019_loader as L

JAX_UTM = "EPSG:32617"
OMA_UTM = "EPSG:32615"


def most_nadir_view(tile: str, rgb_dir: str, metadata_dir: str) -> tuple[str, dict]:
    """
    The view whose look direction is closest to straight down.

    Returns (path, imd_fields). Falls back to the first file when metadata is
    missing, rather than failing -- but reports what it used, since the choice
    materially changes how much the roofs lean.
    """
    paths = sorted(glob.glob(os.path.join(rgb_dir, f"{tile}_*_RGB.tif")))
    if not paths:
        raise FileNotFoundError(f"no RGB views for {tile} in {rgb_dir}")

    best, best_angle, best_imd = paths[0], float("inf"), {}
    for p in paths:
        try:
            imd = L.parse_imd(L.imd_path_for_rgb(p, metadata_dir))
        except Exception:
            continue
        angle = imd.get("meanOffNadirViewAngle")
        if angle is not None and angle < best_angle:
            best, best_angle, best_imd = p, angle, imd
    return best, best_imd


def orthorectify(tile: str, truth_dir: str, rgb_dir: str, metadata_dir: str,
                 out_px: int = 2048, rgb_path: str = None,
                 dem_path: str = None, extent_m: float = None) -> dict:
    """
    Resample one RGB view onto the truth tile's UTM grid.

    out_px sets the output raster size over the truth's fixed ground extent
    (512 px at 0.5 m = 256 m), so out_px=2048 gives a 0.125 m grid. That
    oversamples the ~0.31 m source; it costs nothing in accuracy because the
    truth is still compared at its own 0.5 m, and it gives the mesh a dense
    canvas to carry surface detail.

    Returns the ortho image, the aligned truth arrays, and the transform.
    """
    truth = L.load_tile(tile, truth_dir)
    gsd_t = truth["gsd_m"]
    size_t = truth["dsm"].shape[0]
    truth_extent_m = size_t * gsd_t

    # The truth tile is 256 m, but the RGB frame behind it is a 2048x2048
    # satellite image covering several times that on the ground. Cropping the
    # ortho to the truth extent throws most of the captured city away. A larger
    # extent is centred on the truth tile so the validated area stays in the
    # middle and can still be scored; everything beyond it is reconstruction
    # without ground truth, which callers must not report as validated.
    if extent_m is None:
        extent_m = truth_extent_m
    pad = (extent_m - truth_extent_m) / 2.0

    if rgb_path is None:
        rgb_path, imd = most_nadir_view(tile, rgb_dir, metadata_dir)
    else:
        try:
            imd = L.parse_imd(L.imd_path_for_rgb(rgb_path, metadata_dir))
        except Exception:
            imd = {}

    out_gsd = extent_m / out_px
    # The TXT's northing is the tile's LOWER-left corner, but `from_origin`
    # wants the UPPER-left, so the tile's height is added. Determined
    # empirically, not assumed: warping the same view against all four corner
    # conventions and a centred one, then scoring each against the truth DSM's
    # own gradients, peaks at this convention on both JAX_033 and JAX_167, and
    # a +-40 m sub-tile offset search around it peaks at exactly zero on both.
    # Reading it as the upper-left instead lands the tile 256 m off -- on
    # JAX_167 that puts the whole frame in the river next to the skyline.
    dst_transform = from_origin(truth["utm_x"] - pad,
                                truth["utm_y"] + truth_extent_m + pad,
                                out_gsd, out_gsd)
    dst_crs = CRS.from_string(JAX_UTM if tile.startswith("JAX") else OMA_UTM)

    # The RPC model maps ground (lon, lat, height) to image (col, row), so the
    # warp needs a height to assume for each output pixel. Use the truth DSM's
    # median: it is a single scalar describing the tile's terrain level, not
    # per-pixel truth, so it does not leak the surface being measured.
    finite = truth["dsm"][np.isfinite(truth["dsm"])]
    rpc_height = float(np.median(finite)) if finite.size else 0.0

    ortho = np.zeros((3, out_px, out_px), dtype=np.uint8)
    with rasterio.open(rgb_path) as src:
        if not src.rpcs:
            raise ValueError(f"{rgb_path} carries no RPC model; cannot orthorectify")
        for band in (1, 2, 3):
            reproject(
                source=rasterio.band(src, band),
                destination=ortho[band - 1],
                rpcs=src.rpcs,
                src_crs=CRS.from_epsg(4326),
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.cubic,
                **{"RPC_HEIGHT": rpc_height} if not dem_path else {"RPC_DEM": dem_path},
            )

    image = np.transpose(ortho, (1, 2, 0))
    covered = float((image.sum(axis=2) > 0).mean())

    return {
        "image": image,
        "truth": truth,
        "transform": dst_transform,
        "crs": dst_crs.to_string(),
        "gsd_m": out_gsd,
        "rgb_path": rgb_path,
        "off_nadir_deg": imd.get("meanOffNadirViewAngle"),
        "sun_elev_deg": imd.get("meanSunEl"),
        "sun_azimuth_deg": imd.get("meanSunAz"),
        "rpc_height_m": rpc_height,
        "coverage": covered,
        "scale_truth_to_ortho": out_px / size_t,
        "extent_m": extent_m,
        "truth_extent_m": truth_extent_m,
        "truth_inset_px": int(round(pad / out_gsd)),
    }


if __name__ == "__main__":
    import sys
    from PIL import Image

    tile = sys.argv[1] if len(sys.argv) > 1 else "JAX_167"
    r = orthorectify(tile, "dfc2019_data/truth/Track3-Truth",
                     "dfc2019_data/rgb/Track3-RGB-1",
                     "dfc2019_data/metadata/Track3-Metadata")
    print(f"{tile}: {os.path.basename(r['rgb_path'])} "
          f"off-nadir {r['off_nadir_deg']}deg  sun {r['sun_elev_deg']}deg")
    print(f"  ortho {r['image'].shape} at {r['gsd_m']:.3f} m/px, "
          f"coverage {r['coverage']*100:.1f}%")
    Image.fromarray(r["image"]).save(f"ortho_{tile}.png")
    print(f"  wrote ortho_{tile}.png")
