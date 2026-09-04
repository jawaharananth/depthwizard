"""
Export a calibrated height array as a standard-format GeoTIFF DSM.

Georeferenced inputs (Tier A/B with real bounds) get a real CRS + affine
transform so the file opens correctly in QGIS/ArcGIS/any GIS tool. Tier C
(relative-only, no ground truth) still gets written as a valid GeoTIFF for
pipeline/tooling consistency, but deliberately WITHOUT a CRS -- writing a
fake CRS on unscaled relative heights would silently lie about the file's
accuracy to anyone who opens it in a GIS tool later.
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds, Affine


def export_dsm_geotiff(dsm: np.ndarray, out_path: str, bounds_wgs84: tuple = None,
                        crs: str = "EPSG:4326", nodata: float = -9999.0, tags: dict = None):
    """
    dsm: HxW float array (metric elevation, or relative height if bounds_wgs84 is None).
    bounds_wgs84: (left, bottom, right, top) in WGS84 degrees, or None for an
                  ungeoreferenced (pixel-space only) DSM.
    tags: optional dict of string metadata to embed (e.g. calibration tier, curves).
    """
    dsm_out = np.where(np.isnan(dsm), nodata, dsm).astype(np.float32)
    height, width = dsm_out.shape

    if bounds_wgs84 is not None:
        transform = from_bounds(*bounds_wgs84, width, height)
        out_crs = crs
    else:
        transform = Affine.identity()
        out_crs = None

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype="float32", crs=out_crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(dsm_out, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})

    return out_path


def load_dsm_geotiff(path: str) -> tuple:
    """Returns (dsm array with nodata->NaN, crs, transform, tags) for verification/reuse."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        return arr, src.crs, src.transform, src.tags()


if __name__ == "__main__":
    dsm = np.random.rand(50, 50).astype(np.float32) * 30
    export_dsm_geotiff(dsm, "test_export_georef.tif", bounds_wgs84=(77.0, 12.0, 77.01, 12.01),
                        tags={"tier": "A_georeferenced_per_terrain", "note": "test export"})
    arr, crs, transform, tags = load_dsm_geotiff("test_export_georef.tif")
    print("Georeferenced export: crs=", crs, "shape=", arr.shape,
          "value match=", np.allclose(arr, dsm, atol=0.01), "tags=", tags)

    export_dsm_geotiff(dsm, "test_export_relative.tif", bounds_wgs84=None,
                        tags={"tier": "C_relative_only"})
    arr2, crs2, transform2, tags2 = load_dsm_geotiff("test_export_relative.tif")
    print("Relative-only export: crs=", crs2, "(expect None)", "value match=",
          np.allclose(arr2, dsm, atol=0.01), "tags=", tags2)

    import os
    os.remove("test_export_georef.tif")
    os.remove("test_export_relative.tif")


def export_dsm_geotiff_affine(dsm, out_path, transform=None, crs=None,
                              nodata: float = -9999.0, tags: dict = None):
    """
    Write a DSM using an affine transform directly, rather than WGS84 bounds.

    The bounds-based entry point assumes the caller has degrees. This pipeline
    works in a projected UTM grid and already holds the exact affine transform
    used to build the scene, so converting it to bounds and back would introduce
    a rounding error into a file whose whole purpose is to be measured from.

    Passing crs=None writes a valid GeoTIFF with no projection, which is the
    correct representation of a relative-scale surface -- see the module
    docstring for why a fake CRS is worse than none.
    """
    import numpy as _np
    import rasterio as _rio

    arr = _np.where(_np.isnan(dsm), nodata, dsm).astype(_np.float32)
    h, w = arr.shape
    with _rio.open(out_path, "w", driver="GTiff", height=h, width=w, count=1,
                   dtype="float32", crs=crs, transform=transform,
                   nodata=nodata, compress="deflate") as dst:
        dst.write(arr, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return out_path
