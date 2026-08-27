import numpy as np
import rasterio
from scipy import stats


def fit_scale_shift(relative_depth: np.ndarray, reference_points: list):
    if len(reference_points) < 3:
        raise ValueError("Need at least 3 reference points for a stable fit")

    rel_vals = np.array([relative_depth[r, c] for r, c, _ in reference_points])
    true_vals = np.array([elev for _, _, elev in reference_points])

    a, b, r_value, p_value, std_err = stats.linregress(rel_vals, true_vals)
    return a, b, r_value


def load_copernicus_dem_window(bounds_wgs84: tuple, dem_path: str) -> np.ndarray:
    with rasterio.open(dem_path) as src:
        window = rasterio.windows.from_bounds(*bounds_wgs84, transform=src.transform)
        dem_array = src.read(1, window=window)
    return dem_array


def sample_reference_grid(dem_array: np.ndarray, relative_depth_shape: tuple, n_points: int = 200):
    from skimage.transform import resize
    dem_resized = resize(dem_array, relative_depth_shape, preserve_range=True, anti_aliasing=True)

    rows = np.linspace(0, relative_depth_shape[0] - 1, int(np.sqrt(n_points))).astype(int)
    cols = np.linspace(0, relative_depth_shape[1] - 1, int(np.sqrt(n_points))).astype(int)

    points = []
    for r in rows:
        for c in cols:
            points.append((r, c, float(dem_resized[r, c])))
    return points


def calibrate_georeferenced(relative_depth: np.ndarray, dem_path: str, bounds_wgs84: tuple):
    dem_array = load_copernicus_dem_window(bounds_wgs84, dem_path)
    ref_points = sample_reference_grid(dem_array, relative_depth.shape)
    a, b, r_value = fit_scale_shift(relative_depth, ref_points)

    metric_dsm = a * relative_depth + b
    return metric_dsm, {"tier": "A_georeferenced", "a": a, "b": b, "r_value": r_value}


def estimate_gsd_meters(bounds_wgs84: tuple, shape: tuple) -> tuple:
    """
    Rough meters/pixel from WGS84 bounds + array shape. Good enough for slope
    classification thresholds; not survey-grade (ignores map projection
    distortion), fine at the scale of a single scene.
    """
    left, bottom, right, top = bounds_wgs84
    height, width = shape
    lat_mid = (top + bottom) / 2.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_mid))

    gsd_y_m = abs(top - bottom) * m_per_deg_lat / max(height, 1)
    gsd_x_m = abs(right - left) * m_per_deg_lon / max(width, 1)
    return gsd_x_m, gsd_y_m


def calibrate_georeferenced_per_terrain(relative_depth: np.ndarray, dem_path: str,
                                         bounds_wgs84: tuple, seg_labels: np.ndarray):
    """
    Tier A upgrade: instead of one global scale/shift for the whole scene,
    fit a separate relative-depth -> metric-elevation curve per terrain class
    (urban/hilly/forest/sparse), since e.g. a global linear fit systematically
    under/over-shoots height in dense forest vs open ground.
    """
    from skimage.transform import resize
    import terrain_classify

    dem_array = load_copernicus_dem_window(bounds_wgs84, dem_path)
    dem_resized = resize(dem_array, relative_depth.shape, preserve_range=True,
                          anti_aliasing=True).astype(np.float32)

    gsd_x_m, gsd_y_m = estimate_gsd_meters(bounds_wgs84, relative_depth.shape)
    terrain_mask, terrain_stats = terrain_classify.classify_terrain(
        seg_labels, dem_meters=dem_resized, gsd_x_m=gsd_x_m, gsd_y_m=gsd_y_m)

    calibrated, curves = terrain_classify_curves_fit(relative_depth, dem_resized, terrain_mask)

    return calibrated, {
        "tier": "A_georeferenced_per_terrain",
        "curves": curves,
        "terrain_stats": terrain_stats,
        "gsd_x_m": gsd_x_m,
        "gsd_y_m": gsd_y_m,
    }


def terrain_classify_curves_fit(relative_depth, dem_resized, terrain_mask):
    from calibration.terrain_curves import fit_per_terrain_curves
    return fit_per_terrain_curves(relative_depth, dem_resized, terrain_mask)
