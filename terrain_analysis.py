"""
Terrain analysis layers: slope, and the elevation sampling that cross-section
and line-of-sight tools need.

WHY SLOPE IS COMPUTED HERE RATHER THAN IN THE VIEWER

The problem statement asks the visualisation platform to support analysis of
structural heights AND SLOPES. Slope is a derivative of elevation, so computing
it in the browser from a decimated mesh would measure the mesh's tessellation as
much as the terrain: the ground mesh is adaptively simplified, and a flat area
meshed coarsely would report different slope from the same area meshed finely.

Computing it here, on the full-resolution DTM in real metres, gives a slope field
that means the same thing everywhere regardless of how the mesh was built.

UNITS

Slope is reported in DEGREES from horizontal, not as a percentage or a ratio.
Degrees are what a person reads off a map legend, and the three conventions are
routinely confused -- a 100% grade is 45 degrees, not 90.
"""
import numpy as np
import cv2

# Break points for the colour ramp, in degrees. These are not arbitrary: they
# follow the thresholds used in terrain trafficability and site planning --
# under 5 deg is effectively flat and buildable, 5-15 needs grading, 15-30 is
# difficult, above 30 is generally unusable for construction and impassable for
# most wheeled vehicles.
SLOPE_BREAKS_DEG = (5.0, 15.0, 30.0)


def slope_degrees(ground_m: np.ndarray, gsd_m: float,
                  smooth_px: float = 2.0) -> np.ndarray:
    """
    Slope magnitude in degrees, from a metric elevation raster.

    The gradient is taken with a Sobel operator over a lightly smoothed copy.
    Smoothing first matters: elevation from photogrammetry carries per-pixel
    noise, and differentiating noise amplifies it -- an unsmoothed slope map of a
    flat car park looks like scree.
    """
    z = ground_m.astype(np.float32)
    if smooth_px > 0:
        z = cv2.GaussianBlur(z, (0, 0), sigmaX=smooth_px)

    # Sobel returns a gradient scaled by the kernel; dividing by 8 and by the
    # pixel size converts it to metres of rise per metre of run.
    dzdx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * gsd_m)
    dzdy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * gsd_m)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def colourise_slope(slope_deg: np.ndarray) -> np.ndarray:
    """
    Green flat, yellow moderate, red steep -- the convention every GIS user
    already reads without a legend.

    Colours are assigned by the planning breaks above rather than by stretching
    the scene's own min-max. A stretch would make the flattest tile look
    alarming and the steepest look benign, because the same colour would mean a
    different gradient in every scene.
    """
    s = np.clip(slope_deg, 0, 45.0)
    b1, b2, b3 = SLOPE_BREAKS_DEG
    rgb = np.zeros((*s.shape, 3), np.uint8)

    flat = s <= b1
    mod = (s > b1) & (s <= b2)
    steep = (s > b2) & (s <= b3)
    severe = s > b3

    rgb[flat] = (76, 155, 96)
    # Interpolate within the moderate band so a gradient reads as a gradient
    # rather than a hard step at each break.
    t = np.clip((s - b1) / max(b2 - b1, 1e-6), 0, 1)[mod]
    rgb[mod] = np.stack([76 + t * 155, 155 + t * 45, 96 - t * 40], -1).astype(np.uint8)
    t2 = np.clip((s - b2) / max(b3 - b2, 1e-6), 0, 1)[steep]
    rgb[steep] = np.stack([231 + t2 * 15, 200 - t2 * 130, 56 - t2 * 10], -1).astype(np.uint8)
    rgb[severe] = (166, 58, 46)
    return rgb


def slope_stats(slope_deg: np.ndarray) -> dict:
    s = slope_deg[np.isfinite(slope_deg)]
    if s.size == 0:
        return {}
    b1, b2, b3 = SLOPE_BREAKS_DEG
    return {
        "median_deg": round(float(np.median(s)), 2),
        "p95_deg": round(float(np.percentile(s, 95)), 2),
        "max_deg": round(float(s.max()), 2),
        "frac_flat": round(float((s <= b1).mean()), 3),
        "frac_moderate": round(float(((s > b1) & (s <= b2)).mean()), 3),
        "frac_steep": round(float((s > b2).mean()), 3),
        "breaks_deg": list(SLOPE_BREAKS_DEG),
    }
