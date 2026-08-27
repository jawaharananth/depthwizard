"""
Physically-derived shading maps baked from the DSM itself.

These are not stylistic filters. Each one is computed from the actual
reconstructed geometry, so what you see corresponds to real structure in the
elevation data:

  sky-view factor  -- fraction of the sky hemisphere visible from each point.
                      Standard technique in LiDAR / archaeological terrain
                      visualisation. This is what makes narrow streets read as
                      recessed and building bases sit in contact shadow instead
                      of looking pasted onto a flat plane.

  curvature        -- local convexity/concavity. Sharpens ridge lines, roof
                      edges, and curbs that a plain diffuse texture flattens.

  roughness        -- per-material surface response derived from the semantic
                      classes, so water reflects, vegetation scatters, and
                      roofs sit between the two under image-based lighting.

Baking these offline is deliberate: computing sky-view factor per-frame is
expensive, whereas a heightfield's occlusion is static. It also avoids the
screen-space postprocessing path, which broke rendering when tried earlier
(UnrealBloomPass blacked out half the viewport).
"""
import numpy as np
import cv2
from PIL import Image

import segmentation as seg


def sky_view_factor(dsm: np.ndarray, gsd_m: float = 1.0, n_directions: int = 16,
                    max_radius_px: int = 48, n_steps: int = 12) -> np.ndarray:
    """
    Fraction of the sky hemisphere visible from each cell, in [0, 1].

    For each of n_directions azimuths, march outward and track the maximum
    horizon elevation angle encountered. SVF is then the mean of
    cos^2(horizon_angle) across directions -- the standard formulation, which
    weights by projected solid angle rather than raw angle.

    Vectorised over the whole raster per (direction, step), so cost is
    n_directions * n_steps shifts rather than a per-pixel ray loop.
    """
    h, w = dsm.shape
    z = dsm.astype(np.float32)

    # radii spaced geometrically: near-field occluders matter far more than
    # distant ones, so sample them more densely
    radii = np.unique(np.round(
        np.geomspace(1.0, max(max_radius_px, 2), n_steps)).astype(np.int32))

    svf_accum = np.zeros((h, w), dtype=np.float32)

    for d in range(n_directions):
        theta = 2.0 * np.pi * d / n_directions
        dx, dy = np.cos(theta), np.sin(theta)

        max_tan = np.zeros((h, w), dtype=np.float32)

        for r in radii:
            ox, oy = int(round(dx * r)), int(round(dy * r))
            if ox == 0 and oy == 0:
                continue

            shifted = _shift_replicate(z, oy, ox)
            horizontal_dist = float(np.hypot(ox, oy)) * gsd_m
            if horizontal_dist <= 0:
                continue

            tan_angle = (shifted - z) / horizontal_dist
            np.maximum(max_tan, tan_angle, out=max_tan)

        # horizon angle -> visible sky fraction in this azimuth slice.
        # cos^2 weighting accounts for the projected area of the sky band.
        horizon = np.arctan(np.clip(max_tan, 0, None))
        svf_accum += np.cos(horizon) ** 2

    return np.clip(svf_accum / n_directions, 0.0, 1.0)


def _shift_replicate(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift with edge replication, so tile borders don't manufacture cliffs."""
    return cv2.warpAffine(
        a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def curvature_map(dsm: np.ndarray, gsd_m: float = 1.0, blur_px: int = 3) -> np.ndarray:
    """
    Local convexity in [-1, 1] via a difference-of-means (mean curvature
    proxy). Positive on ridges and roof edges, negative in gutters and
    alleys. Used to sharpen structural edges the diffuse texture flattens.
    """
    z = dsm.astype(np.float32)
    smooth = cv2.GaussianBlur(z, (0, 0), sigmaX=blur_px)
    diff = z - smooth

    scale = np.percentile(np.abs(diff), 98) or 1.0
    return np.clip(diff / scale, -1.0, 1.0)


def flatten_water(dsm: np.ndarray, seg_labels: np.ndarray,
                  percentile: float = 35.0, min_area_px: int = 1500,
                  max_relief_frac: float = 0.10, min_fill_ratio: float = 0.35) -> tuple:
    """
    Level genuinely-identified water bodies to a single elevation.

    Standing water is flat by definition, while a monocular depth model returns
    the same rippled surface it gives everything else, so real rivers and lakes
    come out as textured lumps and lose the mirror reflection that makes water
    read as water.

    THE HARD PART IS TRUSTING THE MASK. On RGB-only input, water and shadow are
    close to indistinguishable -- both are dark, low-saturation and smooth. An
    earlier version of this function flattened any water-classified blob over
    40px, which on a real Jacksonville tile meant 294 "water bodies" out of
    5,384 fragments covering 6.7% of the scene. Most were shadows and dark
    roofs, and levelling them flattened real terrain.

    So a component must now clear three independent gates, not just a size
    threshold:

      1. AREA -- a real lake or river is large and contiguous. Small dark
         patches are overwhelmingly shadow.
      2. EXISTING FLATNESS -- water is already low-relief in the DSM. If a
         region has significant height variation it is terrain in shadow, not
         a water surface, and flattening it would destroy real geometry.
      3. SHAPE PLAUSIBILITY -- the component must fill a reasonable fraction
         of its own bounding box. Shadow masks are typically thin, ragged
         slivers cast along building edges.

    Each surviving body is levelled independently, since separate bodies sit
    at genuinely different elevations. A low percentile is used rather than the
    mean because water masks tend to over-spill onto their banks.

    Returns (levelled_dsm, n_bodies_levelled, n_rejected).
    """
    out = dsm.astype(np.float32).copy()
    water = (seg_labels == seg.CLASS_IDX["water"]).astype(np.uint8)
    if water.sum() == 0:
        return out, 0, 0

    # scene-wide relief sets the scale for "flat enough to be water"
    scene_relief = float(np.percentile(dsm, 98) - np.percentile(dsm, 2)) or 1.0

    n, cc, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    levelled, rejected = 0, 0

    # Only components that clear the cheap area and fill-ratio gates get the
    # expensive per-component pixel work. Building `comp = cc == i` for every
    # label instead touches the whole image once per label, which on a
    # 2048x2048 scene with thousands of shadow slivers does not finish.
    areas = stats[1:, cv2.CC_STAT_AREA]
    bws = stats[1:, cv2.CC_STAT_WIDTH]
    bhs = stats[1:, cv2.CC_STAT_HEIGHT]
    fills = areas / np.maximum(bws * bhs, 1)
    passes_gates = (areas >= min_area_px) & (fills >= min_fill_ratio)
    rejected += int((~passes_gates).sum())

    for i in (np.nonzero(passes_gates)[0] + 1):
        comp = cc == i
        vals = dsm[comp]
        relief = float(np.percentile(vals, 95) - np.percentile(vals, 5))
        if relief > max_relief_frac * scene_relief:
            rejected += 1          # too much internal relief to be a water surface
            continue

        out[comp] = float(np.percentile(vals, percentile))
        levelled += 1

    return out, levelled, rejected


def roughness_from_segmentation(seg_labels: np.ndarray) -> np.ndarray:
    """
    Per-class PBR roughness in [0, 1]. Values chosen to match real surface
    behaviour under image-based lighting rather than for stylistic effect:
    water is near-specular, vegetation fully diffuse, built surfaces between.
    """
    table = {
        seg.CLASS_IDX["water"]: 0.04,        # near-mirror; picks up sky reflection
        seg.CLASS_IDX["road"]: 0.55,         # asphalt: slight sheen
        seg.CLASS_IDX["building"]: 0.70,     # roofing membrane / tile
        seg.CLASS_IDX["bare_earth"]: 0.88,
        seg.CLASS_IDX["vegetation"]: 0.97,   # canopy scatters, no coherent highlight
    }
    out = np.full(seg_labels.shape, 0.85, dtype=np.float32)
    for idx, val in table.items():
        out[seg_labels == idx] = val
    # soften class boundaries so materials transition instead of stair-stepping
    return cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)


def _contrast_stretch(x: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0,
                      out_lo: float = 0.30, out_hi: float = 1.0) -> np.ndarray:
    """
    Expand a physically-computed quantity into a visually useful range.

    Raw sky-view factor over mostly-open terrain concentrates near 1.0 (open
    ground genuinely does see almost the whole sky), measured mean 0.967. That
    is correct physics and useless shading -- the map renders essentially
    white. Every terrain-visualisation toolchain contrast-stretches SVF before
    display for this reason.

    This is a monotonic remap: it preserves the ordering of the physical
    quantity (more-occluded stays more-occluded) and only rescales the
    interval. It is display contrast, not a change to the underlying geometry,
    and the un-stretched values remain available via sky_view_factor().
    """
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi - lo < 1e-6:
        return np.full_like(x, out_hi)
    norm = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return out_lo + norm * (out_hi - out_lo)


def neutralise_facades(image_np: np.ndarray, dsm: np.ndarray,
                       gsd_m: float, low_deg: float = 45.0,
                       high_deg: float = 70.0) -> tuple[np.ndarray, float]:
    """
    Replace the texture on near-vertical surfaces with a neutral facade tone.

    A nadir satellite image contains no facade imagery at all -- nothing is
    visible from the side in a top-down capture. The ground mesh uses planar
    UVs, so a wall rising 30 m samples the handful of texels sitting on its
    roof edge and stretches them down its whole height. Viewed from above this
    is invisible; viewed from street level every building looks like melting
    wax, which is the single most damaging artefact in the render.

    Because the UVs are planar, a texel maps one-to-one onto a DSM pixel, so
    the wall texels are exactly the steep-slope pixels and can be treated
    directly. They are blended toward a desaturated, darkened version of
    themselves: the wall keeps a plausible relationship to the building it
    belongs to, but stops advertising stretched roof detail it does not have.

    Blending across a slope band rather than switching at a threshold avoids a
    hard ring around every rooftop.

    Returns (image, fraction of pixels affected).
    """
    z = dsm.astype(np.float32)
    gy, gx = np.gradient(z, max(gsd_m, 1e-6))
    slope = np.sqrt(gx * gx + gy * gy)          # metres rise per metre run

    lo, hi = np.tan(np.radians(low_deg)), np.tan(np.radians(high_deg))
    t = np.clip((slope - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = t * t * (3 - 2 * t)                      # smoothstep

    if t.shape != image_np.shape[:2]:
        t = cv2.resize(t, (image_np.shape[1], image_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    img = image_np.astype(np.float32)
    # Facade tone: a REGIONAL colour, not the per-pixel one.
    #
    # Using each pixel's own colour leaves every wall striped, because the
    # planar UVs stretch one texel column down the entire wall -- so whatever
    # variation exists along the roof edge becomes a vertical stripe metres
    # wide. Blurring first means neighbouring columns agree, and the wall
    # reads as one surface. The blur is wide relative to a building so the
    # tone follows the district rather than the roofline.
    blur_px = max(3, int(round(12.0 / max(gsd_m, 1e-6))) | 1)
    regional = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_px / 3.0)
    lum = regional @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    facade = 0.72 * (0.30 * regional + 0.70 * lum[..., None])

    t3 = t[..., None]
    out = np.clip(img * (1 - t3) + facade * t3, 0, 255).astype(np.uint8)
    return out, float((t > 0.5).mean())


def steepness_weight(dsm: np.ndarray, gsd_m: float, shape: tuple = None,
                     low_deg: float = 45.0, high_deg: float = 70.0) -> np.ndarray:
    """Smooth 0..1 weight rising across a slope band. 1 = effectively a wall."""
    z = dsm.astype(np.float32)
    gy, gx = np.gradient(z, max(gsd_m, 1e-6))
    slope = np.sqrt(gx * gx + gy * gy)
    lo, hi = np.tan(np.radians(low_deg)), np.tan(np.radians(high_deg))
    t = np.clip((slope - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    if shape is not None and t.shape != shape:
        t = cv2.resize(t, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return t


def calm_steep_map(map_img: np.ndarray, dsm: np.ndarray, gsd_m: float,
                   fill) -> np.ndarray:
    """
    Blend a baked map toward a constant on near-vertical surfaces.

    Neutralising the colour texture on walls is not enough on its own: the AO
    and normal maps are baked per-pixel from the same heightfield and then
    stretched down the wall by the same planar UVs, so their variation becomes
    vertical banding exactly where the colour has been calmed. Measured on
    JAX_068 after the texture fix, standard deviation on steep pixels was 82.9
    for AO against 47.3 on flat ground -- the AO map had become the dominant
    source of the striping, not the texture.

    A wall has no meaningful sky-view factor or surface normal in a
    heightfield derived from a nadir image, so a constant is more honest there
    than baked detail.
    """
    t = steepness_weight(dsm, gsd_m, shape=map_img.shape[:2])
    out = map_img.astype(np.float32)
    fill_arr = np.asarray(fill, dtype=np.float32)
    if out.ndim == 3 and fill_arr.ndim == 0:
        fill_arr = np.full(out.shape[2], float(fill), dtype=np.float32)
    t3 = t[..., None] if out.ndim == 3 else t
    return np.clip(out * (1 - t3) + fill_arr * t3, 0, 255).astype(map_img.dtype)


def bake_terrain_maps(dsm: np.ndarray, seg_labels: np.ndarray, gsd_m: float,
                      ao_path: str = None, roughness_path: str = None,
                      metalness_path: str = None,
                      ao_strength: float = 1.0, curvature_weight: float = 0.55,
                      quality: str = "high", stretch: bool = True) -> dict:
    """
    Computes and writes the shading maps. Returns a stats dict.

    quality: 'fast'  -> 8 directions, 24px radius   (preview)
             'high'  -> 16 directions, 48px radius  (default)
             'ultra' -> 32 directions, 96px radius  (final renders)
    """
    presets = {
        "fast":  (8, 24, 8),
        "high":  (16, 48, 12),
        "ultra": (32, 96, 16),
    }
    n_dir, max_r, n_steps = presets.get(quality, presets["high"])

    svf = sky_view_factor(dsm, gsd_m=gsd_m, n_directions=n_dir,
                          max_radius_px=max_r, n_steps=n_steps)
    curv = curvature_map(dsm, gsd_m=gsd_m)

    # Stretch first, so the broad occlusion term actually occupies a visible
    # range before the fine curvature detail is mixed in. Stretching after
    # would compress the curvature contribution back out again.
    svf_display = _contrast_stretch(svf) if stretch else svf

    # sky-view factor supplies broad occlusion; curvature adds the fine edge
    # definition SVF misses at roof scale. Concavities (alleys, gutters,
    # wall-ground junctions) darken; convex edges stay bright.
    ao = svf_display - curvature_weight * np.clip(-curv, 0, 1)
    ao = np.clip(ao, 0.0, 1.0)

    if ao_strength != 1.0:
        ao = 1.0 - (1.0 - ao) * ao_strength
        ao = np.clip(ao, 0.0, 1.0)

    stats = {
        "svf_mean_physical": float(svf.mean()),
        "svf_min_physical": float(svf.min()),
        "ao_mean_display": float(ao.mean()),
        "ao_min_display": float(ao.min()),
        "contrast_stretched": stretch,
        "quality": quality,
        "directions": n_dir,
        "max_radius_px": max_r,
    }

    if ao_path:
        ao_u8 = calm_steep_map((ao * 255).astype(np.uint8), dsm, gsd_m, 200)
        Image.fromarray(ao_u8).save(ao_path)
        stats["ao_path"] = ao_path

    if roughness_path:
        rough = roughness_from_segmentation(seg_labels)
        Image.fromarray((rough * 255).astype(np.uint8)).save(roughness_path)
        stats["roughness_path"] = roughness_path
        stats["roughness_mean"] = float(rough.mean())

    if metalness_path:
        # Water is given partial metalness purely as a rendering device: a
        # dielectric at grazing incidence reflects strongly (Fresnel), but
        # MeshStandardMaterial's default dielectric response at the near-nadir
        # viewing angles typical here is too weak to read as a water surface.
        # Nudging metalness produces the expected mirror behaviour. It is a
        # shading choice on the water class only -- no geometry or elevation
        # is affected, and every other surface stays fully dielectric.
        metal = np.zeros(seg_labels.shape, dtype=np.float32)
        metal[seg_labels == seg.CLASS_IDX["water"]] = 0.55
        metal = cv2.GaussianBlur(metal, (0, 0), sigmaX=1.5)
        Image.fromarray((metal * 255).astype(np.uint8)).save(metalness_path)
        stats["metalness_path"] = metalness_path

    return stats


if __name__ == "__main__":
    import sys
    import time

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    from depth_model import DepthBackbone

    pil = Image.open(img_path).convert("RGB")
    image_np = np.array(pil)
    dsm = DepthBackbone().predict(pil) * 40.0
    seg_labels, _ = seg.segment(image_np)

    for q in ("fast", "high"):
        t0 = time.time()
        s = bake_terrain_maps(dsm, seg_labels, gsd_m=1.0,
                              ao_path=f"ao_{q}.png", roughness_path=f"rough_{q}.png",
                              quality=q)
        print(f"{q}: {time.time()-t0:.1f}s  {s}")
