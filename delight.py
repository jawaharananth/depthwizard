"""
Geometry-aware delighting of a satellite texture.

A satellite photograph has the capture's shadows baked in. Once the scene is
rendered in 3D with its own shadow-casting sun, those baked shadows double up
with the rendered ones, and they cannot move when the sun does. Removing
them gives an (approximate) albedo texture, so every shadow on screen comes
from real reconstructed geometry and a real sun position -- including at any
other time of day.

  1. WHERE shadows must be: march the reconstructed height field (buildings
     at their TRUE base positions -- roof polygon minus lean x height -- plus
     vegetation) toward the capture sun. A point is geometrically shadowed if
     anything along that ray rises above the sun line.
  2. WHERE shadows are: of those, keep pixels the photo shows as dark relative
     to the local sunlit level. Geometry says where to look; the image says
     what is actually shaded (both errors are then required to coincide).
  3. UNDO them: per-channel gain = sunlit / shadowed, measured locally across
     real shadow boundaries (just outside vs just inside), in blocks, and
     smoothly interpolated. Skylight makes shadows bluish, so per-channel
     gains also remove the cast, not only the darkening.

Standard shadow-compensation practice in remote sensing (boundary-ratio
illumination compensation), with the mask driven by this scene's own 3D
reconstruction and fitted sun instead of image heuristics alone.
"""
import cv2
import numpy as np


def geometric_shadow(height: np.ndarray, gsd_m: float, sun_az_deg: float,
                     sun_el_deg: float, max_dist_m: float = None,
                     tol_m: float = 0.4) -> np.ndarray:
    """uint8 mask, 1 where the height field occludes the sun."""
    h = height.astype(np.float32)
    tan_el = np.tan(np.radians(sun_el_deg))
    relief = float(np.percentile(h, 99.9) - np.percentile(h, 1))
    if max_dist_m is None:
        max_dist_m = relief / max(tan_el, 1e-3)
    ux, uy = np.sin(np.radians(sun_az_deg)), -np.cos(np.radians(sun_az_deg))  # toward sun
    n = int(np.ceil(max_dist_m / gsd_m))
    acc = np.full_like(h, -1e9)
    H, W = h.shape
    for k in range(1, n + 1):
        d_px = float(k)
        M = np.float32([[1, 0, d_px * ux], [0, 1, d_px * uy]])
        s = cv2.warpAffine(h, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
        np.maximum(acc, s - d_px * gsd_m * tan_el, out=acc)
    return (acc > h + tol_m).astype(np.uint8)


def _block_gains(img: np.ndarray, mask: np.ndarray, block: int, r_in: int, r_out: int):
    """Per-block, per-channel sunlit/shadow ratio from boundary rings."""
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode(mask, k(r_in)) & (1 - cv2.erode(mask, k(r_out)))
    outer = cv2.dilate(mask, k(r_out)) & (1 - cv2.dilate(mask, k(r_in)))
    H, W = mask.shape
    by, bx = (H + block - 1) // block, (W + block - 1) // block
    g = np.full((by, bx, 3), np.nan, np.float32)
    for j in range(by):
        for i in range(bx):
            sl = (slice(j * block, (j + 1) * block), slice(i * block, (i + 1) * block))
            a, b = img[sl][outer[sl] > 0], img[sl][inner[sl] > 0]
            if len(a) >= 150 and len(b) >= 150:
                g[j, i] = np.median(a, 0) / np.maximum(np.median(b, 0), 3.0)
    glob = np.nanmedian(g.reshape(-1, 3), 0)
    glob = np.where(np.isfinite(glob), glob, 2.0)
    for c in range(3):
        ch = g[:, :, c]
        ch[~np.isfinite(ch)] = glob[c]
    return np.clip(g, 1.0, 5.0), np.clip(glob, 1.0, 5.0)


def delight(texture: np.ndarray, geo_mask_work: np.ndarray, strip: int = 1024) -> tuple:
    """
    texture: HxWx3 uint8 at any resolution covering the same bounds as the
    working grid geo_mask_work (hxw). Returns (delit uint8, stats dict).
    """
    Ht, Wt = texture.shape[:2]
    h, w = geo_mask_work.shape
    # Working-resolution copy: darkness gate, gains.
    tw = cv2.resize(texture, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    Lw = tw @ np.array([0.299, 0.587, 0.114], np.float32)
    lit = (1 - geo_mask_work).astype(np.float32)
    sig = 12.0
    lit_ref = cv2.GaussianBlur(Lw * lit, (0, 0), sig) / np.maximum(
        cv2.GaussianBlur(lit, (0, 0), sig), 1e-3)
    m_w = (geo_mask_work > 0) & (Lw < 0.80 * lit_ref)
    m_w = cv2.morphologyEx(m_w.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    gains_blk, glob = _block_gains(tw, m_w, block=128, r_in=2, r_out=6)
    # Smooth gain field at working res.
    gain_w = cv2.resize(gains_blk, (w, h), interpolation=cv2.INTER_CUBIC)
    gain_w = np.clip(cv2.GaussianBlur(gain_w, (0, 0), 20), 1.0, 5.0)
    geo_dil = cv2.dilate(geo_mask_work, np.ones((3, 3), np.uint8))

    out = np.empty_like(texture)
    sy = Ht / h
    pad = 16
    for y0 in range(0, Ht, strip):
        y1 = min(Ht, y0 + strip)
        ya, yb = max(0, y0 - pad), min(Ht, y1 + pad)
        wy0, wy1 = int(np.floor(ya / sy)), min(h, int(np.ceil(yb / sy)) + 1)
        sub = texture[ya:yb].astype(np.float32)
        rows = yb - ya
        up = lambda a, interp: cv2.resize(
            a[wy0:wy1], (Wt, int(round((wy1 - wy0) * sy))), interpolation=interp)
        # crop the upsampled working-res strip onto exactly [ya, yb)
        off = int(round(ya - wy0 * sy))
        geo_hi = up(geo_dil, cv2.INTER_NEAREST)[off:off + rows]
        ref_hi = up(lit_ref, cv2.INTER_LINEAR)[off:off + rows]
        g_hi = up(gain_w, cv2.INTER_LINEAR)[off:off + rows]
        L = sub @ np.array([0.299, 0.587, 0.114], np.float32)
        m = ((geo_hi > 0) & (L < 0.80 * ref_hi)).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        res = sub * (1.0 + (g_hi - 1.0) * m[..., None])
        out[y0:y1] = np.clip(res[y0 - ya:y1 - ya], 0, 255).astype(np.uint8)
    stats = {"shadow_fraction_geometric": float(geo_mask_work.mean()),
             "shadow_fraction_removed": float(m_w.mean()),
             "median_gain_rgb": [round(float(v), 2) for v in glob]}
    return out, stats
