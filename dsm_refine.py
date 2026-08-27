"""
Edge-aware DSM refinement.

Monocular depth networks produce smooth, blobby elevation: a building's wall
becomes a soft ramp several pixels wide rather than a step. That blur is the
single most visible reason reconstructed geometry reads as "an AI depth map"
instead of a building -- roof edges round off, walls slope, and small
structures dissolve into the ground.

The RGB image, however, is perfectly registered with the DSM and *does* have a
sharp edge exactly where the wall is. Guided filtering (He, Sun & Tang, 2013)
transfers that edge structure onto the elevation: it smooths the DSM where the
image is flat, and preserves discontinuities where the image has them.

This is a well-established depth-refinement technique, not invention -- the
guide provides evidence for *where* a depth discontinuity lies, while the DSM
still decides *how large* it is. No elevation is created from nothing.

IMPORTANT SCOPE NOTE
--------------------
This modifies elevation values. It is applied to the mesh/visualisation path
only until it has been benchmarked against DFC2019 LiDAR ground truth. If it
measurably improves RMSE it can be promoted to the exported DSM as well; until
that measurement exists, the scientific GeoTIFF stays un-refined.
"""
import numpy as np
import cv2


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8,
                  eps: float = 1e-3) -> np.ndarray:
    """
    He et al. guided filter, grayscale guide.

    Output is a locally-linear function of the guide: q = a*I + b, with a and b
    solved per-window. Where the guide has an edge, `a` is large and the edge
    transfers to the output; where the guide is flat, `a` collapses toward zero
    and the result is a local mean (smoothing).

    guide, src: float32, same shape, guide normalised to roughly [0,1].
    eps: regularisation. Smaller = sharper edge transfer, more noise transfer.
    """
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    k = (radius * 2 + 1, radius * 2 + 1)

    mean_g = cv2.boxFilter(g, -1, k)
    mean_p = cv2.boxFilter(p, -1, k)
    mean_gp = cv2.boxFilter(g * p, -1, k)
    mean_gg = cv2.boxFilter(g * g, -1, k)

    cov_gp = mean_gp - mean_g * mean_p
    var_g = mean_gg - mean_g * mean_g

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = cv2.boxFilter(a, -1, k)
    mean_b = cv2.boxFilter(b, -1, k)
    return mean_a * g + mean_b


def refine_dsm(dsm: np.ndarray, image_np: np.ndarray,
               radius: int = 6, eps: float = 2e-4,
               iterations: int = 2, edge_boost: float = 0.5) -> np.ndarray:
    """
    Sharpen DSM discontinuities using the co-registered image as a guide.

    iterations: repeated guided filtering progressively squares off edges that
                a single pass leaves partially ramped.
    edge_boost: how much of the (refined - original) difference to add back on
                top of the refined result. 0 = plain guided filter; higher
                values push the step steeper. Kept below 1.0 because
                over-boosting manufactures ringing at edges, which looks like
                a wall that leans outward.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Normalise elevation into a comparable range so eps means the same thing
    # regardless of whether the DSM is in metres, relative units, or exaggerated.
    lo, hi = float(np.percentile(dsm, 0.5)), float(np.percentile(dsm, 99.5))
    span = max(hi - lo, 1e-6)
    d = ((dsm - lo) / span).astype(np.float32)

    refined = d
    for _ in range(max(1, iterations)):
        refined = guided_filter(gray, refined, radius=radius, eps=eps)

    if edge_boost > 0:
        # unsharp-style: reinforce where refinement moved the surface, which is
        # precisely at the discontinuities the guide identified
        detail = refined - cv2.GaussianBlur(refined, (0, 0), sigmaX=1.5)
        refined = refined + edge_boost * detail

    return (refined * span + lo).astype(np.float32)


def prismify_buildings(dsm: np.ndarray, seg_labels: np.ndarray,
                       dtm: np.ndarray, min_area_px: int = 400,
                       roof_percentile: float = 85.0) -> tuple[np.ndarray, int]:
    """
    Give each building footprint a flat roof at one height, so its walls
    become vertical.

    Monocular depth returns a smooth field. Where a real building has a
    vertical wall, the model produces a ramp several metres wide, and the
    ground heightfield inherits it -- every roof slumps into the street like
    melting wax when viewed from a low angle. Neutralising the wall texture
    helps the colour but not the shape, because the shape is genuinely wrong:
    buildings are prismatic and the ramp is an artefact of the estimator, not
    a feature of the scene.

    Each segmented building region is therefore set to a single roof height
    (a high percentile of its own pixels, which resists the blurred skirt
    pulling the value down), leaving the step to the neighbouring terrain to
    happen across one pixel.

    This trusts the segmentation: a region that is not a building gets
    flattened anyway, and a building the segmenter misses keeps its ramp. So
    it is applied only to regions large enough to be structures, and the count
    of regions altered is returned for the caller to report.
    """
    mask = (seg_labels == 0).astype(np.uint8)   # CLASS_IDX["building"]
    if not mask.any():
        return dsm.astype(np.float32), 0

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = dsm.astype(np.float32).copy()
    changed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        x1 = x0 + stats[i, cv2.CC_STAT_WIDTH]
        y1 = y0 + stats[i, cv2.CC_STAT_HEIGHT]
        sub = lab[y0:y1, x0:x1] == i
        vals = dsm[y0:y1, x0:x1][sub]
        if vals.size == 0:
            continue
        roof = float(np.percentile(vals, roof_percentile))
        # Never push a roof below the terrain it stands on -- that would bury
        # the building rather than square it off.
        floor = float(np.percentile(dtm[y0:y1, x0:x1][sub], 50))
        out[y0:y1, x0:x1][sub] = max(roof, floor)
        changed += 1

    return out, changed


def edge_sharpness(dsm: np.ndarray) -> float:
    """
    Mean gradient magnitude at strong edges -- a scalar proxy for how crisp
    the elevation discontinuities are. Used to verify refinement actually
    sharpened something rather than just shifting values around.
    """
    gx = cv2.Sobel(dsm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dsm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, 95)
    strong = mag[mag >= thresh]
    return float(strong.mean()) if strong.size else 0.0


if __name__ == "__main__":
    import sys
    from PIL import Image
    from depth_model import DepthBackbone
    import time

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    pil = Image.open(img_path).convert("RGB")
    image_np = np.array(pil)
    dsm = DepthBackbone().predict(pil) * 40.0

    t0 = time.time()
    refined = refine_dsm(dsm, image_np)
    dt = time.time() - t0

    before = edge_sharpness(dsm)
    after = edge_sharpness(refined)
    print(f"refine: {dt:.2f}s")
    print(f"edge sharpness  before {before:.4f}  after {after:.4f}  "
          f"({after / max(before, 1e-9):.2f}x)")
    print(f"elevation range preserved: "
          f"before [{dsm.min():.2f}, {dsm.max():.2f}]  "
          f"after [{refined.min():.2f}, {refined.max():.2f}]")

    # side-by-side visual
    def norm8(a):
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    h, w = dsm.shape
    y0, x0 = h // 3, w // 4
    crop = (slice(y0, y0 + 220), slice(x0, x0 + 300))
    pair = np.hstack([norm8(dsm[crop]), norm8(refined[crop])])
    Image.fromarray(pair).resize((pair.shape[1] * 2, pair.shape[0] * 2),
                                  Image.NEAREST).save("refine_compare.png")
    print("wrote refine_compare.png (left: original, right: refined)")
