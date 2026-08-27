"""
Bare-earth terrain extraction (DSM -> DTM).

A DSM is a *surface* model: it includes buildings, tree canopy, everything the
sensor saw from above. A DTM is the *terrain* underneath.

Why this matters structurally, not cosmetically: the mesh pipeline was building
its ground surface directly from the DSM, so the terrain already bulged upward
at every building. It then extruded a separate building box at the same
location, from a single scene-wide base elevation. Two consequences, both
measured:

  * every building existed twice -- once as a bump in the "ground", once as a
    box occupying the same space;
  * because the box base was a global percentile rather than the local terrain,
    30 of 40 buildings on a test scene had their base BELOW the ground beside
    them, i.e. sunk into the surface instead of standing on it.

Separating terrain from structures fixes both: the ground mesh becomes actual
ground, and each building sits on the terrain height at its own footprint.

Method is morphological, which suits the noisy, relative-scale elevation this
project works with better than a plane-fit or a cloth simulation: buildings are
removed by grey-scale opening at a kernel larger than the biggest building,
then the removed regions are reconstructed by interpolating terrain inward from
their edges.
"""
import numpy as np
import cv2

import segmentation as seg


def estimate_dtm(dsm: np.ndarray, seg_labels: np.ndarray = None,
                 max_structure_px: int = 60, smooth_px: float = 2.0) -> np.ndarray:
    """
    Estimate bare-earth terrain from a surface model.

    max_structure_px: the largest structure to remove, in pixels. Anything
        wider than this is treated as terrain (a hill, an embankment) rather
        than a building, which is the correct behaviour -- a morphological
        filter cannot distinguish a very large flat roof from a plateau, so
        the parameter states the assumption explicitly.
    """
    z = dsm.astype(np.float32)

    # Which pixels are NOT terrain. When a semantic mask is available, use it
    # directly -- it is far more reliable than inferring structures from the
    # elevation alone. Otherwise fall back to morphological detection.
    if seg_labels is not None:
        structure = (
            (seg_labels == seg.CLASS_IDX["building"]) |
            (seg_labels == seg.CLASS_IDX["vegetation"])
        ).astype(np.uint8)
    else:
        k = max(3, int(max_structure_px) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(z, cv2.MORPH_OPEN, kernel)
        structure = ((z - opened) > 0.05 * (z.max() - z.min())).astype(np.uint8)

    if not structure.any():
        return cv2.GaussianBlur(z, (0, 0), sigmaX=smooth_px)

    # Grow the mask so the terrain samples used for interpolation come from
    # clean ground, not the shadowed pixels hugging a wall base.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hole = cv2.dilate(structure, grow).astype(bool)

    # Fill structure areas by pulling terrain inward from the surrounding
    # ground. distanceTransform gives, for every masked pixel, the nearest
    # unmasked pixel; sampling elevation there is a nearest-neighbour
    # extrapolation of the terrain under the structure. This replaces an
    # earlier inpaint-based version that returned values at or above the
    # original surface, leaving buildings with a median height above terrain
    # of exactly 0.00 -- i.e. the "terrain" was still following the rooftops.
    filled = z.copy()
    if hole.any():
        _, labels = cv2.distanceTransformWithLabels(
            hole.astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(~hole)
        # label i corresponds to the i-th zero (unmasked) pixel, 1-indexed
        order = np.argsort(labels[~hole])
        src_y = ys[order]
        src_x = xs[order]
        idx = np.clip(labels[hole] - 1, 0, len(src_y) - 1)
        filled[hole] = z[src_y[idx], src_x[idx]]

    # Terrain is smooth by nature; this also blends the nearest-neighbour
    # patches into a continuous surface.
    dtm = cv2.GaussianBlur(filled, (0, 0), sigmaX=max(smooth_px, 3.0))

    # Terrain can never sit above the observed surface.
    return np.minimum(dtm, z)


def _to_u16(a: np.ndarray):
    """cv2.inpaint needs 8U or 16U; preserve resolution via 16-bit scaling."""
    lo, hi = float(a.min()), float(a.max())
    scale = 65535.0 / max(hi - lo, 1e-6)
    return ((a - lo) * scale).astype(np.uint16)


def _from_u16(u: np.ndarray, ref: np.ndarray):
    lo, hi = float(ref.min()), float(ref.max())
    return u.astype(np.float32) / 65535.0 * max(hi - lo, 1e-6) + lo


def local_ground_height(dtm: np.ndarray, poly_px: np.ndarray,
                        percentile: float = 50.0) -> float:
    """
    Terrain elevation beneath one building footprint.

    This replaces a single scene-wide base elevation. Using a global value
    meant any building standing on higher ground was extruded from below its
    own terrain and appeared buried.
    """
    pts = np.round(poly_px).astype(np.int32)
    x0, y0, w, h = cv2.boundingRect(pts)
    w, h = max(w, 1), max(h, 1)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts - [x0, y0]], 1)

    y1, x1 = min(y0 + h, dtm.shape[0]), min(x0 + w, dtm.shape[1])
    y0c, x0c = max(y0, 0), max(x0, 0)
    if y1 <= y0c or x1 <= x0c:
        return float(np.percentile(dtm, 10))

    sub = dtm[y0c:y1, x0c:x1]
    sub_mask = mask[y0c - y0:y1 - y0, x0c - x0:x1 - x0].astype(bool)
    vals = sub[sub_mask]
    if vals.size == 0:
        return float(np.percentile(dtm, 10))
    return float(np.percentile(vals, percentile))


def structure_height_stats(dsm: np.ndarray, dtm: np.ndarray,
                           seg_labels: np.ndarray) -> dict:
    """Sanity metrics: buildings should stand above the terrain, by a positive amount."""
    building = seg_labels == seg.CLASS_IDX["building"]
    if building.sum() == 0:
        return {"n_building_px": 0}
    agl = (dsm - dtm)[building]
    return {
        "n_building_px": int(building.sum()),
        "mean_agl": float(agl.mean()),
        "median_agl": float(np.median(agl)),
        "negative_fraction": float((agl < 0).mean()),
    }


if __name__ == "__main__":
    import sys
    from PIL import Image
    from depth_model import DepthBackbone

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    pil = Image.open(img_path).convert("RGB")
    image_np = np.array(pil)

    dsm = DepthBackbone().predict(pil) * 40.0
    seg_labels, _ = seg.segment(image_np)

    dtm = estimate_dtm(dsm, seg_labels)
    s = structure_height_stats(dsm, dtm, seg_labels)

    print(f"DSM range {dsm.min():.2f} .. {dsm.max():.2f}")
    print(f"DTM range {dtm.min():.2f} .. {dtm.max():.2f}")
    print(f"building pixels: {s['n_building_px']}")
    print(f"  mean height above terrain   {s['mean_agl']:.2f}")
    print(f"  median height above terrain {s['median_agl']:.2f}")
    print(f"  fraction below terrain      {s['negative_fraction']*100:.1f}%  (want ~0%)")
