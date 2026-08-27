"""
Adaptive tessellation of the height field.

A uniform grid spends its vertex budget evenly, which is exactly wrong for
terrain: a flat car park consumes as many triangles as a roofline. At full
2048px resolution that costs ~4.2M vertices and a ~235MB GLB -- slow to
download and slow to parse -- while most of those vertices describe nothing.

This places vertices by information content instead. Roof edges, building
outlines, curbs and steep relief get sampled at full resolution; flat regions
get the minimum needed to stay planar. The result carries the same visible
detail at a fraction of the geometry.

Method: score every pixel by local curvature and gradient, greedily select
high-scoring pixels subject to a spacing constraint that relaxes in flat
areas, add a coarse background lattice and a dense tile boundary, then
Delaunay-triangulate the selected points in 2D and lift them onto the height
field.

Delaunay is safe here because the domain is a convex rectangle: the
triangulation covers it exactly, with no holes and no self-intersection.
"""
import numpy as np
import cv2
from scipy.spatial import Delaunay


def detail_map(dsm: np.ndarray, seg_labels: np.ndarray = None) -> np.ndarray:
    """
    Per-pixel importance in [0,1]. High where geometry carries information.

    Combines second-derivative response (curvature -- ridges, eaves, curbs)
    with first-derivative magnitude (slope -- walls, embankments). Building
    pixels get a floor applied, because a flat roof is geometrically boring
    but visually load-bearing: under-sampling it makes the silhouette ragged.
    """
    z = dsm.astype(np.float32)

    lap = np.abs(cv2.Laplacian(z, cv2.CV_32F, ksize=3))
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    def norm(a):
        hi = np.percentile(a, 99.0)
        return np.clip(a / max(hi, 1e-6), 0.0, 1.0)

    score = 0.65 * norm(lap) + 0.35 * norm(grad)

    if seg_labels is not None:
        import segmentation as seg
        building = seg_labels == seg.CLASS_IDX["building"]
        # dilate so the vertex ring around a footprint is sampled too --
        # that ring is what forms the wall silhouette
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        near_building = cv2.dilate(building.astype(np.uint8), k).astype(bool)
        score = np.maximum(score, np.where(near_building, 0.55, 0.0))

    return cv2.GaussianBlur(score, (0, 0), sigmaX=1.0)


def _select_points(score: np.ndarray, budget: int, min_spacing: int = 0,
                    coarse_step: int = 24, boundary_step: int = 4):
    """
    Choose sample locations: dense tile boundary, coarse background lattice,
    then the highest-scoring remaining pixels under a DETAIL-DEPENDENT spacing
    constraint.

    Spacing varies with score rather than being a single constant. A fixed
    spacing is just a coarser uniform grid -- measured, spacing=2 capped
    selection at ~22k points on an 850x637 tile no matter how large the budget,
    because each point blocks a 5x5 neighbourhood. Varying it is the whole
    point of adaptive sampling: strong edges get every pixel, moderate detail
    gets every other pixel, flat ground is left to the coarse lattice.

    Spacing is enforced on an occupancy grid rather than by distance queries,
    which keeps selection linear in the number of candidates.
    """
    h, w = score.shape
    taken = np.zeros((h, w), dtype=bool)
    pts = []

    def spacing_for(s: float) -> int:
        if s >= 0.45:
            return min_spacing        # strong edge: sample at full resolution
        if s >= 0.22:
            return max(min_spacing, 1)
        return max(min_spacing, 3)

    def try_add(y, x, sp=None):
        if y < 0 or x < 0 or y >= h or x >= w:
            return False
        if sp is None:
            sp = min_spacing
        if sp <= 0:
            if taken[y, x]:
                return False
        else:
            y0, y1 = max(0, y - sp), min(h, y + sp + 1)
            x0, x1 = max(0, x - sp), min(w, x + sp + 1)
            if taken[y0:y1, x0:x1].any():
                return False
        taken[y, x] = True
        pts.append((x, y))
        return True

    # 1. tile boundary, densely -- a ragged edge is highly visible and Delaunay
    #    needs the hull well-defined
    for x in range(0, w, boundary_step):
        pts.append((x, 0)); taken[0, x] = True
        pts.append((x, h - 1)); taken[h - 1, x] = True
    for y in range(0, h, boundary_step):
        pts.append((0, y)); taken[y, 0] = True
        pts.append((w - 1, y)); taken[y, h - 1 if False else w - 1] = True
    for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if not taken[corner[1], corner[0]]:
            pts.append(corner); taken[corner[1], corner[0]] = True

    # 2. coarse background lattice so flat regions stay planar rather than
    #    collapsing into long slivers
    for y in range(coarse_step, h - 1, coarse_step):
        for x in range(coarse_step, w - 1, coarse_step):
            try_add(y, x, sp=0)

    # 3. detail-driven points, strongest first
    remaining = budget - len(pts)
    if remaining > 0:
        flat = score.ravel()
        # only consider pixels with meaningful score; sorting the full raster
        # is wasteful when most of it is flat ground
        thresh = max(float(np.percentile(flat, 60)), 1e-4)
        cand = np.flatnonzero(flat >= thresh)
        order = cand[np.argsort(-flat[cand])]
        for idx in order:
            if remaining <= 0:
                break
            y, x = divmod(int(idx), w)
            if try_add(y, x, sp=spacing_for(float(flat[idx]))):
                remaining -= 1

    return np.array(pts, dtype=np.float64)


def build_adaptive_ground_mesh(dsm: np.ndarray, gsd_x_m: float, gsd_y_m: float,
                               seg_labels: np.ndarray = None,
                               budget_verts: int = 400_000,
                               min_spacing: int = 0):
    """
    Returns (vertices Nx3, uvs Nx2, faces Mx3) matching build_ground_mesh's
    contract, so it is a drop-in alternative.
    """
    h, w = dsm.shape
    score = detail_map(dsm, seg_labels)
    pts = _select_points(score, budget_verts, min_spacing=min_spacing)

    tri = Delaunay(pts)
    faces = tri.simplices.astype(np.int64)

    xs = pts[:, 0]
    ys = pts[:, 1]
    zi = dsm[np.clip(ys.astype(np.int32), 0, h - 1),
             np.clip(xs.astype(np.int32), 0, w - 1)]

    vertices = np.stack([xs * gsd_x_m, zi, -ys * gsd_y_m], axis=-1).astype(np.float32)
    uvs = np.stack([xs / max(w - 1, 1), 1.0 - ys / max(h - 1, 1)], axis=-1).astype(np.float32)

    # Winding is left as scipy returns it. An earlier version reversed it on
    # the assumption that flipping Z into world space demanded it -- that was
    # a guess, and it was wrong: it pointed every ground normal downward, so
    # the entire terrain was back-face culled and only buildings rendered.
    # Verified directly: with vertices built as (x, height, -y), Delaunay's
    # native simplex order already yields +Y (upward) face normals.
    #
    # Asserted rather than trusted, because a terrain that renders completely
    # invisible is both catastrophic and easy to miss in a headless pipeline.
    _assert_upward(vertices, faces)
    return vertices, uvs, faces


def _assert_upward(vertices: np.ndarray, faces: np.ndarray, sample: int = 4096):
    """Fail loudly if ground faces point downward (would be back-face culled)."""
    if len(faces) == 0:
        return
    idx = faces if len(faces) <= sample else faces[
        np.random.default_rng(0).choice(len(faces), sample, replace=False)]
    v0, v1, v2 = vertices[idx[:, 0]], vertices[idx[:, 1]], vertices[idx[:, 2]]
    ny = np.cross(v1 - v0, v2 - v0)[:, 1]
    up_fraction = float((ny > 0).mean())
    if up_fraction < 0.5:
        raise RuntimeError(
            f"Ground mesh winding is inverted: only {up_fraction:.1%} of faces point "
            f"upward. The terrain would be back-face culled and render invisible.")


if __name__ == "__main__":
    import sys
    import time
    from PIL import Image
    from depth_model import DepthBackbone
    import segmentation as seg
    import dsm_refine
    import mesh_generation as mg

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    pil = Image.open(img_path).convert("RGB")
    image_np = np.array(pil)
    dsm = dsm_refine.refine_dsm(DepthBackbone().predict(pil), image_np) * 40.0
    seg_labels, _ = seg.segment(image_np)

    h, w = dsm.shape
    print(f"source {w}x{h}")

    t0 = time.time()
    uv_, uu_, uf_ = mg.build_ground_mesh(dsm, 1.0, 1.0)
    t_uniform = time.time() - t0
    print(f"uniform : {len(uv_):>8,} verts  {len(uf_):>9,} faces  {t_uniform:5.1f}s")

    for budget in (150_000, 300_000):
        t0 = time.time()
        av, au, af = build_adaptive_ground_mesh(dsm, 1.0, 1.0, seg_labels, budget_verts=budget)
        dt = time.time() - t0
        ratio = len(af) / max(len(uf_), 1)
        print(f"adaptive: {len(av):>8,} verts  {len(af):>9,} faces  {dt:5.1f}s  "
              f"({ratio*100:4.1f}% of uniform faces)")
