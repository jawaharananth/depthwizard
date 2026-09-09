"""Check that scene objects actually rest on the ground mesh that is drawn.

Objects are based on the full-resolution ground field, but the terrain the
viewer renders is that field resampled onto a coarser vertex grid and then
interpolated across triangles by the GPU. Those are two different surfaces, and
an object can be correctly based on the first while visibly hovering above the
second. Nothing in the existing geometry validation catches that: a floating
prism is perfectly well-formed, with no degenerate faces and correct winding.

So this measures the only thing that matters to the eye:

    gap = object_vertex_height - rendered_ground_height_at_that_x_z

Objects are compared using their LOWEST vertices, because that is what makes
contact. A class whose lowest vertices all sit above zero is floating as a
class; scattered positive gaps mean individual objects are.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# Below this the gap reads as contact rather than a hovering object. It is
# roughly one ground-mesh cell of vertical slack, which is the most the
# interpolated surface can legitimately differ from the sampled field.
CONTACT_TOL_M = 0.30


def sample_ground(gsmall: np.ndarray, cell_m: float,
                  x_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    """Bilinearly sample the rendered ground surface at world X/Z.

    Bilinear, not nearest: this must reproduce what the GPU shows across the
    face of a triangle, and nearest-neighbour would report the vertex height
    instead of the interpolated height the object actually sits over.

    World convention matches build_ground_mesh and build_prisms: X runs east
    from the grid origin and Z runs NEGATIVE south, so the row index is -Z.
    """
    h, w = gsmall.shape
    gx = np.clip(x_m / cell_m, 0, w - 1.001)
    gy = np.clip(-z_m / cell_m, 0, h - 1.001)

    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (gx - x0).astype(np.float32)
    fy = (gy - y0).astype(np.float32)

    top = gsmall[y0, x0] * (1 - fx) + gsmall[y0, x1] * fx
    bot = gsmall[y1, x0] * (1 - fx) + gsmall[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def check(verts: np.ndarray, gsmall: np.ndarray, cell_m: float,
          name: str, base_quantile: float = 5.0, flat: bool = False) -> Dict:
    """Measure how the lowest vertices of one mesh class sit on the terrain.

    ``flat`` marks a class whose geometry is a single horizontal surface -- a
    water plane -- where EVERY vertex is a contact vertex. Taking the lowest
    slice of those reports the deepest corner of the plane against the highest
    bank, which reads as a submerged object even when the surface as a whole
    sits correctly. Extruded classes are the opposite: their vertices run from
    the base to the roof, so the low slice is the only meaningful one.
    """
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    if verts.size == 0:
        return {"name": name, "n": 0, "empty": True}

    gy = sample_ground(gsmall, cell_m, verts[:, 0], verts[:, 2])
    gap = verts[:, 1] - gy

    # The base population: the lowest slice of vertices, which is where contact
    # happens. Using every vertex would average roof heights into the answer and
    # report a tower block as floating by its own height.
    if flat:
        base_gap = gap
    else:
        cut = np.percentile(gap, base_quantile)
        base_gap = gap[gap <= cut] if np.isfinite(cut) else gap

    return {
        "name": name,
        "n": int(verts.shape[0]),
        "empty": False,
        "min_gap_m": float(gap.min()),
        "base_median_m": float(np.median(base_gap)),
        "base_p90_m": float(np.percentile(base_gap, 90)),
        "floating": bool(np.median(base_gap) > CONTACT_TOL_M),
        "frac_above_tol": float((gap > CONTACT_TOL_M).mean()),
    }


def report(results) -> bool:
    """Print one line per class. Returns False if any class floats."""
    ok = True
    print("      grounding check (object base vs the terrain actually drawn):")
    for r in results:
        if r.get("empty"):
            print(f"  [ -- ] {r['name']:<12} no geometry")
            continue
        tag = "FLOAT" if r["floating"] else "OK  "
        if r["floating"]:
            ok = False
        print(f"  [{tag}] {r['name']:<12} "
              f"lowest gap {r['min_gap_m']:+6.2f} m   "
              f"base median {r['base_median_m']:+6.2f} m   "
              f"base p90 {r['base_p90_m']:+6.2f} m")
    return ok
