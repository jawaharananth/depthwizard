"""Inundation by flood fill on the real terrain, not a plane at an elevation.

A horizontal plane at height h marks every pixel below h as flooded. That is
wrong in a specific and important way: it floods ground that water cannot
physically reach. A basin on the far side of a ridge, a depression with no
channel to the river, the inside of an embanked compound -- all sit below the
water level and none of them fill, because water arrives by flowing from
somewhere, and a plane has no notion of where it came from.

So this fills from a SEED. Water starts where the operator says the water is and
spreads only to terrain that is both below the level and connected to the seed
through terrain that is also below it. A ridge between the seed and a low basin
keeps the basin dry, which is what actually happens.

WHAT THIS OUTPUTS THAT A PLANE CANNOT

Because the fill knows which cells hold water and how deep each one is, it can
report volume and per-structure depth rather than only an extent:

  * buildings inundated, and how deep the water is at each
  * area affected, in real square metres
  * water volume, by summing depth x cell area

All three are metric quantities and all three are refused on a Tier C scene.
A relative-scale height field has no metres, so an area in "m2" or a volume in
"m3" computed from it would be arithmetic on units that do not exist.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2


def fill_from_seed(ground: np.ndarray, level_m: float,
                   seed_rc: Tuple[int, int],
                   gsd_m: float) -> Dict:
    """Flood fill terrain below ``level_m`` connected to ``seed_rc``.

    ``ground`` is the bare-earth surface in the same units as ``level_m``.
    Returns the wet mask, per-cell depth, and metric summaries.
    """
    if not np.isfinite(level_m):
        raise ValueError("level must be finite")
    r, c = int(seed_rc[0]), int(seed_rc[1])
    H, W = ground.shape
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError(f"seed {seed_rc} outside the {H}x{W} grid")

    below = np.isfinite(ground) & (ground <= level_m)
    if not below[r, c]:
        # The seed is above the water line. Reporting an empty flood is correct
        # but unhelpful, so say why: the operator has almost certainly put the
        # seed on a roof or a bank rather than in the channel.
        return {"wet": np.zeros_like(below), "depth": np.zeros_like(ground),
                "seed_dry": True,
                "reason": f"seed terrain is {float(ground[r, c]):.2f}, above the "
                          f"{level_m:.2f} level -- place the seed in the water",
                "cells": 0, "area_m2": 0.0, "volume_m3": 0.0,
                "max_depth_m": 0.0, "mean_depth_m": 0.0}

    # Connected component containing the seed. 4-connectivity, not 8: a diagonal
    # touch between two cells is not a channel water can flow through, and
    # 8-connectivity leaks a flood across a diagonal ridge line one cell wide.
    n, labels = cv2.connectedComponents(below.astype(np.uint8), connectivity=4)
    wet = labels == labels[r, c]

    depth = np.where(wet, level_m - ground, 0.0).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth = np.maximum(depth, 0.0)

    cell_area = float(gsd_m) * float(gsd_m)
    cells = int(wet.sum())
    return {
        "wet": wet,
        "depth": depth,
        "seed_dry": False,
        "cells": cells,
        "area_m2": cells * cell_area,
        "volume_m3": float(depth.sum()) * cell_area,
        "max_depth_m": float(depth.max()) if cells else 0.0,
        "mean_depth_m": float(depth[wet].mean()) if cells else 0.0,
        # How much a plane would have claimed. The gap between the two is the
        # unreachable ground -- basins below the level with no path to the seed.
        "plane_cells": int(below.sum()),
        "unreachable_cells": int(below.sum()) - cells,
    }


def buildings_inundated(result: Dict, footprints: Sequence[np.ndarray],
                        records: Optional[Sequence[Dict]] = None,
                        min_frac: float = 0.15) -> List[Dict]:
    """Which structures stand in the water, and how deep it is at each.

    Depth is taken at the footprint's own cells rather than at its centroid: a
    building on a slope has water at its downhill wall and none at its uphill
    one, and a single centroid sample reports whichever the centre happened to
    be.
    """
    depth = result["depth"]
    wet = result["wet"]
    H, W = depth.shape
    out = []
    for i, poly in enumerate(footprints):
        p = np.asarray(poly, np.int32)
        if len(p) < 3:
            continue
        x, y, w, h = cv2.boundingRect(p)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, [p - [x0, y0]], 1)
        m = sub.astype(bool)
        if not m.any():
            continue
        wsub = wet[y0:y1, x0:x1][m]
        frac = float(wsub.mean())
        if frac < min_frac:
            continue
        dsub = depth[y0:y1, x0:x1][m][wsub]
        rec = (records[i] if records and i < len(records) else {}) or {}
        out.append({
            "index": i,
            "id": rec.get("id", i),
            "wet_fraction": round(frac, 3),
            "mean_depth_m": round(float(dsub.mean()), 2) if dsub.size else 0.0,
            "max_depth_m": round(float(dsub.max()), 2) if dsub.size else 0.0,
            "height_m": rec.get("height_m"),
        })
    out.sort(key=lambda r: -r["max_depth_m"])
    return out


def summarise(result: Dict, buildings: Sequence[Dict], metric: bool) -> str:
    if result.get("seed_dry"):
        return f"no flood: {result['reason']}"
    if not metric:
        # Tier C: the numbers exist as cell counts but not as metres.
        return (f"{result['cells']:,} cells inundated, {len(buildings)} buildings "
                f"affected -- RELATIVE SCALE, so no area, depth or volume in "
                f"metres is reported")
    lines = [
        f"{result['area_m2']:,.0f} m2 inundated, {len(buildings)} buildings affected",
        f"  volume {result['volume_m3']:,.0f} m3, "
        f"mean depth {result['mean_depth_m']:.2f} m, max {result['max_depth_m']:.2f} m",
    ]
    if result.get("unreachable_cells"):
        lines.append(
            f"  {result['unreachable_cells']:,} cells sit below the level but are "
            f"not connected to the seed -- a plane would have flooded them")
    return "\n".join(lines)
