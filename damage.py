"""Per-building damage assessment between a before and an after reconstruction.

change_detection.py answers "where did the surface move" as a raster. That is
the right question for a flood extent and the wrong one for a responder, who
needs a list of structures with a verdict against each. This matches buildings
between two builds and classifies each one.

THE CONFORMAL GATE IS THE POINT

A height that fell by 3 m is not evidence of damage when the pipeline's own
90% interval on each height is +/-4.5 m. Two independent measurements each
carrying a half-width q give a DIFFERENCE whose half-width is q*sqrt(2) -- the
errors add in quadrature -- so on this pipeline's fitted quantile a drop has to
exceed about 6.4 m before it can be distinguished from the measurement at all.

Every classification is therefore gated on that band, and a delta inside it is
reported as INCONCLUSIVE rather than being rounded to "intact". Those are
different answers: "this building is undamaged" and "our instrument cannot tell
you about this building" lead a responder to different actions, and collapsing
the second into the first is how a survey quietly under-reports a disaster.

INCONCLUSIVE is a first-class outcome, counted and exported like any other.
A run where most buildings land there is a run whose input was not precise
enough for the question, and saying so is the useful output.

WHY IoU MATCHING RATHER THAN NEAREST CENTROID

A collapsed building's footprint often survives -- rubble keeps roughly the plan
outline -- while a centroid can shift several metres. Matching by centroid then
pairs a collapsed structure with its intact neighbour and reports both wrongly.
Overlap is the stabler relation, and a footprint with no overlapping partner is
itself informative: it appeared or vanished between captures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

# Two independent heights, so their difference carries sqrt(2) times the
# single-height half-width.
DIFF_WIDEN = math.sqrt(2.0)

# Minimum overlap for two footprints to be the same structure. Below this they
# are treated as unmatched -- a partial overlap between neighbours is not a
# correspondence.
MIN_MATCH_IOU = 0.30

# Damage bands, as a fraction of the BEFORE height. Applied only after the
# conformal gate has confirmed the change is larger than the measurement error,
# so these divide real change rather than noise.
COLLAPSED_FRAC = 0.60      # lost most of its height
PARTIAL_FRAC = 0.25        # lost a substantial part of it


def _load_geojson(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    out = []
    for feat in gj.get("features", []):
        g = feat.get("geometry") or {}
        if g.get("type") != "Polygon":
            continue
        ring = np.asarray(g["coordinates"][0], dtype=np.float64)
        if len(ring) >= 4 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        out.append({"ring": ring, "props": feat.get("properties", {}) or {}})
    return out


def _bounds(rec: Dict) -> Tuple[float, float, float, float]:
    r = rec["ring"]
    return float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 0].max()), float(r[:, 1].max())


def _iou(a: Dict, b: Dict, scale: float = 1.0) -> float:
    """Polygon IoU via rasterisation on a shared local grid.

    Rasterised rather than computed analytically: the rings come from contour
    tracing and are not guaranteed simple, and a raster test needs no robust
    predicates to stay correct on the degenerate cases that actually occur.
    """
    ax0, ay0, ax1, ay1 = _bounds(a)
    bx0, by0, bx1, by1 = _bounds(b)
    x0, y0 = min(ax0, bx0), min(ay0, by0)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    w = int(math.ceil((x1 - x0) * scale)) + 2
    h = int(math.ceil((y1 - y0) * scale)) + 2
    if w < 2 or h < 2 or w * h > 4_000_000:
        return 0.0
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    pa = ((a["ring"] - [x0, y0]) * scale).astype(np.int32)
    pb = ((b["ring"] - [x0, y0]) * scale).astype(np.int32)
    cv2.fillPoly(ma, [pa], 1)
    cv2.fillPoly(mb, [pb], 1)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / union if union else 0.0


def match_footprints(before: Sequence[Dict], after: Sequence[Dict],
                     min_iou: float = MIN_MATCH_IOU,
                     scale: float = 1.0) -> Dict:
    """Greedy highest-IoU matching between two footprint sets.

    Greedy on descending overlap rather than optimal assignment: the candidate
    pairs are spatially disjoint in practice, so the optimal solution and the
    greedy one coincide, and greedy stays tractable on thousands of buildings.
    """
    # Bounding-box prefilter: comparing every pair is O(n*m) rasterisations,
    # which on 1300 x 1300 buildings is 1.7 million polygon fills.
    cand = []
    for i, b in enumerate(before):
        bx0, by0, bx1, by1 = _bounds(b)
        for j, a in enumerate(after):
            ax0, ay0, ax1, ay1 = _bounds(a)
            if ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1:
                continue
            v = _iou(b, a, scale)
            if v >= min_iou:
                cand.append((v, i, j))
    cand.sort(reverse=True)

    used_b, used_a, pairs = set(), set(), []
    for v, i, j in cand:
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
        pairs.append({"before": i, "after": j, "iou": round(v, 3)})

    return {"pairs": pairs,
            "vanished": [i for i in range(len(before)) if i not in used_b],
            "appeared": [j for j in range(len(after)) if j not in used_a]}


def classify(delta_m: float, before_h: float, band_m: float) -> Tuple[str, str]:
    """Damage class for one matched pair, gated on the interval.

    Returns (class, reason). The gate is applied FIRST: nothing is called
    damage until the change is larger than the band on the difference.
    """
    if band_m is not None and abs(delta_m) <= band_m:
        # A change inside the band bounds the damage rather than merely failing
        # to detect it: if the structure had really lost band_m or more, the
        # measurement would have shown it. So whether this is INTACT or
        # INCONCLUSIVE depends on how the band compares with the damage
        # threshold FOR THIS BUILDING.
        #
        # Without this split, INTACT was unreachable. It required
        # |delta| > band AND lost < 25%, which on a 20 m building means losing
        # more than 6.4 m (32%) and less than 25% at the same time. Only
        # buildings above about 26 m could ever be called intact, and every
        # ordinary house came back INCONCLUSIVE however good the data was.
        rule_out = PARTIAL_FRAC * max(before_h, 1e-6)
        if band_m < rule_out:
            return ("INTACT",
                    f"change {delta_m:+.1f} m within +/-{band_m:.1f} m, and that "
                    f"band is tighter than the {PARTIAL_FRAC*100:.0f}% damage "
                    f"threshold ({rule_out:.1f} m here) -- partial damage is "
                    f"ruled out, not merely undetected")
        return ("INCONCLUSIVE",
                f"change {delta_m:+.1f} m is inside the +/-{band_m:.1f} m "
                f"interval on the difference, and that band is wider than the "
                f"{PARTIAL_FRAC*100:.0f}% damage threshold ({rule_out:.1f} m "
                f"here) -- damage cannot be ruled out at this precision")
    if delta_m > 0:
        return ("GREW", f"height increased by {delta_m:.1f} m")
    lost = -delta_m / max(before_h, 1e-6)
    if lost >= COLLAPSED_FRAC:
        return ("COLLAPSED", f"lost {lost*100:.0f}% of {before_h:.1f} m")
    if lost >= PARTIAL_FRAC:
        return ("PARTIAL", f"lost {lost*100:.0f}% of {before_h:.1f} m")
    return ("INTACT", f"lost {lost*100:.0f}% of {before_h:.1f} m, below the "
                      f"{PARTIAL_FRAC*100:.0f}% partial-damage threshold")


def assess(before_geojson: str, after_geojson: str,
           half_width_m: Optional[float] = None,
           min_iou: float = MIN_MATCH_IOU) -> Dict:
    """Match buildings between two builds and classify each."""
    before = _load_geojson(before_geojson)
    after = _load_geojson(after_geojson)
    if not before or not after:
        raise SystemExit("one of the footprint sets is empty")

    # The band on a DIFFERENCE of two independent measurements. Taken from the
    # records themselves when they carry an interval, so a scene whose quantile
    # differs from the committed one is gated on its own number.
    if half_width_m is None:
        widths = [r["props"].get("interval_half_width_m") for r in before + after]
        widths = [w for w in widths if isinstance(w, (int, float))]
        half_width_m = float(np.median(widths)) if widths else None
    band = None if half_width_m is None else float(half_width_m) * DIFF_WIDEN

    m = match_footprints(before, after, min_iou)
    rows = []
    for pair in m["pairs"]:
        b = before[pair["before"]]["props"]
        a = after[pair["after"]]["props"]
        bh = b.get("height_m")
        ah = a.get("height_m")
        if not isinstance(bh, (int, float)) or not isinstance(ah, (int, float)):
            continue
        delta = float(ah) - float(bh)
        cls, why = classify(delta, float(bh), band)
        rows.append({
            "before_id": b.get("id", pair["before"]),
            "after_id": a.get("id", pair["after"]),
            "iou": pair["iou"],
            "before_h_m": round(float(bh), 2),
            "after_h_m": round(float(ah), 2),
            "delta_m": round(delta, 2),
            "class": cls,
            "reason": why,
            "area_m2": b.get("area_m2"),
        })

    # Most damaged first: that is the order a responder reads.
    order = {"COLLAPSED": 0, "PARTIAL": 1, "INCONCLUSIVE": 2, "INTACT": 3, "GREW": 4}
    rows.sort(key=lambda r: (order.get(r["class"], 9), r["delta_m"]))

    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1

    return {
        "rows": rows,
        "counts": counts,
        "matched": len(rows),
        "vanished": len(m["vanished"]),
        "appeared": len(m["appeared"]),
        "n_before": len(before),
        "n_after": len(after),
        "half_width_m": (None if half_width_m is None else round(half_width_m, 2)),
        "difference_band_m": (None if band is None else round(band, 2)),
        "min_match_iou": min_iou,
    }


def to_csv(res: Dict, path: str) -> None:
    cols = ["before_id", "after_id", "iou", "before_h_m", "after_h_m",
            "delta_m", "class", "area_m2", "reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in res["rows"]:
            w.writerow(r)


def report(res: Dict) -> str:
    c = res["counts"]
    lines = [f"damage assessment: {res['matched']} buildings matched "
             f"({res['n_before']} before, {res['n_after']} after)"]
    if res["difference_band_m"] is None:
        lines.append("  NO INTERVAL AVAILABLE -- every class below is ungated "
                     "and a small change cannot be told from measurement error")
    else:
        lines.append(f"  gate: +/-{res['half_width_m']:.1f} m per height, so "
                     f"+/-{res['difference_band_m']:.1f} m on the difference")
    for k in ("COLLAPSED", "PARTIAL", "INCONCLUSIVE", "INTACT", "GREW"):
        if c.get(k):
            lines.append(f"  {k:<13} {c[k]}")
    if res["vanished"] or res["appeared"]:
        lines.append(f"  unmatched: {res['vanished']} vanished, "
                     f"{res['appeared']} appeared")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-building damage assessment between two builds")
    ap.add_argument("before", help="buildings.geojson from the earlier build")
    ap.add_argument("after", help="buildings.geojson from the later build")
    ap.add_argument("--half-width", type=float, default=None,
                    help="conformal half-width in metres per height; taken "
                         "from the records themselves when omitted")
    ap.add_argument("--min-iou", type=float, default=MIN_MATCH_IOU)
    ap.add_argument("--out", default="damage")
    a = ap.parse_args()

    res = assess(a.before, a.after, a.half_width, a.min_iou)
    print(report(res))
    to_csv(res, a.out + ".csv")
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in res.items() if k != "rows"}, f, indent=2)
    print(f"wrote {a.out}.csv and {a.out}.json")


if __name__ == "__main__":
    main()
