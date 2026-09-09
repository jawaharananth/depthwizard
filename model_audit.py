"""Audit a finished 3D model against the image it was reconstructed from.

The repair loop checks the model against ITSELF -- is the geometry well formed,
does everything rest on the ground. That cannot catch a model which is
internally perfect and describes the wrong city: a clean prism standing on an
empty car park is valid geometry and a false building.

So this audits the model against its own evidence. The model is re-rendered from
directly overhead, orthographically, into a synthetic nadir view; that render is
then compared with the input image and with the measured height field. If the
reconstruction is right, its silhouette lands on the image's buildings, its
outlines follow real image edges, and its roof heights agree with the surface
that was measured.

Five independent checks, chosen because they fail in different ways:

  silhouette   Does the model cover the buildings the image contains?
               Catches wholesale under- or over-building.
  spurious     Does every prism stand on something that looks like a building?
               Catches structures invented on car parks and shadows.
  edges        Do footprint boundaries lie on real intensity edges?
               Catches outlines that are the right area in the wrong place --
               which silhouette IoU alone cannot see.
  heights      Do roof heights agree with the measured height field?
               Catches correct outlines extruded to arbitrary heights.
  shadows      Does the sun direction implied by the geometry match the
               shadows in the image? Reported, never enforced -- see below.

A single number is refused deliberately. These fail independently, and averaging
them into one score hides which one failed and therefore what to do about it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


# Bands for the verdict. Chosen against the benchmarked DFC path, so "good"
# means "comparable with the configuration that measures 2.57 m per building"
# rather than an arbitrary aspiration.
THRESHOLDS = {
    "silhouette_iou": (0.35, 0.50),      # (marginal, good)
    "spurious_frac": (0.35, 0.20),       # lower is better
    "edge_support": (0.35, 0.50),
    "height_corr": (0.50, 0.70),
}


# ---------------------------------------------------------------------------
# Orthographic re-render
# ---------------------------------------------------------------------------

def render_nadir(footprints: Sequence[np.ndarray], binfo: Dict,
                 shape) -> np.ndarray:
    """Rasterise the model's roofs into a synthetic nadir height image.

    A painter's algorithm ordered by roof height is sufficient and exactly
    correct here: every prism is a vertical extrusion, so from directly overhead
    the visible surface at any pixel is simply the highest roof covering it. No
    z-buffer interpolation is needed because roofs are drawn as flat polygons.
    """
    H, W = shape
    out = np.full((H, W), np.nan, np.float32)
    builds = binfo.get("buildings") or []
    order = sorted(builds, key=lambda b: b.get("roof_h", 0.0))
    for b in order:
        pi = b.get("poly_index")
        if pi is None or pi >= len(footprints):
            continue
        poly = np.asarray(footprints[pi], np.int32)
        if len(poly) < 3:
            continue
        cv2.fillPoly(out, [poly], float(b.get("height_m", 0.0)))
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

# WHAT THIS AUDIT CANNOT TELL YOU.
#
# silhouette_iou and spurious_frac are scored against the SEGMENTATION mask, and
# the footprints were derived from the same height field that segmentation sees.
# The two are not independent, so agreement between them is not evidence of
# correctness: if a car park is called a building, a prism is built there and
# this audit calls the prism supported.
#
# Measured against LiDAR -- a genuinely independent reference -- the real
# numbers on JAX_165 are precision 0.762 and recall 0.789, against a spurious
# fraction of 0.148 reported here. Roughly 24% of built area is false, and 53%
# of that sits on bare earth. The audit is a consistency check, not an accuracy
# measurement, and only edge_support and height_agreement use evidence the
# footprints were not derived from.
def _silhouette(model_mask: np.ndarray, truth_mask: np.ndarray) -> Dict:
    inter = float((model_mask & truth_mask).sum())
    union = float((model_mask | truth_mask).sum())
    iou = inter / union if union else 0.0
    recall = inter / float(truth_mask.sum()) if truth_mask.any() else 0.0
    precision = inter / float(model_mask.sum()) if model_mask.any() else 0.0
    return {"silhouette_iou": round(iou, 3),
            "coverage_recall": round(recall, 3),
            "coverage_precision": round(precision, 3),
            "spurious_frac": round(1.0 - precision, 3)}


def _edge_support(footprints: Sequence[np.ndarray], image_np: np.ndarray,
                  tol_px: int = 3) -> Dict:
    """Fraction of footprint boundary that lies on a real image edge.

    A building outline is a physical discontinuity, so it should coincide with
    an intensity edge. This is the check that separates "right area, wrong
    place" from a genuine footprint -- two outlines can score the same IoU while
    one traces the building and the other traces its shadow.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    near = cv2.dilate(edges, np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8))

    H, W = gray.shape
    total = 0
    hit = 0
    for poly in footprints:
        if len(poly) < 3:
            continue
        p = np.asarray(poly, np.int32)
        # Draw into a canvas the size of the FOOTPRINT, not the scene. Allocating
        # a full-frame buffer per building is O(buildings x pixels): on a 2560px
        # tile with 1344 buildings that is 8.8 billion cleared bytes, and it
        # dominated the audit by orders of magnitude.
        x, y, w, h = cv2.boundingRect(p)
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1, y1 = min(x + w + 1, W), min(y + h + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(canvas, [p - [x0, y0]], True, 1, 1)
        ys, xs = np.nonzero(canvas)
        if not len(ys):
            continue
        total += len(ys)
        hit += int((near[y0:y1, x0:x1][ys, xs] > 0).sum())
    return {"edge_support": round(hit / total, 3) if total else 0.0,
            "boundary_px": int(total)}


def _height_agreement(model_h: np.ndarray, measured_ndsm: np.ndarray) -> Dict:
    """Do the extruded heights match the surface that was measured?"""
    m = np.isfinite(model_h) & np.isfinite(measured_ndsm) & (model_h > 0)
    if m.sum() < 100:
        return {"height_corr": 0.0, "height_mae": None, "height_px": int(m.sum())}
    a = model_h[m].astype(np.float64)
    b = measured_ndsm[m].astype(np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {"height_corr": round(corr, 3),
            "height_mae": round(float(np.mean(np.abs(a - b))), 2),
            "height_px": int(m.sum())}


def _shadow_direction(image_np: np.ndarray, model_mask: np.ndarray,
                      sun_azimuth_deg: Optional[float]) -> Dict:
    """Is there dark ground on the anti-sun side of the model's buildings?

    Reported but never used to pass or fail a build. The project's own shadow
    detector was measured against reference shadows at IoU 0.144 -- it finds
    11.7% of the frame where 45.2% was predicted -- so a build must not be
    rejected on its say-so. It is included because a gross sun-direction error
    is worth seeing, not because the number is trustworthy.
    """
    if sun_azimuth_deg is None:
        return {"shadow_agreement": None, "note": "sun azimuth unknown"}
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = gray < np.percentile(gray, 25)

    th = np.deg2rad(sun_azimuth_deg)
    dx, dy = -np.sin(th), np.cos(th)         # away from the sun
    shift = 8
    M = np.float32([[1, 0, dx * shift], [0, 1, dy * shift]])
    shifted = cv2.warpAffine(model_mask.astype(np.uint8), M,
                             (model_mask.shape[1], model_mask.shape[0]))
    ring = (shifted > 0) & (~model_mask)
    if ring.sum() < 50:
        return {"shadow_agreement": None, "note": "too little exposed ground"}
    return {"shadow_agreement": round(float(dark[ring].mean()), 3),
            "note": "advisory only; the shadow detector is unreliable "
                    "(measured IoU 0.144 against reference)"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def audit(footprints: Sequence[np.ndarray], binfo: Dict,
          image_np: np.ndarray, ndsm: np.ndarray,
          building_mask: np.ndarray,
          sun_azimuth_deg: Optional[float] = None) -> Dict:
    shape = ndsm.shape
    model_h = render_nadir(footprints, binfo, shape)
    model_mask = np.isfinite(model_h) & (model_h > 0)

    res: Dict = {}
    res.update(_silhouette(model_mask, building_mask.astype(bool)))
    res.update(_edge_support(footprints, image_np))
    res.update(_height_agreement(model_h, ndsm))
    res.update(_shadow_direction(image_np, model_mask, sun_azimuth_deg))
    res["model_footprint_px"] = int(model_mask.sum())
    res["image_building_px"] = int(building_mask.astype(bool).sum())

    grades = {}
    for key, (marginal, good) in THRESHOLDS.items():
        v = res.get(key)
        if v is None:
            grades[key] = "n/a"
            continue
        if key == "spurious_frac":          # lower is better
            grades[key] = "GOOD" if v <= good else ("OK" if v <= marginal else "POOR")
        else:
            grades[key] = "GOOD" if v >= good else ("OK" if v >= marginal else "POOR")
    res["grades"] = grades
    res["verdict"] = ("POOR" if "POOR" in grades.values()
                      else ("OK" if "OK" in grades.values() else "GOOD"))
    return res


def report(res: Dict) -> str:
    g = res["grades"]
    lines = ["      model-vs-image audit:"]
    lines.append(f"  [{g['silhouette_iou']:>4}] silhouette IoU   {res['silhouette_iou']:.3f}"
                 f"   (recall {res['coverage_recall']:.3f}, "
                 f"precision {res['coverage_precision']:.3f})")
    lines.append(f"  [{g['spurious_frac']:>4}] spurious area    {res['spurious_frac']:.3f}"
                 f"   model footprint with no building evidence")
    lines.append(f"  [{g['edge_support']:>4}] edge support     {res['edge_support']:.3f}"
                 f"   of {res['boundary_px']} boundary px on a real image edge")
    mae = res.get("height_mae")
    lines.append(f"  [{g['height_corr']:>4}] height agreement corr {res['height_corr']:.3f}"
                 + (f", MAE {mae}" if mae is not None else ""))
    sa = res.get("shadow_agreement")
    if sa is not None:
        lines.append(f"  [ -- ] shadow side     {sa:.3f} dark  (advisory only)")
    lines.append(f"      VERDICT: {res['verdict']}")
    return "\n".join(lines)
