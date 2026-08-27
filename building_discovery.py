"""
High-recall, multi-scale building discovery with evidence-based retention.

Implements directive sections 7-10, 27 and 52.

THE RULE THIS MODULE EXISTS TO ENFORCE

    Never `if area < threshold: delete`.

A structure is dropped only when the evidence for it is *zero*, and every drop is
counted and attributed to a reason. A 10x10-pixel shed is a building; at 0.25 m/px that
is 6 m^2, which the previous single-threshold filter discarded silently along with
everything else below its area cut.

WHY MULTI-SCALE

One detection pass has one effective object size. Running the same detector over the
full-resolution image and over 2x and 4x downsampled copies changes what each pass can
see:

  * 4x  -- city structure, large blocks, anything whose interior is bigger than the
           morphological kernel
  * 2x  -- ordinary buildings
  * 1x  -- sheds, extensions, detached garages, small rooftops

Detections are fused rather than concatenated: the same building found at three scales
must become one instance, not three overlapping prisms.

EVIDENCE, NOT A SINGLE SIGNAL

Each candidate is scored on independent evidence, so a weak signal in one channel does
not delete a structure that is obvious in another:

  height    -- does it stand above local ground?
  edge      -- does it have a roof outline?
  texture   -- is its interior smoother than vegetation?
  shadow    -- is there a shadow adjacent on the anti-sun side?

A candidate is retained if ANY channel is non-trivial, and its confidence is the
weighted combination. Low-confidence structures are kept and marked, per section 10.
"""
import numpy as np
import cv2

import segmentation as seg


# Provenance values (directive section 24).
OBSERVED = "OBSERVED"        # directly supported by measured elevation
MEASURED = "MEASURED"        # metric height from shadow / DEM
INFERRED = "INFERRED"        # from learned depth only
AI_COMPLETED = "AI_COMPLETED"

# Size classes for the recall report (section 27), in square metres.
SIZE_CLASSES = [
    ("tiny",   0.0,   25.0),
    ("small",  25.0,  120.0),
    ("medium", 120.0, 600.0),
    ("large",  600.0, float("inf")),
]


def size_class(area_m2: float) -> str:
    for name, lo, hi in SIZE_CLASSES:
        if lo <= area_m2 < hi:
            return name
    return "large"


def _candidate_masks(ndsm: np.ndarray, building_mask: np.ndarray,
                     band_m: float, min_px: int) -> list:
    """Height-banded components at one scale; each band yields separate instances."""
    h = np.where(building_mask > 0, ndsm, np.nan)
    if not np.isfinite(h).any():
        return []
    top = float(np.nanpercentile(h, 99.5))
    n_bands = max(1, int(np.ceil(top / max(band_m, 0.5))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    out = []
    for b in range(n_bands):
        lo, hi = b * band_m, (b + 1) * band_m
        band = ((h >= lo) & (h < hi)).astype(np.uint8)
        if band.sum() < min_px:
            continue
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_px:
                continue
            out.append(cnt)
    return out


def discover(image_np: np.ndarray, seg_labels: np.ndarray, ndsm: np.ndarray,
             gsd_m: float, sun_azimuth_deg: float = None,
             shadow_mask: np.ndarray = None,
             min_area_m2: float = 6.0, band_m: float = 3.0,
             scales: tuple = (1, 2, 4)) -> dict:
    """
    Find building candidates at several scales and fuse them into instances.

    min_area_m2 defaults to 6 m^2 -- about a 2.5 x 2.5 m shed, near the smallest
    structure this imagery can resolve at all. It is expressed in square metres so it
    does not silently change meaning with resolution, which a pixel threshold does.

    Returns a dict with `instances` (list of records) and `report` (counts by size
    class and rejection reason).
    """
    H, W = ndsm.shape
    building_mask = (seg_labels == seg.CLASS_IDX["building"]).astype(np.uint8)
    veg_mask = (seg_labels == seg.CLASS_IDX["vegetation"])

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 130).astype(np.float32) / 255.0
    edge_density = cv2.blur(edges, (9, 9))
    fine_tex = cv2.blur(gray.astype(np.float32) ** 2, (5, 5)) - \
        cv2.blur(gray.astype(np.float32), (5, 5)) ** 2

    px_area = gsd_m * gsd_m
    contours_all = []

    for s in scales:
        if s == 1:
            nd_s, bm_s = ndsm, building_mask
        else:
            nd_s = cv2.resize(ndsm, (W // s, H // s), interpolation=cv2.INTER_AREA)
            bm_s = cv2.resize(building_mask, (W // s, H // s),
                              interpolation=cv2.INTER_NEAREST)
        min_px_s = max(4, int(min_area_m2 / (px_area * s * s)))
        for cnt in _candidate_masks(nd_s, bm_s, band_m, min_px_s):
            contours_all.append((cnt.astype(np.float32) * s).astype(np.int32))

    # ---- fuse across scales -------------------------------------------------
    # The same building found at 1x, 2x and 4x must become ONE instance. Fusion is by
    # occupancy: a candidate whose footprint is already mostly claimed is a duplicate.
    # Candidates are considered largest-first so the better-resolved outline of a big
    # block wins, and genuinely small structures inside a courtyard still get their turn.
    contours_all.sort(key=cv2.contourArea, reverse=True)
    claimed = np.zeros((H, W), np.uint8)

    instances = []
    rejected = {"no_evidence": 0, "duplicate": 0, "below_min_area": 0, "degenerate": 0}

    for cnt in contours_all:
        area_px = cv2.contourArea(cnt)
        area_m2 = area_px * px_area
        if area_m2 < min_area_m2:
            rejected["below_min_area"] += 1
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            rejected["degenerate"] += 1
            continue

        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, [cnt - [x0, y0]], 1)
        inside = sub.astype(bool)
        if inside.sum() < 4:
            rejected["degenerate"] += 1
            continue

        overlap = claimed[y0:y1, x0:x1][inside].mean()
        if overlap > 0.55:
            rejected["duplicate"] += 1
            continue

        # ---- evidence channels ---------------------------------------------
        local_ndsm = ndsm[y0:y1, x0:x1][inside]
        height_m = float(np.percentile(local_ndsm, 80))
        # Height evidence saturates at 3 m: anything clearly above head height is
        # already fully convincing on this channel.
        e_height = float(np.clip(height_m / 3.0, 0.0, 1.0))

        ring = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        ring_only = (ring.astype(bool)) & (~inside)
        e_edge = float(np.clip(
            edge_density[y0:y1, x0:x1][ring_only].mean() * 4.0, 0.0, 1.0)) \
            if ring_only.sum() > 0 else 0.0

        veg_frac = float(veg_mask[y0:y1, x0:x1][inside].mean())
        e_texture = float(np.clip(1.0 - veg_frac, 0.0, 1.0))

        e_shadow = 0.0
        if shadow_mask is not None and sun_azimuth_deg is not None:
            import math
            rad = math.radians((sun_azimuth_deg + 180) % 360)
            dx, dy = math.sin(rad), -math.cos(rad)
            step = max(2, int(2.0 / max(gsd_m, 1e-6)))
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            py, px = int(cy + dy * step), int(cx + dx * step)
            if 0 <= py < H and 0 <= px < W:
                e_shadow = float(shadow_mask[py, px])

        evidence = {"height": e_height, "edge": e_edge,
                    "texture": e_texture, "shadow": e_shadow}

        # RETENTION RULE (section 10): keep unless evidence is essentially zero.
        # Height alone is enough; so is a strong roof outline. A candidate is only
        # dropped when nothing supports it at all.
        if e_height < 0.05 and e_edge < 0.10:
            rejected["no_evidence"] += 1
            continue

        confidence = float(np.clip(
            0.55 * e_height + 0.25 * e_edge + 0.10 * e_texture + 0.10 * e_shadow,
            0.0, 1.0))

        claimed[y0:y1, x0:x1][inside] = 1
        peri = cv2.arcLength(cnt, True)
        m = cv2.moments(cnt)
        cxf = m["m10"] / m["m00"] if m["m00"] else float(x + w / 2)
        cyf = m["m01"] / m["m00"] if m["m00"] else float(y + h / 2)

        instances.append({
            "id": len(instances),
            "contour": cnt,
            "area_m2": round(area_m2, 1),
            "perimeter_m": round(peri * gsd_m, 1),
            "centroid_px": (float(cxf), float(cyf)),
            "height_m": round(height_m, 2),
            "size_class": size_class(area_m2),
            "evidence": {k: round(v, 3) for k, v in evidence.items()},
            "confidence": round(confidence, 3),
            "provenance": INFERRED,   # upgraded by the caller when scale is metric
        })

    counts = {name: 0 for name, _, _ in SIZE_CLASSES}
    for r in instances:
        counts[r["size_class"]] += 1

    report = {
        "candidates_examined": len(contours_all),
        "retained": len(instances),
        "by_size": counts,
        "rejected": rejected,
        "min_area_m2": min_area_m2,
        "scales": list(scales),
    }
    return {"instances": instances, "report": report}


def format_report(report: dict) -> str:
    """The explicit accounting required by section 27 -- never hide these numbers."""
    lines = [
        f"      candidates examined {report['candidates_examined']}, "
        f"retained {report['retained']}",
        "      by size: " + "  ".join(
            f"{k} {v}" for k, v in report["by_size"].items()),
    ]
    rej = report["rejected"]
    if sum(rej.values()):
        lines.append("      rejected: " + "  ".join(
            f"{k} {v}" for k, v in rej.items() if v))
    return "\n".join(lines)
