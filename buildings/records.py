"""
Per-building record with explicit evidence provenance.

The pipeline previously carried buildings as anonymous polygon arrays, so
there was nowhere to attach a confidence, a rejection reason, or a statement
of how a height was obtained. Every downstream requirement -- confidence
overlays, building inspection, honest exports -- needs a per-object identity,
so that is what this module provides.

Central rule: a height derived from a physical measurement (shadow geometry,
DEM difference) and a height guessed by a neural network are NOT
interchangeable, and must never be presented as if they were. Provenance is
recorded per building and travels with it into the mesh, the viewer, and
every export.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import numpy as np


class Provenance(str, Enum):
    """How a building's geometry was arrived at. Ordered strongest to weakest."""
    MEASURED = "MEASURED"        # physical measurement: shadow geometry, DEM difference
    OBSERVED = "OBSERVED"        # directly visible in imagery: footprint from pixels
    INFERRED = "INFERRED"        # learned prior: neural depth, shape completion
    AI_COMPLETED = "AI_COMPLETED"  # synthesised because the sensor could not observe it


class HeightMethod(str, Enum):
    SHADOW = "shadow"              # shadow length x tan(sun elevation)
    DEM_DIFFERENCE = "dem_diff"    # DSM minus bare-earth DEM
    NEURAL_DEPTH = "neural_depth"  # monocular depth model
    FUSED = "fused"                # confidence-weighted combination
    GROUP_MEDIAN = "group_median"  # imputed from neighbours, no direct evidence
    NONE = "none"


@dataclass
class Evidence:
    """
    Independent signals that a candidate region is a building. Each is in
    [0, 1]. They are deliberately kept separate rather than pre-combined so
    that a rejection can name which signals were absent.
    """
    edge: float = 0.0        # crisp boundary / rectilinear edges
    shadow: float = 0.0      # cast shadow present in the anti-sun direction
    height: float = 0.0      # elevated relative to local surroundings
    texture: float = 0.0     # roof-like surface vs vegetation speckle
    shape: float = 0.0       # compactness / rectilinearity
    spectral: float = 0.0    # non-vegetation per NDVI, when bands allow

    def total(self) -> float:
        """Max-weighted sum: any one strong signal can carry a candidate."""
        vals = [self.edge, self.shadow, self.height, self.texture, self.shape, self.spectral]
        return float(max(vals) * 0.5 + (sum(vals) / len(vals)) * 0.5)

    def present(self) -> list:
        names = ["edge", "shadow", "height", "texture", "shape", "spectral"]
        return [n for n, v in zip(names, [self.edge, self.shadow, self.height,
                                           self.texture, self.shape, self.spectral]) if v > 0.05]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Building:
    """One detected structure. Geometry in pixel coordinates unless noted."""
    id: int
    polygon: np.ndarray                 # Nx2 float32, pixel coords, CCW, no repeated last point
    area_px: float
    perimeter_px: float
    centroid: tuple                     # (x, y) pixel
    orientation_deg: float              # long-axis angle, 0 = +x

    evidence: Evidence = field(default_factory=Evidence)
    detection_scale: int = 1            # which pyramid level found it (1, 2, 4)

    height_m: Optional[float] = None
    height_confidence: float = 0.0
    height_method: HeightMethod = HeightMethod.NONE
    provenance: Provenance = Provenance.OBSERVED

    # populated by shadow analysis when available
    shadow_length_px: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None

    roof_type: str = "unknown"
    roof_confidence: float = 0.0

    notes: list = field(default_factory=list)

    @property
    def is_metric(self) -> bool:
        """True only when the height is in real metres from a real reference."""
        return self.height_m is not None and self.height_method in (
            HeightMethod.SHADOW, HeightMethod.DEM_DIFFERENCE, HeightMethod.FUSED)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "polygon": self.polygon.tolist(),
            "area_px": round(float(self.area_px), 2),
            "perimeter_px": round(float(self.perimeter_px), 2),
            "centroid": [round(float(c), 2) for c in self.centroid],
            "orientation_deg": round(float(self.orientation_deg), 1),
            "evidence": {k: round(float(v), 3) for k, v in self.evidence.as_dict().items()},
            "evidence_total": round(self.evidence.total(), 3),
            "evidence_present": self.evidence.present(),
            "detection_scale": self.detection_scale,
            "height_m": None if self.height_m is None else round(float(self.height_m), 2),
            "height_confidence": round(float(self.height_confidence), 3),
            "height_method": self.height_method.value,
            "provenance": self.provenance.value,
            "is_metric": self.is_metric,
            "roof_type": self.roof_type,
            "roof_confidence": round(float(self.roof_confidence), 3),
            "notes": self.notes,
        }


@dataclass
class Rejection:
    """
    A candidate that was NOT kept, and why.

    This type exists because the previous pipeline discarded 85.8% of building
    candidates through bare `continue` statements that recorded nothing. A
    rejection must always be explainable.
    """
    area_px: float
    centroid: tuple
    reason: str
    evidence_total: float
    detection_scale: int = 1

    def to_dict(self) -> dict:
        return {
            "area_px": round(float(self.area_px), 2),
            "centroid": [round(float(c), 2) for c in self.centroid],
            "reason": self.reason,
            "evidence_total": round(float(self.evidence_total), 3),
            "detection_scale": self.detection_scale,
        }


@dataclass
class DetectionReport:
    """Full funnel, per the small-object recall requirement."""
    buildings: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    raw_candidates: int = 0
    scales_run: list = field(default_factory=list)
    merged_duplicates: int = 0
    instances_split: int = 0

    def size_buckets(self) -> dict:
        buckets = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
        for b in self.buildings:
            a = b.area_px
            if a < 15:
                buckets["tiny"] += 1
            elif a < 50:
                buckets["small"] += 1
            elif a < 200:
                buckets["medium"] += 1
            else:
                buckets["large"] += 1
        return buckets

    def rejection_summary(self) -> dict:
        out = {}
        for r in self.rejections:
            out[r.reason] = out.get(r.reason, 0) + 1
        return out

    def summary_text(self) -> str:
        b = self.size_buckets()
        retained = len(self.buildings)
        total = self.raw_candidates
        pct = (retained / total * 100) if total else 0.0
        lines = [
            f"Buildings retained: {retained} of {total} candidates ({pct:.1f}%)",
            f"  large  (>200px)   {b['large']}",
            f"  medium (50-200px) {b['medium']}",
            f"  small  (15-50px)  {b['small']}",
            f"  tiny   (<15px)    {b['tiny']}",
            f"  scales run: {self.scales_run}",
            f"  duplicates merged across scales: {self.merged_duplicates}",
            f"  merged blobs split into instances: {self.instances_split}",
        ]
        rs = self.rejection_summary()
        if rs:
            lines.append("  rejected:")
            for reason, n in sorted(rs.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {n:5d}  {reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "raw_candidates": self.raw_candidates,
            "retained": len(self.buildings),
            "size_buckets": self.size_buckets(),
            "scales_run": self.scales_run,
            "merged_duplicates": self.merged_duplicates,
            "instances_split": self.instances_split,
            "rejection_summary": self.rejection_summary(),
            "buildings": [b.to_dict() for b in self.buildings],
            "rejections": [r.to_dict() for r in self.rejections],
        }
