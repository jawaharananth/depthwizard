"""VLM-as-judge audit: render the finished model from nadir, place it beside
the source image, and ask a vision-language model to flag discrepancies a
numeric metric can't articulate -- wrong roof form, a car park read as a
building, a structure missing entirely, a height that "looks" wrong even
where model_audit's height correlation looks fine in aggregate.

This is Layer 2 on top of model_audit.py's Layer 1 (silhouette IoU, edge
support, height correlation, shadow direction -- all deterministic,
re-derived from the same evidence the model was built from). Layer 2 adds an
independent judge: a general-purpose vision model that has not seen this
project's pipeline and is not scored against pass/fail thresholds.

Hard rule, same one the rest of this project follows: this is a REPORTING
layer, not a correction layer. Nothing here writes back into dsm, ndsm,
heights, or footprints. A VLM's visual read is not a Tier A/B measurement --
it is a QA pass that tells a human where to look, exactly like model_audit's
worst_buildings() list, and is logged the same way: flagged, never silently
applied.

Requires ANTHROPIC_API_KEY in the environment and the `anthropic` package.
Both are checked explicitly and the call fails loudly if either is missing --
no fallback, no fabricated result, because a synthetic "audit passed" is
worse than no audit at all.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


VLM_MODEL = "claude-sonnet-5"

JUDGE_PROMPT = """You are auditing a 3D city reconstruction against the satellite image it was built from. Both images are the SAME scene from directly overhead (nadir).

Image 1: the original satellite/aerial photo.
Image 2: a synthetic top-down render of the reconstructed 3D model (buildings as flat-roofed colored blocks, height encoded as shading -- darker/taller blocks read as taller structures; this is a schematic render, not a photorealistic one, so judge STRUCTURE and PLACEMENT, not rendering style).

Compare them and report ONLY concrete, visually-grounded discrepancies:
- A structure clearly visible in Image 1 that has no corresponding block in Image 2 (missing building)
- A block in Image 2 sitting where Image 1 shows no structure (false building -- e.g. on a car park, road, or bare ground)
- A block whose footprint shape/size clearly does not match the structure's outline in Image 1
- A block whose relative height/shading looks implausible next to its neighbors given what Image 1 shows (e.g. a clearly single-storey structure rendered as tall as a multi-storey one, or vice versa)

Do NOT comment on color, texture, or rendering style -- the render is intentionally schematic. Do NOT invent issues if the two images broadly agree.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"discrepancies": [{"location": "brief pixel-region description, e.g. 'top-left quadrant, cluster of 3 buildings'", "issue": "missing_building | false_building | shape_mismatch | height_mismatch", "detail": "one sentence, specific to what you see"}], "summary": "one sentence overall verdict"}

If there are no real discrepancies, return {"discrepancies": [], "summary": "..."}.
"""


def render_schematic_rgb(footprints: Sequence[np.ndarray], binfo: Dict,
                         image_np: np.ndarray, shape) -> np.ndarray:
    """Colour each footprint by the real mean image colour under it, then
    darken taller buildings -- so height differences are visible in the
    render without needing a full 3D renderer for what is a 2D judge pass.
    Colour is sampled from the source image, not invented.
    """
    H, W = shape
    out = np.zeros((H, W, 3), np.uint8)
    out[:] = (30, 30, 28)  # background reads as ground, not "empty/missing"

    builds = binfo.get("buildings") or []
    if not builds:
        return out
    heights = np.array([b.get("height_m", 0.0) for b in builds], np.float32)
    hi = float(np.percentile(heights, 95)) if len(heights) else 1.0
    hi = max(hi, 1e-3)

    order = sorted(builds, key=lambda b: b.get("height_m", 0.0))
    for b in order:
        pi = b.get("poly_index")
        if pi is None or pi >= len(footprints):
            continue
        poly = np.asarray(footprints[pi], np.int32)
        if len(poly) < 3:
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        mean_color = image_np[ys, xs].mean(axis=0)
        shade = 1.0 - 0.5 * min(float(b.get("height_m", 0.0)) / hi, 1.0)
        color = tuple(int(c * shade) for c in mean_color[::-1])  # RGB->BGR
        cv2.fillPoly(out, [poly], color)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def save_comparison(image_np: np.ndarray, rendered_rgb: np.ndarray,
                    out_dir: str, name: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    orig_path = os.path.join(out_dir, f"{name}_vlm_original.png")
    rend_path = os.path.join(out_dir, f"{name}_vlm_render.png")
    cv2.imwrite(orig_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(rend_path, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
    return {"original": orig_path, "render": rend_path}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def call_vlm_judge(original_path: str, render_path: str,
                   api_key: Optional[str] = None) -> Dict:
    """Sends both images to Claude, returns the parsed discrepancy list.

    Raises RuntimeError (not a silent fallback) if the SDK or API key is
    missing -- an audit step that pretends to have run when it didn't is a
    worse failure mode than one that visibly refuses.
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "vlm_audit requires the 'anthropic' package (pip install anthropic)"
        ) from e

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "vlm_audit requires ANTHROPIC_API_KEY in the environment. "
            "Not calling with a missing key -- no fallback result is produced."
        )

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=VLM_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": JUDGE_PROMPT},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                              "data": _b64(original_path)}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                              "data": _b64(render_path)}},
            ],
        }],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"VLM judge returned non-JSON output: {raw[:300]!r}") from e
    return parsed


def audit(footprints: Sequence[np.ndarray], binfo: Dict, image_np: np.ndarray,
         shape, out_dir: str, name: str, api_key: Optional[str] = None) -> Dict:
    """Full Layer-2 pass: render, save both images, call the judge, return
    its raw verdict plus the paths so a human can look at exactly what the
    model saw. Never writes back into the scene's own data.
    """
    rendered = render_schematic_rgb(footprints, binfo, image_np, shape)
    paths = save_comparison(image_np, rendered, out_dir, name)
    verdict = call_vlm_judge(paths["original"], paths["render"], api_key=api_key)
    return {"paths": paths, **verdict}


def report(res: Dict) -> str:
    lines = ["      VLM judge audit (Layer 2, reporting only):"]
    disc = res.get("discrepancies") or []
    if not disc:
        lines.append(f"      no discrepancies flagged -- {res.get('summary', '')}")
        return "\n".join(lines)
    lines.append(f"      {len(disc)} flagged -- {res.get('summary', '')}")
    for d in disc:
        lines.append(f"        [{d.get('issue', '?')}] {d.get('location', '?')}: "
                     f"{d.get('detail', '')}")
    return "\n".join(lines)
