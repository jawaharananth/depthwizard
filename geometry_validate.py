"""
Formal geometry validation for exported meshes.

WHY A SEPARATE VALIDATOR

The pipeline already checks a handful of invariants inline, but scattered across
the code where each is convenient, and each only where someone remembered. A
mesh can satisfy every one of those and still be unusable: degenerate triangles
that render as nothing, duplicate vertices that break smoothing, faces wound
inconsistently so half a surface culls away, or NaN coordinates that silently
corrupt a bounding box.

Every defect checked here has actually occurred in this project:

  - a ground mesh wound so every face pointed downward, rendering invisible
  - fan-triangulated concave footprints producing self-intersecting shards
  - roof planes extrapolating to 79 m spikes between buildings
  - NaN heights propagating from an unfilled MVS surface

The point is that geometry which LOOKS fine in one view can be broken in a way
that only shows from another angle, so the checks are numeric rather than
visual.

Returns findings rather than raising: a caller may reasonably ship a mesh with a
handful of degenerate triangles, but should never do so without being told.
"""
import numpy as np


def _tri_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def validate(verts: np.ndarray, faces: np.ndarray, name: str = "mesh",
             expect_upward: bool = False, area_eps: float = 1e-9) -> dict:
    """
    expect_upward: for a ground surface, most faces should point up. A mesh
    wound the other way renders as nothing under backface culling while every
    other check still passes, which is exactly how that bug survived before.
    """
    findings = []
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.size == 0 or f.size == 0:
        return {"name": name, "ok": True, "empty": True, "findings": []}

    # --- coordinate sanity -------------------------------------------------
    n_nan = int((~np.isfinite(v)).any(axis=1).sum())
    if n_nan:
        findings.append(("FAIL", f"{n_nan} vertices contain NaN or inf"))

    # --- index integrity ---------------------------------------------------
    if f.max() >= len(v) or f.min() < 0:
        findings.append(("FAIL", f"face index out of range: [{f.min()}, {f.max()}] "
                                 f"for {len(v)} vertices"))
        return {"name": name, "ok": False, "findings": findings}

    # --- degenerate triangles ----------------------------------------------
    repeated = ((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))
    n_rep = int(repeated.sum())
    if n_rep:
        findings.append(("WARN", f"{n_rep} faces reuse a vertex (zero-area by construction)"))

    areas = _tri_areas(v, f)
    n_zero = int((areas <= area_eps).sum())
    if n_zero:
        findings.append(("WARN", f"{n_zero} faces have near-zero area "
                                 f"({n_zero / len(f) * 100:.2f}%)"))

    # --- duplicate vertices ------------------------------------------------
    # Exact duplicates split what should be one surface, which breaks smooth
    # shading and inflates the file. Rounded before comparison because float
    # coordinates that differ in the last bit are duplicates in every sense
    # that matters here.
    uniq = np.unique(np.round(v, 4), axis=0)
    n_dup = len(v) - len(uniq)
    if n_dup > len(v) * 0.5:
        findings.append(("WARN", f"{n_dup} of {len(v)} vertices are duplicates "
                                 f"({n_dup / len(v) * 100:.0f}%) -- mesh is unwelded"))

    # --- winding consistency ----------------------------------------------
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    up_frac = float((nrm[ok, 1] > 0).mean()) if ok.any() else 0.0
    if expect_upward and up_frac < 0.5:
        findings.append(("FAIL", f"only {up_frac*100:.0f}% of faces point upward -- "
                                 f"surface is wound inside-out and will backface-cull"))

    # --- extent sanity -----------------------------------------------------
    finite = v[np.isfinite(v).all(axis=1)]
    if len(finite):
        span = finite.max(axis=0) - finite.min(axis=0)
        if span[1] > 1000:
            findings.append(("WARN", f"vertical span {span[1]:.0f} m is implausible "
                                     f"for a city scene"))

    hard = [m for lvl, m in findings if lvl == "FAIL"]
    return {
        "name": name,
        "ok": not hard,
        "n_verts": int(len(v)),
        "n_faces": int(len(f)),
        "degenerate": n_zero + n_rep,
        "duplicate_verts": int(n_dup),
        "upward_fraction": round(up_frac, 3),
        "findings": findings,
    }


def report(results: list) -> bool:
    """Print a summary. Returns True when every mesh passed its hard checks."""
    all_ok = True
    for r in results:
        if r.get("empty"):
            print(f"  {r['name']:16s} empty (skipped)")
            continue
        status = "OK  " if r["ok"] else "FAIL"
        print(f"  [{status}] {r['name']:16s} {r['n_verts']:>7} verts  "
              f"{r['n_faces']:>7} faces  degenerate {r['degenerate']}  "
              f"up {r['upward_fraction']:.2f}")
        for lvl, msg in r["findings"]:
            print(f"           {lvl}: {msg}")
        all_ok &= r["ok"]
    return all_ok
