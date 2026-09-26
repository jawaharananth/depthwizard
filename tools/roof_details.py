"""
Rooftop structures (water tanks, stair/lift rooms, plant rooms) for a scene.

    venv\\Scripts\\python.exe tools\\roof_details.py hitec_2x2 [--overlay-dir DIR]

Two real-data methods, tagged per structure ("src"):

  "dsm_residual"  DSM minus the roof plane (median DSM inside the footprint
                  eroded 1 px); connected residual > 1.2 m, area >= 3 m^2,
                  height = blob p90 residual. On HITEC this finds ~nothing:
                  the exported DSM is prismified (one flat value per building,
                  heights fused from Open Buildings 2.5D), so it carries no
                  rooftop relief. Kept so a scene with a real DSM gets it.
  "shadow"        Measured from each structure's own shadow on its roof in
                  the display imagery (0.3 m/px) under the scene's fitted sun
                  (scene.json sun_azimuth_deg / sun_elevation_deg, fitted to
                  300 building shadows). Along the anti-sun direction, the
                  dark run from the structure's top edge is
                      run = h * (1/tan(el) - lean . shadow_dir)
                  (the oblique view hides the first lean*h of the shadow
                  under the structure's own displaced top; scene.json
                  lean_m_per_m). Structures >= 1.5 m, <= 8 m and >= 1.5 m^2
                  plan (a 1000-2000 L tank is ~1.2-1.4 m across) are emitted.
                  Plan shape: the lit region immediately sun-side of the
                  shadow, depth from the colour edge where one exists, else
                  the shadow's own width (square plan) -- INFERRED outline.

Placement: each building's roof is registered in the image (the footprint
polygons sit at the building BASE; its photographed roof is displaced by
lean*h). Detected structures are mapped back into the prism's frame (the
terrain.glb buildings stand on the footprints) so they sit on the prism roof.

Outputs (scenes/<scene>/):
  roof_details.glb   one mesh "roof_details": POSITION, NORMAL, TEXCOORD_0
                     into the scene texture (same UV convention as the
                     building prisms: u = col/(W-1), v = 1 - row/(H-1) on the
                     4096 grid). Top faces sample the structure's own
                     photographed top; walls sample its photographed facade /
                     shadow strip. No material (viewer assigns, like
                     structures.glb). World frame = terrain.glb (see
                     tools/scene_frame.py).
  roof_details.json  per-structure records + stats (schema in the header).
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_frame import ROOT, SceneFrame  # noqa: E402

sys.path.insert(0, ROOT)
from city_model import _ear_clip  # noqa: E402
from glb_export import export_meshes_glb  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ring_px(sf, f):
    g = f["geometry"]
    ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
    xy = np.asarray(ring, np.float64)
    c, r = sf.px_from_crs(xy[:, 0], xy[:, 1])
    pts = np.stack([c, r], 1)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def outline_points(poly, step=1.0):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(2, int(np.hypot(*(b - a)) / step))
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        out.append(a + (b - a) * t)
    return np.vstack(out)


# ------------------------------------------------------------ registration --
def register_roof(grad, poly8, pred, x0, y0):
    """Shift (px, 8192 grid) of the footprint outline that best lies on image
    edges, searched around the lean prediction."""
    pts = outline_points(poly8) - [x0, y0]
    R = int(max(4, 0.3 * np.hypot(*pred)))
    best, bs = -1.0, pred
    H, W = grad.shape
    for dy in range(int(pred[1]) - R, int(pred[1]) + R + 1):
        yy = np.round(pts[:, 1] + dy).astype(int)
        for dx in range(int(pred[0]) - R, int(pred[0]) + R + 1):
            xx = np.round(pts[:, 0] + dx).astype(int)
            ok = (xx >= 0) & (yy >= 0) & (xx < W) & (yy < H)
            if ok.mean() < 0.9:
                continue
            s = float(grad[yy[ok], xx[ok]].mean())
            if s > best:
                best, bs = s, (dx, dy)
    return np.array(bs, np.float64), best


# ------------------------------------------------------------ shadow method -
def shadow_structures(rgb, roof, sun_img, k_run, px_m, min_h=1.5, max_h=8.0, rel=0.65, cap=130.0):
    """rgb/roof: crop in image frame. Returns list of (caster_mask, shadow_mask,
    height_m, n_scanlines, plan_from) in the crop frame."""
    Hc, Wc = roof.shape
    ang = math.degrees(math.atan2(sun_img[1], sun_img[0]))   # image angle of sun dir
    ctr = (Wc / 2.0, Hc / 2.0)
    D = int(np.ceil(np.hypot(Hc, Wc))) + 4
    M = cv2.getRotationMatrix2D(ctr, ang, 1.0)                # CCW by ang -> sun on +x
    M[:, 2] += [D / 2 - ctr[0], D / 2 - ctr[1]]
    v = M[:, :2] @ np.asarray(sun_img)
    assert v[0] > 0.99, v
    rr = cv2.warpAffine(rgb, M, (D, D), flags=cv2.INTER_LINEAR)
    rm = cv2.warpAffine(roof.astype(np.uint8), M, (D, D), flags=cv2.INTER_NEAREST).astype(bool)
    Y = rr.astype(np.float32) @ np.float32([0.2126, 0.7152, 0.0722])
    if rm.sum() < 50:
        return []
    ref = float(np.median(Y[rm]))
    S = rm & (Y < min(rel * ref, cap))
    S = ndi.binary_opening(S, np.ones((2, 2)))
    lab, n = ndi.label(S)
    if n == 0:
        return []
    Dmax = int(round(8.0 / px_m))
    runs = {}   # blob -> list of (row, a, b, depth, hit_edge)
    rows = np.nonzero(S.any(1))[0]
    for y in rows:
        s = S[y]
        d = np.diff(np.r_[0, s.astype(np.int8), 0])
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        for a, b in zip(starts, ends):
            if a - 1 < 0 or b + 3 >= D:
                continue
            if not (rm[y, a - 1] and rm[y, b + 1] and rm[y, b + 2]):
                continue            # shadow must start at a caster and end on lit roof
            c0 = rr[y, b + 1:b + 3].astype(np.float32).mean(0)
            j = b + 1
            while (j < D and j - b <= Dmax and rm[y, j] and not S[y, j]
                   and np.abs(rr[y, j].astype(np.float32) - c0).max() < 40):
                j += 1
            depth = j - b - 1
            runs.setdefault(int(lab[y, a]), []).append((y, a, b, depth, j - b > Dmax))
    out = []
    Minv = cv2.invertAffineTransform(M)
    for bl, rs in runs.items():
        if len(rs) < 3:
            continue
        rs = np.array(rs, np.float64)
        run_px = np.percentile(rs[:, 2] - rs[:, 1] + 1, 75)   # core of the shadow, not its tapered ends
        h = run_px * px_m / k_run
        if not (min_h <= h <= max_h):
            continue
        width = rs[:, 0].max() - rs[:, 0].min() + 1
        med_depth = np.median(rs[:, 3])
        edge_frac = np.mean(rs[:, 4])
        if edge_frac > 0.5 or med_depth < 2:
            depth, plan = min(width, Dmax), "square"      # no colour edge: assume square plan
        else:
            # textured roofs stop the colour march early: never thinner than
            # 1.5 m (or the shadow's own width if smaller)
            depth, plan = max(med_depth, min(width, round(1.5 / px_m))), "colour_edge"
        cm = np.zeros((D, D), np.uint8)
        for y, a, b, _, _ in rs:
            y, b = int(y), int(b)
            cm[y, b + 1:min(D, b + 1 + int(depth))] = 1
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cm &= rm.astype(np.uint8)
        cm = ndi.binary_fill_holes(cm).astype(np.uint8)
        back = cv2.warpAffine(cm, Minv, (Wc, Hc), flags=cv2.INTER_NEAREST)
        sm = cv2.warpAffine((lab == bl).astype(np.uint8), Minv, (Wc, Hc), flags=cv2.INTER_NEAREST)
        out.append((back.astype(bool), sm.astype(bool), float(h), len(rs), plan))
    return out


# ------------------------------------------------------------ mesh ----------
def prism_mesh(top_xz, uv_top, uv_base, y0, y1):
    """Flat-shaded closed-top prism: returns verts, normals, uvs, faces."""
    n = len(top_xz)
    area = 0.5 * np.sum(top_xz[:, 0] * np.roll(top_xz[:, 1], -1) - np.roll(top_xz[:, 0], -1) * top_xz[:, 1])
    if area < 0:   # make the ring CCW in (x, z)
        top_xz, uv_top, uv_base = top_xz[::-1], uv_top[::-1], uv_base[::-1]
    V, N, U, F = [], [], [], []
    tris = _ear_clip(top_xz.astype(np.float64))
    for i in range(n):
        V.append((top_xz[i, 0], y1, top_xz[i, 1]))
        N.append((0, 1, 0))
        U.append(uv_top[i])
    for a, b, c in tris:
        pa, pb, pc = (np.array(V[k]) for k in (a, b, c))
        if np.cross(pb - pa, pc - pa)[1] < 0:
            b, c = c, b
        F.append((a, b, c))
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = top_xz[i], top_xz[j]
        e = p1 - p0
        L = np.hypot(*e)
        if L < 1e-6:
            continue
        nrm = np.array([e[1], 0.0, -e[0]]) / L          # outward for a CCW (x,z) ring
        base = len(V)
        V += [(p0[0], y0, p0[1]), (p1[0], y0, p1[1]), (p1[0], y1, p1[1]), (p0[0], y1, p0[1])]
        N += [tuple(nrm)] * 4
        U += [uv_base[i], uv_base[j], uv_top[j], uv_top[i]]
        tri = [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        va, vb, vc = (np.array(V[k]) for k in tri[0])
        if np.dot(np.cross(vb - va, vc - va), nrm) < 0:
            tri = [(t[0], t[2], t[1]) for t in tri]
        F += tri
    return (np.array(V, np.float32), np.array(N, np.float32), np.array(U, np.float32),
            np.array(F, np.int64))


# ------------------------------------------------------------ main ----------
def run(scene, overlay_dir=None, min_area_m2=3.0, resid_thr=1.2, min_area_shadow_m2=1.5):
    t0 = time.time()
    sf = SceneFrame(os.path.join("scenes", scene))
    gsd, W, H = sf.gsd, sf.W, sf.H
    datum, datum_note = sf.datum()
    dsm = sf.read_dsm()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    tex = np.asarray(Image.open(sf.texture_path).convert("RGB"))
    S8 = tex.shape[0] / H                  # texture px per working px (2 on HITEC)
    px_m = gsd / S8
    az, el = math.radians(sf.meta["sun_azimuth_deg"]), math.radians(sf.meta["sun_elevation_deg"])
    sun_img = np.array([math.sin(az), -math.cos(az)])          # (col, row) toward the sun
    lean = np.array(sf.meta.get("lean_m_per_m") or [0.0, 0.0])
    lean_img = -lean                                            # roof displaced NW of base (measured)
    k_run = 1.0 / math.tan(el) - float(lean_img @ (-sun_img))
    log(f"sun az {math.degrees(az):.2f} el {math.degrees(el):.2f}; lean {lean}; run/h = {k_run:.3f}")

    feats = json.load(open(os.path.join(sf.dir, "buildings.geojson")))["features"]
    recs, meshes = [], []
    overlay_items = []
    n_resid = 0
    reg_shifts = []
    for bi, f in enumerate(feats):
        props = f.get("properties", {})
        hb = float(props.get("height_m") or 0.0)
        poly = ring_px(sf, f)
        if len(poly) < 3:
            continue
        # ---- roof level (prism top) from the DSM, which is flat per prism
        x0, y0 = np.floor(poly.min(0)).astype(int) - 2
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 3
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
        me = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)
        if me.sum() < 8:
            continue
        d = dsm[y0:y1, x0:x1]
        roof_abs = float(np.nanmedian(d[me]))
        roof_y = roof_abs - datum
        area_m2 = float(m.sum()) * gsd * gsd
        # ---- method A: DSM residual
        res = np.where(me, d - roof_abs, 0)
        bl, nb = ndi.label(res > resid_thr)
        rim = me & ~cv2.erode(me.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        for k in range(1, nb + 1):
            bm = bl == k
            if (bm & rim).sum() > 0.3 * bm.sum():
                continue            # hugs the footprint edge: an adjacent prism, not a roof structure
            if bm.sum() * gsd * gsd < min_area_m2:
                continue
            if np.percentile(res[bm], 90) > 8.0:
                continue            # a neighbouring taller prism clipped by this footprint, not a roof structure
            n_resid += 1
            cs, _ = cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True).reshape(-1, 2).astype(np.float64)
            c += [x0 + 0.5, y0 + 0.5]
            h = float(np.percentile(res[bm], 90))
            recs.append(dict(b=int(props.get("id", bi)), src="dsm_residual", img=c * S8, world=c.copy(),
                             h=h, roof_y=roof_y, base_img=c * S8, pts=int(bm.sum())))
        if area_m2 < 30 or hb < 2.5:
            continue
        # ---- method B: shadows in the display image
        poly8 = poly * S8
        pred = lean_img * hb / px_m
        pad = int(np.hypot(*pred)) + 20
        X0 = int(max(0, poly8[:, 0].min() - pad))
        Y0 = int(max(0, poly8[:, 1].min() - pad))
        X1 = int(min(tex.shape[1], poly8[:, 0].max() + pad))
        Y1 = int(min(tex.shape[0], poly8[:, 1].max() + pad))
        crop = tex[Y0:Y1, X0:X1]
        L = cv2.GaussianBlur(crop.astype(np.float32) @ np.float32([0.2126, 0.7152, 0.0722]), (0, 0), 1.0)
        grad = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
        shift, score = register_roof(grad, poly8, pred, X0, Y0)
        reg_shifts.append((hb, *(shift * px_m)))
        roof8 = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(roof8, [np.round(poly8 + shift - [X0, Y0]).astype(np.int32)], 1)
        roof8 = cv2.erode(roof8, np.ones((5, 5), np.uint8)).astype(bool)   # drop parapet edge
        found = shadow_structures(crop, roof8, sun_img, k_run, px_m)
        for cmask, smask, h, nsl, plan in found:
            if cmask.sum() * px_m * px_m < min_area_shadow_m2:
                continue
            cs, _ = cv2.findContours(cmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True).reshape(-1, 2).astype(np.float64)
            if len(c) < 3 or cv2.contourArea(c.astype(np.float32)) * px_m * px_m < min_area_shadow_m2:
                continue
            c_img = c + [X0 + 0.5, Y0 + 0.5]                     # top outline, 8192 px
            # image top = base_world + lean_img*(hb + h); roof registration
            # measured lean_img*hb directly
            world8 = c_img - shift - lean_img * h / px_m
            base_img = c_img - lean_img * h / px_m               # where the structure's base shows
            col = np.median(crop[cmask], axis=0)
            recs.append(dict(b=int(props.get("id", bi)), src="shadow", img=c_img, world=world8 / S8,
                             h=h, roof_y=roof_y, base_img=base_img, pts=int(cmask.sum()),
                             rgb=col, plan=plan, scanlines=nsl, reg_score=score))
            if overlay_dir:
                overlay_items.append((X0, Y0, cmask, smask))
        if bi % 1000 == 0:
            log(f"{bi}/{len(feats)} buildings, {len(recs)} structures")

    # ---- colours for residual blobs, meshes, records --------------------
    V, N, U, F = [], [], [], []
    nv = 0
    out = []
    for r in recs:
        if "rgb" not in r:
            mm = np.zeros(tex.shape[:2], np.uint8)
            cv2.fillPoly(mm, [np.round(r["img"]).astype(np.int32)], 1)
            r["rgb"] = np.median(tex[mm > 0], axis=0)
        wc = r["world"]
        xz = np.stack([wc[:, 0] * gsd, -wc[:, 1] * gsd], 1)
        # Drop what cannot be a rooftop structure: slivers (a shadow edge or a
        # solar-array seam traced as an outline) and anything taller than a
        # stair room or tank plausibly stands.
        _a = 0.5 * abs(np.sum(xz[:, 0] * np.roll(xz[:, 1], -1) - np.roll(xz[:, 0], -1) * xz[:, 1]))
        _p = float(np.sum(np.hypot(*(np.roll(xz, -1, 0) - xz).T)))
        if _p <= 0 or 4 * np.pi * _a / _p ** 2 < 0.15 or 2 * _a / _p < 0.6 or r["h"] > 8.0:
            continue
        uv_t = np.stack([r["img"][:, 0] / S8 / (W - 1), 1 - r["img"][:, 1] / S8 / (H - 1)], 1)
        uv_b = np.stack([r["base_img"][:, 0] / S8 / (W - 1), 1 - r["base_img"][:, 1] / S8 / (H - 1)], 1)
        y0, y1 = r["roof_y"] - 0.2, r["roof_y"] + r["h"]
        v, n_, u, fc = prism_mesh(xz, uv_t, uv_b, y0, y1)
        if not len(fc):
            continue
        V.append(v); N.append(n_); U.append(u); F.append(fc + nv)
        nv += len(v)
        a = 0.5 * abs(np.sum(xz[:, 0] * np.roll(xz[:, 1], -1) - np.roll(xz[:, 0], -1) * xz[:, 1]))
        out.append({"building_id": r["b"], "src": r["src"], "h": round(r["h"], 2),
                    "base_y": round(r["roof_y"], 2), "area_m2": round(float(a), 1),
                    "rgb": [int(round(c)) for c in r["rgb"]],
                    "plan": r.get("plan", "dsm"),
                    "xz": np.round(xz, 2).tolist()})
    glb = os.path.join(sf.dir, "roof_details.glb")
    if V:
        export_meshes_glb(glb, [("roof_details", np.vstack(V), np.vstack(N), "TEXCOORD_0",
                                 np.vstack(U), np.vstack(F))])
    hs = np.array([o["h"] for o in out]) if out else np.zeros(1)
    per_b = {}
    for o in out:
        per_b[o["building_id"]] = per_b.get(o["building_id"], 0) + 1
    rs = np.array(reg_shifts)
    stats = {
        "structures": len(out),
        "by_src": {s: sum(o["src"] == s for o in out) for s in ("shadow", "dsm_residual")},
        "dsm_residual_blobs": n_resid,
        "buildings_with_structures": len(per_b),
        "buildings_examined": len(feats),
        "height_m": {"p10": float(np.percentile(hs, 10)), "p50": float(np.median(hs)),
                     "p90": float(np.percentile(hs, 90)), "max": float(hs.max())},
        "height_hist_m": {"bins": [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8],
                          "counts": np.histogram(hs, [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8])[0].tolist()},
        "area_m2": {"p10": float(np.percentile([o["area_m2"] for o in out] or [0], 10)),
                    "p50": float(np.median([o["area_m2"] for o in out] or [0])),
                    "p90": float(np.percentile([o["area_m2"] for o in out] or [0], 90))},
        "plan": {p: sum(o["plan"] == p for o in out) for p in ("colour_edge", "square", "dsm")},
        "roof_registration_lean_m_per_m_measured": (
            [round(float(np.median(-rs[rs[:, 0] > 15, 1] / rs[rs[:, 0] > 15, 0])), 3),
             round(float(np.median(-rs[rs[:, 0] > 15, 2] / rs[rs[:, 0] > 15, 0])), 3)]
            if len(rs) and (rs[:, 0] > 15).any() else None),
        "glb_vertices": int(nv), "seconds": round(time.time() - t0, 1),
    }
    doc = {
        "schema": "depthwizard.roof_details/1",
        "scene": scene,
        "frame": {"units": "metres, viewer world = terrain.glb frame (x east, y up, z = -row*gsd north)",
                  "x": f"X_crs - {sf.left}", "z": f"Y_crs - {sf.top}", "y": f"elevation - {datum:.3f}",
                  "datum_source": datum_note,
                  "uv": "TEXCOORD_0 into scene texture_file: u = col/(W-1), v = 1-row/(H-1), 4096 grid"},
        "method": {"shadow": f"h = shadow run along anti-sun / {k_run:.3f} (1/tan(el) - lean.shadow_dir); "
                             f"sun az {math.degrees(az):.2f} el {math.degrees(el):.2f} (scene.json); "
                             "plan outline inferred (colour_edge or square)",
                   "dsm_residual": f"DSM - median roof > {resid_thr} m, area >= {min_area_m2} m^2, h = p90"},
        "glb": "roof_details.glb (mesh 'roof_details': POSITION, NORMAL, TEXCOORD_0; no material)",
        "record": {"building_id": "buildings.geojson properties.id", "src": "shadow | dsm_residual",
                   "h": "structure height above roof (m)", "base_y": "prism roof y (m); mesh base sunk 0.2 m",
                   "area_m2": "plan area", "rgb": "median texture colour of its top, sRGB",
                   "plan": "colour_edge | square | dsm", "xz": "plan polygon [[x,z],...] world m"},
        "stats": stats,
        "structures": out,
    }
    with open(os.path.join(sf.dir, "roof_details.json"), "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    log(f"wrote roof_details.glb/.json: {len(out)} structures")
    print(json.dumps(stats, indent=1))
    if overlay_dir:
        write_overlay(scene, tex, recs, overlay_items, overlay_dir)
    return doc


def write_overlay(scene, tex, recs, items, odir, crops=((3000, 5000), (5400, 4200), (2000, 2000), (5200, 6000))):
    from PIL import Image
    os.makedirs(odir, exist_ok=True)
    for i, (x, y) in enumerate(crops):
        w = 1200
        img = tex[y:y + w, x:x + w].copy()
        o = img.copy()
        for X0, Y0, cm, sm in items:
            ys, xs = np.nonzero(sm)
            ys, xs = ys + Y0 - y, xs + X0 - x
            k = (ys >= 0) & (xs >= 0) & (ys < w) & (xs < w)
            o[ys[k], xs[k]] = (o[ys[k], xs[k]] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
        for r in recs:
            c = np.round(r["img"] - [x, y]).astype(np.int32)
            if (c < -50).all() or (c > w + 50).all():
                continue
            cv2.polylines(o, [c], True, (0, 255, 255) if r["src"] == "shadow" else (255, 128, 0), 2)
            cv2.putText(o, f"{r['h']:.1f}", tuple(c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        Image.fromarray(np.concatenate([img, o], 1)).save(os.path.join(odir, f"{scene}_roof_crop{i}.jpg"),
                                                         quality=90)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--overlay-dir", default=None)
    a = ap.parse_args()
    run(a.scene, overlay_dir=a.overlay_dir)
