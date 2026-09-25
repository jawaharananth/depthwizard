"""
Per-building roof (and, for tall buildings, facade) material classification
from the real pixels of the ~0.3 m display image.

Classes: concrete/plaster, metal sheet, glass/curtain wall, solar panel,
tarp/membrane, vegetation (roof garden).

Features are simple and inspectable -- no learned model:
  colour     median HSV, excess-green fraction, saturated-blue fraction,
             dark-blue "panel" fraction, bright/specular fraction
  texture    mean gradient energy, structure-tensor coherence (one dominant
             stripe direction = sheet seams / panel rows)
  period     share of FFT power in one narrow peak at 1.5-10 px periods
             (corrugation / standing seams / panel rows repeat regularly)

Each class maps to literature PBR values (glTF metallic-roughness):
  concrete/plaster   rough 0.85-0.90, metal 0.0   (Physically Based DB:
                     concrete ~0.9; Unreal/Substance defaults ~0.8-0.9)
  metal sheet        rough 0.45, metal 0.9        (galvanised/painted steel,
                     coated -> not mirror; Substance "painted metal" 0.4-0.5)
  glass/curtain wall rough 0.05, metal 0.0, specular F0 0.04 (IOR 1.5);
                     coated low-e glass reads metallic-ish in renders -> the
                     viewer may use metal up to 0.3 (noted, not applied here)
  solar panel        rough 0.25, metal 0.1        (tempered glass over dark
                     cells; F0 0.04 glass, slight metallic frame/cells)
  tarp/membrane      rough 0.65, metal 0.0        (PVC/PE coated fabric)
  vegetation         rough 0.95, metal 0.0        (leaf litter/grass, diffuse)

Output (one record per buildings.geojson feature, SAME ORDER as the features
array -- which is also the order of the prisms in terrain.glb's "buildings"
mesh, see build_city_image.py's export loop):

    {"feature_index", "id", "centroid_px", "centroid_world_xz",
     "roof": {"class", "confidence", "roughness", "metalness", "specular",
              "features": {...}},
     "facade": {...} | null,
     "tag": "classified"}

    python tools/roof_materials.py scenes/hitec_2x2 hitec2x2_z19.png \
        [--sheet viz_dir] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

CLASSES = ["concrete/plaster", "metal sheet", "glass/curtain wall",
           "solar panel", "tarp/membrane", "vegetation"]
PBR = {  # roughness, metalness, specular (F0-ish, glTF KHR_materials_specular factor)
    "concrete/plaster":   (0.88, 0.00, 0.50),
    "metal sheet":        (0.45, 0.90, 0.50),
    "glass/curtain wall": (0.05, 0.00, 1.00),
    "solar panel":        (0.25, 0.10, 0.80),
    "tarp/membrane":      (0.65, 0.00, 0.50),
    "vegetation":         (0.95, 0.00, 0.35),
}


def _periodicity(g: np.ndarray, m: np.ndarray):
    """Share of spectral power in the strongest narrow peak with period
    1.5-10 px, measured over the largest inscribed-ish square of the mask."""
    ys, xs = np.nonzero(m)
    if ys.size < 256:
        return 0.0, 0.0
    # bounding square of the mask interior, mask-weighted
    s = int(min(64, max(16, np.sqrt(ys.size))))
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, cy - s // 2), max(0, cx - s // 2)
    patch = g[y0:y0 + s, x0:x0 + s].astype(np.float32)
    pm = m[y0:y0 + s, x0:x0 + s]
    if patch.shape != (s, s) or pm.mean() < 0.8:
        return 0.0, 0.0
    patch = (patch - patch.mean()) * np.outer(np.hanning(s), np.hanning(s))
    P = np.abs(np.fft.fftshift(np.fft.fft2(patch))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(s)),
                         np.fft.fftshift(np.fft.fftfreq(s)), indexing="ij")
    fr = np.hypot(fx, fy)
    band = (fr >= 0.1) & (fr <= 0.66)          # period 1.5 .. 10 px
    tot = P[fr > 0.03].sum() + 1e-9
    if not band.any():
        return 0.0, 0.0
    Pb = np.where(band, P, 0)
    k = np.unravel_index(np.argmax(Pb), P.shape)
    # peak = the max bin and its 8-neighbours (and the symmetric twin)
    y, x = k
    peak = Pb[max(0, y - 1):y + 2, max(0, x - 1):x + 2].sum() * 2
    return float(peak / tot), float(1.0 / max(fr[k], 1e-6))


def roof_features(rgb: np.ndarray, m: np.ndarray) -> dict:
    px = rgb[m].astype(np.float32)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[m].astype(np.float32)
    V, S = hsv[:, 2], hsv[:, 1] / 255.0
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gf = g.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2)[m].mean()
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2)[m].mean()
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2)[m].mean()
    tr = Jxx + Jyy + 1e-6
    coh = float(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / tr)
    per, period = _periodicity(g, m)
    exg = 2 * G - R - B
    return {
        "n_px": int(m.sum()),
        "v_med": float(np.median(V)), "v_std": float(np.std(V)),
        "s_med": float(np.median(S)),
        "rgb_med": [float(np.median(R)), float(np.median(G)), float(np.median(B))],
        "veg_frac": float(np.mean((exg > 18) & (G > R) & (G > B - 5))),
        "blue_tarp_frac": float(np.mean((B > R + 45) & (B > G + 10) & (V > 110) & (S > 0.35))),
        "panel_frac": float(np.mean((V < 115) & (B >= R + 3) & (B >= G - 6) & (S < 0.45))),
        "dark_frac": float(np.mean(V < 70)),
        "bright_frac": float(np.mean(V > 235)),
        "grad": float(np.hypot(gx, gy)[m].mean()),
        "coherence": coh, "periodicity": per, "period_px": period,
    }


def classify(f: dict, area_m2: float):
    """(class, confidence) from the features above. Order matters: the most
    specific evidence is tested first."""
    sc = {}
    sc["vegetation"] = np.clip((f["veg_frac"] - 0.30) / 0.30, 0, 1)
    sc["tarp/membrane"] = np.clip((f["blue_tarp_frac"] - 0.20) / 0.30, 0, 1)
    # solar: dark blue-grey panels, regular rows
    sc["solar panel"] = (np.clip((f["panel_frac"] - 0.30) / 0.35, 0, 1)
                         * (0.6 + 0.4 * np.clip(f["periodicity"] / 0.08, 0, 1))
                         * (1.0 if f["veg_frac"] < 0.2 else 0.3))
    # metal sheet: one strong stripe direction + a periodic seam pattern,
    # smooth colour, a roof big enough to be a shed/industrial span
    sc["metal sheet"] = (np.clip((f["coherence"] - 0.25) / 0.35, 0, 1)
                         * np.clip(f["periodicity"] / 0.06, 0, 1)
                         * np.clip((area_m2 - 30) / 120, 0, 1)
                         * (1.0 if f["panel_frac"] < 0.35 else 0.4)
                         * (1.0 if f["veg_frac"] < 0.2 else 0.2))
    # glass roof / skylight / atrium: dark, smooth, low saturation, not panels
    sc["glass/curtain wall"] = (np.clip((70 - f["v_med"]) / 30, 0, 1)
                                * np.clip((14 - f["grad"]) / 8, 0, 1)
                                * (1.0 if f["panel_frac"] < 0.5 else 0.3))
    best = max(sc, key=sc.get)
    if sc[best] < 0.35:
        # default class; confidence = how clearly nothing else fits
        return "concrete/plaster", float(np.clip(0.9 - sc[best], 0.5, 0.9)), sc
    others = sorted(sc.values())[-2] if len(sc) > 1 else 0
    return best, float(np.clip(0.5 + 0.5 * (sc[best] - others), 0.4, 0.95)), sc


def facade_class(strip_rgb: np.ndarray, m: np.ndarray):
    """Facade strip seen obliquely: glass curtain wall = darker, bluish/teal,
    smooth, low saturation; else concrete/plaster (punched windows)."""
    f = roof_features(strip_rgb, m)
    r, g, b = f["rgb_med"]
    glass = (np.clip((125 - f["v_med"]) / 40, 0, 1)
             * (1.0 if b >= r - 4 else 0.4)
             * np.clip((f["dark_frac"] + 0.3), 0, 1))
    if glass >= 0.45:
        return "glass/curtain wall", float(np.clip(0.45 + 0.5 * glass, 0.45, 0.9)), f
    return "concrete/plaster", float(np.clip(0.85 - glass, 0.45, 0.85)), f


def run(scene_dir: str, display_path: str, sheet_dir: str = None, limit: int = None,
        lean_px_per_m=None, out_name="materials.json", seed=0):
    meta = json.load(open(os.path.join(scene_dir, "scene.json")))
    a, _, c, _, e, f0 = meta["affine_transform"]
    Wk, Hk = meta["resolution"]
    gj = json.load(open(os.path.join(scene_dir, "buildings.geojson")))
    feats = gj["features"][:limit] if limit else gj["features"]
    img = np.array(Image.open(display_path).convert("RGB"))
    k = img.shape[1] / Wk                    # display px per working px
    gsd_d = abs(a) / k
    if lean_px_per_m is None:
        lx, ly = meta["lean_m_per_m"]
        lean_px_per_m = (lx / abs(a), -ly / abs(a))    # working px per metre
    lvec = np.array(lean_px_per_m, float) * k          # display px per metre
    out, t0 = [], time.time()
    for fi, ft in enumerate(feats):
        pr = ft["properties"]
        ring = np.array(ft["geometry"]["coordinates"][0], float)
        pw = np.column_stack([(ring[:, 0] - c) / a, (ring[:, 1] - f0) / e])
        pd = pw * k
        x0, y0 = np.floor(pd.min(0)).astype(int) - 2
        x1, y1 = np.ceil(pd.max(0)).astype(int) + 3
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        rec = {"feature_index": fi, "id": pr.get("id"),
               "centroid_px": [round(float(pw[:, 0].mean()), 2), round(float(pw[:, 1].mean()), 2)],
               "centroid_world_xz": [round(float(pw[:, 0].mean() * abs(a)), 2),
                                     round(float(-pw[:, 1].mean() * abs(a)), 2)],
               "roof": None, "facade": None, "tag": "classified"}
        if x1 - x0 < 4 or y1 - y0 < 4:
            out.append(rec); continue
        crop = img[y0:y1, x0:x1]
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(pd - [x0, y0]).astype(np.int32)], 1)
        me = cv2.erode(m, np.ones((5, 5), np.uint8))
        mm = (me if me.sum() >= 40 else m).astype(bool)
        area_m2 = float(m.sum()) * gsd_d * gsd_d
        if mm.sum() < 12:
            out.append(rec); continue
        fr = roof_features(crop, mm)
        cls, conf, sc = classify(fr, area_m2)
        rgh, met, spec = PBR[cls]
        rec["roof"] = {"class": cls, "confidence": round(conf, 3),
                       "roughness": rgh, "metalness": met, "specular": spec,
                       "scores": {kk: round(float(v), 3) for kk, v in sc.items()},
                       "features": {kk: (round(v, 4) if isinstance(v, float) else v)
                                    for kk, v in fr.items()}}
        # Facade: the swept strip between the roof outline and its base,
        # lean * height away (only where that strip is >= 6 display px wide).
        h = float(pr.get("height_m") or 0)
        sh = -lvec * h
        if np.hypot(*sh) >= 6:
            pts = np.vstack([pd, pd + sh]).astype(np.float32)
            hull = cv2.convexHull(pts).reshape(-1, 2)
            fx0, fy0 = np.maximum(np.floor(hull.min(0)).astype(int), 0)
            fx1 = int(min(np.ceil(hull[:, 0].max()) + 1, img.shape[1]))
            fy1 = int(min(np.ceil(hull[:, 1].max()) + 1, img.shape[0]))
            if fx1 - fx0 > 3 and fy1 - fy0 > 3:
                fm = np.zeros((fy1 - fy0, fx1 - fx0), np.uint8)
                cv2.fillPoly(fm, [np.round(hull - [fx0, fy0]).astype(np.int32)], 1)
                cv2.fillPoly(fm, [np.round(pd - [fx0, fy0]).astype(np.int32)], 0)
                cv2.fillPoly(fm, [np.round(pd + sh - [fx0, fy0]).astype(np.int32)], 0)
                if fm.sum() >= 60:
                    fc, fconf, ff = facade_class(img[fy0:fy1, fx0:fx1], fm.astype(bool))
                    rgh, met, spec = PBR[fc]
                    rec["facade"] = {"class": fc, "confidence": round(fconf, 3),
                                     "roughness": rgh, "metalness": met, "specular": spec,
                                     "note": "strip between roof outline and base "
                                             "(roof - lean*height); footprint "
                                             "misregistration lowers reliability"}
        out.append(rec)
        if fi % 1000 == 0:
            print(f"  {fi}/{len(feats)}  {time.time() - t0:.0f}s", flush=True)
    counts = {}
    for r in out:
        if r["roof"]:
            counts[r["roof"]["class"]] = counts.get(r["roof"]["class"], 0) + 1
    fcounts = {}
    for r in out:
        if r["facade"]:
            fcounts[r["facade"]["class"]] = fcounts.get(r["facade"]["class"], 0) + 1
    doc = {"schema": "depthwizard.materials/1",
           "order": "same order as buildings.geojson features (= prism order in "
                    "terrain.glb 'buildings' mesh); match on feature_index or id",
           "classes": CLASSES,
           "pbr": {kk: {"roughness": v[0], "metalness": v[1], "specular": v[2]}
                   for kk, v in PBR.items()},
           "method": "rule-based on real 0.3 m pixels: colour stats, gradient "
                     "energy, structure-tensor coherence, FFT stripe periodicity",
           "source_image": os.path.basename(display_path),
           "lean_px_per_m_working": [float(x) for x in lean_px_per_m],
           "counts_roof": counts, "counts_facade": fcounts,
           "tag": "classified", "buildings": out}
    if out_name:
        with open(os.path.join(scene_dir, out_name), "w") as fh:
            json.dump(doc, fh)
    if sheet_dir:
        contact_sheet(out, feats, img, k, a, c, e, f0, sheet_dir, seed=seed)
    print("roof classes:", counts, " facade:", fcounts, f" {time.time() - t0:.0f}s")
    return doc


def contact_sheet(out, feats, img, k, a, c, e, f0, sheet_dir, n=48, seed=0,
                  per_class_min=4, name="materials_sheet"):
    """Random roof crops with their label; guarantees a few of each class."""
    os.makedirs(sheet_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    valid = [i for i, r in enumerate(out) if r["roof"] and r["roof"]["features"]["n_px"] > 150]
    pick = []
    for cl in CLASSES[1:]:
        cand = [i for i in valid if out[i]["roof"]["class"] == cl]
        pick += list(rng.choice(cand, min(per_class_min, len(cand)), replace=False)) if cand else []
    rest = [i for i in valid if i not in pick]
    pick += list(rng.choice(rest, max(0, n - len(pick)), replace=False))
    tiles, T = [], 192
    from PIL import ImageDraw
    for i in pick:
        ft = feats[i]
        ring = np.array(ft["geometry"]["coordinates"][0], float)
        pd = np.column_stack([(ring[:, 0] - c) / a, (ring[:, 1] - f0) / e]) * k
        cx, cy = pd.mean(0)
        s = max(24, int(max(np.ptp(pd[:, 0]), np.ptp(pd[:, 1])) * 0.6 + 8))
        xa, ya = int(max(cx - s, 0)), int(max(cy - s, 0))
        xb, yb = int(min(cx + s, img.shape[1])), int(min(cy + s, img.shape[0]))
        crop = img[ya:yb, xa:xb].copy()
        cv2.polylines(crop, [np.round(pd - [xa, ya]).astype(np.int32)], True, (255, 0, 255), 1)
        t = Image.fromarray(crop).resize((T, T), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        r = out[i]["roof"]
        d.rectangle([0, T - 26, T, T], fill=(0, 0, 0))
        d.text((3, T - 25), f"#{i} {r['class'][:14]} {r['confidence']:.2f}", fill=(255, 255, 0))
        fa = out[i]["facade"]
        d.text((3, T - 13), f"fac: {fa['class'][:12] if fa else '-'}", fill=(0, 255, 255))
        tiles.append(t)
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, rows * T))
    for j, t in enumerate(tiles):
        sheet.paste(t, ((j % cols) * T, (j // cols) * T))
    p = os.path.join(sheet_dir, f"{name}_s{seed}.png")
    sheet.save(p)
    print("contact sheet:", p, "indices:", [int(i) for i in pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("display_image")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lean", type=float, nargs=2, default=None,
                    help="working px per metre (x, y); default from scene.json")
    ap.add_argument("--out", default="materials.json", help="'' = do not write")
    a = ap.parse_args()
    run(a.scene_dir, a.display_image, a.sheet, a.limit, a.lean, a.out or None, a.seed)


if __name__ == "__main__":
    sys.exit(main())
