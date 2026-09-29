"""Tree-or-building for a traced candidate region, from what an upload has.

    python scripts/train_tree_filter.py

The traced building candidates come from the height map, so a tree crown
standing 8 m above a lawn is a candidate exactly like a house. On the held-out
DFC2019 tiles most drawn "buildings" in wooded suburbs were trees (JAX_017:
44 % of drawn area on LiDAR vegetation, 19 % on buildings). The one existing
check -- mean excess-green -- misses crowns in shadow.

This fits a logistic model on region features an upload actually has: the
Meta/WRI CHMv2 canopy height (when the image is georeferenced), colour
chromaticity, brightness, saturation and texture. Training regions are LiDAR
components (class 5 vegetation vs class 6 building, >= 20 m^2) on the 35
training areas of training/dfc_tiles, imagery degraded to ISRO's 0.6 m; the
score is on the 18 held-out benchmark tiles, which training never sees.
Writes calibration/tree_filter.json (coefficients, no sklearn at run time).
"""
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import canopy_source as cs              # noqa: E402
import dfc2019_height as DH             # noqa: E402
import dfc2019_loader as L              # noqa: E402
import tree_filter as TF                # noqa: E402

TRUTH = os.path.join(ROOT, "dfc2019_data", "truth", "Track3-Truth")
BLIND = os.path.join(ROOT, "final_out", "blind_input")
OUT = os.path.join(ROOT, "calibration", "tree_filter.json")


def degrade(img, px_m):
    """To 0.6 m and back up to the grid the build works on (0.3 m)."""
    h, w = img.shape[:2]
    f = px_m / 0.6
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def truth_bounds(gt):
    t = gt["transform"]
    H, W = gt["cls"].shape
    # the loader anchors its transform at the tile's lower edge
    left, bottom = t.c, t.f
    return left, bottom, left + W * t.a, bottom + H * abs(t.e)


def regions(cls, keep):
    out = []
    for c, lab in ((5, 1), (6, 0)):
        n, cc, st, _ = cv2.connectedComponentsWithStats((cls == c).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= keep:
                out.append((cc == i, lab))
    return out


def train_samples():
    X, Xn, y = [], [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "training", "dfc_tiles", "*", "*_v0_RGB.h5"))):
        tile = os.path.basename(p).split("_v")[0].replace("DFC_", "")
        agl = h5py.File(p.replace("_RGB", "_AGL"))["image"][()].astype(np.float32)
        img = h5py.File(p)["image"][()]
        rows = np.where(np.isfinite(agl).any(1))[0]
        if not len(rows):
            continue
        inset = int(rows[0])
        H, W = agl.shape
        inner = (H - 2 * inset, W - 2 * inset)
        gt = L.load_tile(tile, TRUTH)
        cls = np.zeros((H, W), np.uint8)
        cls[inset:inset + inner[0], inset:inset + inner[1]] = cv2.resize(
            gt["cls"].astype(np.uint8), (inner[1], inner[0]), interpolation=cv2.INTER_NEAREST)
        l, b, r, t = truth_bounds(gt)
        px = (r - l) / inner[1]
        tr = rasterio.transform.from_bounds(l - inset * px, b - inset * px, r + inset * px, t + inset * px, W, H)
        crs = gt["crs"] if not isinstance(gt["crs"], int) else f"EPSG:{gt['crs']}"
        chm = cs.fetch(tr, crs, W, H)
        img = degrade(img, px)
        for m, lab in regions(cls, int(20 / px ** 2)):
            f = TF.features(img, m, chm)
            X.append(f); y.append(lab)
        print(f"  {tile}: {len(y)} regions so far", flush=True)
    return np.array(X), np.array(y)


def test_samples():
    X, y = [], []
    for tile in sorted(DH.benchmark_tiles()):
        p = os.path.join(ROOT, "work", "bench_buildings", f"{tile}_0p6.tif")
        if not os.path.exists(p):
            continue
        gt = L.load_tile(tile, TRUTH)
        H, W = gt["cls"].shape
        with rasterio.open(p) as s:
            img = s.read([1, 2, 3]).transpose(1, 2, 0)
            chm = cs.fetch(s.transform, s.crs, s.width, s.height)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        chm = cv2.resize(chm, (W, H)) if chm is not None else None
        for m, lab in regions(gt["cls"], int(20 / 0.25)):
            X.append(TF.features(img, m, chm)); y.append(lab)
    return np.array(X), np.array(y)


def fit(X, y, cols):
    Xs = X[:, cols]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Z = (Xs - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    wpos = (y == 0).sum() / max(1, (y == 1).sum())       # balance classes
    sw = np.where(y == 1, wpos, 1.0)
    for _ in range(3000):                                 # plain gradient descent, L2
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= 0.1 * (Z.T @ g / len(y) + 1e-3 * w)
        b -= 0.1 * g.mean()
    return {"cols": [TF.FEATURES[c] for c in cols], "mean": mu.tolist(), "std": sd.tolist(),
            "w": w.tolist(), "b": float(b)}


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main():
    X, y = train_samples()
    Xt, yt = test_samples()
    print(f"train {len(y)} regions ({y.mean():.0%} trees), test {len(yt)} ({yt.mean():.0%} trees)")
    all_cols = list(range(len(TF.FEATURES)))
    no_chm = [i for i, n in enumerate(TF.FEATURES) if not n.startswith("chm")]
    has = np.isfinite(X).all(1)
    has_t = np.isfinite(Xt).all(1)
    models = {"with_chm": fit(X[has], y[has], all_cols),
              "no_chm": fit(np.nan_to_num(X), y, no_chm)}
    report = {}
    for k, m in models.items():
        Xe = Xt[has_t] if k == "with_chm" else np.nan_to_num(Xt)
        ye = yt[has_t] if k == "with_chm" else yt
        p = TF.predict(m, Xe)
        rows = {"auc": auc(p, ye)}
        for thr in (0.6, 0.7, 0.8, 0.9):
            rows[f"thr{thr}"] = {"trees_removed": float((p[ye == 1] >= thr).mean()),
                                 "buildings_lost": float((p[ye == 0] >= thr).mean())}
        report[k] = rows
        print(k, json.dumps(rows))
    exg = np.array([f[TF.FEATURES.index("exg")] for f in Xt])
    report["baseline_exg_auc"] = auc(exg, yt)
    print("baseline excess-green AUC", report["baseline_exg_auc"])
    json.dump({"models": models, "held_out_report": report,
               "trained_on": "DFC2019 training areas (training/dfc_tiles v0), imagery at 0.6 m",
               "scored_on": "18 held-out DFC2019 benchmark tiles"}, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
