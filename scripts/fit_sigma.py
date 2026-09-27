"""Fit the per-pixel uncertainty calibration for a checkpoint (coverage-based).

    python scripts/fit_sigma.py checkpoints/height_best

The pipeline's 1-sigma height uncertainty is

    sigma = sqrt((k * tta_std)^2 + s0^2 + (c * height)^2)

where height is the predicted height above ground (error grows with it:
tall structures are under-called) and tta_std is the spread of the model over 4 rotated views (what
build_city_image.py computes). k and s0 are fitted by Gaussian negative log
likelihood of the product's error (predict_tiled height minus LiDAR height
above ground) on half of the DFC2019 validation views, and scored on the
other half, which the fit never saw:

  * ECE: mean |observed - nominal| coverage of |error| <= z_p * sigma over
    p = 0.1 .. 0.9;
  * rank correlation (Spearman) between sigma and |error|.

Validation views are the DFC2019 satellite views training/prepare_dfc.py set
aside for epoch selection; no benchmark tile is among them. Writes
sigma_calibration.json into the checkpoint directory.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image
from scipy.stats import norm, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import depth_model as dm  # noqa: E402

VAL = os.path.join(ROOT, "training", "dfc_tiles", "val")


def views():
    for f in sorted(glob.glob(os.path.join(VAL, "*_RGB.h5"))):
        with h5py.File(f) as h:
            rgb = h["image"][:]
        with h5py.File(f.replace("_RGB", "_AGL")) as h:
            agl = h["image"][:].astype(np.float32)
        yield os.path.basename(f)[:-7], rgb, agl


def ece(err, sig):
    ps = np.linspace(0.1, 0.9, 9)
    return float(np.mean([abs(np.mean(np.abs(err) <= norm.ppf(0.5 + p / 2) * sig) - p) for p in ps]))


def main():
    ck = os.path.abspath(sys.argv[1])
    dm.DepthBackbone.METRIC_CKPT = ck
    bb = dm.DepthBackbone(metric=True)
    cache = os.path.join(ck, "_sigma_fit_samples.npz")
    z = np.load(cache, allow_pickle=True) if os.path.exists(cache) else None
    if z is not None and "P" in z.files:
        E, S, P = list(z["E"]), list(z["S"]), list(z["P"])
    else:
        E, S, P = [], [], []
        for name, rgb, agl in views():
            im = Image.fromarray(rgb)
            pred = bb.predict_tiled(im, raw_metric=True)
            _, std = bb.predict_tta(im, raw_metric=True, views=4)
            m = np.isfinite(agl)
            E.append((pred - agl)[m][::7]); S.append(std[m][::7]); P.append(np.maximum(pred, 0)[m][::7])
            print(f"  {name}", flush=True)
        np.savez(cache, E=np.array(E, dtype=object), S=np.array(S, dtype=object),
                 P=np.array(P, dtype=object))
    idx = np.arange(len(E))
    fit, test = idx[::2], idx[1::2]
    ef, sf = np.concatenate([E[i] for i in fit]), np.concatenate([S[i] for i in fit])
    pf = np.concatenate([P[i] for i in fit])
    best = None
    for k in np.arange(0.0, 6.01, 0.25):
      for s0 in np.arange(0.1, 3.01, 0.1):
        for c in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            sig = np.sqrt((k * sf) ** 2 + s0 ** 2 + (c * pf) ** 2)
            # Coverage, not likelihood: the NLL is dominated by the few huge
            # errors at building edges and inflates sigma everywhere
            # (k=5, s0=2.5 -> held-out ECE 0.188); what a user reads is
            # "68 % of pixels within +/- 1 sigma", so that is what is fitted.
            score = ece(ef, sig)
            if best is None or score < best[0]:
                best = (score, float(k), float(s0), float(c))
    _, k, s0, c = best
    et, st = np.concatenate([E[i] for i in test]), np.concatenate([S[i] for i in test])
    pt = np.concatenate([P[i] for i in test])
    sig = np.sqrt((k * st) ** 2 + s0 ** 2 + (c * pt) ** 2)
    out = {"k": k, "s0": s0, "c_height": c, "fit_views": int(len(fit)), "test_views": int(len(test)),
           "ece_heldout": ece(et, sig), "rank_corr": float(spearmanr(sig, np.abs(et))[0]),
           "fitted_by": "scripts/fit_sigma.py (coverage ECE, DFC2019 validation views, alternate halves)"}
    json.dump(out, open(os.path.join(ck, "sigma_calibration.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
