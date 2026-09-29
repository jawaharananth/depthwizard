"""
Tall-building physics fusion benchmark (DFC2019 Track 3, held-out tiles only).

    python scripts/benchmark_tall.py              # predict (cached) + fuse + score
    python scripts/benchmark_tall.py --overlays   # also write overlay PNGs
    python scripts/benchmark_tall.py --diag       # + physics-only sanity check on
                                                  #   the tall training tiles

WHAT IS MEASURED

Every monocular height model compresses tall buildings. physics/tall_buildings.py
measures candidate tall buildings a second, independent way from the SAME
image (shadow length + relief displacement, with the view's own IMD sun /
satellite geometry) and fuses the two per building. This script reports
per-building error (LiDAR building instances, median heights) by height band
BEFORE vs AFTER that fusion, plus whole-tile RMSE.

TRUTH DISCIPLINE (same rules as scripts/benchmark_blind.py)

  * predict_stage(): truth-free ortho (benchmark_blind._orthorectify_blind --
    SRTM terrain prior, not the LiDAR median), the metric checkpoint, the
    view's own IMD. No truth is loaded. The prediction is hashed.
  * fuse_stage(): physics.tall_buildings.refine() on (image, prediction, IMD
    geometry). No truth.
  * score_stage(): the ONLY call to dfc2019_loader.load_tile. Hashes of both
    the prediction and the fused map are re-checked first.

Scored tiles are dfc2019_height.benchmark_tiles() ONLY -- the 18 tiles the
metric checkpoint never saw (its training used split_tiles() train + val,
which exclude exactly these). The benchmark set contains a single LiDAR
building > 20 m (JAX_122), which is stated in the report rather than hidden.

--diag additionally runs the physics estimator on the tall downtown tiles
(JAX_164..168, 207, 260). Those tiles WERE in the model's training set, so
nothing model-dependent from them is reported as a score; only the physics
height against LiDAR (the physics has no trained parameters) and, labelled
in-sample, the fusion delta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import dfc2019_height as DH
import dfc2019_loader as L
from physics import tall_buildings as TB

CACHE = os.path.join(ROOT, "cache", "tall_bench")
OVERLAYS = os.path.join(ROOT, "final_out", "tall_overlays")
METADATA = os.path.join(ROOT, "dfc2019_data", "metadata", "Track3-Metadata")
DIAG_TILES = ["JAX_164", "JAX_165", "JAX_166", "JAX_167", "JAX_168", "JAX_207", "JAX_260"]
BANDS = [(0, 6), (6, 10), (10, 20), (20, 1e9)]
OUT_PX = 1024


def _fp(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# 1. prediction -- no truth
# --------------------------------------------------------------------------

_MODEL = None


def predict_stage(tile: str) -> dict:
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{tile}_{OUT_PX}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        geom = json.loads(str(z["geom"]))
        return {"tile": tile, "image": z["image"], "pred": z["pred"],
                "gsd": float(z["gsd"]), "geom": geom}

    global _MODEL
    from PIL import Image
    from scripts.benchmark_blind import _orthorectify_blind
    from depth_model import DepthBackbone
    o = _orthorectify_blind(tile, OUT_PX)
    if _MODEL is None:
        _MODEL = DepthBackbone(metric=True, htc=False)
    pred = _MODEL.predict_tiled(Image.fromarray(o["image"]), raw_metric=True)
    pred = np.maximum(pred, 0.0).astype(np.float32)
    geom = TB.read_imd_geometry(L.imd_path_for_rgb(o["rgb_path"], METADATA))
    geom["view"] = os.path.basename(o["rgb_path"])
    geom["terrain_prior"] = o["terrain_height_source"]
    np.savez_compressed(p, image=o["image"], pred=pred, gsd=o["gsd_m"],
                        geom=json.dumps(geom))
    return {"tile": tile, "image": o["image"], "pred": pred, "gsd": o["gsd_m"], "geom": geom}


# --------------------------------------------------------------------------
# 2. physics fusion -- no truth
# --------------------------------------------------------------------------

def fuse_stage(r: dict) -> dict:
    g = r["geom"]
    fused, rep = TB.refine(r["image"], r["pred"], gsd_m=r["gsd"],
                           sun_el_deg=g["sun_el"], sun_az_deg=g["sun_az"],
                           sat_el_deg=g.get("sat_el"), sat_az_deg=g.get("sat_az"),
                           off_nadir_deg=g.get("off_nadir"))
    r["fused"], r["report"] = fused, rep
    r["fp_pred"], r["fp_fused"] = _fp(r["pred"]), _fp(fused)
    return r


# --------------------------------------------------------------------------
# 3. scoring -- the only truth read
# --------------------------------------------------------------------------

def _gt(tile):
    from rescore_baseline import ground_truth_ndsm
    t = L.load_tile(tile, DH.TRUTH_DIR)          # the one ground-truth load site
    dsm = t["dsm"].copy()
    dsm[~t["valid_mask"]] = np.nan
    return ground_truth_ndsm(dsm, t["cls"]), t["cls"], t["valid_mask"]


def score_stage(r: dict) -> dict:
    if _fp(r["pred"]) != r["fp_pred"] or _fp(r["fused"]) != r["fp_fused"]:
        raise AssertionError(f"{r['tile']}: prediction changed between fusion and scoring")
    gt, cls, valid = _gt(r["tile"])
    n = gt.shape[0]
    pred = cv2.resize(r["pred"], (n, n), interpolation=cv2.INTER_AREA)
    fused = cv2.resize(r["fused"], (n, n), interpolation=cv2.INTER_AREA)
    fin = np.isfinite(gt) & valid
    out = {"tile": r["tile"],
           "rmse_before": float(np.sqrt(np.mean((pred[fin] - gt[fin]) ** 2))),
           "rmse_after": float(np.sqrt(np.mean((fused[fin] - gt[fin]) ** 2))),
           "n_px": int(fin.sum())}
    bmask = ((cls == 6) & fin).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(bmask, connectivity=4)
    rows = []
    for i in range(1, nl):
        if st[i, cv2.CC_STAT_AREA] < 40:            # < 10 m2 at 0.5 m
            continue
        m = lab == i
        rows.append((float(np.median(gt[m])), float(np.median(pred[m])),
                     float(np.median(fused[m])), int(m.sum())))
    out["buildings"] = rows
    # per-candidate: physics estimate vs LiDAR (resampled to the truth grid)
    s = n / r["pred"].shape[0]
    for c in r["report"]["candidates"]:
        m = cv2.resize(c["_mask"].astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
        m &= fin
        c["lidar_p90_in_mask"] = float(np.percentile(gt[m], 90)) if m.sum() > 10 else None
        # LiDAR building height under the candidate: the tallest building
        # instance the candidate overlaps most
        ids, cnt = np.unique(lab[m & (cls == 6)], return_counts=True)
        keep = ids > 0
        if keep.any():
            j = ids[keep][np.argmax(cnt[keep])]
            c["lidar_bldg_median"] = float(np.median(gt[lab == j]))
        else:
            c["lidar_bldg_median"] = None
    del s
    return out


# --------------------------------------------------------------------------

def band_table(rows):
    rows = np.array(rows) if rows else np.zeros((0, 4))
    lines = ["| LiDAR height band | n bldg | bias before | MAE before | bias after | MAE after |",
             "|---|---|---|---|---|---|"]
    for lo, hi in BANDS:
        sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi) if len(rows) else np.zeros(0, bool)
        if not sel.any():
            lines.append(f"| {lo:g}-{'' if hi > 1e8 else f'{hi:g}'} m | 0 | | | | |")
            continue
        g, p, f = rows[sel, 0], rows[sel, 1], rows[sel, 2]
        lab = f"> {lo:g} m" if hi > 1e8 else f"{lo:g}-{hi:g} m"
        lines.append(f"| {lab} | {sel.sum()} | {np.mean(p - g):+.2f} | {np.mean(np.abs(p - g)):.2f} "
                     f"| {np.mean(f - g):+.2f} | {np.mean(np.abs(f - g)):.2f} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--tiles", nargs="*")
    args = ap.parse_args()

    bench = args.tiles or DH.benchmark_tiles()
    leak = set(bench) - set(DH.benchmark_tiles())
    if leak and not args.diag:
        print(f"WARNING: {sorted(leak)} are not held-out benchmark tiles")
    sets = [("bench", bench)] + ([("diag", DIAG_TILES)] if args.diag else [])
    all_res = {}
    for name, tiles in sets:
        res = []
        for t in tiles:
            t0 = time.time()
            r = fuse_stage(predict_stage(t))
            if args.overlays:
                os.makedirs(OVERLAYS, exist_ok=True)
                TB.draw_overlay(r["image"], r["report"],
                                os.path.join(OVERLAYS, f"{t}.png"))
            sc = score_stage(r)
            sc["report"] = r["report"]
            sc["geom"] = r["geom"]
            res.append(sc)
            nf = sum(c["fired"] for c in r["report"]["candidates"])
            print(f"[{name}] {t}: cand={len(r['report']['candidates'])} fired={nf} "
                  f"RMSE {sc['rmse_before']:.3f} -> {sc['rmse_after']:.3f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            for c in r["report"]["candidates"]:
                print("    ", TB.fmt_candidate(c),
                      f"| LiDAR bldg {c['lidar_bldg_median']} p90mask {c['lidar_p90_in_mask']}")
        all_res[name] = res
    write_report(all_res)


def _agg_rmse(res, key):
    sq = sum(r[key] ** 2 * r["n_px"] for r in res)
    return float(np.sqrt(sq / max(1, sum(r["n_px"] for r in res))))


def _phys_rows(res):
    out = []
    for r in res:
        for c in r["report"]["candidates"]:
            if c.get("lidar_bldg_median") is None:
                continue
            out.append((r["tile"], c))
    return out


def write_report(all_res):
    js = {k: [{kk: vv for kk, vv in r.items() if kk != "report"} |
              {"candidates": [{kk: vv for kk, vv in c.items() if not kk.startswith("_")}
                              for c in r["report"]["candidates"]]} for r in v]
          for k, v in all_res.items()}
    os.makedirs(os.path.join(ROOT, "final_out"), exist_ok=True)
    with open(os.path.join(ROOT, "final_out", "benchmark_tall.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)
    for name, res in all_res.items():
        rows = [b for r in res for b in r["buildings"]]
        print(f"\n== {name}: {len(res)} tiles, {len(rows)} LiDAR buildings ==")
        print("\n".join(band_table(rows)))
        print(f"whole-tile RMSE (pixel-weighted): {_agg_rmse(res, 'rmse_before'):.3f} -> "
              f"{_agg_rmse(res, 'rmse_after'):.3f} m")
        pr = _phys_rows(res)
        nf = sum(c["fired"] for _, c in pr)
        print(f"candidates with a LiDAR building under them: {len(pr)}, fired: {nf}")
        for kind in ("shadow", "disp"):
            e = [(c[f"h_{kind}"] - c["lidar_bldg_median"]) for _, c in pr
                 if c.get(f"h_{kind}") is not None and c.get(f"conf_{kind}", 0) > 0]
            em = [(c["h_model"] - c["lidar_bldg_median"]) for _, c in pr
                  if c.get(f"h_{kind}") is not None and c.get(f"conf_{kind}", 0) > 0]
            if e:
                print(f"  {kind}: n={len(e)} physics bias {np.mean(e):+.2f} MAE {np.mean(np.abs(e)):.2f} | "
                      f"model on same bldgs bias {np.mean(em):+.2f} MAE {np.mean(np.abs(em)):.2f}")


if __name__ == "__main__":
    main()
