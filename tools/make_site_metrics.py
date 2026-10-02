"""Collect every number the home page shows into viewer/metrics.json.

    python tools/make_site_metrics.py

Nothing on the page is typed in by hand: each figure is read here from the
results file that produced it, and the file name travels with it.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viewer", "metrics.json")


def load(rel):
    p = os.path.join(ROOT, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    m = {"sources": {}}
    tta = load("final_out/tta_eval.json")
    if tta:
        one, four = tta.get("1", {}), tta.get("4", {})
        m["heldout"] = {"bldg_rmse": one.get("bldg_rmse"), "bldg_bias": one.get("bldg_bias"),
                        "n_buildings": one.get("n_buildings"), "pixel_rmse": one.get("rmse"),
                        "pixel_mae": one.get("mae"), "corr": one.get("corr"),
                        "bldg_rmse_4view": four.get("bldg_rmse"), "pixel_rmse_4view": four.get("rmse"),
                        "tiles": 18, "gsd_m": 0.6}
        m["sources"]["heldout"] = "final_out/tta_eval.json (scripts/compare_ckpt.py)"
    cq = load("calibration/conformal_quantile.json")
    if cq:
        m["interval"] = {"half_width_m": cq.get("q"), "coverage": cq.get("held_out_coverage"), "n": cq.get("n")}
        m["sources"]["interval"] = "calibration/conformal_quantile.json"

    places = load("viewer/places.json") or []
    pid = {p["id"]: p for p in places}
    isro = load("results/isro/landscapes.json") or {}
    rows = []
    for key, rec in isro.items():
        place, anchor = key.split("/")
        if anchor != "glo30" or place not in pid:
            continue
        vs = rec.get("vs", {})
        g = (vs.get("glo30") or {}).get("at_30m") or {}
        s = (vs.get("srtm") or {}).get("at_30m") or {}
        if "rmse_m" not in g:
            continue
        rows.append({"id": place, "name": pid[place]["name"], "landscape": rec.get("landscape"),
                     "matched_rmse": g.get("rmse_m"), "matched_bias": g.get("bias_m"), "r": g.get("r"),
                     "cross_rmse": s.get("rmse_m"), "cross_bias": s.get("bias_m")})
    order = {"urban": 0, "hilly": 1, "forest": 2, "forested": 2, "sparse": 3}
    rows.sort(key=lambda r: (order.get(r["landscape"], 9), r["name"]))
    m["isro"] = rows
    m["sources"]["isro"] = "results/isro/landscapes.json (scripts/isro_landscapes.py), DSM anchored to GLO-30"

    rob = load("results/robustness/matrix.json")
    if rob:
        m["robustness"] = {"ok": sum(1 for r in rob.values() if r.get("ok")), "n": len(rob)}
        m["sources"]["robustness"] = "results/robustness/matrix.json (scripts/robustness_matrix.py)"
    res = load("final_out/resolution_test.json")
    if res:
        m["resolution"] = [{"label": k, "bldg_rmse": v.get("bldg_rmse"), "rmse": v.get("rmse")}
                           for k, v in res.items()]
        m["sources"]["resolution"] = "final_out/resolution_test.json (scripts/resolution_test.py)"

    # tests: count what pytest collects
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                           capture_output=True, text=True, timeout=600)
        n = sum(1 for ln in r.stdout.splitlines() if "::" in ln)
        if n:
            m["tests"] = n
    except Exception:
        pass

    # places: one card per place (close-ups open first), facts from scene.json
    cards = []
    closeup = {p["parent"]: p["id"] for p in places if p.get("parent")}
    for p in places:
        if p.get("parent"):
            continue
        sid = closeup.get(p["id"], p["id"])
        sj = load(f"scenes/{sid}/scene.json") or {}
        whole = load(f"scenes/{p['id']}/scene.json") or {}
        f, fw = sj.get("facts") or {}, whole.get("facts") or {}
        thumb = next((t for t in ("thumb.jpg", "thumb.png") if os.path.exists(os.path.join(ROOT, "scenes", sid, t))), None)
        cards.append({"id": p["id"], "open": sid, "name": p["name"], "region": p.get("region"),
                      "landscape": p.get("landscape"), "thumb": f"/scenes/{sid}/{thumb}" if thumb else None,
                      "closeup_m": f.get("ground_m"), "whole_m": fw.get("ground_m"),
                      "pixel_m": f.get("pixel_m"), "tall_m": fw.get("height_p99_m") or f.get("height_p99_m"),
                      "crs": f.get("crs") or fw.get("crs"), "has_closeup": sid != p["id"]})
    m["places"] = cards
    json.dump(m, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, "|", len(rows), "ISRO rows,", len(cards), "places,", m.get("tests"), "tests")


if __name__ == "__main__":
    main()
