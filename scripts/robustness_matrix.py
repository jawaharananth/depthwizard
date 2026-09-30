"""Build every case from make_cartosat_like.py (or make_upload_matrix.py)
through the upload pipeline and check the product, as the server would.

    DW_MAX_PX=2048 python scripts/robustness_matrix.py work/cartosat_like

Per case: finished without an exception; which product was written (DSM for
georeferenced input, rDSM otherwise); for a DSM, a CRS and heights within
DEM +/- 300 m (the median of a real DSM sits near the DEM); build seconds.
Writes results/robustness/matrix.json.
"""
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def check_dsm(path):
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        tags = s.tags()
        return {"crs": s.crs.to_string() if s.crs else None,
                "datum": tags.get("VERTICAL_DATUM"),
                "min": float(np.nanmin(a)), "median": float(np.nanmedian(a)), "max": float(np.nanmax(a)),
                "valid_frac": float(np.isfinite(a).mean()), "shape": list(a.shape)}


def main(d):
    import server
    import build_city_image as b
    man = json.load(open(os.path.join(d, "manifest.json")))
    out_p = os.path.join(ROOT, "results", "robustness", "matrix.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for f, expect in man.items():
        if f in res:
            continue
        path = os.path.join(d, f)
        kw = server.upload_build_kwargs()
        kw.update(sr_texture=False, pbr_maps=False)
        if not server._is_georeferenced(path):
            kw.update(gsd_m=server.DEFAULT_UPLOAD_GSD_M, gsd_assumed=True)
        name = "rob_" + os.path.splitext(f)[0]
        t = time.time()
        rec = {"expect": expect}
        try:
            b.build(path, name, **kw)
            rec["ok"] = True
            dsm = glob.glob(os.path.join(b.OUT_DIR, f"city_{name}_DSM.tif"))
            rdsm = glob.glob(os.path.join(b.OUT_DIR, f"city_{name}_rDSM.tif"))
            # only files this build wrote, not ones left by an earlier run
            dsm = [p for p in dsm if os.path.getmtime(p) >= t]
            rdsm = [p for p in rdsm if os.path.getmtime(p) >= t]
            rec["product"] = "DSM" if dsm else ("rDSM" if rdsm else None)
            if dsm or rdsm:
                rec["raster"] = check_dsm((dsm or rdsm)[0])
            if dsm:
                import sys as _s
                _s.path.insert(0, os.path.join(ROOT, "scripts"))
                from isro_protocol_eval import evaluate
                e = evaluate(dsm[0], "glo30")["at_30m"]
                if e is None:   # the GLO-30 reference could not be read (network)
                    rec["vs_glo30_30m"] = None
                    rec["eval_note"] = "GLO-30 reference not available; rerun this case"
                else:
                    rec["vs_glo30_30m"] = {k: round(v, 3) for k, v in e.items() if isinstance(v, float)}
                    rec["plausible"] = bool(abs(e["bias_m"]) < 300)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()[-1500:]
        rec["seconds"] = round(time.time() - t)
        res[f] = rec
        json.dump(res, open(out_p, "w"), indent=1)
        print("CASE", f, "ok" if rec.get("ok") else "FAIL", rec.get("product"), rec.get("seconds"), "s",
              (rec.get("vs_glo30_30m") or {}).get("rmse_m", ""), rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
