"""ISRO-protocol table across landscapes: rebuild each showcase place through
the upload pipeline anchored to GLO-30 and to SRTM, and score each DSM against
both DEMs at the 30 m grid (scripts/isro_protocol_eval.py).

    DW_MAX_PX=2048 python scripts/isro_landscapes.py [place ...]

Writes results/isro/landscapes.json (one record per place x anchor DEM).
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

PLACES = {  # id: (source image, landscape)
    "hitec_2x2": ("hitec2x2.tif", "urban"),
    "mumbai_lower_parel": ("work/showcase/mumbai_lower_parel/mumbai_lower_parel.tif", "urban"),
    "gangtok": ("work/showcase/gangtok/gangtok.tif", "hilly"),
    "darjeeling": ("work/showcase/darjeeling/darjeeling.tif", "hilly"),
    "periyar_forest": ("periyar.tif", "forest"),
    "kanha": ("work/showcase/kanha/kanha.tif", "forest"),
    "jaisalmer_village": ("work/showcase/jaisalmer_village/jaisalmer_village.tif", "sparse"),
}


def main(names):
    import server
    import build_city_image as b
    from isro_protocol_eval import evaluate
    out_p = os.path.join(ROOT, "results", "isro", "landscapes.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for name in names:
        src, land = PLACES[name]
        for anchor in ("glo30", "srtm"):
            key = f"{name}/{anchor}"
            if key in res:
                continue
            kw = server.upload_build_kwargs()
            kw.update(sr_texture=False, pbr_maps=False, dem_source=anchor)
            t = time.time()
            try:
                b.build(src, f"isro_{name}_{anchor}", **kw)
                dsm = f"final_out/city_isro_{name}_{anchor}_DSM.tif"
                # a DSM left by an earlier run must not be scored when this
                # build fell back to an rDSM (e.g. the DEM tile failed to load)
                if not os.path.exists(dsm) or os.path.getmtime(dsm) < t:
                    raise RuntimeError("no DSM written by this build (anchor DEM not usable?)")
                rec = {"place": name, "landscape": land, "anchor_dem": anchor,
                       "build_s": round(time.time() - t),
                       "vs": {d: evaluate(dsm, d) for d in ("glo30", "srtm")}}
            except Exception as e:
                rec = {"place": name, "landscape": land, "anchor_dem": anchor, "error": f"{type(e).__name__}: {e}"}
            res[key] = rec
            json.dump(res, open(out_p, "w"), indent=1)
            v = rec.get("vs", {})
            print("RES", key, land, {d: round(v[d]["at_30m"]["rmse_m"], 2) if v[d]["at_30m"] else None
                                     for d in v} or rec.get("error"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(PLACES))
