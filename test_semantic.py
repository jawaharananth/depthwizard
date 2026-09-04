"""Does road width recover a KNOWN GSD? The tile's GSD is measured, so this is checkable."""
import numpy as np, cv2
import ortho, segmentation as seg, height_cache, semantic_scale as ss
TRUTH="dfc2019_data/truth/Track3-Truth"
for tile in ("JAX_165","JAX_068"):
    try:
        o=ortho.orthorectify(tile,TRUTH,"dfc2019_data/rgb/Track3-RGB-1",
            "dfc2019_data/metadata/Track3-Metadata",out_px=1280,extent_m=640.0)
    except Exception as e:
        print(f"{tile}: {e}"); continue
    img=o["image"]; gsd=o["gsd_m"]
    h=height_cache.load(tile,"tiled_e640",2560)
    if h is not None: h=cv2.resize(h,(1280,1280),interpolation=cv2.INTER_AREA)
    lab,_=seg.segment(img,height=h)
    r=ss.estimate_scale(lab, gsd_m=gsd)
    if r["m_per_px"] is None:
        print(f"{tile}: {r['reason']}"); continue
    print(f"{tile}: true GSD {gsd:.3f} m/px   estimated {r['m_per_px']:.3f}   "
          f"error {(r['m_per_px']-gsd)/gsd*100:+5.1f}%   spread {r['spread_ratio']:.2f}   "
          f"n={r['n']}  median road {r['median_width_m']:.1f} m")
