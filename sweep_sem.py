import numpy as np, cv2
import ortho, segmentation as seg, height_cache, semantic_scale as ss
TRUTH="dfc2019_data/truth/Track3-Truth"
scenes={}
for tile in ("JAX_165","JAX_068"):
    o=ortho.orthorectify(tile,TRUTH,"dfc2019_data/rgb/Track3-RGB-1",
        "dfc2019_data/metadata/Track3-Metadata",out_px=1280,extent_m=640.0)
    h=height_cache.load(tile,"tiled_e640",2560)
    if h is not None: h=cv2.resize(h,(1280,1280),interpolation=cv2.INTER_AREA)
    lab,_=seg.segment(o["image"],height=h)
    scenes[tile]=(lab,o["gsd_m"])
print(f"{'elong':>6} | " + "  ".join(f"{t:>22s}" for t in scenes))
for el in (1.5,2.0,2.5,3.0):
    row=[]
    for tile,(lab,gsd) in scenes.items():
        w=ss.measure_road_widths_px(lab,gsd_m=gsd,min_elongation=el)
        if w.size<50: row.append("   too little road   "); continue
        est=[]
        for wp in w:
            wm=wp*gsd
            L=min(ss.PLAUSIBLE_LANES,key=lambda L: abs(wm-L*ss.LANE_WIDTH_M))
            est.append((L*ss.LANE_WIDTH_M)/max(wp,1e-6))
        m=float(np.median(est))
        row.append(f"{m:.3f} ({(m-gsd)/gsd*100:+5.1f}%) n={w.size:5d}")
    print(f"{el:6.1f} | " + "  ".join(row))
