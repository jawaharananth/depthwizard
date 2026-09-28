"""End to end, as a judge would: upload each file to the running server, wait
for the model, then check what the viewer is about to show.

    python scripts/upload_matrix_test.py DIR [--port 8009]

Checks per upload: it finishes (no error), how long it took, the tier, that
the cursor rasters are sane (no 655 m codes, heights within 0-300 m, sigma
present when metric), that the 3D mesh and texture exist, and the number of
buildings. Writes final_out/upload_matrix.json.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = sys.argv[1]
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8009
only = [a for a in sys.argv[2:] if not a.startswith("--") and not a.isdigit()]
B = f"http://127.0.0.1:{port}"
OUT = os.path.join(ROOT, "viewer", "output")


def get(path):
    with urllib.request.urlopen(B + path, timeout=30) as r:
        return json.load(r)


def upload(path):
    data = open(path, "rb").read()
    req = urllib.request.Request(f"{B}/api/upload?name={os.path.basename(path)}", data=data,
                                 headers={"Content-Type": "application/octet-stream"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"http": e.code, **json.load(e)}


def check_output():
    m = json.load(open(os.path.join(OUT, "scene.json")))
    r = {"tier": (m.get("tier") or "")[:60], "product": m.get("dsm_product"),
         "metric": m.get("height_is_metric"), "buildings": m.get("buildings_extruded"),
         "facts": {k: m.get("facts", {}).get(k) for k in ("height_p99_m", "height_max_m", "sigma_median_m",
                                                           "pixel_m", "ground_m")}}
    hv = m.get("hover")
    problems = []
    if not hv:
        problems.append("no cursor rasters")
    else:
        a = np.fromfile(os.path.join(OUT, hv["agl"]), "<u2").reshape(hv["height"], hv["width"])
        real = a[a != 65535] / 100.0
        if (a == 65534).any():
            problems.append("655 m code present")
        if real.size == 0:
            problems.append("no height anywhere")
        elif real.max() > 300:
            problems.append(f"height {real.max():.0f} m")
        r["agl_p99"] = round(float(np.percentile(real, 99)), 1) if real.size else None
        r["nodata_share"] = round(float((a == 65535).mean()), 3)
        if hv.get("sigma"):
            s = np.fromfile(os.path.join(OUT, hv["sigma"]), "<u2")
            sr = s[s != 65535] / 100.0
            r["sigma_p50"] = round(float(np.median(sr)), 2) if sr.size else None
        elif m.get("height_is_metric"):
            problems.append("metric but no sigma raster")
    for f in ("terrain.glb", m.get("texture_file") or "terrain_texture.png"):
        if not os.path.exists(os.path.join(OUT, f)):
            problems.append(f"missing {f}")
    r["problems"] = problems
    return r


results = []
files = sorted(f for f in os.listdir(d) if not only or f in only)
for i, f in enumerate(files):
    if i:
        time.sleep(21)                         # the server's one-upload-per-20-s rule
    t0 = time.time()
    j = upload(os.path.join(d, f))
    row = {"file": f, "mb": round(os.path.getsize(os.path.join(d, f)) / 1e6, 1)}
    if "job" not in j:
        row.update({"state": "refused", "message": j.get("error"), "seconds": round(time.time() - t0, 1)})
    else:
        while True:
            s = get(f"/api/status/{j['job']}")
            if s.get("state") in ("done", "failed", "error"):
                break
            time.sleep(3)
        row.update({"state": s.get("state"), "message": s.get("message"), "seconds": round(time.time() - t0, 1)})
        if s.get("state") == "done":
            row.update(check_output())
    results.append(row)
    print(json.dumps(row), flush=True)
json.dump(results, open(os.path.join(ROOT, "final_out", "upload_matrix.json"), "w"), indent=2)
