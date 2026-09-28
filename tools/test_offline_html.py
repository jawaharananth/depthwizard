"""Open an offline DepthWizard HTML from disk with the network cut off and
check the scene actually loads (no screenshot: it only needs the page's own
readiness flags and its console).

    python tools/test_offline_html.py dist/DepthWizard_HITEC_offline.html
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

path = os.path.abspath(sys.argv[1])
errors, requests = [], []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    ctx = b.new_context(offline=True, viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    pg.on("request", lambda r: requests.append(r.url) if r.url.startswith("http") else None)
    t0 = time.time()
    pg.goto("file:///" + path.replace("\\", "/"), timeout=300000)
    ok = True
    try:
        pg.wait_for_function("() => window.__dwxReady && window.__dwxReady.rasters", timeout=600000)
    except Exception as e:
        ok = False
        errors.append(f"not ready: {e}")
    state = pg.evaluate("""() => ({dwx: window.__dwxReady || null,
        meshes: (() => { let n = 0; try { window.__dwScene && window.__dwScene.traverse(o => { if (o.isMesh) n++; }); } catch (e) {} return n; })()})""")
    b.close()
print(f"loaded in {time.time() - t0:.0f}s, ready={ok}, state={state}")
print(f"network requests attempted: {len(requests)}", requests[:5])
for e in errors[:15]:
    print("  error:", e[:250])
