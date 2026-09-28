"""Screenshot the viewer with the real GPU (Chrome, visible window).

    python tools/render_view.py OUT.png "http://127.0.0.1:8009/viewer/index.html?scene=hitec_2x2&help=0&dwxshot=auto" [--w 1440 --h 810 --wait 25]

Headless Chrome renders WebGPU/WebGL on the CPU (SwiftShader), which takes
tens of minutes for a city scene on this laptop; a normal window uses the
iGPU and takes seconds. Waits for the scene (window.__dwxReady) plus a settle
time, and prints the page's console errors so a broken load is visible.
"""
import argparse
import time

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("url")
ap.add_argument("--w", type=int, default=1440)
ap.add_argument("--h", type=int, default=810)
ap.add_argument("--wait", type=float, default=25.0)
ap.add_argument("--timeout", type=float, default=240.0)
a = ap.parse_args()

errors = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=False,
                          args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
                                # keep painting when the window is covered or the screen is off
                                "--disable-backgrounding-occluded-windows",
                                "--disable-renderer-backgrounding",
                                "--disable-background-timer-throttling",
                                "--disable-features=CalculateNativeWinOcclusion",
                                f"--window-size={a.w},{a.h + 120}"])
    pg = b.new_page(viewport={"width": a.w, "height": a.h})
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    t0 = time.time()
    pg.goto(a.url, wait_until="load", timeout=a.timeout * 1000)
    try:
        pg.wait_for_function("() => window.__dwxReady && window.__dwxReady.rasters !== undefined",
                             timeout=a.timeout * 1000)
    except Exception as e:
        errors.append(f"not ready: {e}")
    ready = time.time() - t0
    time.sleep(a.wait)
    pg.screenshot(path=a.out, timeout=a.timeout * 1000)
    info = pg.evaluate("() => window.__dwxReady || null")
    b.close()
print(f"ready in {ready:.1f}s; wrote {a.out}; dwx {info}")
for e in errors[:12]:
    print("  page error:", e[:300])
