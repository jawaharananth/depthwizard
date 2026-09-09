"""Capture presentation screenshots of a rendered scene, using Microsoft Edge.

Screenshots of a WebGPU viewer cannot be produced headlessly with any
reliability -- the headless GPU stack falls back to a software rasteriser that
either fails outright or renders without the post chain, which would show
something the project does not actually produce. So Edge is driven in a real
window against the real GPU, and the frames are the same ones a judge sees.

Each shot waits for the mesh to load AND for several frames to be presented
before capturing. A WebGPU canvas is not finished when the DOM says it is: the
post chain (GTAO, denoise, SSR, bloom, SMAA) resolves over a few frames, and a
capture taken too early shows an un-composited image.

    python capture.py jax165
    python capture.py jax165 --out docs/shots --width 2560 --height 1440
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Camera positions, chosen to show different claims rather than to look varied:
#   overview    the whole reconstruction, so extent and density read at once
#   oblique     building volume and vertical walls -- the thing a heightfield
#               cannot produce
#   low_angle   the skyline, where relative heights are legible
#   street      close enough to see individual roofs and footprint edges
# Distances are tighter than the app's default framing. The viewer sizes its
# camera so a user can orbit without the scene leaving frame; a still does not
# need that headroom, and the empty margin reads as a small model on a slide.
SHOTS = [
    ("01_overview",   {"az": 35,  "el": 55, "d": 0.78}, "rgb"),
    ("02_oblique",    {"az": 145, "el": 28, "d": 0.72}, "rgb"),
    ("03_low_angle",  {"az": 250, "el": 13, "d": 0.62}, "rgb"),
    ("04_close",      {"az": 60,  "el": 22, "d": 0.32}, "rgb"),
    ("05_top_down",   {"az": 0,   "el": 88, "d": 0.72}, "rgb"),
    # No heatmap shot: build_city does not write terrain_heatmap.png, so that
    # mode falls back to RGB and would produce a duplicate of 01_overview.
    ("07_confidence", {"az": 35,  "el": 55, "d": 0.78}, "confidence"),
    ("08_slope",      {"az": 35,  "el": 55, "d": 0.78}, "slope"),
    ("09_vs_lidar",   {"az": 35,  "el": 55, "d": 0.78}, "error"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Screenshot a rendered scene")
    ap.add_argument("scene", help="scene id under scenes/, e.g. jax165")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=1350)
    ap.add_argument("--settle", type=float, default=2.5,
                    help="seconds to let the post chain resolve per shot")
    a = ap.parse_args()

    out = a.out or os.path.join("docs", "shots", a.scene)
    os.makedirs(out, exist_ok=True)

    from playwright.sync_api import sync_playwright

    url = f"{a.url}/viewer/index.html?scene={a.scene}"
    print(f"capturing {url}")
    print(f"  -> {out}  at {a.width}x{a.height}")

    with sync_playwright() as pw:
        # Edge, headed. WebGPU needs a real GPU context; headless gives a
        # software fallback that renders without the post chain or not at all.
        browser = pw.chromium.launch(
            channel="msedge",
            headless=False,
            args=["--enable-unsafe-webgpu",
                  "--enable-features=Vulkan",
                  "--hide-scrollbars"],
        )
        page = browser.new_page(viewport={"width": a.width, "height": a.height},
                                device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="load", timeout=120000)

        print("  waiting for the mesh ...")
        try:
            page.wait_for_function("window.__dw && window.__dw.ready === true",
                                   timeout=180000)
        except Exception:
            print("  ERROR: scene never became ready")
            for e in errors[:5]:
                print(f"    page error: {e[:160]}")
            browser.close()
            sys.exit(1)

        # Hide the HUD and push the fog back. Both are right for the live app
        # and wrong for a still: the panel covers a quarter of the frame, and
        # fog tuned for an orbiting camera washes out the far half of a single
        # fixed shot.
        page.evaluate("() => { window.__dw.setChrome(false);"
                      " window.__dw.setFog(1.2, 6.0); }")

        # Let the first frames composite before touching anything.
        page.wait_for_timeout(3000)

        made = []
        for name, cam, mode in SHOTS:
            ok = page.evaluate(
                "([az, el, d, m]) => { const r = window.__dw.frame(az, el, d);"
                " window.__dw.setTextureMode(m); return r; }",
                [cam["az"], cam["el"], cam["d"], mode])
            if not ok:
                print(f"  {name}: could not frame, skipped")
                continue
            page.wait_for_timeout(int(a.settle * 1000))
            path = os.path.join(out, f"{name}.png")
            page.screenshot(path=path)
            made.append(path)
            print(f"  {name:<14} {mode:<10} -> {os.path.basename(path)}")

        browser.close()

    print(f"\n{len(made)} screenshot(s) in {out}")
    if errors:
        print(f"({len(errors)} page error(s) during capture; first: {errors[0][:120]})")


if __name__ == "__main__":
    main()
