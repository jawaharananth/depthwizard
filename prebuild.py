"""Pre-build a gallery of scenes so a demo renders instantly.

A reconstruction takes about a minute on this hardware. That is fine for someone
who uploaded their own image and is watching the stages tick past; it is wrong
for a judge who clicked a thumbnail and is waiting in front of an audience.

When the images are known in advance, the reconstruction can be done in advance.
Each one is built once, stored complete under scenes/<id>/, and served from
there -- the viewer loads it in under a second because nothing is being computed.

WHAT THIS IS NOT: a way to make the demo look faster than it is. The gallery is
labelled as pre-built in the UI, and the upload path beside it runs the same
pipeline live on whatever anyone drops in. The claim being demonstrated is the
reconstruction; the thumbnail is just a way to reach a finished one quickly.

    python prebuild.py gallery/*.tif
    python prebuild.py --list
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

SCENES_DIR = "scenes"
VIEWER_OUT = "viewer/output"
THUMB_PX = 320

# Everything the viewer may read. Missing entries are skipped rather than fatal,
# so a scene built by an older pipeline still serves.
ASSETS = [
    "terrain.glb", "terrain_texture.png", "scene.json", "buildings.geojson",
    "terrain_heatmap.png", "terrain_confidence.png", "terrain_error.png",
    "terrain_slope.png", "terrain_normal.png", "terrain_ao.png",
    "terrain_roughness.png", "terrain_metalness.png",
]


def _thumb(src_image: str, dst: str) -> bool:
    """A thumbnail of the SOURCE image, not the render.

    The gallery tile should show what the judge is choosing -- the satellite
    photograph -- not the result. Showing the output would tell them what they
    are about to get and remove the point of picking.
    """
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(src_image) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
            im.save(dst, format="JPEG", quality=88)
        return True
    except Exception as e:
        print(f"      thumbnail failed: {e}")
        return False


def build_one(image_path: str, scene_id: str, title: str, px: int) -> dict:
    import build_city_image as bci

    t0 = time.time()
    print(f"\n=== {scene_id}  <- {os.path.basename(image_path)}")
    bci.build(image_path, f"gallery_{scene_id}", max_px=px, stage=True)
    took = time.time() - t0

    dst = os.path.join(SCENES_DIR, scene_id)
    os.makedirs(dst, exist_ok=True)
    copied = []
    for a in ASSETS:
        src = os.path.join(VIEWER_OUT, a)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, a))
            copied.append(a)

    meta = {"id": scene_id, "title": title,
            "source": os.path.basename(image_path),
            "built_seconds": round(took, 1),
            "assets": copied}
    scene_json = os.path.join(dst, "scene.json")
    if os.path.exists(scene_json):
        try:
            with open(scene_json, encoding="utf-8") as f:
                sj = json.load(f)
            meta["buildings"] = sj.get("buildings_extruded")
            meta["tier"] = sj.get("tier")
            meta["extent_m"] = sj.get("extent_m")
            meta["metric"] = bool(sj.get("height_is_metric"))
        except Exception:
            pass
    if _thumb(image_path, os.path.join(dst, "thumb.jpg")):
        meta["thumb"] = "thumb.jpg"

    with open(os.path.join(dst, "gallery.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    -> scenes/{scene_id}  ({len(copied)} assets, {took:.0f}s)")
    return meta


def list_scenes() -> list:
    out = []
    if not os.path.isdir(SCENES_DIR):
        return out
    for name in sorted(os.listdir(SCENES_DIR)):
        gj = os.path.join(SCENES_DIR, name, "gallery.json")
        glb = os.path.join(SCENES_DIR, name, "terrain.glb")
        if not os.path.exists(glb):
            continue
        meta = {"id": name, "title": name}
        if os.path.exists(gj):
            try:
                with open(gj, encoding="utf-8") as f:
                    meta.update(json.load(f))
            except Exception:
                pass
        meta["size_mb"] = round(os.path.getsize(glb) / 1e6, 1)
        out.append(meta)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-build gallery scenes")
    ap.add_argument("images", nargs="*", help="image files, or a directory")
    ap.add_argument("--px", type=int, default=2048)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or not a.images:
        rows = list_scenes()
        if not rows:
            print("no pre-built scenes yet")
            return
        print(f"{'id':<22}{'buildings':>10}{'tier':>6}{'MB':>7}  title")
        for r in rows:
            tier = (r.get("tier") or "?")[:1]
            print(f"{r['id']:<22}{str(r.get('buildings','?')):>10}{tier:>6}"
                  f"{r['size_mb']:>7}  {r.get('title','')}")
        return

    paths = []
    for p in a.images:
        if os.path.isdir(p):
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
                paths.extend(sorted(glob.glob(os.path.join(p, ext))))
        else:
            paths.extend(sorted(glob.glob(p)))
    if not paths:
        raise SystemExit("no images matched")

    print(f"pre-building {len(paths)} scene(s) at {a.px}px")
    done, failed = [], []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        sid = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).lower()
        title = stem.replace("_", " ").replace("-", " ").title()
        try:
            done.append(build_one(path, sid, title, a.px))
        except SystemExit as e:
            # The pipeline refuses obliques and undersized images on purpose.
            print(f"    REFUSED: {str(e)[:120]}")
            failed.append((path, str(e)[:120]))
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed.append((path, f"{type(e).__name__}: {e}"))

    print(f"\nbuilt {len(done)} scene(s), {len(failed)} skipped")
    for p, why in failed:
        print(f"  skipped {os.path.basename(p)}: {why[:90]}")


if __name__ == "__main__":
    main()
