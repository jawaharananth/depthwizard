"""Rebuilds the priority ("best") demo scenes with the current pipeline code,
then copies the fresh output into scenes/<id>/ so the gallery serves the
rebuilt version -- build_city_image.build()'s own stage=True only writes to
viewer/output (the "currently open" scene), it does not touch the saved
gallery copy, so that copy step has to happen here explicitly.

Deliberately NOT touching the 8 kn_* Karnataka scenes in this pass -- they
were built via build_karnataka_gt.py with a <tag> argument that isn't
recorded anywhere on disk, and guessing it risks silently swapping in the
wrong ground-truth building mask for that scene. Left for a later pass once
the real tag-to-scene mapping is confirmed.

What this DOES change: the new region_footprints.py height_model_estimate
field, the compare-slider assets for any scene with real ground truth,
whatever bug fixes landed in the pipeline since these scenes were last
built. What this does NOT change: the actual predicted heights or accuracy
-- the model checkpoint itself has not been retrained, so the underlying
numbers are identical, just recomputed with current code.
"""
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prebuild

VIEWER_OUT = "viewer/output"
SCENES_DIR = "scenes"

JOBS = [
    {"id": "agra_v2", "cmd": [sys.executable, "build_city_image.py",
                              "india/agra.png", "--name", "agra_v2", "--px", "2560"]},
    {"id": "varanasi_v2", "cmd": [sys.executable, "build_city_image.py",
                                  "india/varanasi.tif", "--name", "varanasi_v2", "--px", "2560"]},
    {"id": "mumbai_best", "cmd": [sys.executable, "build_city_image.py",
                                  "india/mumbai_1_10.png", "--name", "mumbai_best", "--px", "2560"]},
    {"id": "mumbai_colaba_full", "cmd": [sys.executable, "build_mumbai_bmc_combined.py"]},
]


def copy_to_scenes(scene_id: str) -> int:
    """Mirrors viewer/output into scenes/<id>/ -- and only mirrors, never
    unions. A real bug caught the hard way: viewer/output is a SHARED
    staging directory, and build_city_image.py only ever overwrites the
    specific files it produces (terrain.glb, terrain_texture.png,
    buildings.geojson, scene.json) -- it has no reason to know about, let
    alone delete, an error/confidence/slope map a PREVIOUS, unrelated build
    left behind. The first version of this function copied whatever existed
    in prebuild.ASSETS unconditionally, which silently carried an earlier
    build_city.py test run's ground-truth error map into a plain image scene
    that has no ground truth at all -- wrong data presented as real, not
    just a cosmetic leftover. Same fix already applied and tested in
    server.py's _copy_cached_result: for every possible asset, copy it if
    the current build produced it, remove it from the destination if not.
    """
    dst = os.path.join(SCENES_DIR, scene_id)
    os.makedirs(dst, exist_ok=True)
    copied = 0
    for name in prebuild.ASSETS:
        src = os.path.join(VIEWER_OUT, name)
        dst_file = os.path.join(dst, name)
        if os.path.exists(src):
            shutil.copy2(src, dst_file)
            copied += 1
        elif os.path.exists(dst_file):
            os.remove(dst_file)   # this build did not produce it -- don't leave a stale one
    return copied


def main() -> None:
    results = []
    for job in JOBS:
        t0 = time.time()
        print(f"\n=== rebuilding {job['id']} ===", flush=True)
        print("  ", " ".join(job["cmd"]), flush=True)
        r = subprocess.run(job["cmd"], capture_output=True, text=True)
        took = time.time() - t0
        if r.returncode != 0:
            print(f"  FAILED ({took:.0f}s): {r.stderr[-800:]}", flush=True)
            results.append((job["id"], False, took, r.stderr[-300:]))
            continue
        n = copy_to_scenes(job["id"])
        print(f"  OK ({took:.0f}s), {n} assets copied to scenes/{job['id']}/", flush=True)
        results.append((job["id"], True, took, None))

    print("\n=== summary ===")
    for sid, ok, took, err in results:
        status = "OK" if ok else "FAILED"
        print(f"  {sid:<22} {status:<7} {took:.0f}s" + (f"  -- {err}" if err else ""))


if __name__ == "__main__":
    main()
