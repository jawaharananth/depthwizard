"""
Save and restore finished scenes.

A build takes minutes and overwrites `viewer/output/` in place, so the previous
scene is destroyed every time a new one is made. This stores a complete,
self-contained copy of a finished scene and can put it back in seconds.

  python scenes.py save jax165 "Downtown Jacksonville, 910 buildings"
  python scenes.py list
  python scenes.py show jax165          # restore + start the server

A saved scene holds everything the viewer needs -- mesh, texture, metadata,
per-building GeoJSON -- so restoring never re-runs inference and never depends
on the source dataset still being present.
"""
import json
import os
import shutil
import subprocess
import sys
import time

SCENES_DIR = "scenes"
VIEWER_DIR = "viewer/output"
PORT = 8800

# Everything the viewer reads. Missing entries are skipped rather than fatal, so
# a scene built by an older pipeline still restores.
ASSETS = [
    "terrain.glb", "terrain_texture.png", "scene.json", "buildings.geojson",
    "terrain_heatmap.png", "terrain_normal.png", "terrain_ao.png",
    "terrain_roughness.png", "terrain_metalness.png",
]


def save(name: str, description: str = "") -> str:
    src = VIEWER_DIR
    if not os.path.isdir(src):
        raise SystemExit(f"nothing staged in {src} -- build a scene first")
    dst = os.path.join(SCENES_DIR, name)
    os.makedirs(dst, exist_ok=True)

    saved = []
    for a in ASSETS:
        p = os.path.join(src, a)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, a))
            saved.append(a)

    meta = {}
    sj = os.path.join(src, "scene.json")
    if os.path.exists(sj):
        with open(sj) as f:
            meta = json.load(f)
    meta["_saved_name"] = name
    meta["_saved_description"] = description
    meta["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["_saved_files"] = saved
    meta["_saved_mb"] = round(
        sum(os.path.getsize(os.path.join(dst, a)) for a in saved) / 1e6, 1)
    with open(os.path.join(dst, "scene.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"saved '{name}' ({meta['_saved_mb']} MB, {len(saved)} files) -> {dst}")
    if description:
        print(f"  {description}")
    return dst


def list_scenes() -> list:
    if not os.path.isdir(SCENES_DIR):
        print("no saved scenes yet")
        return []
    rows = []
    for name in sorted(os.listdir(SCENES_DIR)):
        sj = os.path.join(SCENES_DIR, name, "scene.json")
        if not os.path.exists(sj):
            continue
        with open(sj) as f:
            m = json.load(f)
        rows.append((name, m))
        print(f"  {name:16s} {m.get('_saved_mb', '?')} MB  "
              f"{m.get('tile', '?')}  {m.get('buildings_extruded', '?')} buildings  "
              f"tier {m.get('tier', '?')}")
        if m.get("_saved_description"):
            print(f"                   {m['_saved_description']}")
    return rows


def restore(name: str) -> None:
    src = os.path.join(SCENES_DIR, name)
    if not os.path.isdir(src):
        raise SystemExit(f"no saved scene '{name}' -- run `python scenes.py list`")
    os.makedirs(VIEWER_DIR, exist_ok=True)

    # Clear first: a stale map from the previous scene would otherwise be lit
    # onto this one's geometry.
    for a in ASSETS:
        p = os.path.join(VIEWER_DIR, a)
        if os.path.exists(p):
            os.remove(p)
    for a in os.listdir(src):
        shutil.copy2(os.path.join(src, a), os.path.join(VIEWER_DIR, a))
    print(f"restored '{name}' -> {VIEWER_DIR}")


def _server_running() -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def show(name: str) -> None:
    restore(name)
    if _server_running():
        print(f"server already up")
    else:
        subprocess.Popen([sys.executable, "-m", "http.server", str(PORT)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        print("started server")
    print(f"\n  http://localhost:{PORT}/viewer/index.html\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "save":
        save(sys.argv[2], " ".join(sys.argv[3:]))
    elif cmd == "list":
        list_scenes()
    elif cmd in ("show", "restore"):
        show(sys.argv[2]) if cmd == "show" else restore(sys.argv[2])
    else:
        print(__doc__)
