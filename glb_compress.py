"""Meshopt-compress a scene GLB for the viewer (gltfpack), or copy it unchanged.

    import glb_compress
    glb_compress.compress(src_glb, dst_glb)      # -> "meshopt" or "copied (reason)"

The same geometry in about a third of the bytes: positions quantised to 16
bits over the scene (~3.6 cm steps across a 2.3 km scene; measured on HITEC:
every object's extent and height percentiles within 2.1 cm of the original),
normals to 10 bits, texture coordinates to 14. Names are kept on their nodes
(-kn/-km); the viewer copies them back onto the meshes when it loads.

gltfpack comes from tools/gltfpack (npm, WebAssembly -- runs wherever Node
does). Any failure -- no Node, no package, a timeout, an empty output --
falls back to copying the original, so a scene is never lost to compression.
"""
from __future__ import annotations

import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
ARGS = ["-cc", "-kn", "-km", "-ke", "-vp", "16", "-vt", "14", "-vn", "10"]


def _tool():
    base = os.path.join(ROOT, "tools", "gltfpack", "node_modules", ".bin")
    for name in ("gltfpack.cmd", "gltfpack"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return shutil.which("gltfpack")


def compress(src: str, dst: str, timeout_s: int = 300) -> str:
    if os.environ.get("DW_NO_GLB_COMPRESS"):
        shutil.copy2(src, dst)
        return "copied (compression off)"
    tool = _tool()
    if tool is None:
        shutil.copy2(src, dst)
        return "copied (gltfpack not installed)"
    tmp = dst + ".tmp.glb"
    try:
        pr = subprocess.Popen([tool, "-i", src, "-o", tmp] + ARGS, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        try:
            out, err = pr.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # on Windows the tool is a .cmd wrapper around node: killing the
            # wrapper alone leaves node holding the pipes, so end the whole tree
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pr.pid)], capture_output=True)
            else:
                pr.kill()
            pr.communicate(timeout=30)
            raise
        if pr.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1024:
            raise RuntimeError((err or out or "no output").strip()[:200])
        os.replace(tmp, dst)
        return "meshopt"
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        shutil.copy2(src, dst)
        return f"copied ({type(e).__name__}: {e})"
