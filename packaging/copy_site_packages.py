"""Copy the serving dependencies from this venv into the embedded runtime.

    venv\\Scripts\\python.exe packaging/copy_site_packages.py

Same interpreter build (CPython 3.13.7 amd64) on both sides, so installed
wheels are binary-compatible. Only the dependency closure of the serving
requirements is copied (resolved from installed metadata), not the whole venv.
"""
import importlib.metadata as md
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "dist_backend", "backend", "python", "Lib", "site-packages")
TOP = ["torch", "transformers", "numpy", "opencv-python-headless", "rasterio", "pillow",
       "scipy", "matplotlib", "safetensors", "huggingface-hub", "pyproj", "shapely",
       "pip-system-certs", "requests", "h5py"]


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def closure():
    seen, stack = {}, [norm(t) for t in TOP]
    dists = {norm(d.metadata["Name"]): d for d in md.distributions()}
    while stack:
        n = stack.pop()
        if n in seen or n not in dists:
            continue
        d = dists[n]
        seen[n] = d
        for r in d.requires or []:
            if "extra ==" in r:
                continue
            stack.append(norm(re.split(r"[ ;<>=!~\[(]", r, 1)[0]))
    return seen


def main():
    ds = closure()
    missing = [t for t in TOP if norm(t) not in ds]
    if missing:
        sys.exit(f"not installed in this venv: {missing}")
    total = 0
    for name, d in sorted(ds.items()):
        n = 0
        for f in d.files or []:
            src = os.path.normpath(os.path.join(str(d.locate_file("")), str(f)))
            if not os.path.isfile(src) or "__pycache__" in src or os.sep + "Scripts" + os.sep in src:
                continue
            rel = os.path.relpath(src, str(d.locate_file("")))
            if rel.startswith(".."):
                continue
            out = os.path.join(DST, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(src, out)
            n += 1
            total += os.path.getsize(src)
        print(f"  {name:28s} {n} files")
    print(f"{len(ds)} distributions, {total / 1e9:.2f} GB -> {DST}")


if __name__ == "__main__":
    main()
