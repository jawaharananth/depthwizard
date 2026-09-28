"""Pack exactly what the server container needs into one archive for the VM.

    python deploy/azure/make_bundle.py            # -> dist/depthwizard_bundle.tar.gz

Reads the Dockerfile's own COPY lines, so the bundle and the image can never
disagree, and refuses anything that must not leave this machine: the IEEE
DFC2019 data (licensed, not redistributable), training tiles, uploads, the
Kaggle token, virtualenvs and caches.
"""
import os
import re
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dist", "depthwizard_bundle.tar.gz")
NEVER = re.compile(r"(^|/)(dfc2019_data|training|uploads|venv|\.git|__pycache__|\.kaggle|cache|final_out|work)(/|$)"
                   r"|\.(pyc|log|npy|npz)$|_sigma_fit_samples")


def copy_sources():
    txt = open(os.path.join(ROOT, "Dockerfile"), encoding="utf8").read()
    txt = txt.replace("\\\n", " ")
    items = ["Dockerfile", "requirements-serve.txt", ".dockerignore"]
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("COPY "):
            parts = line.split()[1:-1]
            items += [p for p in parts if not p.startswith("--")]
    items += ["deploy/azure/docker-compose.yml", "deploy/azure/Caddyfile"]
    return items


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n, size = 0, 0
    with tarfile.open(OUT, "w:gz", compresslevel=6) as tar:
        for item in copy_sources():
            src = os.path.join(ROOT, item.rstrip("/"))
            if not os.path.exists(src):
                sys.exit(f"missing: {item}")
            for dirpath, _, files in (os.walk(src) if os.path.isdir(src) else [(os.path.dirname(src), [], [os.path.basename(src)])]):
                for f in files:
                    full = os.path.join(dirpath, f)
                    rel = os.path.relpath(full, ROOT).replace("\\", "/")
                    if NEVER.search(rel) and not rel.startswith("checkpoints/height_best/"):
                        continue
                    if rel.startswith("checkpoints/height_best/") and rel.endswith(".npz"):
                        continue
                    tar.add(full, arcname=rel)
                    n += 1
                    size += os.path.getsize(full)
    print(f"{n} files, {size / 1e9:.2f} GB raw -> {OUT} ({os.path.getsize(OUT) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
