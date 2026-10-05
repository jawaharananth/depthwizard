"""Build dist_hf/: exactly what the Hugging Face Docker Space needs, nothing that must stay private.

    python deploy/hf/make_space.py

File list = the Dockerfile's COPY lines (deploy/azure/make_bundle.copy_sources), same NEVER filter
(no DFC2019 data, training tiles, uploads, caches, venvs, tokens). Then writes the HF Dockerfile and
README, and scans every text file for secrets before anything can be pushed.
"""
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deploy", "azure"))
import make_bundle as MB  # noqa: E402

OUT = os.path.join(ROOT, "dist_hf")
EXTRA_NEVER = re.compile(r"(^|/)(_private_notes|\.env|\.huggingface|\.cache)(/|$)|\.(pem|key)$")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_]{10,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}"
                    r"|ghp_[A-Za-z0-9]{30,}|\"key\"\s*:\s*\"[A-Za-z0-9]{24,}\")")

DOCKERFILE = open(os.path.join(ROOT, "Dockerfile"), encoding="utf8").read()


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    n = size = 0
    for item in MB.copy_sources():
        if item.startswith("deploy/"):
            continue
        src = os.path.join(ROOT, item.rstrip("/"))
        if not os.path.exists(src):
            sys.exit(f"missing: {item}")
        walk = os.walk(src) if os.path.isdir(src) else [(os.path.dirname(src), [], [os.path.basename(src)])]
        for dirpath, _, files in walk:
            for f in files:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, ROOT).replace("\\", "/")
                if (MB.NEVER.search(rel) and not rel.startswith("checkpoints/height_best/")) or EXTRA_NEVER.search(rel):
                    continue
                if rel.startswith("checkpoints/height_best/") and rel.endswith(".npz"):
                    continue
                dst = os.path.join(OUT, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(full, dst)
                n += 1
                size += os.path.getsize(full)
    # HF Dockerfile: same image, plus a writable /app for user 1000 and the public limits
    df = DOCKERFILE.replace('CMD python server.py --host 0.0.0.0 --port ${PORT:-8080}', '''# Hugging Face Spaces run the container as uid 1000: give it the app folder (uploads/, cache/,
# viewer/output/ and final_out/ are written at run time), a home and a writable HF cache.
RUN useradd -m -u 1000 user && mkdir -p /app/uploads /app/cache /app/final_out /app/viewer/output \
    && chown -R 1000:1000 /app
ENV HOME=/home/user HF_HOME=/tmp/hf XDG_CACHE_HOME=/tmp/cache \
    DW_MAX_UPLOAD_MB=95 DW_MAX_JOB_SECONDS=2700 DW_MAX_PX=4096
USER 1000
CMD python server.py --host 0.0.0.0 --port ${PORT:-8080}''')
    assert "USER 1000" in df
    open(os.path.join(OUT, "Dockerfile"), "w", encoding="utf8").write(df)
    shutil.copy2(os.path.join(ROOT, "deploy", "hf", "README_space.md"), os.path.join(OUT, "README.md"))
    # secret scan over every text file
    bad = []
    for dirpath, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(dirpath, f)
            if os.path.getsize(p) > 5_000_000:
                continue
            try:
                txt = open(p, encoding="utf8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if SECRET.search(txt):
                bad.append(os.path.relpath(p, OUT))
    if bad:
        sys.exit(f"SECRETS FOUND, not safe to push: {bad}")
    print(f"{n} files, {size / 1e9:.2f} GB -> {OUT}; secret scan clean")


if __name__ == "__main__":
    main()
