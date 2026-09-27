"""Kaggle GPU run 04: run 03 recipe + Swiss federal LiDAR tiles (tall buildings, tall forest, mountains), initialised from run 02.

Swiss tiles are built first (prepare_swiss.py, inside the session: nothing is uploaded
from home), then GAMUS fills the remaining disk.

GAMUS is 80 GB upstream, so it is pulled inside the Kaggle session rather than
uploaded from a home connection. Heights are rewritten as float16 on arrival:
metres to ~1 cm precision is far below the label noise, and it halves the disk
footprint so the full train split fits.

Everything except the checkpoint stays out of /kaggle/working, because Kaggle
saves that folder as the run's output.
"""
import concurrent.futures as cf
import os
import random
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import requests
from huggingface_hub import HfApi

T0 = time.time()
REPO = "earthflow/GAMUS"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
SESSION_H = 12.0
SAFETY_H = 0.6            # leave time to write outputs before Kaggle kills the session
HEADROOM_GB = 6.0         # free space to keep on the data disk
MB_PER_TILE = 5.4         # RGB 3.1 MB + float16 AGL 2.1 MB, plus slack


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def pick_data_dir():
    best, best_free = None, -1.0
    for d in ("/kaggle/tmp", "/kaggle/temp", "/tmp"):
        try:
            os.makedirs(d, exist_ok=True)
            f = free_gb(d)
            print(f"  {d}: {f:.1f} GB free")
            if f > best_free:
                best, best_free = d, f
        except OSError:
            pass
    return os.path.join(best, "gamus"), best_free


def find_code():
    for root, _, files in os.walk("/kaggle/input"):
        if "train_height.py" in files:
            return root
    sys.exit("train_height.py not found under /kaggle/input -- attach the code dataset")


def find_dfc():
    """Folder holding train/ and val/ with DFC_*_AGL.h5 (zipped dirs are extracted by Kaggle)."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs and "val" in dirs:
            t = os.path.join(root, "train")
            if any(f.startswith("DFC_") for f in os.listdir(t)):
                return root
    sys.exit("DFC tiles not found under /kaggle/input")


def find_init():
    for root, _, files in os.walk("/kaggle/input"):
        if "model.safetensors" in files and "config.json" in files:
            return root
    sys.exit("run02 checkpoint not found under /kaggle/input")


def list_split(api, split):
    """(stem, rgb_name) pairs for tiles that have both imagery and heights."""
    imgs = [os.path.basename(f.path) for f in
            api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"images/{split}")]
    hts = {os.path.basename(f.path) for f in
           api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"heights/{split}")}
    out = []
    for n in imgs:
        stem = n[:-len("_RGB.h5")]
        if f"{stem}_AGL.h5" in hts:
            out.append((stem, n))
    return out


SESSION = requests.Session()


def fetch(url, path, tries=6):
    for k in range(tries):
        try:
            with SESSION.get(url, timeout=120, stream=True) as r:
                if r.status_code == 429:
                    time.sleep(10 * (k + 1))
                    continue
                r.raise_for_status()
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                os.replace(tmp, path)
                return True
        except Exception as e:
            if k == tries - 1:
                print(f"    give up {os.path.basename(path)}: {e}", flush=True)
            time.sleep(2 * (k + 1))
    return False


def get_tile(split, stem, rgb_name, dest):
    rgb_p = os.path.join(dest, rgb_name)
    agl_p = os.path.join(dest, f"{stem}_AGL.h5")
    if os.path.exists(rgb_p) and os.path.exists(agl_p):
        return True
    if not fetch(f"{BASE}/images/{split}/{rgb_name}", rgb_p):
        return False
    raw = agl_p + ".f32"
    if not fetch(f"{BASE}/heights/{split}/{stem}_AGL.h5", raw):
        os.remove(rgb_p)
        return False
    with h5py.File(raw, "r") as h:
        a = h["image"][:].astype(np.float16)
    with h5py.File(agl_p, "w") as h:
        h.create_dataset("image", data=a)
    os.remove(raw)
    return True


def download(api, split, root, cap_tiles=None):
    tiles = list_split(api, split)
    # Interleave cities so a disk-capped download stays balanced across them.
    by_city = {}
    for t in tiles:
        by_city.setdefault(t[0].split("_")[0], []).append(t)
    for v in by_city.values():
        random.Random(1337).shuffle(v)
    order = []
    while any(by_city.values()):
        for c in sorted(by_city):
            if by_city[c]:
                order.append(by_city[c].pop())
    if cap_tiles is not None:
        order = order[:cap_tiles]
    counts = {}
    for s, _ in order:
        counts[s.split("_")[0]] = counts.get(s.split("_")[0], 0) + 1
    print(f"{split}: fetching {len(order)} of {len(tiles)} tiles {counts}", flush=True)

    dest = os.path.join(root, split)
    os.makedirs(dest, exist_ok=True)
    ok, t0 = 0, time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(get_tile, split, s, n, dest) for s, n in order]
        for i, f in enumerate(cf.as_completed(futs), 1):
            ok += bool(f.result())
            if i % 250 == 0 or i == len(futs):
                print(f"  {split} {i}/{len(futs)}  ok {ok}  {time.time()-t0:.0f}s  "
                      f"free {free_gb(dest):.1f} GB", flush=True)
    return ok


def main():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    print("python", sys.version.split()[0])
    import torch, transformers
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "transformers", transformers.__version__)

    code = find_code()
    print("code:", code, sorted(os.listdir(code)))
    data, free = pick_data_dir()
    print("data dir:", data)

    t_sw = time.time()
    sh(f"cd {code} && {sys.executable} -u prepare_swiss.py {data} --max-tiles 330")
    print(f"swiss tiles built in {(time.time()-t_sw)/60:.1f} min", flush=True)

    api = HfApi()
    n_val = download(api, "val", data)
    free = free_gb(os.path.dirname(data))
    cap = int(max(0.0, free - HEADROOM_GB) * 1000 / MB_PER_TILE)
    print(f"disk allows ~{cap} train tiles", flush=True)
    n_train = download(api, "train", data, cap_tiles=cap)
    print(f"data ready: {n_train} train, {n_val} val tiles, "
          f"{(time.time()-T0)/60:.1f} min elapsed", flush=True)

    budget = SESSION_H - SAFETY_H - (time.time() - T0) / 3600
    out = "/kaggle/working/ckpt"
    cmd = (f"cd {code} && {sys.executable} -u train_height.py "
           f"--data-root {data} --out {out} "
           f"--model depth-anything/Depth-Anything-V2-Small-hf "
           f"--init-from {find_init()} --extra-root {find_dfc()} --extra-crops-per-tile 16 "
           f"--epochs 6 --batch 8 --accum 1 --crops-per-tile 2 --workers 4 "
           f"--lr-head 1e-4 --lr-backbone 1e-5 --sensor-aug "
           f"--time-budget-h {budget:.2f}")
    sh(cmd)
    sh(f"ls -la {out} {out}/height_best 2>/dev/null")
    print(f"total {(time.time()-T0)/3600:.2f} h")


if __name__ == "__main__":
    main()
