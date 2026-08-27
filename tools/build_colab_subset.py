"""
Packages the minimal DFC2019 subset needed for GPU work on Colab.

The full Track 3 RGB set is 8.17 GB, which is painful to move through Drive.
Only one source image per tile is needed for segmentation benchmarking and
for LoRA supervision, which brings the upload to roughly 0.6 GB:

    53 tiles with matched RGB + DSM + CLS
    440 MB  one RGB image per tile
    144 MB  all truth (DSM elevation + CLS labels + coordinate TXT)

Run:  python tools/build_colab_subset.py
Out:  colab_subset/  and  colab_subset.zip
"""
import os
import glob
import shutil
import collections
import zipfile

TRUTH_DIR = "dfc2019_data/truth/Track3-Truth"
RGB_DIR = "dfc2019_data/rgb/Track3-RGB-1"
META_DIR = "dfc2019_data/metadata/Track3-Metadata"
OUT_DIR = "colab_subset"


def build(images_per_tile: int = 1):
    truth_tiles = sorted({os.path.basename(p)[:7]
                          for p in glob.glob(os.path.join(TRUTH_DIR, "*_DSM.tif"))})

    by_tile = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RGB_DIR, "*_RGB.tif"))):
        by_tile[os.path.basename(p)[:7]].append(p)

    paired = [t for t in truth_tiles if t in by_tile]
    print(f"{len(truth_tiles)} truth tiles, {len(paired)} have matching RGB")

    for sub in ("rgb", "truth", "metadata"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    total = 0
    for tile in paired:
        for src in by_tile[tile][:images_per_tile]:
            dst = os.path.join(OUT_DIR, "rgb", os.path.basename(src))
            shutil.copy2(src, dst)
            total += os.path.getsize(dst)

        for suffix in ("_DSM.tif", "_CLS.tif", "_DSM.txt"):
            src = os.path.join(TRUTH_DIR, tile + suffix)
            if os.path.exists(src):
                dst = os.path.join(OUT_DIR, "truth", tile + suffix)
                shutil.copy2(src, dst)
                total += os.path.getsize(dst)

    # metadata is tiny and carries sun angle + GSD, which the shadow and
    # calibration paths both need
    if os.path.isdir(META_DIR):
        for root, _, files in os.walk(META_DIR):
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(OUT_DIR, "metadata", f)
                shutil.copy2(src, dst)
                total += os.path.getsize(dst)

    print(f"staged {total/1e9:.2f} GB into {OUT_DIR}/")

    zip_path = OUT_DIR + ".zip"
    print(f"zipping -> {zip_path} (stored, not deflated: TIFFs are already compressed)")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, OUT_DIR))

    print(f"done: {os.path.getsize(zip_path)/1e9:.2f} GB")
    print(f"tiles: {len(paired)}  images/tile: {images_per_tile}")
    return zip_path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    build(images_per_tile=n)
