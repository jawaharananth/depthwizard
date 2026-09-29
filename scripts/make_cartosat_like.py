"""Cartosat-style and awkward inputs not covered by make_upload_matrix.py.

    python scripts/make_cartosat_like.py OUT_DIR

Cut from the same real scene (HITEC City, EPSG:3857, 0.6 m). Each file is a
way ISRO's hidden set, or a tool a judge might use, could write an image:
11/12-bit counts in uint16 with tagged band order, BGR band order, India's
own projection (EPSG:7755), a rotated grid, a CRS with no geotransform, a
nodata tag that is not 0, a 8k BigTIFF with JPEG compression, and JPEG/PNG
variants (EXIF-rotated, CMYK, greyscale, 16-bit PNG, a WebP named .jpg).
A manifest (manifest.json) lists each case and what a correct result is.
"""
import json
import os
import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "hitec2x2.tif")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
manifest = {}

with rasterio.open(SRC) as s:
    T0, crs = s.transform, s.crs
    full = s.read([1, 2, 3])                     # uint8 RGB

X, Y, W, H = 1200, 1400, 1800, 1600
rgb = full[:, Y:Y + H, X:X + W]
tf = T0 * Affine.translation(X, Y)
nir = np.clip(rgb[1].astype(int) * 1.4 - rgb[0] * 0.3 + 20, 0, 255).astype(np.uint8)   # plausible NIR


def write(name, arr, tf_, crs_, expect, ci=None, desc=None, **kw):
    p = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
             dtype=arr.dtype, transform=tf_, crs=crs_)
    p.update(kw)
    with rasterio.open(os.path.join(out, name), "w", **p) as d:
        d.write(arr)
        if ci:
            d.colorinterp = ci
        if desc:
            for i, t in enumerate(desc):
                d.set_band_description(i + 1, t)
    manifest[name] = expect


def counts(a, bits):
    return (a.astype(np.float32) / 255 * (2 ** bits - 1)).astype(np.uint16)


# 11-bit (Cartosat-2S MX quantisation) B,G,R,NIR in uint16, band order tagged
bgrn = np.stack([rgb[2], rgb[1], rgb[0], nir])
write("c1_bgrn_11bit_tagged.tif", counts(bgrn, 11), tf, crs, "DSM",
      ci=[ColorInterp.blue, ColorInterp.green, ColorInterp.red, ColorInterp.undefined],
      desc=["blue", "green", "red", "nir"])
# 12-bit RGB in uint16, untagged, 3 bands
write("c2_rgb_12bit.tif", counts(rgb, 12), tf, crs, "DSM")
# 3-band BGR, tagged
write("c3_bgr_tagged.tif", rgb[::-1].copy(), tf, crs, "DSM",
      ci=[ColorInterp.blue, ColorInterp.green, ColorInterp.red])
# nodata tag 255 with a white collar
col = rgb.copy(); col[:, :, :120] = 255; col[:, :90, :] = 255
write("c4_nodata255_collar.tif", col, tf, crs, "DSM", nodata=255)

# India's own projection: EPSG:7755 (WGS 84 / India NSF LCC)
for name, dst in (("c5_epsg7755_india_lcc.tif", "EPSG:7755"),):
    t2, w2, h2 = calculate_default_transform(crs, dst, W, H, *rasterio.transform.array_bounds(H, W, tf))
    arr = np.zeros((3, h2, w2), np.uint8)
    for i in range(3):
        reproject(rgb[i], arr[i], src_transform=tf, src_crs=crs, dst_transform=t2, dst_crs=dst,
                  resampling=Resampling.bilinear)
    write(name, arr, t2, dst, "DSM", nodata=0)

# rotated geotransform (15 degrees), same pixels
rot = tf * Affine.rotation(15)
write("c6_rotated_15deg.tif", rgb, rot, crs, "DSM or clear refusal")
# a CRS but no geotransform (identity)
write("c7_crs_no_transform.tif", rgb, Affine.identity(), crs, "rDSM or clear refusal")

# 8k BigTIFF, JPEG-compressed, tiled (cut from the full scene)
big = full[:, :, :]
if big.shape[1] >= 4096:
    big = np.concatenate([big, big[:, ::-1]], axis=1)[:, :8192, :]
    big = np.concatenate([big, big[:, :, ::-1]], axis=2)[:, :, :8192]
write("c8_8k_bigtiff_jpeg.tif", np.ascontiguousarray(big), T0, crs, "DSM or size refusal",
      compress="jpeg", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES", photometric="YCBCR")

img = Image.fromarray(np.transpose(rgb, (1, 2, 0)))
# JPEG stored sideways with an EXIF orientation tag (camera/phone export)
ex = Image.Exif(); ex[0x0112] = 6
img.transpose(Image.ROTATE_90).save(os.path.join(out, "c9_exif_rotated.jpg"), quality=90, exif=ex)
manifest["c9_exif_rotated.jpg"] = "rDSM, upright"
img.convert("CMYK").save(os.path.join(out, "c10_cmyk.jpg"), quality=90)
manifest["c10_cmyk.jpg"] = "rDSM"
img.convert("L").save(os.path.join(out, "c11_gray.jpg"), quality=90)
manifest["c11_gray.jpg"] = "rDSM"
Image.fromarray((np.transpose(rgb, (1, 2, 0)).astype(np.uint16) * 257)).save(
    os.path.join(out, "c12_rgb16.png")) if False else None
# 16-bit PNG (PIL writes 16-bit single channel only; greyscale 16-bit is the common case)
Image.fromarray((rgb.mean(0) * 257).astype(np.uint16)).save(os.path.join(out, "c12_gray16.png"))
manifest["c12_gray16.png"] = "rDSM"
img.save(os.path.join(out, "c13_webp_named.jpg"), format="WEBP", quality=90)
manifest["c13_webp_named.jpg"] = "rDSM or clear refusal"

json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(f"{len(manifest)} cases in {out}")
