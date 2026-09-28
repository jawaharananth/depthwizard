"""Test inputs a judge could plausibly upload, cut from imagery we hold.

    python scripts/make_upload_matrix.py OUT_DIR

Every case is a real scene (HITEC City, Esri World Imagery 0.6 m) written the
way a different product or tool would write it: Cartosat-style 4-band 16-bit
with a black collar, single-band panchromatic, UTM, geographic lat/lon,
float reflectance, compressed/tiled, no CRS at all, PNG with and without
transparency, grayscale PNG, JPEG, a small crop and a large one.
"""
import os
import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "hitec2x2.tif")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

with rasterio.open(SRC) as s:
    prof = s.profile
    T0, crs = s.transform, s.crs
    full = s.read([1, 2, 3])                     # uint8 RGB, EPSG:3857


def win(x, y, w, h):
    return full[:, y:y + h, x:x + w], T0 * Affine.translation(x, y)


def write(name, arr, tf, crs_, **kw):
    p = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
             dtype=arr.dtype, transform=tf, crs=crs_)
    p.update(kw)
    with rasterio.open(os.path.join(out, name), "w", **p) as d:
        d.write(arr)
        if kw.pop("colorinterp_bgrn", False):
            pass


rgb, tf = win(1200, 1500, 1800, 1600)            # ~1.1 x 1.0 km, 2.9 MP

# 1. plain 3-band 8-bit GeoTIFF, Web Mercator
write("g1_rgb8_3857.tif", rgb, tf, crs)

# 2. Cartosat-style: 4-band 16-bit B,G,R,NIR (untagged), 10-bit DN range, black collar
dn = (rgb.astype(np.float32) * 3.9 + 40).astype(np.uint16)
nir = (dn[1] * 0.6 + dn[0] * 0.5).astype(np.uint16)
bgrn = np.stack([dn[2], dn[1], dn[0], nir])
bgrn[:, :90, :] = 0; bgrn[:, :, -140:] = 0
write("g2_cartosat_bgrn16.tif", bgrn, tf, crs)

# 3. single-band panchromatic 16-bit (Cartosat-2S PAN is one band)
pan = (0.299 * dn[0] + 0.587 * dn[1] + 0.114 * dn[2]).astype(np.uint16)[None]
write("g3_pan16.tif", pan, tf, crs)

# 4. UTM zone 44N, LZW-compressed and tiled
dst_crs = "EPSG:32644"
t2, w2, h2 = calculate_default_transform(crs, dst_crs, rgb.shape[2], rgb.shape[1],
                                         *rasterio.transform.array_bounds(rgb.shape[1], rgb.shape[2], tf))
utm = np.zeros((3, h2, w2), np.uint8)
for b in range(3):
    reproject(rgb[b], utm[b], src_transform=tf, src_crs=crs, dst_transform=t2, dst_crs=dst_crs,
              resampling=Resampling.bilinear)
write("g4_utm44n_lzw_tiled.tif", utm, t2, dst_crs, compress="lzw", tiled=True, blockxsize=256, blockysize=256)

# 5. geographic lat/lon (EPSG:4326)
t3, w3, h3 = calculate_default_transform(crs, "EPSG:4326", rgb.shape[2], rgb.shape[1],
                                         *rasterio.transform.array_bounds(rgb.shape[1], rgb.shape[2], tf))
geo = np.zeros((3, h3, w3), np.uint8)
for b in range(3):
    reproject(rgb[b], geo[b], src_transform=tf, src_crs=crs, dst_transform=t3, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear)
write("g5_wgs84_latlon.tif", geo, t3, "EPSG:4326")

# 6. float32 reflectance 0..1 with a nodata value
refl = (rgb.astype(np.float32) / 255.0) * 0.6 + 0.02
refl[:, -120:, :] = -9999.0
write("g6_float_reflectance.tif", refl, tf, crs, nodata=-9999.0)

# 7. a TIFF with no georeference at all
with rasterio.open(os.path.join(out, "g7_tiff_no_crs.tif"), "w", driver="GTiff", width=rgb.shape[2],
                   height=rgb.shape[1], count=3, dtype="uint8") as d:
    d.write(rgb)

img = Image.fromarray(np.transpose(rgb, (1, 2, 0)))
# 8. PNG, 9. PNG with transparency, 10. grayscale PNG, 11. JPEG
img.save(os.path.join(out, "p8_rgb.png"))
rgba = img.convert("RGBA"); a = np.array(rgba); a[:, :150, 3] = 0; Image.fromarray(a).save(os.path.join(out, "p9_rgba_transparent.png"))
img.convert("L").save(os.path.join(out, "p10_gray.png"))
img.save(os.path.join(out, "j11_rgb.jpg"), quality=88)
# 12. small crop, 13. large (the whole 4096 px scene, 16.8 MP)
Image.fromarray(np.transpose(full[:, 2000:2600, 1600:2300], (1, 2, 0))).save(os.path.join(out, "p12_small_700x600.png"))
write("g13_large_4096.tif", full, T0, crs, compress="deflate", tiled=True)
for f in sorted(os.listdir(out)):
    print(f"{f:28s} {os.path.getsize(os.path.join(out, f)) / 1e6:6.1f} MB")
