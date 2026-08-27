"""
Loader for DFC2019 Track 3 ground-truth tiles (DSM + CLS + coordinate TXT).

Official format (github.com/pubgeo/dfc2019, data/README.md):
  DSM: float32, ground-truth WGS84 Z coordinate (meters) -- real elevation,
       not relative. Negative values in Jacksonville are legitimate: Florida's
       geoid undulation is ~-25m, so ~0-5m MSL ground maps to ~-23m WGS84 ellipsoidal height.
  CLS: uint8 classification, LAS-spec codes:
       2=Ground, 5=Trees, 6=Buildings, 9=Water, 17=Bridge/elevated road, 65=Unlabeled
  TXT: 4 lines -- UTM easting (top-left), UTM northing (top-left), size_px, gsd_m
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

import segmentation as seg
import terrain_classify

# DFC2019 LAS-spec class codes -> our 5-class segmentation scheme
CLS_TO_SEG = {
    6: seg.CLASS_IDX["building"],
    17: seg.CLASS_IDX["road"],
    9: seg.CLASS_IDX["water"],
    5: seg.CLASS_IDX["vegetation"],
    2: seg.CLASS_IDX["bare_earth"],
    # 65 (Unlabeled) intentionally has no mapping -- callers must exclude it via the valid mask
}
CLS_UNLABELED = 65

JACKSONVILLE_UTM_EPSG = "EPSG:32617"  # UTM zone 17N
OMAHA_UTM_EPSG = "EPSG:32615"          # UTM zone 15N


def parse_dsm_txt(txt_path: str) -> dict:
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    utm_x, utm_y, size_px, gsd_m = float(lines[0]), float(lines[1]), float(lines[2]), float(lines[3])
    return {"utm_x": utm_x, "utm_y": utm_y, "size_px": size_px, "gsd_m": gsd_m}


def load_tile(tile_prefix: str, truth_dir: str) -> dict:
    """
    tile_prefix e.g. 'JAX_004' -> reads {truth_dir}/JAX_004_DSM.tif, _CLS.tif, _DSM.txt.
    Returns dsm (meters), cls (raw LAS codes), seg_labels (our scheme, unlabeled=-1),
    terrain_mask (urban/hilly/forest/sparse via real metric slope), gsd_m, crs, transform.
    """
    import os
    dsm_path = os.path.join(truth_dir, f"{tile_prefix}_DSM.tif")
    cls_path = os.path.join(truth_dir, f"{tile_prefix}_CLS.tif")
    txt_path = os.path.join(truth_dir, f"{tile_prefix}_DSM.txt")

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
    with rasterio.open(cls_path) as src:
        cls = src.read(1)

    coords = parse_dsm_txt(txt_path)
    gsd = coords["gsd_m"]
    epsg = JACKSONVILLE_UTM_EPSG if tile_prefix.startswith("JAX") else OMAHA_UTM_EPSG
    transform = from_origin(coords["utm_x"], coords["utm_y"], gsd, gsd)

    seg_labels = np.full(cls.shape, -1, dtype=np.int32)
    for las_code, seg_idx in CLS_TO_SEG.items():
        seg_labels[cls == las_code] = seg_idx
    valid_mask = cls != CLS_UNLABELED

    # Real metric slope for hilly detection -- exactly the terrain_classify path
    # already verified against synthetic flat/steep ground truth.
    seg_labels_filled = np.where(seg_labels == -1, seg.CLASS_IDX["bare_earth"], seg_labels)
    terrain_mask, terrain_stats = terrain_classify.classify_terrain(
        seg_labels_filled, dem_meters=dsm, gsd_x_m=gsd, gsd_y_m=gsd)

    return {
        "dsm": dsm, "cls": cls, "seg_labels": seg_labels, "valid_mask": valid_mask,
        "terrain_mask": terrain_mask, "terrain_stats": terrain_stats,
        "gsd_m": gsd, "crs": epsg, "transform": transform,
        "utm_x": coords["utm_x"], "utm_y": coords["utm_y"],
    }


def parse_imd(imd_path: str) -> dict:
    """Pull sun angle + GSD out of a real WorldView IMD metadata file."""
    fields = {}
    with open(imd_path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"')
            if key in ("meanSunAz", "meanSunEl", "meanProductGSD", "meanOffNadirViewAngle", "meanSatEl"):
                try:
                    fields[key] = float(val)
                except ValueError:
                    pass
    return fields


def imd_path_for_rgb(rgb_path: str, metadata_dir: str) -> str:
    """JAX_004_006_RGB.tif -> source image number 006 -> {metadata_dir}/JAX/06.IMD"""
    import os
    fname = os.path.basename(rgb_path)
    parts = fname.split("_")
    location, image_num = parts[0], parts[2]
    return os.path.join(metadata_dir, location, f"{int(image_num):02d}.IMD")


def list_available_tiles(truth_dir: str) -> list:
    import os, re
    tiles = set()
    for fname in os.listdir(truth_dir):
        m = re.match(r"([A-Z]{3}_\d+)_DSM\.tif$", fname)
        if m:
            tiles.add(m.group(1))
    return sorted(tiles)


if __name__ == "__main__":
    import sys
    truth_dir = sys.argv[1] if len(sys.argv) > 1 else "dfc2019_data/truth/Track3-Truth"
    tiles = list_available_tiles(truth_dir)
    print(f"Found {len(tiles)} tiles:", tiles[:10], "..." if len(tiles) > 10 else "")

    t = load_tile(tiles[0], truth_dir)
    print(f"\nTile {tiles[0]}:")
    print("  DSM shape:", t["dsm"].shape, "range:", t["dsm"].min(), t["dsm"].max())
    print("  gsd_m:", t["gsd_m"], "crs:", t["crs"])
    print("  valid (labeled) fraction:", t["valid_mask"].mean())
    print("  terrain_stats:", t["terrain_stats"])
