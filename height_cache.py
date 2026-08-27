"""
Disk cache for the height field.

Tiled inference runs the backbone 25 times over one tile -- around five
minutes on CPU -- and it is deterministic for a given image and mode. Without
a cache every re-verification pays that cost again, which in practice means
verification gets run less often than it should.

The key includes the tile, the inference mode and the raster size, so a
different ortho resolution or a switch between whole-image and tiled
inference cannot silently reuse the wrong field.
"""
import os
import numpy as np

CACHE_DIR = "cache"


def path_for(tile: str, mode: str, size: int) -> str:
    return os.path.join(CACHE_DIR, f"{tile}_{mode}_{size}.npy")


def load(tile: str, mode: str, size: int):
    p = path_for(tile, mode, size)
    if os.path.exists(p):
        arr = np.load(p)
        if arr.shape == (size, size):
            return arr
    return None


def save(tile: str, mode: str, size: int, arr: np.ndarray) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path_for(tile, mode, size), arr.astype(np.float32))
