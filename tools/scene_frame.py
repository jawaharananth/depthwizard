"""
Shared scene-frame helpers for the detail-layer tools (tree_crowns.py,
roof_details.py). Read-only: nothing here writes into a scene.

VIEWER WORLD FRAME (the frame terrain.glb / buildings / structures.glb use)
---------------------------------------------------------------------------
  Units   metres, Y up, right-handed (three.js).
  x       = col * gsd                     (east;   col = 4096-grid pixel coord)
  z       = -row * gsd                    (north;  row = 4096-grid pixel coord)
  y       = DSM_absolute - datum_m        (the scene is built with ~0 at its low point)

  with (col, row) CONTINUOUS pixel coordinates of the scene's working grid
  (scene.json "resolution", 4096^2): pixel (i, j) covers [i, i+1) so its
  centre is at i + 0.5. Equivalently, in the source CRS (scene.json "crs",
  EPSG:3857, "affine_transform" = [gsd, 0, left, 0, -gsd, top]):
        x = X_crs - left            z = Y_crs - top
  i.e. world metres are MAP-UNIT metres (Web Mercator scale is not corrected
  for these GT-grid builds -- same convention as build_prisms/context).
  Texture UV (same convention as city_model.build_prisms):
        u = col / (W - 1)           v = 1 - row / (H - 1)
  datum_m: the DSM GeoTIFF's VERTICAL_DATUM tag offset when it states one
  (hitec: +576.88 m); otherwise measured as the mode of DSM-ground minus
  terrain.glb ground-vertex y (periyar: Tier A DEM datum, ~859.14 m).
"""
import json
import os
import re
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_glb_node(path, node_name, attrs=("POSITION",)):
    """Return {attr: ndarray} for one named node of a GLB (float accessors)."""
    with open(path, "rb") as f:
        f.read(12)
        jl, _ = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(jl))
        f.read(8)
        off0 = f.tell()
        for n in j["nodes"]:
            if n.get("name") != node_name:
                continue
            prim = j["meshes"][n["mesh"]]["primitives"][0]["attributes"]
            out = {}
            for k in attrs:
                if k not in prim:
                    continue
                a = j["accessors"][prim[k]]
                bv = j["bufferViews"][a["bufferView"]]
                nc = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[a["type"]]
                f.seek(off0 + bv.get("byteOffset", 0) + a.get("byteOffset", 0))
                out[k] = np.frombuffer(f.read(a["count"] * nc * 4), np.float32).reshape(-1, nc)
            return out
    return None


class SceneFrame:
    def __init__(self, scene_dir):
        self.dir = scene_dir if os.path.isabs(scene_dir) else os.path.join(ROOT, scene_dir)
        self.meta = json.load(open(os.path.join(self.dir, "scene.json")))
        tf = self.meta["affine_transform"]
        self.gsd = float(tf[0])
        self.left, self.top = float(tf[2]), float(tf[5])
        self.W, self.H = [int(v) for v in self.meta["resolution"]]
        dsm = [f for f in os.listdir(self.dir) if f.endswith("_DSM.tif")]
        self.dsm_path = os.path.join(self.dir, dsm[0]) if dsm else None
        self.texture_path = os.path.join(self.dir, self.meta.get("texture_file") or "terrain_texture.png")
        self.terrain_glb = os.path.join(self.dir, "terrain.glb")
        self._datum = None
        self._ground_grid = None

    # ---- coordinates -------------------------------------------------------
    def world_xz(self, col, row):
        return np.asarray(col) * self.gsd, -np.asarray(row) * self.gsd

    def px_from_crs(self, X, Y):
        return (np.asarray(X) - self.left) / self.gsd, (self.top - np.asarray(Y)) / self.gsd

    def uv(self, col, row):
        return np.asarray(col) / (self.W - 1), 1.0 - np.asarray(row) / (self.H - 1)

    # ---- rasters -----------------------------------------------------------
    def read_dsm(self):
        import rasterio
        with rasterio.open(self.dsm_path) as r:
            a = r.read(1).astype(np.float32)
            nd = r.nodata
            self.dsm_tags = r.tags()
        if nd is not None:
            a[a == nd] = np.nan
        return a

    def read_texture(self, size=None):
        """Scene texture as uint8 RGB; area-downsampled to `size` if given."""
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(self.texture_path).convert("RGB")
        if size is not None and im.size != (size, size):
            im = im.reduce(im.size[0] // size) if im.size[0] % size == 0 else im.resize((size, size), Image.BOX)
        return np.asarray(im)

    # ---- vertical datum ----------------------------------------------------
    def ground_grid(self):
        """terrain.glb ground node as a (n, n) y-grid + its x/z spacing."""
        if self._ground_grid is None:
            p = read_glb_node(self.terrain_glb, "ground")["POSITION"]
            n = int(round(np.sqrt(len(p))))
            g = p.reshape(n, n, 3)
            assert np.allclose(g[0, :, 2], g[0, 0, 2]) and g[0, 1, 0] > g[0, 0, 0], "ground not row-major grid"
            self._ground_grid = (g[:, :, 1].copy(), float(g[0, 1, 0] - g[0, 0, 0]),
                                 float(g[0, 0, 2] - g[1, 0, 2]))
        return self._ground_grid

    def terrain_y(self, x, z):
        """Bilinear sample of the rendered terrain surface (terrain.glb ground)."""
        yg, dx, dz = self.ground_grid()
        n = yg.shape[0]
        fx = np.clip(np.asarray(x) / dx, 0, n - 1.001)
        fz = np.clip(-np.asarray(z) / dz, 0, n - 1.001)
        i0, j0 = fz.astype(int), fx.astype(int)
        tz, tx = fz - i0, fx - j0
        return ((yg[i0, j0] * (1 - tx) + yg[i0, j0 + 1] * tx) * (1 - tz)
                + (yg[i0 + 1, j0] * (1 - tx) + yg[i0 + 1, j0 + 1] * tx) * tz)

    def datum(self, ground_abs=None):
        """Metres to subtract from absolute DSM elevation to get viewer y."""
        if self._datum is not None:
            return self._datum
        import rasterio
        with rasterio.open(self.dsm_path) as r:
            tag = r.tags().get("VERTICAL_DATUM", "")
        m = re.search(r"offset\s*([+-]?\d+(?:\.\d+)?)\s*m", tag)
        if m:
            self._datum = (float(m.group(1)), "DSM tag VERTICAL_DATUM: " + tag)
            return self._datum
        # Measured: bare-ground (pipeline ground) elevation vs rendered ground y.
        yg, dx, dz = self.ground_grid()
        n = yg.shape[0]
        ii = np.arange(n)
        cols = np.clip(((ii + 0.5) * dx / self.gsd).astype(int), 0, self.W - 1)
        rows = np.clip(((ii + 0.5) * dz / self.gsd).astype(int), 0, self.H - 1)
        src = ground_abs if ground_abs is not None else self.read_dsm()
        d = (src[rows[:, None], cols[None, :]] - yg).ravel()
        d = d[np.isfinite(d)]
        h, e = np.histogram(d, bins=np.arange(np.percentile(d, 0.5), np.percentile(d, 99.5), 0.02))
        k = int(h.argmax())
        sel = d[np.abs(d - e[k]) < 0.3]
        self._datum = (float(np.median(sel)), "measured: mode of (ground elevation - terrain.glb y), "
                       f"{len(sel) / len(d):.0%} of ground vertices within 0.3 m")
        return self._datum
