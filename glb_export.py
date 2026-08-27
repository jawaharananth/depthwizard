"""
Minimal, spec-correct GLB (binary glTF 2.0) writer for the terrain + building
meshes.

Why not keep OBJ: OBJ is a text format -- at the vertex densities needed for
a crisp close-up terrain (500k+ verts, 1M+ faces) it produces ~80MB files
that the browser must parse line-by-line as text, which is both a slow load
and a large download. The same geometry as GLB is roughly 5x smaller and
parses as raw typed arrays. GLB also carries real vertex normals, which OBJ
export here never did (Three.js was left to guess them per-face, giving the
faceted look).

Deliberately supports only what this project emits (indexed triangles,
POSITION/NORMAL/TEXCOORD_0, one optional base-color texture per mesh) rather
than pulling in a general glTF dependency.
"""
import json
import struct
import numpy as np

# glTF componentType constants
FLOAT = 5126
UNSIGNED_INT = 5125

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def _pad4(b: bytes, pad_byte: bytes = b"\x00") -> bytes:
    remainder = len(b) % 4
    return b if remainder == 0 else b + pad_byte * (4 - remainder)


class _BufferBuilder:
    def __init__(self):
        self.chunks = []
        self.offset = 0
        self.buffer_views = []
        self.accessors = []

    def add_view(self, data: np.ndarray, target: int) -> int:
        raw = data.tobytes()
        # bufferView offsets must be 4-byte aligned
        pad = (4 - (self.offset % 4)) % 4
        if pad:
            self.chunks.append(b"\x00" * pad)
            self.offset += pad

        self.buffer_views.append({
            "buffer": 0, "byteOffset": self.offset, "byteLength": len(raw), "target": target,
        })
        self.chunks.append(raw)
        self.offset += len(raw)
        return len(self.buffer_views) - 1

    def add_accessor(self, data: np.ndarray, component_type: int, type_str: str, target: int,
                      include_bounds: bool = False) -> int:
        view_idx = self.add_view(data, target)
        accessor = {
            "bufferView": view_idx,
            "componentType": component_type,
            "count": int(data.shape[0]) if data.ndim > 1 else int(data.size),
            "type": type_str,
        }
        if include_bounds:
            # POSITION accessors are REQUIRED by the glTF spec to carry min/max
            accessor["min"] = [float(v) for v in data.min(axis=0)]
            accessor["max"] = [float(v) for v in data.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def data(self) -> bytes:
        return b"".join(self.chunks)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (the standard smooth-shading computation)."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    # cross product magnitude is proportional to triangle area, so accumulating
    # un-normalized face normals gives area weighting for free
    face_n = np.cross(v1 - v0, v2 - v0)

    np.add.at(normals, faces[:, 0], face_n)
    np.add.at(normals, faces[:, 1], face_n)
    np.add.at(normals, faces[:, 2], face_n)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(lengths, 1e-12)).astype(np.float32)


def export_glb(out_path: str,
               ground_verts: np.ndarray, ground_uvs: np.ndarray, ground_faces: np.ndarray,
               building_verts: np.ndarray, building_faces: np.ndarray,
               texture_bytes: bytes = None, texture_mime: str = "image/png",
               building_uvs: np.ndarray = None,
               building_color=(0.82, 0.78, 0.70, 1.0),
               building_colors: np.ndarray = None,
               extra_meshes: list = None):
    """
    extra_meshes: optional [(name, verts, faces, rgba), ...] emitted as further
    flat-coloured nodes. Used for scene elements that are neither terrain nor
    buildings -- tree canopy, water surfaces -- which need their own material
    and so cannot be merged into either existing mesh.
    """
    buf = _BufferBuilder()
    meshes, nodes, materials = [], [], []
    images, samplers, textures = [], [], []

    # ---- ground mesh (textured) ----
    ground_verts = np.ascontiguousarray(ground_verts, dtype=np.float32)
    ground_uvs = np.ascontiguousarray(ground_uvs, dtype=np.float32)
    ground_faces = np.ascontiguousarray(ground_faces, dtype=np.uint32)
    ground_normals = compute_vertex_normals(ground_verts, ground_faces)

    pos_acc = buf.add_accessor(ground_verts, FLOAT, "VEC3", ARRAY_BUFFER, include_bounds=True)
    nrm_acc = buf.add_accessor(ground_normals, FLOAT, "VEC3", ARRAY_BUFFER)
    uv_acc = buf.add_accessor(ground_uvs, FLOAT, "VEC2", ARRAY_BUFFER)
    idx_acc = buf.add_accessor(ground_faces.reshape(-1), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    ground_material = {
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.95},
        "name": "terrain",
    }
    if texture_bytes is not None:
        img_view = buf.add_view(np.frombuffer(texture_bytes, dtype=np.uint8), ARRAY_BUFFER)
        # image bufferViews must not declare a target
        buf.buffer_views[img_view].pop("target", None)
        images.append({"bufferView": img_view, "mimeType": texture_mime})
        samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071})
        textures.append({"sampler": 0, "source": 0})
        ground_material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    materials.append(ground_material)
    meshes.append({
        "primitives": [{
            "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc, "TEXCOORD_0": uv_acc},
            "indices": idx_acc, "material": 0, "mode": 4,
        }],
        "name": "ground",
    })
    nodes.append({"mesh": 0, "name": "ground"})

    # ---- buildings mesh (flat color) ----
    if building_verts is not None and len(building_verts) > 0:
        building_verts = np.ascontiguousarray(building_verts, dtype=np.float32)
        building_faces = np.ascontiguousarray(building_faces, dtype=np.uint32)
        building_normals = compute_vertex_normals(building_verts, building_faces)

        b_pos = buf.add_accessor(building_verts, FLOAT, "VEC3", ARRAY_BUFFER, include_bounds=True)
        b_nrm = buf.add_accessor(building_normals, FLOAT, "VEC3", ARRAY_BUFFER)
        b_idx = buf.add_accessor(building_faces.reshape(-1), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

        b_attrs = {"POSITION": b_pos, "NORMAL": b_nrm}
        b_pbr = {"metallicFactor": 0.0, "roughnessFactor": 0.8}
        # Per-vertex colour, used to give every building the colour of its own
        # roof as measured from the satellite image. glTF multiplies COLOR_0
        # into baseColorFactor, so the factor is left white and the attribute
        # carries the whole colour.
        if building_colors is not None and len(building_colors) == len(building_verts):
            b_col = buf.add_accessor(
                np.ascontiguousarray(building_colors, dtype=np.float32),
                FLOAT, "VEC3", ARRAY_BUFFER)
            b_attrs["COLOR_0"] = b_col
            b_pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        if building_uvs is not None and len(building_uvs) == len(building_verts) and textures:
            # share the ground's satellite texture so roofs show real imagery
            b_uv = buf.add_accessor(
                np.ascontiguousarray(building_uvs, dtype=np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
            b_attrs["TEXCOORD_0"] = b_uv
            b_pbr["baseColorTexture"] = {"index": 0}
        else:
            b_pbr["baseColorFactor"] = list(building_color)

        materials.append({
            "pbrMetallicRoughness": b_pbr,
            "doubleSided": True,  # footprint winding from OpenCV contours isn't guaranteed
            "name": "buildings",
        })
        meshes.append({
            "primitives": [{
                "attributes": b_attrs,
                "indices": b_idx, "material": len(materials) - 1, "mode": 4,
            }],
            "name": "buildings",
        })
        nodes.append({"mesh": len(meshes) - 1, "name": "buildings"})

    # ---- extra flat-coloured meshes (vegetation, water, ...) ----
    for name, ev, ef, rgba in (extra_meshes or []):
        if ev is None or len(ev) == 0 or ef is None or len(ef) == 0:
            continue
        ev = np.ascontiguousarray(ev, dtype=np.float32)
        ef = np.ascontiguousarray(ef, dtype=np.uint32)
        en = compute_vertex_normals(ev, ef)
        e_pos = buf.add_accessor(ev, FLOAT, "VEC3", ARRAY_BUFFER, include_bounds=True)
        e_nrm = buf.add_accessor(en, FLOAT, "VEC3", ARRAY_BUFFER)
        e_idx = buf.add_accessor(ef.reshape(-1), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorFactor": list(rgba),
                "metallicFactor": 0.0, "roughnessFactor": 0.9,
            },
            "doubleSided": True,
            "name": name,
        })
        meshes.append({
            "primitives": [{
                "attributes": {"POSITION": e_pos, "NORMAL": e_nrm},
                "indices": e_idx, "material": len(materials) - 1, "mode": 4,
            }],
            "name": name,
        })
        nodes.append({"mesh": len(meshes) - 1, "name": name})

    bin_data = buf.data()
    gltf = {
        "asset": {"version": "2.0", "generator": "DepthWizard"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": buf.accessors,
        "bufferViews": buf.buffer_views,
        "buffers": [{"byteLength": len(bin_data)}],
    }
    if images:
        gltf["images"] = images
        gltf["samplers"] = samplers
        gltf["textures"] = textures

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = _pad4(bin_data)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))          # 'glTF', version 2, total length
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))    # JSON chunk header
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))     # BIN chunk header
        f.write(bin_bytes)

    return out_path
