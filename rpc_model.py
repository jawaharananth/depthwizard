"""
RPC00B rational polynomial camera model (standard DigitalGlobe/WorldView RPB
format) -- forward projection (object space -> image) and inverse
localization (image + height -> object space), both needed for multi-view
triangulation.

Forward projection is a direct polynomial evaluation. Inverse localization
has no closed form (this is true of every real RPC implementation, including
GDAL's own RPC transformer) -- solved here by 2D Newton-Raphson using a
numerically estimated Jacobian, which is what GDAL does internally too.
"""
import re
import numpy as np


class RPCModel:
    def __init__(self, params: dict):
        self.line_off, self.samp_off = params["lineOffset"], params["sampOffset"]
        self.line_scale, self.samp_scale = params["lineScale"], params["sampScale"]
        self.lat_off, self.lon_off, self.height_off = params["latOffset"], params["longOffset"], params["heightOffset"]
        self.lat_scale, self.lon_scale, self.height_scale = params["latScale"], params["longScale"], params["heightScale"]
        self.line_num = np.array(params["lineNumCoef"])
        self.line_den = np.array(params["lineDenCoef"])
        self.samp_num = np.array(params["sampNumCoef"])
        self.samp_den = np.array(params["sampDenCoef"])

    @staticmethod
    def _poly_terms(x, y, z):
        # standard 20-term RPC00B monomial basis, x=lon(P), y=lat(L), z=height(H)
        return np.stack([
            np.ones_like(x), x, y, z, x * y, x * z, y * z, x ** 2, y ** 2, z ** 2,
            x * y * z, x ** 3, x * y ** 2, x * z ** 2, x ** 2 * y, y ** 3, y * z ** 2,
            x ** 2 * z, y ** 2 * z, z ** 3,
        ], axis=-1)

    def project(self, lat, lon, height):
        """object space (deg, deg, m) -> image (line, sample), vectorized over arrays."""
        lat, lon, height = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64), np.asarray(height, dtype=np.float64)
        x = (lon - self.lon_off) / self.lon_scale
        y = (lat - self.lat_off) / self.lat_scale
        z = (height - self.height_off) / self.height_scale

        terms = self._poly_terms(x, y, z)
        line_n = terms @ self.line_num
        line_d = terms @ self.line_den
        samp_n = terms @ self.samp_num
        samp_d = terms @ self.samp_den

        line = (line_n / line_d) * self.line_scale + self.line_off
        samp = (samp_n / samp_d) * self.samp_scale + self.samp_off
        return line, samp

    def localize(self, line, sample, height, max_iter=15, tol=1e-8):
        """image (line, sample) + height -> (lat, lon) via Newton-Raphson. Vectorized."""
        line, sample, height = np.asarray(line, dtype=np.float64), np.asarray(sample, dtype=np.float64), np.asarray(height, dtype=np.float64)
        lat = np.full_like(line, self.lat_off, dtype=np.float64)
        lon = np.full_like(line, self.lon_off, dtype=np.float64)

        eps_deg = 1e-6
        for _ in range(max_iter):
            l0, s0 = self.project(lat, lon, height)
            res_l, res_s = line - l0, sample - s0
            if np.all(np.abs(res_l) < tol) and np.all(np.abs(res_s) < tol):
                break

            l_dlat, s_dlat = self.project(lat + eps_deg, lon, height)
            l_dlon, s_dlon = self.project(lat, lon + eps_deg, height)
            J11, J21 = (l_dlat - l0) / eps_deg, (s_dlat - s0) / eps_deg  # d(line,samp)/d(lat)
            J12, J22 = (l_dlon - l0) / eps_deg, (s_dlon - s0) / eps_deg  # d(line,samp)/d(lon)

            det = J11 * J22 - J12 * J21
            det = np.where(np.abs(det) < 1e-20, 1e-20, det)
            dlat = (J22 * res_l - J12 * res_s) / det
            dlon = (J11 * res_s - J21 * res_l) / det
            lat, lon = lat + dlat, lon + dlon

        return lat, lon


def from_rasterio_rpc(rpc) -> RPCModel:
    """
    Build from rasterio's parsed RPC tags (src.rpcs), which GDAL reads
    directly from the NITF/TIFF metadata embedded in each image file --
    crucially, these are already CROP-adjusted for this specific tile crop.
    The standalone .RPB files in Track3-Metadata describe the full,
    uncropped satellite scene (tens of thousands of pixels) and give
    wildly out-of-bounds coordinates if used directly on a 2048x2048 crop.
    """
    params = {
        "lineOffset": rpc.line_off, "sampOffset": rpc.samp_off,
        "lineScale": rpc.line_scale, "sampScale": rpc.samp_scale,
        "latOffset": rpc.lat_off, "longOffset": rpc.long_off, "heightOffset": rpc.height_off,
        "latScale": rpc.lat_scale, "longScale": rpc.long_scale, "heightScale": rpc.height_scale,
        "lineNumCoef": rpc.line_num_coeff, "lineDenCoef": rpc.line_den_coeff,
        "sampNumCoef": rpc.samp_num_coeff, "sampDenCoef": rpc.samp_den_coeff,
    }
    return RPCModel(params)


def parse_rpb(rpb_path: str) -> RPCModel:
    with open(rpb_path) as f:
        text = f.read()

    scalar_fields = ["lineOffset", "sampOffset", "latOffset", "longOffset", "heightOffset",
                      "lineScale", "sampScale", "latScale", "longScale", "heightScale"]
    params = {}
    for field in scalar_fields:
        m = re.search(rf"{field}\s*=\s*([-\d.Ee+]+)", text)
        params[field] = float(m.group(1))

    coef_fields = ["lineNumCoef", "lineDenCoef", "sampNumCoef", "sampDenCoef"]
    for field in coef_fields:
        m = re.search(rf"{field}\s*=\s*\(([^)]+)\)", text, re.S)
        values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+[Ee][-+]?\d+|[-+]?\d*\.\d+|[-+]?\d+", m.group(1))]
        if len(values) != 20:
            raise ValueError(f"{field} in {rpb_path}: expected 20 coefficients, got {len(values)}")
        params[field] = values

    return RPCModel(params)


if __name__ == "__main__":
    import sys
    rpb_path = sys.argv[1] if len(sys.argv) > 1 else "dfc2019_data/metadata/Track3-Metadata/JAX/06.RPB"
    rpc = parse_rpb(rpb_path)

    # Self-consistency check: project a grid of (lat,lon,height) points, localize
    # back, verify recovery within sub-pixel/sub-cm tolerance. This is the only
    # honest way to know the RPC math is actually correct before trusting any
    # triangulation built on top of it.
    rng = np.random.RandomState(0)
    n = 200
    lat = rpc.lat_off + rng.uniform(-1, 1, n) * rpc.lat_scale * 0.8
    lon = rpc.lon_off + rng.uniform(-1, 1, n) * rpc.lon_scale * 0.8
    height = rpc.height_off + rng.uniform(-1, 1, n) * rpc.height_scale * 0.5

    line, samp = rpc.project(lat, lon, height)
    lat2, lon2 = rpc.localize(line, samp, height)

    lat_err_deg = np.abs(lat - lat2)
    lon_err_deg = np.abs(lon - lon2)
    # convert degree error to meters roughly (111320 m/deg)
    lat_err_m = lat_err_deg * 111320
    lon_err_m = lon_err_deg * 111320 * np.cos(np.radians(rpc.lat_off))

    print(f"Forward->inverse self-consistency over {n} random points:")
    print(f"  max lat error: {lat_err_m.max():.6f} m, max lon error: {lon_err_m.max():.6f} m")
    print(f"  mean lat error: {lat_err_m.mean():.6f} m, mean lon error: {lon_err_m.mean():.6f} m")
