"""The vectorised Open Buildings height check must give exactly what the
per-candidate loop it replaced gave (kept below, verbatim, as the reference)."""
import numpy as np
import pytest
from affine import Affine

import build_city_image as bci


def _reference_fuse(dsm, ground, footprints, dst, align, cand):
    H, W = dsm.shape
    n_fused = 0
    for poly in footprints:
        crop = bci._poly_crop_mask(poly, H, W)
        if crop is None:
            continue
        win, mask = crop
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        ys += win[0].start
        xs += win[1].start
        (ox, oy), (vx, vy) = align["offset_px"], align["lean_px_per_m"]
        consistent = []
        min_ok = max(4, int(bci.OPEN_BUILDINGS_MIN_FOOTPRINT_COVERAGE * ys.size))
        shift_f = {}
        for h in cand:
            sx, sy = int(round(ox + vx * h)), int(round(oy + vy * h))
            if (sx, sy) not in shift_f:
                f = None
                yy, xx = ys - sy, xs - sx
                ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                if ok.sum() >= min_ok:
                    v = dst[yy[ok], xx[ok]]
                    v = v[np.isfinite(v)]
                    if v.size >= 4:
                        f = float(np.median(v))
                shift_f[(sx, sy)] = f
            f = shift_f[(sx, sy)]
            if f is not None and abs(f - h) <= max(2.0, 0.12 * h):
                consistent.append(f)
        real_h = None
        if consistent:
            d = float(np.median((dsm[win] - ground[win])[mask]))
            real_h = (max(consistent) if d >= bci.DEPTH_SATURATION_M
                      else min(consistent, key=lambda c: abs(c - d)))
        if real_h is None or real_h < 1.0:
            continue
        dsm[win][mask] = float(np.median(ground[win][mask])) + real_h
        n_fused += 1
    return n_fused


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_vectorised_matches_loop(seed, monkeypatch):
    rng = np.random.RandomState(seed)
    H = W = 700
    heights = np.full((H, W), np.nan, np.float32)
    fps = []
    align = {"offset_px": (3.0, -5.0), "lean_px_per_m": (0.4, -0.6),
             "residual_px": 1.0, "bands": 4}
    for _ in range(300):
        x, y = rng.randint(30, W - 90), rng.randint(90, H - 60)
        w, h = rng.randint(5, 50), rng.randint(5, 50)
        hgt = float(rng.choice([3, 6, 9, 12, 20, 35, 60])) + float(rng.rand())
        fps.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32))
        if rng.rand() < 0.8:
            sx, sy = int(round(3 + 0.4 * hgt)), int(round(-5 - 0.6 * hgt))
            heights[max(0, y - sy):y - sy + h, max(0, x - sx):x - sx + w] = hgt
    ground = rng.rand(H, W).astype(np.float32)
    dsm0 = ground + (rng.rand(H, W) * 14).astype(np.float32)
    monkeypatch.setattr(bci, "estimate_ob_alignment", lambda *a, **k: align)
    tf = Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 2000000.0)
    got = dsm0.copy()
    n_new = bci._fuse_ob_grid(got, ground, fps, tf, "EPSG:32644", None,
                              {"heights": heights, "transform": tf}, 32644)
    ref = dsm0.copy()
    cand = np.arange(0.0, float(np.nanmax(heights)) + 1.0, 1.0)
    n_ref = _reference_fuse(ref, ground, fps, heights, align, cand)
    assert n_new == n_ref and n_new > 50
    assert np.array_equal(got, ref)
