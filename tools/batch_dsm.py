"""One command for a folder of satellite images -> one height product per image.

    python tools/batch_dsm.py IN_DIR OUT_DIR [--dem glo30|srtm] [--dem-file DEM.tif]
                              [--gsd 0.6] [--force] [--extras] [--fast]

For every image in IN_DIR (GeoTIFF, JPEG, PNG, JPEG 2000, BMP):

  * runs the same pipeline the upload page uses (build_city_image.build);
  * writes OUT_DIR/<name>_DSM.tif when the image is georeferenced: absolute
    elevation in metres, anchored to the chosen DEM (GLO-30 by default; SRTM,
    or any DEM GeoTIFF such as CartoDEM with --dem-file);
  * writes OUT_DIR/<name>_rDSM.tif when it is not (a JPEG or PNG): height
    above local ground in metres, ground sample from --gsd (Cartosat-2S
    0.6 m by default);
  * puts every product on the input's own pixel grid -- same width, height,
    CRS and geotransform -- so it overlays the image pixel for pixel;
  * checks every file it writes (opens, one float32 band, nodata set, grid
    identical to the input, CRS present for a DSM, enough valid pixels,
    heights in a physically possible range);
  * carries on when one image fails and records why.

Writes OUT_DIR/batch_log.csv and OUT_DIR/batch_log.json. Rerunning skips the
images already done, so an interrupted run resumes where it stopped (--force
rebuilds everything). --extras also writes the 1-sigma uncertainty and the
building outlines to OUT_DIR/extras/.

    python tools/batch_dsm.py --check OUT_DIR [IN_DIR]

only re-runs the checks on products already written.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

IMAGE_EXT = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".bmp")
NODATA = -9999.0
# Physically possible surfaces: the Dead Sea shore to above Everest, and an
# above-ground height no taller than the tallest building on Earth.
DSM_RANGE = (-450.0, 9000.0)
RDSM_RANGE = (-50.0, 900.0)
MIN_VALID_FRAC = 0.5


def find_images(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXT) and not p.lower().endswith(".aux.xml"):
            out.append(p)
    return out


def input_grid(path):
    """The input's pixel grid: (width, height, crs, transform), with crs and
    transform None when the file carries no usable georeference."""
    try:
        import rasterio
        with rasterio.open(path) as s:
            geo = s.crs is not None and not s.transform.is_identity
            return s.width, s.height, (s.crs if geo else None), (s.transform if geo else None)
    except Exception:
        pass
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = ImageOps.exif_transpose(im).size     # as the pipeline sees it: upright
    return w, h, None, None


def safe_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    return keep or "image"


def to_input_grid(src_path, dst_path, grid, tags_extra=None):
    """Write src_path's band onto the input's exact grid at dst_path."""
    import cv2
    import rasterio
    from rasterio.warp import Resampling, reproject
    w, h, crs, tf = grid
    with rasterio.open(src_path) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        tags = s.tags()
        s_crs, s_tf = s.crs, s.transform
    if crs is not None and s_crs is not None:
        out = np.full((h, w), np.nan, np.float32)
        reproject(a, out, src_transform=s_tf, src_crs=s_crs, src_nodata=np.nan,
                  dst_transform=tf, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    else:
        # No georeference: pixel grid to pixel grid. Resample values and the
        # nodata mask separately so nodata never bleeds into heights.
        valid = np.isfinite(a)
        fill = np.where(valid, a, np.nanmedian(a) if valid.any() else 0.0).astype(np.float32)
        out = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
        vm = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out = np.where(vm, out, np.nan).astype(np.float32)
        crs, tf = None, None
    tags.update(tags_extra or {})
    prof = dict(driver="GTiff", width=w, height=h, count=1, dtype="float32", nodata=NODATA,
                compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512,
                BIGTIFF="IF_SAFER")
    if crs is not None:
        prof.update(crs=crs, transform=tf)
    with rasterio.open(dst_path, "w", **prof) as d:
        d.write(np.where(np.isfinite(out), out, NODATA).astype(np.float32), 1)
        d.update_tags(**{k: str(v) for k, v in tags.items()})
        d.set_band_description(1, "elevation (m)" if tags.get("PRODUCT") == "DSM" else "height (m)")
    return dst_path


def check_product(path, grid=None, input_valid_frac=None):
    """Checks one written product. Returns (ok, report dict)."""
    import rasterio
    rep = {"file": os.path.basename(path)}
    problems = []
    try:
        with rasterio.open(path) as s:
            rep.update(bands=s.count, dtype=s.dtypes[0], width=s.width, height=s.height,
                       crs=s.crs.to_string() if s.crs else None, nodata=s.nodata)
            tags = s.tags()
            a = s.read(1, masked=True).filled(np.nan).astype(np.float32)
            tf = s.transform
    except Exception as e:
        return False, {"file": os.path.basename(path), "problems": [f"cannot open: {e}"]}
    product = tags.get("PRODUCT") or ("rDSM" if path.endswith("_rDSM.tif") else "DSM")
    rep["product"] = product
    rep["vertical_datum"] = tags.get("VERTICAL_DATUM")
    if rep["bands"] != 1:
        problems.append(f"{rep['bands']} bands, expected 1")
    if rep["dtype"] != "float32":
        problems.append(f"dtype {rep['dtype']}, expected float32")
    if rep["nodata"] is None:
        problems.append("no nodata value set")
    if grid is not None:
        w, h, crs, gtf = grid
        if (rep["width"], rep["height"]) != (w, h):
            problems.append(f"grid {rep['width']}x{rep['height']} differs from input {w}x{h}")
        if crs is not None:
            if rep["crs"] is None:
                problems.append("input is georeferenced but the product has no CRS")
            elif not (tf.almost_equals(gtf, precision=1e-6)):
                problems.append("geotransform differs from the input's")
    if product == "DSM" and rep["crs"] is None:
        problems.append("a DSM without a CRS cannot be compared with SRTM/Copernicus")
    fin = np.isfinite(a)
    rep["valid_frac"] = round(float(fin.mean()), 4)
    need = MIN_VALID_FRAC * (input_valid_frac if input_valid_frac else 1.0)
    if rep["valid_frac"] < need:
        problems.append(f"only {100 * rep['valid_frac']:.0f} % valid pixels")
    if fin.any():
        v = a[fin]
        lo, med, hi = (float(x) for x in np.percentile(v, [0.1, 50, 99.9]))
        rep.update(p001=round(lo, 2), median=round(med, 2), p999=round(hi, 2))
        rng = DSM_RANGE if product == "DSM" else RDSM_RANGE
        if lo < rng[0] or hi > rng[1]:
            problems.append(f"heights {lo:.0f}..{hi:.0f} m outside the possible {rng[0]:.0f}..{rng[1]:.0f} m")
    rep["problems"] = problems
    return not problems, rep


def input_valid_fraction(path):
    """Share of the input that carries image (not nodata/collar), for the
    valid-pixel check; 1.0 when it cannot be read cheaply."""
    try:
        import rasterio
        with rasterio.open(path) as s:
            f = max(1, max(s.width, s.height) // 1024)
            m = s.dataset_mask(out_shape=(max(1, s.height // f), max(1, s.width // f)))
            return max(0.01, float((m > 0).mean()))
    except Exception:
        return 1.0


def run_one(path, out_dir, args):
    import server
    import build_city_image as b
    name = "batch_" + safe_name(path)
    grid = input_grid(path)
    kw = server.upload_build_kwargs()
    # The scored product is the DSM; display-only work is skipped.
    kw.update(sr_texture=False, pbr_maps=False, stage=False, dem_source=args.dem)
    if args.dem_file:
        kw["dem_path"] = args.dem_file
    if grid[2] is None:
        kw.update(gsd_m=args.gsd, gsd_assumed=args.gsd_assumed)
    t = time.time()
    for suf in ("_DSM.tif", "_rDSM.tif", "_sigma.tif"):
        p = os.path.join(b.OUT_DIR, f"city_{name}{suf}")
        if os.path.exists(p):
            os.remove(p)                       # never pick up an earlier run's file
    b.build(path, name, **kw)
    made = [s for s in ("_DSM.tif", "_rDSM.tif")
            if os.path.exists(os.path.join(b.OUT_DIR, f"city_{name}{s}"))]
    if not made:
        raise RuntimeError("the pipeline finished but wrote no DSM/rDSM")
    suf = made[0]
    out = os.path.join(out_dir, safe_name(path) + suf)
    to_input_grid(os.path.join(b.OUT_DIR, f"city_{name}{suf}"), out, grid,
                  {"SOURCE": os.path.basename(path), "GRID": "input image grid"})
    if args.extras:
        ex = os.path.join(out_dir, "extras")
        os.makedirs(ex, exist_ok=True)
        sig = os.path.join(b.OUT_DIR, f"city_{name}_sigma.tif")
        if os.path.exists(sig):
            to_input_grid(sig, os.path.join(ex, safe_name(path) + "_sigma.tif"), grid)
        gj = os.path.join(b.OUT_DIR, f"city_{name}_buildings.geojson")
        if os.path.exists(gj):
            shutil.copy2(gj, os.path.join(ex, safe_name(path) + "_buildings.geojson"))
    return out, grid, round(time.time() - t, 1)


def run_child(path, args):
    """Process one image in a child process, stopped after args.timeout s.
    The child's log streams through; its last line carries the result row."""
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), "--one", path, "--dem", args.dem,
           "--gsd", str(args.gsd), "--timeout", str(args.timeout), "x", args.out_dir]
    if args.dem_file:
        cmd += ["--dem-file", args.dem_file]
    if args.extras:
        cmd.append("--extras")
    if not args.gsd_assumed:
        cmd.append("--gsd-given")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    row, t0 = None, time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    import threading

    def pump():
        nonlocal row
        for line in proc.stdout:
            if line.startswith("@@ROW@@"):
                row = json.loads(line[7:])
            elif line.strip():
                print("      " + line.rstrip()[:240], flush=True)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        th.join(5)
        return {"input": os.path.basename(path), "status": "timeout",
                "error": f"no result after {args.timeout:.0f} s", "seconds": round(time.time() - t0)}
    th.join(10)
    return row or {"input": os.path.basename(path), "status": "failed",
                   "error": f"worker exited with code {proc.returncode} and no result"}


def write_logs(out_dir, rows):
    with open(os.path.join(out_dir, "batch_log.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    cols = ["input", "status", "product", "output", "seconds", "width", "height", "crs",
            "median", "p001", "p999", "valid_frac", "vertical_datum", "problems", "error"]
    with open(os.path.join(out_dir, "batch_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["problems"] = "; ".join(r.get("problems") or [])
            w.writerow(r2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("in_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--dem", default="glo30", choices=("glo30", "srtm"),
                    help="global DEM the absolute DSM is anchored to (default: Copernicus GLO-30)")
    ap.add_argument("--dem-file", default=None, help="anchor to this DEM GeoTIFF instead (e.g. CartoDEM)")
    ap.add_argument("--gsd", type=float, default=0.6,
                    help="ground sample (m/px) for images without a georeference (default 0.6, Cartosat-2S)")
    ap.add_argument("--force", action="store_true", help="rebuild images that already have a product")
    ap.add_argument("--extras", action="store_true", help="also write uncertainty and building outlines")
    ap.add_argument("--check", metavar="OUT_DIR", help="only check products already in OUT_DIR")
    ap.add_argument("--timeout", type=float, default=1800,
                    help="seconds one image may take before it is stopped and retried once "
                         "(a stalled download or a sleeping laptop must not hang the whole run)")
    ap.add_argument("--one", help=argparse.SUPPRESS)        # internal: process one image, print a JSON row
    ap.add_argument("--fast", action="store_true",
                    help="one model pass per image instead of the 4-view average (about 4x quicker, "
                         "slightly less accurate)")
    ap.add_argument("--gsd-given", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    # Accuracy mode: the product height is the mean of 4 rotated views
    # (held-out DFC2019 at 0.6 m: per-building RMSE 1.87 -> 1.84 m).
    os.environ.setdefault("DW_PRODUCT_TTA", "1" if args.fast else "4")
    # The product's heights come from the model pass on the source grid; the
    # 3D scene's working grid only carries the smooth DEM ground, so it is
    # kept small here to save time.
    os.environ.setdefault("DW_MAX_PX", "2048")
    if args.one:                         # the parent says whether --gsd was given
        args.gsd_assumed = not args.gsd_given
    else:
        args.gsd_assumed = not any(a.startswith("--gsd=") or a == "--gsd" for a in sys.argv[1:])

    if args.check:
        prods = sorted(glob.glob(os.path.join(args.check, "*_DSM.tif")) +
                       glob.glob(os.path.join(args.check, "*_rDSM.tif")))
        src = {safe_name(p): p for p in find_images(args.in_dir)} if args.in_dir else {}
        bad = 0
        for p in prods:
            key = os.path.basename(p).rsplit("_", 1)[0]
            grid = input_grid(src[key]) if key in src else None
            ok, rep = check_product(p, grid)
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {rep['file']}  {rep.get('product')}  "
                  f"median {rep.get('median')}  valid {rep.get('valid_frac')}  "
                  + "; ".join(rep.get("problems", [])))
        print(f"{len(prods) - bad}/{len(prods)} products pass")
        sys.exit(1 if bad else 0)

    if args.one:
        # Child process: one image, one JSON row on the last stdout line.
        row = {"input": os.path.basename(args.one)}
        try:
            out, grid, secs = run_one(args.one, args.out_dir, args)
            ok, rep = check_product(out, grid, input_valid_fraction(args.one))
            row.update(status="ok" if ok else "check_failed", output=os.path.basename(out), seconds=secs,
                       **{k: rep.get(k) for k in ("product", "width", "height", "crs", "median", "p001",
                                                  "p999", "valid_frac", "vertical_datum", "problems")})
        except BaseException as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1200:])
        print("\n@@ROW@@" + json.dumps(row, default=str), flush=True)
        return

    if not (args.in_dir and args.out_dir):
        ap.error("IN_DIR and OUT_DIR are required (or use --check OUT_DIR)")
    os.makedirs(args.out_dir, exist_ok=True)
    images = find_images(args.in_dir)
    if not images:
        sys.exit(f"no images ({', '.join(IMAGE_EXT)}) in {args.in_dir}")
    log_p = os.path.join(args.out_dir, "batch_log.json")
    prev = {}
    if os.path.exists(log_p) and not args.force:
        try:
            prev = {r["input"]: r for r in json.load(open(log_p, encoding="utf-8"))}
        except Exception:
            prev = {}
    print(f"{len(images)} images -> {args.out_dir}  (DEM: {args.dem_file or args.dem}; "
          f"un-georeferenced images at {args.gsd} m/px)", flush=True)
    rows, t_all = [], time.time()
    for i, p in enumerate(images, 1):
        key = os.path.basename(p)
        old = prev.get(key)
        if old and old.get("status") == "ok" and old.get("output") and \
                os.path.exists(os.path.join(args.out_dir, old["output"])):
            rows.append(old)
            print(f"[{i}/{len(images)}] {key}: already done, skipped", flush=True)
            continue
        print(f"[{i}/{len(images)}] {key}", flush=True)
        row = None
        for attempt in (1, 2):
            row = run_child(p, args)
            if row.get("status") != "timeout":
                break
            print(f"    no result after {args.timeout:.0f} s -- stopped; "
                  + ("retrying once" if attempt == 1 else "giving up on this image"), flush=True)
        rows.append(row)
        write_logs(args.out_dir, rows)
        print(f"    {row['status']}  {row.get('product', '')}  {row.get('seconds', '')} s  "
              f"median {row.get('median', '')} m  {row.get('error', '')}"
              + ("  problems: " + "; ".join(row["problems"]) if row.get("problems") else ""), flush=True)
    n_ok = sum(r.get("status") == "ok" for r in rows)
    print(f"\n{n_ok}/{len(rows)} images -> checked products in {time.time() - t_all:.0f} s. "
          f"Log: {os.path.join(args.out_dir, 'batch_log.csv')}")
    sys.exit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
