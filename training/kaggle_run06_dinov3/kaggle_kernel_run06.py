"""Kaggle GPU run 06: height above ground from a SATELLITE-pretrained encoder.

Encoder: Meta DINOv3 ViT-L/16 pretrained on SAT-493M (493 million Maxar
satellite images) -- DINOv3 License (commercial use allowed; ships with its
licence; no military / ITAR end use). Weights: the Meta checkpoint as
mirrored in giswqs/geoai (dinov3_vitl16_sat493m.pth). Model code: the
official facebookresearch/dinov3 repository (torch.hub, local).

Head: a DPT-style fusion of four encoder depths (blocks 5, 12, 18, 24) up to
full resolution; output softplus metres above ground.

Data (all tiles never overlap our 18 held-out DFC benchmark tiles, which are
not in the attached datasets):
  * DFC2019 Track 3 train + Track 1 train (WorldView-3, airborne LiDAR)   [Kaggle datasets]
  * GAMUS (aerial, LiDAR nDSM), pulled inside the session                  [Hugging Face]
  * NEW: USDA NAIP 0.6 m imagery + USGS 3DEP LiDAR height above ground (2 m),
    fetched inside the session from Microsoft Planetary Computer: real 0.6 m
    sensor imagery -- the Cartosat-2S ground sample -- over tall cities,
    tall forest, hills and farmland. The 2 m labels are compared with the
    prediction area-averaged to 2 m, so the coarse label never blurs the
    target at full resolution.

Loss: Huber on height + scale-invariant log term + gradient matching (full-
resolution labels only), each pixel weighted 1 + min(h, 40)/20 so tall
buildings and canopy -- where every model under-calls -- count more.
Augmentation: D4 flips/rotations, 0.3-1.0 m resolution simulation, colour.
Model selection: DFC Track 3 validation tiles (same split as runs 03-05).
Outputs (/kaggle/working): run06_best.pt, training_report.json.
"""
import concurrent.futures as cf
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

T0 = time.time()
SESSION_H = float(os.environ.get("RUN_HOURS", "11.2"))   # Kaggle kills at 12 h
TRAIN_H = float(os.environ.get("TRAIN_HOURS", "3.4"))    # wall time spent in the loop
WORK = "/kaggle/working"
TMP = "/kaggle/tmp" if os.path.isdir("/kaggle") else "/tmp/run06"
os.makedirs(TMP, exist_ok=True)
CROP = 448
BATCH = int(os.environ.get("BATCH", "8"))
SEED = 1337
random.seed(SEED); np.random.seed(SEED)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)


def el():
    return f"{(time.time() - T0) / 60:6.1f} min"


# ------------------------------------------------------------------ setup
def setup_model_code():
    repo = os.path.join(TMP, "dinov3")
    if not os.path.isdir(repo):
        sh(f"git clone -q --depth 1 https://github.com/facebookresearch/dinov3 {repo}")
    from huggingface_hub import hf_hub_download
    w0 = hf_hub_download("giswqs/geoai", "dinov3_vitl16_sat493m.pth", local_dir=TMP)
    # the hub loader reads the release hash from the file name; eadcf0ff = SAT-493M
    # ViT-L, which also selects its untied global/local cls norm layout
    w = os.path.join(TMP, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    if not os.path.exists(w):
        os.symlink(w0, w)
    try:
        urllib.request.urlretrieve("https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md",
                                   os.path.join(WORK, "DINOv3_LICENSE.md"))
    except Exception:
        for n in ("LICENSE.md", "LICENSE"):
            if os.path.exists(os.path.join(repo, n)):
                shutil.copy(os.path.join(repo, n), os.path.join(WORK, "DINOv3_LICENSE.md")); break
    return repo, w


# ------------------------------------------------------------------ data
def find_dfc():
    t3 = t1 = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs and "val" in dirs and t3 is None:
            tr = os.path.join(root, "train")
            if any(f.startswith("DFC_") for f in os.listdir(tr)):
                t3 = root
        if any(f.startswith("DFC1_") and f.endswith("_RGB.h5") for f in files):
            t1 = root
    return t3, t1


def pairs_in(d, prefix=""):
    out = []
    for p in sorted(glob.glob(os.path.join(d, f"{prefix}*_RGB.h5"))):
        a = p[:-len("_RGB.h5")] + "_AGL.h5"
        if os.path.exists(a):
            out.append(("h5", p, a))
    return out


GAMUS = "earthflow/GAMUS"


def gamus(cap):
    import requests
    from huggingface_hub import HfApi
    api = HfApi()
    dest = os.path.join(TMP, "gamus", "train"); os.makedirs(dest, exist_ok=True)
    imgs = [os.path.basename(f.path) for f in api.list_repo_tree(GAMUS, repo_type="dataset", path_in_repo="images/train")]
    random.Random(SEED).shuffle(imgs)
    imgs = imgs[:cap]
    base = f"https://huggingface.co/datasets/{GAMUS}/resolve/main"
    S = requests.Session()

    def get(n):
        stem = n[:-len("_RGB.h5")]
        rp, ap = os.path.join(dest, n), os.path.join(dest, f"{stem}_AGL.h5")
        if os.path.exists(rp) and os.path.exists(ap):
            return True
        for url, p in ((f"{base}/images/train/{n}", rp), (f"{base}/heights/train/{stem}_AGL.h5", ap)):
            for k in range(5):
                try:
                    r = S.get(url, timeout=120); r.raise_for_status()
                    open(p, "wb").write(r.content); break
                except Exception:
                    time.sleep(3 * (k + 1))
            else:
                return False
        return True
    with cf.ThreadPoolExecutor(16) as ex:
        ok = sum(ex.map(get, imgs))
    print(f"[{el()}] GAMUS {ok}/{len(imgs)} tiles", flush=True)
    return pairs_in(dest)


# NAIP 0.6 m + 3DEP LiDAR HAG (2 m) from Planetary Computer.
# (name, lon, lat): tall cities, tall forest, hills, suburbs, farmland, desert.
SITES = [
    ("nyc_midtown", -73.985, 40.755), ("nyc_downtown", -74.010, 40.708), ("chicago_loop", -87.630, 41.880),
    ("seattle_dt", -122.335, 47.607), ("sf_dt", -122.400, 37.790), ("la_dt", -118.253, 34.050),
    ("houston_dt", -95.367, 29.758), ("denver_dt", -104.990, 39.745), ("atlanta_mid", -84.385, 33.781),
    ("philly_cc", -75.165, 39.952), ("boston_bb", -71.075, 42.350), ("minneapolis", -93.270, 44.977),
    ("pittsburgh", -79.997, 40.441), ("portland_or", -122.677, 45.520), ("dallas_dt", -96.800, 32.780),
    ("miami_brickell", -80.193, 25.763), ("phoenix_dt", -112.074, 33.450), ("las_vegas", -115.170, 36.118),
    ("olympic_forest", -123.900, 47.860), ("redwoods", -124.010, 41.360), ("sierra_forest", -120.230, 38.470),
    ("appalachian", -83.510, 35.610), ("oregon_cascade", -122.100, 44.200), ("smoky_hills", -83.300, 35.700),
    ("rockies_town", -106.820, 39.190), ("boulder_hills", -105.300, 40.010), ("sf_hills", -122.450, 37.760),
    ("iowa_farm", -93.600, 42.000), ("kansas_farm", -97.300, 38.500), ("central_valley", -120.500, 37.000),
    ("atlanta_suburb", -84.200, 33.950), ("dallas_suburb", -96.700, 33.050), ("orlando_suburb", -81.300, 28.550),
    ("tucson", -110.950, 32.230), ("albuquerque", -106.650, 35.090), ("salt_lake", -111.890, 40.760),
    ("nashville", -86.780, 36.160), ("new_orleans", -90.070, 29.950), ("detroit", -83.050, 42.330),
    ("cincinnati", -84.510, 39.100),
]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
CROPAPI = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"


def _post(q):
    req = urllib.request.Request(STAC, data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["features"]


def naip_site(site, per_site=9, tile_m=307.2):
    """per_site tiles in a 3x3 block around the site; returns saved npz paths."""
    from pyproj import Transformer
    import rasterio
    name, lon, lat = site
    zone = int((lon + 180) // 6) + 1
    epsg = f"EPSG:{32600 + zone}"
    x0, y0 = Transformer.from_crs("EPSG:4326", epsg, always_xy=True).transform(lon, lat)
    out = []
    d = os.path.join(TMP, "naip"); os.makedirs(d, exist_ok=True)
    for i in range(per_site):
        cx, cy = x0 + ((i % 3) - 1) * tile_m, y0 + ((i // 3) - 1) * tile_m
        path = os.path.join(d, f"{name}_{i}.npz")
        if os.path.exists(path):
            out.append(("npz", path, None)); continue
        l, b, r, t = cx - tile_m / 2, cy - tile_m / 2, cx + tile_m / 2, cy + tile_m / 2
        ll = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        (lo0, la0), (lo1, la1) = ll.transform(l, b), ll.transform(r, t)
        try:
            hag = _post({"collections": ["3dep-lidar-hag"], "bbox": [lo0, la0, lo1, la1], "limit": 5})
            naip = _post({"collections": ["naip"], "bbox": [lo0, la0, lo1, la1], "limit": 20})
            naip = [f for f in naip if abs(float(f["properties"].get("gsd", 0)) - 0.6) < 0.01]
            if not hag or not naip:
                continue
            naip.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
            c = epsg.replace(":", "%3A")
            img_u = (f"{CROPAPI}/{l},{b},{r},{t}/512x512.tif?collection=naip&item={naip[0]['id']}"
                     f"&assets=image&asset_bidx=image%7C1,2,3&coord_crs={c}&dst_crs={c}&resampling=average")
            n2 = int(round(tile_m / 2.0))
            hag_u = (f"{CROPAPI}/{l},{b},{r},{t}/{n2}x{n2}.tif?collection=3dep-lidar-hag&item={hag[0]['id']}"
                     f"&assets=data&coord_crs={c}&dst_crs={c}&resampling=average")
            with rasterio.MemoryFile(urllib.request.urlopen(img_u, timeout=120).read()) as mf, mf.open() as s:
                im = s.read()
            with rasterio.MemoryFile(urllib.request.urlopen(hag_u, timeout=120).read()) as mf, mf.open() as s:
                h = s.read(1).astype(np.float32)
                m = s.read_masks(1) > 0
            if im.shape[0] > 3:
                valid_img = im[3] > 0; im = im[:3]
            else:
                valid_img = im.sum(0) > 0
            if valid_img.mean() < 0.95:
                continue
            h = np.where(m & (h > -5) & (h < 400), np.clip(h, 0, None), np.nan).astype(np.float32)
            if np.isfinite(h).mean() < 0.8:
                continue
            np.savez_compressed(path, rgb=im.transpose(1, 2, 0).astype(np.uint8), agl=h.astype(np.float16))
            out.append(("npz", path, None))
        except Exception as e:
            print(f"   {name}_{i}: {type(e).__name__}", flush=True)
    return out


def naip_all():
    with cf.ThreadPoolExecutor(8) as ex:
        res = [p for lst in ex.map(naip_site, SITES) for p in lst]
    print(f"[{el()}] NAIP+3DEP {len(res)} tiles from {len(SITES)} sites", flush=True)
    return res


def load(item):
    kind, a, b = item
    if kind == "npz":
        z = np.load(a)
        return z["rgb"], z["agl"].astype(np.float32), 2.0   # label pixel 2 m
    import h5py
    with h5py.File(a, "r") as h:
        rgb = h["image"][:]
    with h5py.File(b, "r") as h:
        agl = h["image"][:].astype(np.float32)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, agl, None                                   # label on the image grid


# ------------------------------------------------------------------ dataset
def make_dataset(items, train=True):
    import cv2
    import torch

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items) * (4 if train else 1)

        def __getitem__(self, i):
            rgb, agl, lab_m = load(items[i % len(items)])
            H, W = rgb.shape[:2]
            if lab_m is None and agl.shape != (H, W):
                agl = cv2.resize(agl, (W, H), interpolation=cv2.INTER_NEAREST)
            coarse = lab_m is not None
            if train:
                # random crop in image pixels (the coarse label follows by ratio)
                c = min(CROP, H, W)
                y, x = random.randint(0, H - c), random.randint(0, W - c)
                rgb = rgb[y:y + c, x:x + c]
                if coarse:
                    f = agl.shape[0] / H
                    ly, lx, lc = int(y * f), int(x * f), max(4, int(round(c * f)))
                    agl = agl[ly:ly + lc, lx:lx + lc]
                else:
                    agl = agl[y:y + c, x:x + c]
                k = random.randint(0, 3)
                rgb, agl = np.rot90(rgb, k), np.rot90(agl, k)
                if random.random() < 0.5:
                    rgb, agl = rgb[:, ::-1], agl[:, ::-1]
                # simulate a coarser sensor (0.3 m source -> 0.45..1.0 m) and back
                if random.random() < 0.6:
                    s = random.uniform(0.3, 0.9)
                    small = cv2.resize(np.ascontiguousarray(rgb), (max(8, int(c * s)), max(8, int(c * s))),
                                       interpolation=cv2.INTER_AREA)
                    rgb = cv2.resize(small, (c, c), interpolation=cv2.INTER_CUBIC)
                rgb = np.ascontiguousarray(rgb).astype(np.float32)
                rgb = rgb * random.uniform(0.8, 1.2) + random.uniform(-15, 15)
                if c != CROP:
                    rgb = cv2.resize(rgb, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
                    if not coarse:
                        agl = cv2.resize(np.ascontiguousarray(agl), (CROP, CROP), interpolation=cv2.INTER_NEAREST)
            else:
                c = (min(H, W) // 16) * 16
                rgb, agl = rgb[:c, :c].astype(np.float32), agl[:c, :c] if not coarse else agl
            x = torch.from_numpy(np.clip(np.ascontiguousarray(rgb), 0, 255) / 255.0).permute(2, 0, 1).float()
            x = (x - torch.tensor([0.430, 0.411, 0.296])[:, None, None]) / torch.tensor([0.213, 0.156, 0.143])[:, None, None]
            agl = np.ascontiguousarray(agl, dtype=np.float32)
            if train:
                # labels differ in size (2 m grids); pad to a fixed block so batches collate
                lh, lw = min(agl.shape[0], CROP), min(agl.shape[1], CROP)
                pad = np.full((CROP, CROP), np.nan, np.float32)
                pad[:lh, :lw] = agl[:lh, :lw]
                agl = pad
            else:
                lh, lw = agl.shape
            y = torch.from_numpy(agl)[None]
            return x, y, torch.tensor([1 if coarse else 0, lh, lw])

    return DS()


# ------------------------------------------------------------------ model
def build_model(repo, weights):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    enc = torch.hub.load(repo, "dinov3_vitl16", source="local", weights=weights)

    class Fuse(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.r1 = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))
            self.r2 = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

        def forward(self, x, skip=None):
            if skip is not None:
                x = x + self.r1(skip)
            x = x + self.r2(x)
            return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    class HeightNet(nn.Module):
        LAYERS = [4, 11, 17, 23]

        def __init__(self, enc, d=1024, c=256):
            super().__init__()
            self.enc = enc
            self.proj = nn.ModuleList([nn.Conv2d(d, c, 1) for _ in range(4)])
            # DPT-style reassemble: 1/4, 1/8, 1/16, 1/32 of the input
            self.resample = nn.ModuleList([nn.ConvTranspose2d(c, c, 4, 4), nn.ConvTranspose2d(c, c, 2, 2),
                                           nn.Identity(), nn.Conv2d(c, c, 3, 2, 1)])
            self.fuse = nn.ModuleList([Fuse(c) for _ in range(4)])
            self.out = nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(c // 2, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))

        def forward(self, x):
            H, W = x.shape[-2:]
            feats = self.enc.get_intermediate_layers(x, n=self.LAYERS, reshape=True, norm=True)
            f = [r(p(t)) for t, p, r in zip(feats, self.proj, self.resample)]
            y = self.fuse[3](f[3])
            y = self.fuse[2](y, f[2])
            y = self.fuse[1](y, f[1])
            y = self.fuse[0](y, f[0])
            y = F.interpolate(self.out(y), size=(H, W), mode="bilinear", align_corners=False)
            return F.softplus(y) * 10.0   # metres, >= 0

    m = HeightNet(enc)
    # train the last half of the encoder; the first half keeps its satellite features
    for i, blk in enumerate(m.enc.blocks):
        for p in blk.parameters():
            p.requires_grad = i >= 12
    for n, p in m.enc.named_parameters():
        if not n.startswith("blocks."):
            p.requires_grad = False
    return m


# ------------------------------------------------------------------ loss / eval
def loss_fn(pred, y, coarse):
    import torch
    import torch.nn.functional as F
    tot, n = 0.0, 0
    for i in range(pred.shape[0]):
        lh, lw = int(coarse[i][1]), int(coarse[i][2])
        p, t = pred[i:i + 1], y[i:i + 1, :, :lh, :lw]
        if coarse[i][0]:
            p = F.adaptive_avg_pool2d(p, t.shape[-2:])
        m = torch.isfinite(t)
        if m.sum() < 16:
            continue
        pv, tv = p[m], t[m]
        w = 1.0 + torch.clamp(tv, 0, 40) / 20.0
        hub = (F.smooth_l1_loss(pv, tv, reduction="none", beta=1.0) * w).sum() / w.sum()
        d = torch.log(pv + 1.0) - torch.log(tv + 1.0)
        silog = torch.sqrt(torch.clamp((d ** 2).mean() - 0.85 * d.mean() ** 2, min=1e-8))
        l = hub + 0.5 * silog
        if not coarse[i][0]:
            tt = torch.nan_to_num(t, nan=0.0)
            mm = m.float()
            gx = ((p[..., 1:] - p[..., :-1]) - (tt[..., 1:] - tt[..., :-1])).abs() * mm[..., 1:] * mm[..., :-1]
            gy = ((p[..., 1:, :] - p[..., :-1, :]) - (tt[..., 1:, :] - tt[..., :-1, :])).abs() * mm[..., 1:, :] * mm[..., :-1, :]
            l = l + 0.5 * (gx.sum() + gy.sum()) / (mm.sum() + 1)
        tot = tot + l; n += 1
    return tot / max(n, 1)


def evaluate(model, items, dev):
    import torch
    ds = make_dataset(items, train=False)
    model.eval()
    se = ae = nn_ = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(len(ds)):
            x, y, _ = ds[i]
            p = model(x[None].to(dev)).float().cpu()[0]
            m = torch.isfinite(y)
            e = (p[m] - y[m]).double()
            se += float((e ** 2).sum()); ae += float(e.abs().sum()); nn_ += int(m.sum())
    model.train()
    return {"rmse_m": math.sqrt(se / max(nn_, 1)), "mae_m": ae / max(nn_, 1), "pixels": nn_}


# ------------------------------------------------------------------ main
def main():
    import torch
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    repo, w = setup_model_code()
    t3, t1 = find_dfc()
    train = pairs_in(os.path.join(t3, "train")) + (pairs_in(t1, "DFC1_") if t1 else [])
    val = pairs_in(os.path.join(t3, "val"))
    print(f"[{el()}] DFC train {len(train)}, val {len(val)}", flush=True)
    with cf.ThreadPoolExecutor(2) as ex:
        fg = ex.submit(gamus, int(os.environ.get("GAMUS_CAP", "1600")))
        fn = ex.submit(naip_all)
        g, nai = fg.result(), fn.result()
    # NAIP tiles are few but in-domain for 0.6 m: repeat them
    train = train + g + nai * 3
    random.shuffle(train)
    print(f"[{el()}] training items {len(train)} (DFC {len(train) - len(g) - 3 * len(nai)}, "
          f"GAMUS {len(g)}, NAIP x3 {3 * len(nai)})", flush=True)

    dev = "cuda"
    model = build_model(repo, w).to(dev)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": head_p, "lr": 2e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    dl = torch.utils.data.DataLoader(make_dataset(train), batch_size=BATCH, shuffle=True, num_workers=4,
                                     drop_last=True, persistent_workers=True)
    t_start, step, best, hist = time.time(), 0, 1e9, []
    budget = min(TRAIN_H, SESSION_H - (time.time() - T0) / 3600 - 0.8) * 3600
    last_eval = time.time()
    print(f"[{el()}] training for {budget / 3600:.2f} h", flush=True)
    done = False
    while not done:
        for x, y, c in dl:
            frac = (time.time() - t_start) / budget
            if frac >= 1:
                done = True; break
            warm = min(1.0, step / 300)
            for gi, base in enumerate((2e-5, 2e-4)):
                opt.param_groups[gi]["lr"] = base * warm * 0.5 * (1 + math.cos(math.pi * min(frac, 1)))
            with torch.autocast("cuda", dtype=torch.float16):
                p = net(x.to(dev, non_blocking=True))
            l = loss_fn(p.float(), y.to(dev), c)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"[{el()}] step {step} loss {float(l):.3f} ({frac * 100:.0f}% of budget)", flush=True)
            if time.time() - last_eval > 40 * 60 or (frac > 0.97 and time.time() - last_eval > 300):
                r = evaluate(model, val, dev); r["step"] = step; hist.append(r); last_eval = time.time()
                print(f"[{el()}] VAL {r}", flush=True)
                if r["mae_m"] < best:
                    best = r["mae_m"]
                    sd = {k: v.half() for k, v in model.state_dict().items()}
                    torch.save(sd, os.path.join(WORK, "run06_best.pt"))
                    print(f"    saved best (val MAE {best:.3f})", flush=True)
    r = evaluate(model, val, dev); r["step"] = step; hist.append(r)
    if r["mae_m"] < best:
        best = r["mae_m"]
        torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(WORK, "run06_best.pt"))
    json.dump({"encoder": "DINOv3 ViT-L/16 SAT-493M (Meta, DINOv3 License)", "head": "DPT-style 4-level fusion",
               "layers": [4, 11, 17, 23], "crop": CROP, "steps": step, "best_val_mae_m": best,
               "val": "DFC2019 Track 3 validation tiles (as runs 03-05)", "history": hist,
               "train_items": len(train), "gamus_tiles": len(g), "naip_3dep_tiles": len(nai),
               "normalisation": {"mean": [0.430, 0.411, 0.296], "std": [0.213, 0.156, 0.143]}},
              open(os.path.join(WORK, "training_report.json"), "w"), indent=1)
    shutil.copy(__file__, os.path.join(WORK, "kaggle_kernel_run06.py"))
    print(f"[{el()}] done: best val MAE {best:.3f} m", flush=True)


if __name__ == "__main__":
    main()
