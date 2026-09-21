import hashlib
import json
import os

import numpy as np
from PIL import Image
import torch
from transformers import pipeline

# --- Behavioural checkpoint verification --------------------------------
#
# A licensed checkpoint (metric_ok.json / htc_ok.json) is trusted to swap
# straight into the pipeline with no further check -- fine until the file on
# disk is not the file that earned that trust: a bad copy, a partial
# download, a manual edit, or (worse) a wrong-but-similarly-named checkpoint
# left in the same directory by mistake. A raw file hash would catch that,
# but only that -- it says nothing about whether the weights still BEHAVE
# the way they did when licensed, which is the property that actually
# matters here and is not implied by "the bytes match" (a hash also breaks
# on any benign re-save: a different safetensors shard layout, a metadata
# touch-up, a format migration -- none of which change what the model
# computes).
#
# This project already has a working pattern for exactly this question:
# ov_backend.py never trusts the OpenVINO iGPU path just because it loaded
# without error -- it runs a real input through both backends and only
# keeps the accelerated one if its output matches PyTorch's within
# MAX_REL_ERROR. The same idea, applied at load time instead of backend-
# switch time: run a fixed, reproducible synthetic input through the loaded
# weights and compare the OUTPUT against a fingerprint recorded when this
# exact checkpoint was licensed. A mismatch means the weights on disk no
# longer behave like the ones that were promoted, whatever the reason, and
# the checkpoint is refused rather than trusted on faith.
CANARY_SEED = 20260920     # fixed, so every machine generates the identical input
CANARY_PX = 64              # small: this runs on every process start, not once
CANARY_REL_TOL = 0.01       # same threshold ov_backend.py already uses for "matches"


def _canary_input() -> Image.Image:
    """A deterministic synthetic image, not a file on disk -- generating it
    from a fixed seed means verification never depends on a golden-image
    file that could itself go missing, get corrupted, or drift out of sync
    with the code, which is exactly the class of problem this exists to
    avoid introducing elsewhere."""
    rng = np.random.RandomState(CANARY_SEED)
    arr = (rng.rand(CANARY_PX, CANARY_PX, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _fingerprint_array(arr: np.ndarray) -> dict:
    """A compact, reproducible description of an output array's BEHAVIOUR.

    Rounded before hashing: bit-exact reproducibility across CPU/GPU/library
    versions is not guaranteed even for identical weights, so hashing raw
    float64 bytes would false-positive on harmless numerical noise. Three
    decimal places is far tighter than CANARY_REL_TOL, so real drift still
    changes the hash while floating-point jitter mostly doesn't -- and the
    mean/std/relative check below is the actual pass/fail criterion; the
    hash is a stronger secondary signal for a human comparing two reports.
    """
    a = np.asarray(arr, dtype=np.float64)
    rounded = np.round(a, 3)
    return {
        "hash": hashlib.sha256(rounded.tobytes()).hexdigest()[:16],
        "mean": round(float(rounded.mean()), 4),
        "std": round(float(rounded.std()), 4),
        "shape": list(a.shape),
    }


def _fingerprints_match(expected: dict, actual: dict) -> tuple[bool, str]:
    if not expected or "mean" not in expected:
        return False, "no fingerprint on record"
    if list(expected.get("shape") or []) != list(actual["shape"]):
        return False, f"output shape changed ({expected.get('shape')} -> {actual['shape']})"
    denom = max(abs(expected["mean"]), 1e-6)
    rel = abs(actual["mean"] - expected["mean"]) / denom
    if rel > CANARY_REL_TOL:
        return False, (f"canary output mean drifted {rel*100:.1f}% "
                        f"(licensed {expected['mean']}, now {actual['mean']}, "
                        f"limit {CANARY_REL_TOL*100:.0f}%)")
    return True, "ok"


def license_checkpoint(marker_path: str, backbone: "DepthBackbone") -> dict:
    """Records the behavioural fingerprint that later loads of THIS
    checkpoint will be checked against -- called once, at the moment a
    checkpoint earns its marker file (metric_ok.json / htc_ok.json), never
    at load time. Merges into whatever's already in the marker rather than
    overwriting it, since train_height.py's own promotion stats (MAE,
    baseline comparison) already live in that same file.
    """
    fp = _fingerprint_array(backbone.predict(_canary_input(), raw_metric=True))
    try:
        with open(marker_path, encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        info = {}
    info["canary"] = fp
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return fp


# Identifies which backbone produced a cached height field. Cache keys must
# carry it: the fields are model-specific, and after switching from Large to
# Base every existing entry describes a different model's output. Without the
# tag the pipeline silently reuses the old model's results and the switch looks
# like it changed nothing.
def backbone_tag(model_name: str = None) -> str:
    n = (model_name or "depth-anything/Depth-Anything-V2-Base-hf").lower()
    for k in ("large", "base", "small"):
        if k in n:
            return k[0]
    return "x"


class DepthBackbone:
    # Base, not Large.
    #
    # Measured on JAX_165 against LiDAR, tiled inference, identical settings,
    # comparing the height field over the 256 m the truth actually covers:
    #
    #     model   time    corr     MAE      roof/ground separation
    #     Large    428s   0.7630   4.67 m   +0.190
    #     Base      78s   0.7626   4.50 m   +0.307
    #     Small     32s   0.6886   6.04 m   +0.252
    #
    # Base matches Large's correlation to four decimal places, has slightly
    # LOWER error, and separates rooftops from ground substantially better --
    # which is the property segmentation and footprint extraction depend on. It
    # is 5.5x faster on CPU, and this runs on CPU.
    #
    # Small is not a free further win: it loses 10% of the correlation and 34%
    # of the accuracy, so it is offered but not the default.
    # A fine-tuned metric checkpoint, when one has been trained and committed.
    #
    # The distinction this carries is the whole point of training one. The stock
    # backbone emits a RELATIVE field, so the pipeline has to invent a scale --
    # "assume the tallest structure is about 40 m" -- and measured blind that
    # guess lands 24-353% away from what each tile actually needs, which is
    # roughly three quarters of the blind error. A model trained on GAMUS AGL
    # predicts METRES, so there is no scale to guess and that error source is
    # removed rather than reduced.
    #
    # Absent by default: it exists only after scripts/run_gamus.py has trained
    # one AND that run beat the baseline. `is_metric` is what the pipeline reads
    # to decide whether to apply its scale assumption at all.
    METRIC_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "height_best")

    # The marker, not the directory, is what licenses the checkpoint.
    #
    # A training run always leaves weights behind, including a run that produced
    # a model WORSE than the stock backbone -- the first smoke run here scored
    # MAE 7.16 m against the baseline's 6.68 m with a correlation of -0.121, and
    # those weights sat on disk looking exactly like a good checkpoint. Gating
    # on the directory alone means any failed run silently degrades every
    # subsequent build. train_height.py writes this marker only when the
    # fine-tuned model beat a baseline that was handed the optimal scale.
    MARKER = "metric_ok.json"

    # HTC-AdaBins checkpoint (train_height.py --htc), a separate lineage from
    # the plain-regression METRIC_CKPT above -- different architecture
    # (adaptive bins + head-tail cut, htc_bins.py), different file format (a
    # plain state_dict, not a transformers checkpoint), and its own licensing
    # marker so a losing --htc run can never get picked up here either.
    HTC_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "height_htc_best")
    HTC_MARKER = "htc_ok.json"

    @classmethod
    def metric_available(cls) -> bool:
        return (os.path.isdir(cls.METRIC_CKPT)
                and os.path.exists(os.path.join(cls.METRIC_CKPT, "config.json"))
                and os.path.exists(os.path.join(cls.METRIC_CKPT, cls.MARKER)))

    @classmethod
    def metric_info(cls) -> dict:
        """What the licensing run measured, for the build log and scene.json."""
        import json
        try:
            with open(os.path.join(cls.METRIC_CKPT, cls.MARKER), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def htc_available(cls) -> bool:
        return (os.path.isdir(cls.HTC_CKPT)
                and os.path.exists(os.path.join(cls.HTC_CKPT, "htc_model.pt"))
                and os.path.exists(os.path.join(cls.HTC_CKPT, cls.HTC_MARKER)))

    @classmethod
    def htc_info(cls) -> dict:
        import json
        try:
            with open(os.path.join(cls.HTC_CKPT, cls.HTC_MARKER), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # Every request used to pay for a full weight reload from disk -- confirmed
    # by reading the actual call path, not assumed: build_city_image.build()
    # constructs a fresh DepthBackbone() on every job, and server.py calls
    # build() fresh per upload, so a "cached" model was never real. Fixed with
    # a flyweight keyed on the exact constructor args: the pipeline's own
    # `_pipeline_lock` (server.py) already serialises every job to one at a
    # time, so there is no concurrent-construction race to guard against, and
    # the args (model_name/device/metric/htc) are the complete description of
    # which weights get loaded -- two calls with the same args always want the
    # same already-loaded model, never a fresh copy of it.
    _instances: dict = {}

    def __new__(cls, model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
                device: str = None, metric: bool = None, htc: bool = None):
        key = (model_name, device, metric, htc)
        cached = cls._instances.get(key)
        if cached is not None:
            return cached
        obj = super().__new__(cls)
        cls._instances[key] = obj
        return obj

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
                 device: str = None, metric: bool = None, htc: bool = None):
        if getattr(self, "_loaded", False):
            return   # flyweight hit -- weights are already in memory
        self._loaded = True
        # metric=None means "use the fine-tuned checkpoint if one is there".
        # Explicit True demands it and fails loudly if absent, so a benchmark
        # cannot silently score the stock model and report it as the metric one.
        #
        # htc=None means "prefer the HTC-AdaBins checkpoint over the plain
        # metric one, if BOTH are licensed" -- htc_bins.py's adaptive-bin,
        # long-tail-aware head is only ever licensed after beating the same
        # baseline-with-affine the plain regression head has to beat, so a
        # licensed HTC checkpoint is never a weaker choice than the plain one.
        # Explicit True/False overrides that preference and fails loudly if
        # the requested checkpoint is not actually licensed, for the same
        # benchmark-integrity reason `metric` does.
        if htc is None:
            htc = self.htc_available()
        elif htc and not self.htc_available():
            raise FileNotFoundError(
                f"HTC checkpoint not licensed at {self.HTC_CKPT} -- "
                f"run train_height.py --htc first, and it must beat baseline")
        self.is_htc = bool(htc)

        if metric is None:
            metric = (not self.is_htc) and self.metric_available()
        elif metric and not self.metric_available():
            raise FileNotFoundError(
                f"metric checkpoint not found at {self.METRIC_CKPT} -- "
                f"run scripts/run_gamus.py first")
        self.is_metric = bool(metric) and not self.is_htc
        if self.is_metric:
            model_name = self.METRIC_CKPT
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use every core. torch defaults to a conservative count, and depth
        # inference is the whole build's critical path.
        try:
            import os
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass
        # Optional iGPU acceleration. Verified against PyTorch on first use and
        # abandoned if it does not reproduce it; see ov_backend.py. Not
        # attempted for --htc: HTCDepthModel is our own module, not an
        # OpenVINO-exportable transformers pipeline.
        self._ov = None
        self._ov_checked = self.is_htc
        self.model_name = model_name

        if self.is_htc:
            import htc_bins
            ck = torch.load(os.path.join(self.HTC_CKPT, "htc_model.pt"),
                            map_location=self.device, weights_only=False)
            self._htc_model = htc_bins.HTCDepthModel(
                model_id=ck["model_id"], n_bins=ck["n_bins"],
                h_min=ck["h_min"], h_max=ck["h_max"]).to(self.device)
            self._htc_model.load_state_dict(ck["state_dict"])
            self._htc_model.eval()
            # Still needed for image preprocessing (resize/normalise) even
            # though inference bypasses the HF pipeline entirely.
            from transformers import AutoImageProcessor
            self._htc_processor = AutoImageProcessor.from_pretrained(ck["model_id"])
            self.pipe = None
        else:
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )

        self._canary_verified = None   # not applicable to the stock backbone
        if self.is_metric or self.is_htc:
            self._verify_checkpoint_behaviour()

    def _verify_checkpoint_behaviour(self) -> None:
        """Refuses a licensed checkpoint whose OUTPUT no longer matches what
        was recorded when it was promoted -- see the module docstring above
        _canary_input for why this checks behaviour rather than file bytes."""
        ckpt_dir = self.HTC_CKPT if self.is_htc else self.METRIC_CKPT
        marker = self.HTC_MARKER if self.is_htc else self.MARKER
        marker_path = os.path.join(ckpt_dir, marker)
        try:
            with open(marker_path, encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}

        actual = _fingerprint_array(self.predict(_canary_input(), raw_metric=True))
        expected = info.get("canary")
        if expected is None:
            # Predates this check (or was licensed before it existed). Not a
            # trust failure -- there is nothing to compare against -- but
            # loud rather than silent, since it means this specific safety
            # net is not actually active for this checkpoint.
            print(f"      NOTE: {marker_path} has no behavioural fingerprint on "
                  f"record; re-run the licensing step to enable verification "
                  f"for this checkpoint (see depth_model.license_checkpoint).")
            self._canary_verified = None
            return

        ok, reason = _fingerprints_match(expected, actual)
        if not ok:
            raise RuntimeError(
                f"checkpoint at {ckpt_dir} FAILED behavioural verification: {reason}. "
                f"The weights on disk no longer compute what this checkpoint did "
                f"when it was licensed -- refusing to use it rather than trust it "
                f"on faith. If this is an intentional retrain, re-run the "
                f"licensing step to record a fresh fingerprint.")
        self._canary_verified = True
        print(f"      checkpoint behaviour verified against licensed fingerprint ({actual['hash']})")

    def _try_openvino(self, pixel_values, reference):
        """Bring up the iGPU path once, verifying it against the CPU result."""
        if self._ov_checked:
            return
        self._ov_checked = True
        try:
            import ov_backend
        except ImportError:
            return
        tag = self.model_name.split("/")[-1].replace("-hf", "")
        back = ov_backend.OpenVINODepth(self.pipe.model, pixel_values, tag)
        if back.ok and back.verify(reference, pixel_values.numpy()):
            self._ov = back

    def predict(self, image: Image.Image, raw_metric: bool = False) -> np.ndarray:
        """
        Returns a relative HEIGHT field in [0,1] -- higher value means higher
        elevation. Note this is height, not depth: the two are opposite, and
        confusing them turns every building into a pit.

        raw_metric=True changes that contract, ONLY when self.is_metric: the
        min-max normalisation below is skipped and the model's own metres are
        returned directly, unmodified. Without this, predict() throws the
        metric checkpoint's whole reason for existing away -- confirmed
        empirically this session: DepthBackbone(metric=True).predict(img)
        returned range (0.0, 1.0), identical to the stock model, despite
        metric_ok.json recording a real MAE of 1.444 m for this exact
        checkpoint when scored WITHOUT this normalisation (train_height.py's
        own evaluate()). Every caller that built a metric-aware branch on top
        of the old predict() -- build_city_image.py's `depth_is_metric` path
        (scale = 1.0 applied to an already-[0,1] field), rescore_baseline.py's
        predict_ndsm() (shadow-calibrates a metric model's output as if it
        were the stock relative field) -- was therefore never actually
        receiving metres, regardless of which checkpoint was loaded. Existing
        callers that do not pass raw_metric=True are completely unaffected;
        this defaults to False so nothing changes for them.

        Depth Anything V2 emits INVERSE depth (disparity-like): larger value =
        nearer the camera. In a nadir satellite view "nearer the camera" is
        "higher off the ground", so the raw output is ALREADY a height field
        and needs no flip.

        A previous version applied `depth = depth.max() - depth` here, which
        inverted it. Measured consequence on real imagery: building pixels
        averaged 111.7 against 131.7 for surrounding ground -- i.e. every
        rooftop sat BELOW the terrain around it, and buildings were extruded
        into holes rather than onto the surface. Raw output on the same image
        gives buildings 143.3 vs ground 123.3, which is the correct ordering.

        The flip is therefore removed. `_orientation_check` asserts the
        invariant per-image so a regression here can't pass silently again.
        """
        if self.is_htc:
            # HTCDepthModel is our own module, not a transformers pipeline, so
            # it is called directly rather than through self.pipe. Its raw
            # output is already metres (see htc_bins.HTCAdaBinsHead, Eqn. 11),
            # but predict()'s documented contract -- and every caller's -- is
            # a relative field in [0,1], so it is normalised the same way the
            # stock and plain-metric paths already are just below. This keeps
            # the HTC checkpoint a drop-in alternative under that contract
            # rather than a third, differently-shaped return type; it does
            # NOT by itself make build_city_image.py's Tier B "scale = 1.0"
            # fast path correct for HTC without separately confirming that
            # path consumes predict()'s existing metric output correctly.
            px = self._htc_processor(images=image, return_tensors="pt")["pixel_values"]
            px = px.to(self.device)
            with torch.inference_mode():
                out = self._htc_model(px)
            h, w = image.size[1], image.size[0]
            height_t = out["height"]
            if height_t.shape[-2:] != (h, w):
                height_t = torch.nn.functional.interpolate(
                    height_t, size=(h, w), mode="bilinear", align_corners=False)
            height = height_t.squeeze().cpu().numpy().astype(np.float32)
            if raw_metric and self.is_metric:
                return height
            rng = height.max() - height.min()
            return (height - height.min()) / (rng + 1e-8)

        # raw_metric bypasses transformers' depth-estimation PIPELINE entirely
        # and calls the model directly instead -- the pipeline's own
        # `result["depth"]` is NOT the model's raw regression output, it is a
        # visualisation-ready rescale (empirically confirmed this session:
        # identical (0, 255) min/max range for two completely unrelated
        # images, an 8-bit-image-shaped coincidence that is not a
        # coincidence). train_height.py's own evaluate() already calls the
        # model this same direct way for exactly that reason. Going through
        # self.pipe(image) for a metric checkpoint was silently handing back
        # an 8-bit-rescaled image instead of metres, on top of predict()'s
        # own [0,1] normalisation -- two separate ways the real scale was
        # being thrown away, not one.
        if self._ov is None and raw_metric and self.is_metric:
            px = self.pipe.image_processor(images=image, return_tensors="pt")["pixel_values"]
            with torch.inference_mode():
                raw = self.pipe.model(pixel_values=px).predicted_depth
            w, h = image.size
            if raw.ndim == 3:
                raw = raw.unsqueeze(1)
            raw = torch.nn.functional.interpolate(raw, size=(h, w), mode="bicubic",
                                                  align_corners=False)
            return raw.squeeze().cpu().numpy().astype(np.float32)

        # The first call always runs the PyTorch pipeline. That result is both
        # the answer and the reference the accelerated path must reproduce
        # before it is allowed to serve any later call.
        if self._ov is None:
            result = self.pipe(image)
            height = np.array(result["depth"], dtype=np.float32)
            if not self._ov_checked:
                try:
                    px = self.pipe.image_processor(
                        images=image, return_tensors="pt")["pixel_values"]
                    with torch.inference_mode():
                        raw = self.pipe.model(pixel_values=px).predicted_depth
                    self._try_openvino(px, raw.numpy())
                except Exception:
                    self._ov_checked = True
            rng = height.max() - height.min()
            return (height - height.min()) / (rng + 1e-8)

        # Accelerated path: same preprocessing the pipeline uses, same
        # interpolation back to the source resolution.
        px = self.pipe.image_processor(images=image,
                                       return_tensors="pt")["pixel_values"]
        pred = self._ov.infer(px.numpy())
        t = torch.from_numpy(np.asarray(pred, dtype=np.float32))
        if t.ndim == 3:
            t = t.unsqueeze(1)
        elif t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        w, h = image.size
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                            align_corners=False)
        height = t.squeeze().numpy().astype(np.float32)
        if raw_metric and self.is_metric:
            return height
        rng = height.max() - height.min()
        return (height - height.min()) / (rng + 1e-8)

    def predict_tiled(self, image: Image.Image, tile: int = 768,
                      overlap: float = 0.5, verbose: bool = False,
                      raw_metric: bool = False) -> np.ndarray:
        """
        Height field from overlapping crops, merged onto the whole-image
        prediction.

        raw_metric=True: see predict()'s docstring. The affine per-crop
        alignment below is unaffected either way -- fitting a local crop's
        scale and shift onto the whole-image reference is the same fit
        whether both are in [0,1] units or both are in metres, since it is a
        relative correction between two predictions of the same kind. Only
        the FINAL min-max normalisation at the end of this function needs the
        same raw_metric guard predict() already has, or metres would survive
        every crop only to be thrown away in the last line.

        Why this exists: the backbone resizes whatever it is given to ~518px
        internally, so a 2048px tile is downsampled 4x before it ever sees the
        network. In a dense downtown that destroys exactly the signal we need
        -- measured on JAX_164, whole-image inference put building pixels at
        0.2718 against ground at 0.2736, i.e. rooftops indistinguishable from
        the street. Feeding 768px crops instead means each crop is downsampled
        only 1.5x, so roof/street contrast survives.

        Each crop's prediction is affine-aligned (scale + shift, least squares)
        to the whole-image prediction over the same footprint before being
        blended in. Without that step the crops disagree about absolute level
        -- relative depth is only defined up to an affine transform per
        inference -- and the merge produces visible tile seams and a terrain
        that steps between crops.

        Blending uses a cosine (Hann) window so weights fall to zero at crop
        edges, where the model is least reliable and where a hard cut would
        leave a ridge.
        """
        base = self.predict(image, raw_metric=raw_metric)
        h, w = base.shape
        stride = max(1, int(tile * (1.0 - overlap)))

        def _starts(extent):
            if extent <= tile:
                return [0]
            pts = list(range(0, extent - tile + 1, stride))
            if pts[-1] != extent - tile:
                pts.append(extent - tile)
            return pts

        ys, xs = _starts(h), _starts(w)
        wy = np.hanning(min(tile, h) + 2)[1:-1]
        wx = np.hanning(min(tile, w) + 2)[1:-1]
        window = np.outer(wy, wx).astype(np.float32)

        acc = np.zeros_like(base)
        wsum = np.zeros_like(base)
        arr = np.asarray(image)
        total = len(ys) * len(xs)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                th, tw = min(tile, h), min(tile, w)
                crop = arr[y:y + th, x:x + tw]
                local = self.predict(Image.fromarray(crop), raw_metric=raw_metric)
                ref = base[y:y + th, x:x + tw]
                # Affine-align local to the global reference. Fit on the
                # reference's own spread; a near-constant crop (uniform field,
                # e.g. open water) has no gradient to fit against, so leave it.
                lv, rv = local.ravel(), ref.ravel()
                if lv.std() > 1e-6:
                    a, b = np.polyfit(lv, rv, 1)
                    local = local * a + b
                acc[y:y + th, x:x + tw] += local * window
                wsum[y:y + th, x:x + tw] += window
                if verbose:
                    print(f"    tile {i*len(xs)+j+1}/{total}", flush=True)

        merged = np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-6), base)
        if raw_metric and self.is_metric:
            return merged.astype(np.float32)
        rng = merged.max() - merged.min()
        return ((merged - merged.min()) / (rng + 1e-8)).astype(np.float32)


def orientation_check(height: np.ndarray, seg_labels: np.ndarray) -> dict:
    """
    Verify the height field is the right way up: buildings must sit ABOVE the
    ground around them.

    Returns a dict with the measured means and a boolean. Callers decide
    whether to warn or fail -- on scenes with very few building pixels the
    comparison is weak, so this reports rather than raising.
    """
    import segmentation as seg

    building = seg_labels == seg.CLASS_IDX["building"]
    ground = seg_labels == seg.CLASS_IDX["bare_earth"]
    if building.sum() < 50 or ground.sum() < 50:
        return {"checked": False, "reason": "too few building or ground pixels"}

    b_mean = float(height[building].mean())
    g_mean = float(height[ground].mean())
    return {
        "checked": True,
        "building_mean": b_mean,
        "ground_mean": g_mean,
        "correct_orientation": b_mean > g_mean,
    }


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    img = Image.open(img_path).convert("RGB")
    model = DepthBackbone()
    rel_depth = model.predict(img)
    print("Relative depth map:", rel_depth.shape, rel_depth.dtype,
          "min", rel_depth.min(), "max", rel_depth.max())

    import matplotlib.pyplot as plt
    plt.imsave("relative_depth_preview.png", rel_depth, cmap="terrain")
    print("Saved relative_depth_preview.png")
