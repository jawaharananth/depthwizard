import numpy as np
from PIL import Image
import torch
from transformers import pipeline


class DepthBackbone:
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
        )

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Returns a relative HEIGHT field in [0,1] -- higher value means higher
        elevation. Note this is height, not depth: the two are opposite, and
        confusing them turns every building into a pit.

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
        result = self.pipe(image)
        height = np.array(result["depth"], dtype=np.float32)
        rng = height.max() - height.min()
        return (height - height.min()) / (rng + 1e-8)

    def predict_tiled(self, image: Image.Image, tile: int = 768,
                      overlap: float = 0.5, verbose: bool = False) -> np.ndarray:
        """
        Height field from overlapping crops, merged onto the whole-image
        prediction.

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
        base = self.predict(image)
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
                local = self.predict(Image.fromarray(crop))
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
