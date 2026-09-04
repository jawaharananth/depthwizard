"""
Does shadow consistency recover a scale it was not given?

Synthetic first: build a height field, cast its shadows with a KNOWN sun, then
hand the refiner a deliberately wrong scale and see whether the shadows pull it
back. If it cannot solve a case with no noise and no segmentation error, it
cannot be trusted on a real one.
"""
import numpy as np, shadow_consistency as sc

gsd, el, az = 0.5, 35.0, 150.0
H = np.zeros((300, 300), np.float32)
rng = np.random.default_rng(0)
for _ in range(18):                     # a synthetic city of blocks
    y, x = rng.integers(20, 260, 2)
    hh, ww = rng.integers(20, 45, 2)
    H[y:y+hh, x:x+ww] = rng.uniform(8, 45)

TRUE = 1.0
obs = sc.cast_shadows(H * TRUE, gsd, el, az)
print(f"synthetic: {obs.mean()*100:.1f}% of pixels shadowed at the true scale")

for wrong in (0.6, 0.8, 1.25, 1.6):
    r = sc.refine_scale(H, obs, gsd, el, az, scale_guess=wrong)
    err = (r["scale"] - TRUE) / TRUE * 100
    print(f"  start {wrong:4.2f} -> recovered {r['scale']:.3f} "
          f"({err:+5.1f}% from truth)  IoU {r['iou']:.3f}  "
          f"constrained={r['constrained']}")

# Negative control: shadows that contain no information must be reported as
# unconstrained rather than yielding a confident wrong answer.
noise = np.zeros_like(obs); noise[::7, ::7] = True
r = sc.refine_scale(H, noise, gsd, el, az, scale_guess=1.0)
print(f"\nnegative control (meaningless shadow mask):")
print(f"  IoU {r['iou']:.3f}  constrained={r['constrained']}  "
      f"prominence {r['peak_prominence']:.4f}")
