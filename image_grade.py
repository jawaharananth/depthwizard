"""Display grading for satellite imagery, with the atmospheric cast removed first.

Satellite imagery arrives with a blue cast. It is not a white-balance error in
the usual sense: Rayleigh scattering is stronger at short wavelengths, so the
blue channel carries atmospheric path radiance the ground did not reflect.
Measured on the Jacksonville upload the source sits at R-B = -27.6.

The previous grade made that worse rather than better. It boosted saturation by
1.12 -- which amplifies whatever cast is already present -- and then applied a
deliberate cool multiplier of [0.94, 0.985, 1.06], cutting red and lifting blue.
The result measured R-B = -48.3 and saturation 49.9 against the source's 30.7:
a 75% stronger cast and a scene that renders as grey-blue monochrome instead of
a city with brick, render, membrane and painted-metal roofs.

So the cast is neutralised BEFORE any stylistic adjustment, using the standard
remote-sensing approach: align the channels on a robust bright reference rather
than on their means. A mean-based (grey-world) correction is dragged by whatever
dominates the frame -- vegetation biases green, water biases blue -- while the
bright end of a built scene is dominated by concrete, render and rooftops, which
really are close to neutral.
"""

import numpy as np


# Percentile used as the neutral reference.
#
# 75, not the usual 95-99. Measured on the Jacksonville upload, 9.7% of pixels
# are at or near 255 in some channel, and the per-channel percentiles read
# [240, 251, 255] at p97 -- the bright end is CLIPPED, so it carries almost no
# colour information and a reference taken there barely moves the cast (R-B
# went -27.4 to only -18.9 at full strength).
#
# The cast is steady at about -20 R-B from p50 to p90, so any unclipped bright
# band measures it correctly. p75 sits on concrete, render and rooftops --
# genuinely near-neutral materials -- while staying clear of the clipping.
WHITE_PCTILE = 75.0

# Pixels at or above this in ANY channel are excluded from the reference: a
# clipped pixel's colour is an artefact of the clip, not of the surface.
CLIP_GUARD = 250.0

# How completely to remove the cast. Full correction can look sterile and
# discards the fact that the scene really was photographed through atmosphere;
# this keeps a trace of it.
CAST_STRENGTH = 0.85

SATURATION = 1.06        # was 1.12, which amplified the cast it should not have
CONTRAST = 1.14
PIVOT = 0.42
LIFT = 0.46


def white_balance(rgb_u8: np.ndarray, strength: float = CAST_STRENGTH) -> np.ndarray:
    """Neutralise the atmospheric colour cast. Input and output are uint8 RGB."""
    a = rgb_u8.astype(np.float32)
    unclipped = a.max(axis=2) < CLIP_GUARD
    if unclipped.sum() < 1000:
        unclipped = np.ones(a.shape[:2], bool)
    ref = np.array([np.percentile(a[:, :, c][unclipped], WHITE_PCTILE)
                    for c in range(3)], dtype=np.float32)
    ref = np.maximum(ref, 1.0)
    target = float(ref.mean())
    gain = target / ref
    # Blend toward full correction rather than applying it outright.
    gain = 1.0 + (gain - 1.0) * float(strength)
    return np.clip(a * gain, 0, 255).astype(np.uint8)


def grade(rgb_u8: np.ndarray) -> np.ndarray:
    """White balance, then a restrained display grade. Returns uint8 RGB.

    Nothing measured is derived from this: the texture is decoration on the
    ground mesh, and no height, footprint or metric reads it.
    """
    g = white_balance(rgb_u8).astype(np.float32) / 255.0
    lum = (g * np.array([0.299, 0.587, 0.114], np.float32)).sum(axis=2, keepdims=True)
    g = lum + (g - lum) * SATURATION
    g = np.clip((g - PIVOT) * CONTRAST + LIFT, 0, 1)
    return (g * 255).astype(np.uint8)
