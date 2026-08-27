"""
Detect and neutralise map-overlay graphics painted onto satellite imagery.

WHY THIS IS NEEDED

Screenshots taken from a mapping application often carry pins, labels, route
lines, POI icons and watermarks. These are opaque graphics composited over the
imagery, but nothing downstream knows that:

  * the depth backbone sees a high-contrast blob and gives it height;
  * segmentation sees saturated colour and hard edges and calls it a building;
  * prism extrusion then builds a solid block standing on nothing;
  * the vehicle detector's size and elongation gates fit a map pin almost
    exactly;
  * and the real roof underneath is hidden either way.

The result is phantom structures that look confident and are pure artefact.

HOW THEY ARE TOLD APART

Overlay graphics differ from imagery in a way that is measurable rather than
stylistic. Satellite and aerial imagery of built environments is close to
neutral -- measured on the DFC2019 Jacksonville orthos, the red, green and blue
channels sit within a few levels of each other at every percentile. UI graphics
are the opposite: strongly saturated, and painted in flat uniform colour with
almost no internal texture.

So a region is flagged when it is simultaneously:

  * highly saturated (far from grey), AND
  * internally flat (a painted fill, not a photographed surface).

Both conditions are required. Saturation alone would flag a blue swimming pool,
a red-tile roof or a sports pitch; flatness alone would flag tarmac and water.
Together they describe a painted graphic and very little else.

Vegetation is excluded explicitly: healthy canopy is both green-saturated and
smooth at some scales, and it is the one natural surface that can otherwise
trip both tests.

WHAT IS DONE WITH THEM

Detected pixels are removed from the building mask so nothing is extruded from
them, and the texture is inpainted from surrounding pixels so the scene does not
show a floating icon. The count and coverage are returned so a caller can warn
-- or refuse -- rather than silently producing a scene built partly from UI.
"""
import numpy as np
import cv2


def detect_overlays(image_np: np.ndarray, veg_mask: np.ndarray = None,
                    sat_percentile: float = 97.0,
                    min_sat: int = 90,
                    max_area_frac: float = 0.02) -> dict:
    """
    Find painted map graphics.

    sat_percentile adapts the saturation cut to the image's own distribution, so
    a naturally colourful scene is not wholesale flagged; min_sat is an absolute
    floor beneath which nothing counts as a painted graphic however unusual it is
    for that image.

    max_area_frac bounds how large a single overlay component may be. Map pins,
    labels and icons are small relative to the frame; a huge saturated flat
    region is far more likely to be a real surface (a lake, a painted court)
    and is left alone.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    texture = np.clip(cv2.blur(gray * gray, (5, 5)) - mean * mean, 0, None)

    sat_cut = max(float(np.percentile(sat, sat_percentile)), float(min_sat))
    tex_cut = float(np.percentile(texture, 35))

    # Two rules, not one.
    #
    # RULE 1 -- saturated AND flat: pin bodies, route lines, filled icons.
    # RULE 2 -- extremely saturated, whatever the local texture: TEXT labels.
    #   Label strokes are only a few pixels wide, so almost every pixel of a
    #   glyph is an edge and the flatness test rejects all of it. Measured: with
    #   rule 1 alone, painted "Main Building" text survived cleaning untouched.
    #   Real imagery essentially never reaches this saturation -- the DFC and
    #   drone orthos here are near-neutral -- so the absolute cut is safe.
    strong_cut = max(float(np.percentile(sat, 99.5)), 150.0)
    rule_flat = (sat > sat_cut) & (texture < tex_cut)
    rule_strong = sat > strong_cut
    candidate = (rule_flat | rule_strong) & (val > 40)

    if veg_mask is not None:
        # Healthy vegetation is green-saturated and smooth at this scale; it is
        # the one natural surface that satisfies both tests, so it is excluded
        # rather than being inpainted away.
        candidate &= ~veg_mask

    # Close BEFORE opening. Letters are separate components until joined, so
    # opening first erases the thin strokes that closing would have merged into
    # a word-sized region, and the area filter then never sees the label at all.
    m = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    total_px = image_np.shape[0] * image_np.shape[1]
    keep = np.zeros(n, dtype=bool)
    kept = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 12:
            continue                       # speckle, not a graphic
        if area > max_area_frac * total_px:
            continue                       # too large to be an icon or label
        keep[i] = True
        kept += 1

    mask = keep[lab]
    return {
        "mask": mask,
        "count": kept,
        "coverage": float(mask.mean()),
        "sat_cut": round(sat_cut, 1),
    }


def clean(image_np: np.ndarray, veg_mask: np.ndarray = None) -> tuple:
    """
    Remove overlay graphics from an image.

    Returns (cleaned_image, info). Inpainting uses the surrounding imagery, so a
    pin sitting on a roof is replaced by roof rather than by a grey patch. This
    restores plausible appearance; it does NOT recover what the graphic hid, and
    the affected region should be treated as having no evidence rather than as
    observed surface.
    """
    info = detect_overlays(image_np, veg_mask=veg_mask)
    if info["count"] == 0:
        return image_np, info

    # Dilate before inpainting: an icon usually carries a soft anti-aliased
    # fringe and often a drop shadow, and leaving those behind produces a halo
    # exactly where a building edge would be detected.
    m = cv2.dilate(info["mask"].astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cleaned = cv2.inpaint(image_np, m, 5, cv2.INPAINT_TELEA)
    info["mask"] = m.astype(bool)
    return cleaned, info


if __name__ == "__main__":
    import sys
    from PIL import Image
    path = sys.argv[1]
    img = np.array(Image.open(path).convert("RGB"))
    out, info = clean(img)
    print(f"{path}: {info['count']} overlay components, "
          f"{info['coverage']*100:.3f}% of pixels (saturation cut {info['sat_cut']})")
    if info["count"]:
        Image.fromarray(out).save("overlay_cleaned.png")
        print("wrote overlay_cleaned.png")
