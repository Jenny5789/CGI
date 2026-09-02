"""Stage 7: foreground appearance adaptation (color/exposure matching).

The highest-priority gap identified in docs/architecture.md: extraction and
placement can be geometrically perfect and the result still reads as
"pasted" if the subject's color/exposure doesn't match its new
surroundings -- the subject was photographed under different light. This
stage nudges color and exposure toward the target scene without touching
shape, identity, or fine texture (that's transform.py's and matting.py's
job respectively; this stage must not undo either).

Deterministic (Reinhard-style LAB color-statistics transfer), matching
docs/architecture.md's stage classification -- no model needed for this.
A `strength` parameter keeps it a *nudge*, not a repaint: at strength=1.0
the subject's color distribution would fully take on the background
context's mean/spread, which usually over-corrects (e.g. a genuinely brown
bear would go color-neutral-gray if the background happens to be gray
sand) -- default is a partial pull, matching the "adaptation, not
regeneration" principle.
"""
import cv2
import numpy as np

from compositing.scene_analysis import analyze_region, context_ring_mask


def match_appearance(fg_rgb, fg_alpha, bg_rgb, bbox, strength=0.5, margin_frac=0.5):
    """Returns an adapted copy of fg_rgb (uint8, same shape).

    fg_rgb, fg_alpha: canvas-sized, e.g. compositing.transform.TransformResult's
        .rgb / .alpha (already placed, not yet composited).
    bg_rgb: the background canvas (same size).
    bbox: the placed subject's bbox (transform.TransformResult.bbox).
    strength: 0 = no change, 1 = fully match the background context's LAB
        mean/spread.
    """
    if strength <= 0:
        return fg_rgb.copy()

    solid_mask = fg_alpha > 0.5
    if solid_mask.sum() < 10:
        return fg_rgb.copy()

    context_mask = context_ring_mask(bg_rgb.shape, bbox, margin_frac, exclude_mask=None)
    if context_mask.sum() < 10:
        return fg_rgb.copy()

    target = analyze_region(bg_rgb, context_mask)
    source = analyze_region(fg_rgb, solid_mask)

    fg_lab = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Reinhard transfer: recenter+rescale the subject's LAB distribution
    # onto the background context's, per channel.
    ratio = target.std_lab / source.std_lab
    transferred = (fg_lab - source.mean_lab) * ratio + target.mean_lab

    blended_lab = fg_lab * (1 - strength) + transferred * strength
    blended_lab = np.clip(blended_lab, 0, 255).astype(np.uint8)
    adapted_rgb = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2RGB)

    out = fg_rgb.copy()
    mask3 = (fg_alpha > 0.02)[..., None]
    out = np.where(mask3, adapted_rgb, out)
    return out
