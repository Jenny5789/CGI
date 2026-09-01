"""Stage 4: edge decontamination.

Removes the ORIGINAL source photo's background color bleed from
semi-transparent boundary pixels (hair/fur/fine-structure edges). This is
conceptually the opposite of, and independent from,
engine/AnyDoor/scripts/direct_composite.py's apply_edge_light_bleed, which
deliberately blends the NEW target background into the rim to fake a
natural-looking seam -- that's a stylistic addition for the pasted-look
problem, not a fix for contamination already present in the source pixels.
Neither AnyDoor nor VACE do this (confirmed absent in both during the
architecture review); it did not exist anywhere in this codebase before.

Deterministic image processing, no model:
1. Estimate the "background, as if the subject weren't there" by masking out
   the foreground and inpainting the hole -- this gives a local (spatially
   varying, not single-color) background-color estimate at every foreground
   pixel location, which matters because real backgrounds are rarely flat
   (bushes, sky gradient, etc.).
2. For each boundary pixel with alpha strictly between 0 and 1, invert the
   standard alpha-compositing equation
       observed = alpha * true_fg + (1 - alpha) * local_bg
   to solve for true_fg, undoing the background's contribution.
3. Leave alpha ~1 (solid interior) and alpha ~0 (pure background) pixels
   untouched -- only the genuine boundary band is corrected.
"""
import cv2
import numpy as np


def decontaminate(rgb, alpha, solid_threshold=0.98, transparent_threshold=0.02):
    """Returns a decontaminated copy of `rgb` (uint8, same shape).

    rgb: (H, W, 3) uint8 -- the ORIGINAL crop (as extracted from its source
        photo), not yet placed on any new background.
    alpha: (H, W) float32 in [0, 1] -- typically the output of a matting
        stage (compositing.matting), which is what makes this step
        meaningful: a hard binary mask has no boundary band to decontaminate.
    """
    h, w = alpha.shape
    boundary_mask = (alpha > transparent_threshold) & (alpha < solid_threshold)

    # Inpaint away the foreground (interior + boundary), filling that hole
    # from the surrounding genuine background pixels (alpha ~0) -- this
    # gives a local, spatially-varying background-color estimate at every
    # foreground pixel location, driven by real background context rather
    # than by the subject's own (already-contaminated) edge pixels.
    inpaint_hole = ((alpha > transparent_threshold).astype(np.uint8)) * 255
    local_bg = cv2.inpaint(rgb, inpaint_hole, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    out = rgb.astype(np.float32).copy()
    a = alpha[..., None]
    local_bg_f = local_bg.astype(np.float32)
    observed = rgb.astype(np.float32)

    # true_fg = (observed - (1-alpha)*local_bg) / alpha, only where alpha is
    # large enough that division stays numerically stable.
    safe_a = np.clip(a, 0.15, 1.0)
    decontaminated = (observed - (1 - a) * local_bg_f) / safe_a

    boundary = boundary_mask[..., None]
    out = np.where(boundary, decontaminated, out)
    return np.clip(out, 0, 255).astype(np.uint8)
