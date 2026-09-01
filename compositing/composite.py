"""Stage 10: alpha compositing.

Straight (non-premultiplied) alpha blend. This exact formula was verified
correct and consistent in engine/AnyDoor/scripts/direct_composite.py this
session (no premultiplied/straight mismatch found) -- ported here unchanged,
generalized to canvas-sized inputs (i.e. meant to run on
transform.transform_subject's output, already placed/rotated/scaled onto a
canvas the size of the background).
"""
import numpy as np


def alpha_composite(background_rgb, fg_rgb, fg_alpha):
    """out = bg*(1-alpha) + fg*alpha, per-pixel, straight alpha.

    background_rgb: (H, W, 3) uint8.
    fg_rgb: (H, W, 3) uint8, same size as background_rgb.
    fg_alpha: (H, W) float32 in [0, 1], same H/W.
    """
    if background_rgb.shape[:2] != fg_rgb.shape[:2] or background_rgb.shape[:2] != fg_alpha.shape[:2]:
        raise ValueError("background_rgb, fg_rgb, fg_alpha must all share the same H, W")

    bg = background_rgb.astype(np.float32)
    fg = fg_rgb.astype(np.float32)
    a = fg_alpha.astype(np.float32)[..., None]
    out = bg * (1 - a) + fg * a
    return np.clip(out, 0, 255).astype(np.uint8)
