"""Stage 3: alpha matting.

ViTMatte (trimap-based, hair/fur-capable), run in the same isolated
.venv-seg environment as compositing.segmentation for the same reason (needs
a recent transformers/torch, incompatible with AnyDoor's pinned stack in the
main .venv). The trimap ViTMatte needs is generated automatically from the
segmentation mask (compositing.segmentation.segment's output) by eroding for
definite-foreground / definite-background and leaving an unknown band for
ViTMatte to resolve -- callers never draw a trimap by hand.

This is a genuinely different operation from segmentation (see
docs/architecture.md section B): segmentation gives a hard region, this
gives a continuous alpha with real sub-pixel/strand structure at the
boundary. rembg's bundled `alpha_matting` mode was tried earlier this
session as a shortcut for this and failed catastrophically (mostly erased
the subject) -- this module exists because that shortcut doesn't work.
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEG_VENV_PYTHON = os.path.join(_ROOT, ".venv-seg", "Scripts", "python.exe")
_MATTE_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_matte_worker.py")


def matte(image_rgb, mask_binary, unknown_band_px=15):
    """Returns a continuous alpha (float32, [0, 1]) the same H/W as image_rgb.

    image_rgb: (H, W, 3) uint8 -- should be the ORIGINAL, un-cropped photo
        (or at least a crop with real background context around the
        subject), since ViTMatte needs real pixel context in the unknown
        boundary band to resolve fine structure correctly.
    mask_binary: (H, W) uint8, 0/1 -- typically compositing.segmentation.segment's
        output. Only used to derive the trimap; ViTMatte itself produces the
        actual alpha values.
    """
    if not os.path.exists(_SEG_VENV_PYTHON):
        raise RuntimeError(
            f"Isolated segmentation venv not found at {_SEG_VENV_PYTHON}. "
            "Run: py -3.10 -m venv .venv-seg, then install the matting deps into it."
        )

    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "image.png")
        mask_path = os.path.join(tmp, "mask.png")
        alpha_path = os.path.join(tmp, "alpha.png")
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask_binary * 255)

        cmd = [_SEG_VENV_PYTHON, _MATTE_WORKER, image_path, mask_path, alpha_path, str(unknown_band_px)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ViTMatte worker failed:\n{result.stderr}")

        alpha_u8 = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        return alpha_u8.astype(np.float32) / 255.0
