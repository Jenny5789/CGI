"""Stage 2: segmentation.

SAM2 (promptable: a point or box), run in an isolated environment
(.venv-seg) because it requires torch>=2.5 while the main .venv is pinned to
torch==2.0.0 for AnyDoor's xformers build. Mixing the two in one venv
corrupted the shared environment once already this session (torch, numpy,
and setuptools all got silently force-upgraded installing SAM2 directly) --
see compositing/README.md. This module talks to that isolated environment
over a subprocess + temp-file boundary so the rest of compositing/ stays
venv-agnostic and callers never have to think about the split.

Wraps the raw SAM2 mask with a deterministic connected-component cleanup:
regression test for the "tight bbox got wrecked by a stray noise blob"
failure class hit earlier this session while segmenting the bear photo.
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEG_VENV_PYTHON = os.path.join(_ROOT, ".venv-seg", "Scripts", "python.exe")
_SEG_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_seg_worker.py")


def _largest_component(mask_binary, close_kernel_px=15):
    """Keep only the largest connected component -- discards stray noise
    blobs that would otherwise blow out a downstream tight-bbox calculation
    far past the actual subject (the exact failure mode found and confirmed
    absent-by-luck on the bear test photo this session; guarding against it
    generally rather than relying on that luck again).

    Connectivity is judged on a morphologically-CLOSED copy of the mask, not
    the raw mask directly: a textured part of the real object (e.g. a plant
    pot's soil, mottled fur) can come back from SAM2 full of small holes,
    which fragments it into many tiny "components" under strict pixel
    connectivity -- naive largest-component picking then discards the whole
    textured part as noise (confirmed: a pot connected to its plant's stem
    in SAM2's raw mask was entirely dropped this way). Closing bridges gaps
    up to ~close_kernel_px to find which raw-mask pixels belong to the same
    real region; the actual returned pixels are still exactly the original
    mask's (intersected with that region), so this never invents new area,
    it just stops texture-vs-noise from being misjudged as separate blobs.
    """
    closed = cv2.morphologyEx(
        mask_binary, cv2.MORPH_CLOSE, np.ones((close_kernel_px, close_kernel_px), np.uint8)
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return mask_binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    region = (labels == largest_label).astype(np.uint8)
    return mask_binary & region


def segment(image_rgb, point=None, box=None):
    """Returns a binary mask (uint8, 0/1) the same H/W as image_rgb.

    Exactly one of `point` (x, y) or `box` (x1, y1, x2, y2) must be given --
    SAM2 is promptable, not a from-scratch detector (see docs/architecture.md
    section D for why: neither AnyDoor's nor a from-scratch approach was
    adopted for this stage).
    """
    if not os.path.exists(_SEG_VENV_PYTHON):
        raise RuntimeError(
            f"Isolated segmentation venv not found at {_SEG_VENV_PYTHON}. "
            "Run: py -3.10 -m venv .venv-seg, then install sam2 into it."
        )
    if (point is None) == (box is None):
        raise ValueError("segment() needs exactly one of point= or box=")

    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "image.png")
        mask_path = os.path.join(tmp, "mask.png")
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        cmd = [_SEG_VENV_PYTHON, _SEG_WORKER, image_path, mask_path]
        if point is not None:
            cmd += ["--point", f"{point[0]},{point[1]}"]
        else:
            cmd += ["--box", ",".join(str(v) for v in box)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SAM2 worker failed:\n{result.stderr}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 127).astype(np.uint8)
        return _largest_component(mask_binary)
