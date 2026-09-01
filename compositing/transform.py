"""Stage 5: deterministic subject transform.

Pure image-processing, no models. Takes a foreground (RGB uint8) + alpha
(float32 in [0,1], same H/W as the RGB) and places it onto a canvas the size
of the target background according to explicit, controllable geometry:
position, scale, rotation, horizontal flip, and a grounding point.

This generalizes the bbox/pad/crop primitives in
engine/AnyDoor/datasets/data_utils.py (get_bbox_from_mask, pad_to_square) but
strips their internal randomness (AnyDoor's expand_bbox draws a random ratio
via np.random on every call, which makes its own placement non-deterministic)
and adds rotation/flip, which AnyDoor's own pipeline does not expose (flip
exists only as an unused/dead training-augmentation stub there).

A later pipeline stage is free to adapt *appearance* (color, sharpness,
shadows) but must not need to touch anything in this module's output shape
or position -- geometry is decided once, here, and locked.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def get_bbox_from_alpha(alpha, threshold=0.02):
    """Tight bounding box (y1, y2, x1, x2), half-open, of alpha > threshold.

    Ported from AnyDoor's get_bbox_from_mask (data_utils.py) -- same
    algorithm, generalized to a continuous alpha channel instead of a binary
    mask (thresholded only to decide bbox *extent*, not to discard the real
    alpha values themselves).
    """
    mask = alpha > threshold
    if mask.sum() < 10:
        raise ValueError("alpha channel is (near-)empty -- nothing to transform")
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(y2) + 1, int(x1), int(x2) + 1


@dataclass
class Placement:
    """Explicit, deterministic placement parameters.

    position: (x, y) pixel coordinates IN THE BACKGROUND CANVAS where the
        grounding_point should land.
    scale: multiplier applied to the foreground's own (already tight-cropped)
        pixel size. 1.0 = native extracted size.
    rotation_deg: counter-clockwise rotation in degrees, applied about the
        grounding point.
    flip_horizontal: mirror the foreground left-right before scale/rotation.
    grounding_point: (gx, gy) in NORMALIZED [0,1] coordinates within the
        foreground's own tight bbox -- e.g. (0.5, 1.0) is bottom-center,
        the natural "feet" contact point for a standing subject.
    """

    position: tuple
    scale: float = 1.0
    rotation_deg: float = 0.0
    flip_horizontal: bool = False
    grounding_point: tuple = (0.5, 1.0)


@dataclass
class TransformResult:
    rgb: np.ndarray          # canvas-sized (H, W, 3) uint8
    alpha: np.ndarray        # canvas-sized (H, W) float32 in [0, 1]
    bbox: tuple              # (y1, y2, x1, x2) of the placed subject on the canvas
    grounding_point_xy: tuple  # where the grounding point actually landed (should equal position)


def transform_subject(fg_rgb, fg_alpha, canvas_size, placement: Placement) -> TransformResult:
    """Place a segmented/matted subject onto a canvas per `placement`.

    fg_rgb: (h, w, 3) uint8, arbitrary background outside the subject (it's
        masked by fg_alpha at composite time -- this stage does not require
        fg_rgb to already be premultiplied or background-cleaned).
    fg_alpha: (h, w) float32 in [0, 1], same h/w as fg_rgb.
    canvas_size: (width, height) of the target background.
    """
    if fg_rgb.shape[:2] != fg_alpha.shape[:2]:
        raise ValueError(
            f"fg_rgb {fg_rgb.shape[:2]} and fg_alpha {fg_alpha.shape[:2]} must match"
        )
    canvas_w, canvas_h = canvas_size

    y1, y2, x1, x2 = get_bbox_from_alpha(fg_alpha)
    crop_rgb = fg_rgb[y1:y2, x1:x2]
    crop_alpha = fg_alpha[y1:y2, x1:x2]
    crop_h, crop_w = crop_alpha.shape

    if placement.flip_horizontal:
        crop_rgb = cv2.flip(crop_rgb, 1)
        crop_alpha = cv2.flip(crop_alpha, 1)

    gx = placement.grounding_point[0] * crop_w
    gy = placement.grounding_point[1] * crop_h

    # One affine transform: rotate + scale about the grounding point, then
    # translate so the grounding point lands exactly at `placement.position`
    # in canvas coordinates. Doing this as a single warpAffine (rather than
    # crop->paste by hand) keeps sub-pixel rotation/scale mathematically
    # exact instead of accumulating separate rounding errors per step.
    M = cv2.getRotationMatrix2D((gx, gy), placement.rotation_deg, placement.scale)
    M[0, 2] += placement.position[0] - gx
    M[1, 2] += placement.position[1] - gy

    canvas_rgb = cv2.warpAffine(
        crop_rgb, M, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    canvas_alpha = cv2.warpAffine(
        crop_alpha, M, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    canvas_alpha = np.clip(canvas_alpha, 0.0, 1.0)

    placed_bbox = get_bbox_from_alpha(canvas_alpha)
    return TransformResult(
        rgb=canvas_rgb,
        alpha=canvas_alpha,
        bbox=placed_bbox,
        grounding_point_xy=tuple(placement.position),
    )
