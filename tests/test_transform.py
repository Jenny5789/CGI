import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from compositing.transform import Placement, get_bbox_from_alpha, transform_subject


def make_square_subject(size=40, color=(200, 50, 50)):
    """A solid square subject on a fully-transparent (alpha=0) canvas."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.float32)
    rgb[:, :] = color
    alpha[:, :] = 1.0
    return rgb, alpha


def test_bbox_matches_known_extent():
    alpha = np.zeros((100, 100), dtype=np.float32)
    alpha[20:50, 30:70] = 1.0  # rows 20-49, cols 30-69
    y1, y2, x1, x2 = get_bbox_from_alpha(alpha)
    assert (y1, y2, x1, x2) == (20, 50, 30, 70)


def test_bbox_raises_on_empty_alpha():
    alpha = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        get_bbox_from_alpha(alpha)


def test_identity_placement_lands_grounding_point_exactly():
    rgb, alpha = make_square_subject(size=40)
    placement = Placement(position=(100, 150), scale=1.0, rotation_deg=0.0, grounding_point=(0.5, 1.0))
    result = transform_subject(rgb, alpha, canvas_size=(300, 300), placement=placement)

    # grounding point = bottom-center of the 40x40 square -> should land at
    # exactly (100, 150) on the canvas.
    assert result.alpha[150, 100] > 0.9 or result.alpha[149, 100] > 0.9
    y1, y2, x1, x2 = result.bbox
    cx = (x1 + x2) / 2
    assert abs(cx - 100) < 2
    assert abs(y2 - 150) < 2  # bottom edge of the placed subject


def test_scale_doubles_placed_extent():
    rgb, alpha = make_square_subject(size=40)
    p1 = Placement(position=(150, 150), scale=1.0)
    p2 = Placement(position=(150, 150), scale=2.0)
    r1 = transform_subject(rgb, alpha, (300, 300), p1)
    r2 = transform_subject(rgb, alpha, (300, 300), p2)

    h1 = r1.bbox[1] - r1.bbox[0]
    h2 = r2.bbox[1] - r2.bbox[0]
    assert abs(h2 - 2 * h1) <= 2  # allow 1-2px rounding from the affine warp


def test_flip_horizontal_mirrors_asymmetric_subject():
    # Left half red, right half blue -- flipping should swap which side is
    # which color, verifying flip actually happened (not a no-op).
    size = 40
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, : size // 2] = (255, 0, 0)
    rgb[:, size // 2 :] = (0, 0, 255)
    alpha = np.ones((size, size), dtype=np.float32)

    p_normal = Placement(position=(100, 100), grounding_point=(0.5, 0.5))
    p_flipped = Placement(position=(100, 100), grounding_point=(0.5, 0.5), flip_horizontal=True)
    r_normal = transform_subject(rgb, alpha, (200, 200), p_normal)
    r_flipped = transform_subject(rgb, alpha, (200, 200), p_flipped)

    # Sample a point clearly on the left side of the placed subject.
    left_x = 100 - size // 4
    assert tuple(r_normal.rgb[100, left_x]) == (255, 0, 0)
    assert tuple(r_flipped.rgb[100, left_x]) == (0, 0, 255)


def test_rotation_90_swaps_width_and_height():
    rgb = np.full((20, 60, 3), 255, dtype=np.uint8)  # wide rectangle
    alpha = np.ones((20, 60), dtype=np.float32)
    placement = Placement(position=(150, 150), rotation_deg=90, grounding_point=(0.5, 0.5))
    result = transform_subject(rgb, alpha, (300, 300), placement)
    y1, y2, x1, x2 = result.bbox
    placed_h, placed_w = y2 - y1, x2 - x1
    # After a 90-degree rotation, the originally-wide (60x20) rect should be
    # tall-and-narrow (~20x60).
    assert placed_h > placed_w
    assert abs(placed_h - 60) <= 3
    assert abs(placed_w - 20) <= 3


def test_mismatched_shapes_raise():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    alpha = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        transform_subject(rgb, alpha, (100, 100), Placement(position=(50, 50)))
