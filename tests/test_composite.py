import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from compositing.composite import alpha_composite


def test_full_opacity_shows_only_foreground():
    bg = np.full((10, 10, 3), 100, dtype=np.uint8)
    fg = np.full((10, 10, 3), 200, dtype=np.uint8)
    alpha = np.ones((10, 10), dtype=np.float32)
    out = alpha_composite(bg, fg, alpha)
    assert np.all(out == 200)


def test_zero_opacity_shows_only_background():
    bg = np.full((10, 10, 3), 100, dtype=np.uint8)
    fg = np.full((10, 10, 3), 200, dtype=np.uint8)
    alpha = np.zeros((10, 10), dtype=np.float32)
    out = alpha_composite(bg, fg, alpha)
    assert np.all(out == 100)


def test_half_opacity_is_exact_average():
    bg = np.full((4, 4, 3), 0, dtype=np.uint8)
    fg = np.full((4, 4, 3), 200, dtype=np.uint8)
    alpha = np.full((4, 4), 0.5, dtype=np.float32)
    out = alpha_composite(bg, fg, alpha)
    assert np.all(out == 100)


def test_hand_computed_blend_value():
    bg = np.array([[[0, 100, 200]]], dtype=np.uint8)
    fg = np.array([[[255, 0, 0]]], dtype=np.uint8)
    alpha = np.array([[0.25]], dtype=np.float32)
    out = alpha_composite(bg, fg, alpha)
    # out = bg*0.75 + fg*0.25
    expected = np.array([[[0 * 0.75 + 255 * 0.25, 100 * 0.75 + 0, 200 * 0.75 + 0]]])
    assert np.allclose(out.astype(np.float32), expected, atol=1.0)


def test_shape_mismatch_raises():
    bg = np.zeros((10, 10, 3), dtype=np.uint8)
    fg = np.zeros((5, 5, 3), dtype=np.uint8)
    alpha = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        alpha_composite(bg, fg, alpha)


def test_gradient_alpha_round_trip_no_premultiplied_artifact():
    # A synthetic soft edge: alpha ramps 0->1 across a row. If straight vs
    # premultiplied alpha were being mixed up, this would produce a visible
    # dark or bright fringe (values outside the [bg, fg] range) at
    # intermediate alpha values.
    bg = np.full((1, 20, 3), 10, dtype=np.uint8)
    fg = np.full((1, 20, 3), 250, dtype=np.uint8)
    alpha = np.linspace(0, 1, 20, dtype=np.float32).reshape(1, 20)
    out = alpha_composite(bg, fg, alpha).astype(np.float32)
    assert np.all(out >= 10 - 1)
    assert np.all(out <= 250 + 1)
    # Strictly increasing along the ramp (monotonic blend, no fringe dip/spike).
    assert np.all(np.diff(out[0, :, 0]) >= -0.5)
