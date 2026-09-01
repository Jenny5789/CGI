import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from compositing.decontaminate import decontaminate


def make_contaminated_fixture():
    """Synthetic ground-truth fixture: a red square on a solid green
    background, with a boundary band where the observed color is a known
    alpha-blend of true red foreground and green background (simulating
    the color bleed a real soft-matted edge would have picked up from its
    original photo background).
    """
    size = 60
    true_fg_color = np.array([220, 30, 30], dtype=np.float32)  # red
    bg_color = np.array([30, 200, 30], dtype=np.float32)  # green

    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[10:50, 10:50] = 1.0  # solid interior
    # Boundary band: a 3px ring around the square with partial alpha.
    for i, a in enumerate([0.25, 0.5, 0.75]):
        ring = i + 1
        alpha[10 - ring, 10 - ring : 50 + ring] = a
        alpha[50 + ring - 1, 10 - ring : 50 + ring] = a
        alpha[10 - ring : 50 + ring, 10 - ring] = a
        alpha[10 - ring : 50 + ring, 50 + ring - 1] = a

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[:, :] = bg_color
    a3 = alpha[..., None]
    # Observed = true contamination process: alpha*fg + (1-alpha)*bg.
    rgb = a3 * true_fg_color + (1 - a3) * bg_color
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, alpha, true_fg_color


def test_decontaminate_recovers_true_foreground_color_at_boundary():
    rgb, alpha, true_fg_color = make_contaminated_fixture()
    out = decontaminate(rgb, alpha)

    boundary_mask = (alpha > 0.02) & (alpha < 0.98)
    boundary_pixels = out[boundary_mask].astype(np.float32)
    mean_recovered = boundary_pixels.mean(axis=0)

    # Recovered color should be much closer to true red than the raw
    # (still-green-contaminated) observed color was.
    raw_boundary_mean = rgb[boundary_mask].astype(np.float32).mean(axis=0)
    err_before = np.abs(raw_boundary_mean - true_fg_color).sum()
    err_after = np.abs(mean_recovered - true_fg_color).sum()
    assert err_after < err_before


def test_solid_interior_left_unchanged():
    rgb, alpha, _ = make_contaminated_fixture()
    out = decontaminate(rgb, alpha)
    solid_mask = alpha >= 0.98
    assert np.array_equal(out[solid_mask], rgb[solid_mask])


def test_output_shape_and_dtype_preserved():
    rgb, alpha, _ = make_contaminated_fixture()
    out = decontaminate(rgb, alpha)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
