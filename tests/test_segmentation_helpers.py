"""Unit tests for segmentation.py's deterministic post-processing
(no model/venv needed -- unlike test_segmentation_matting.py, this always
runs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from compositing.segmentation import _largest_component


def test_drops_a_genuinely_separate_noise_blob():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 1  # main object, 900px
    mask[80:85, 80:85] = 1  # far-away stray noise blob, 25px
    out = _largest_component(mask)
    assert out[82, 82] == 0
    assert out[25, 25] == 1


def test_keeps_a_textured_region_full_of_small_holes():
    # A region riddled with small gaps (e.g. soil texture in a real SAM2
    # mask) must NOT be treated as a swarm of separate tiny components and
    # discarded -- this is the exact bug found on the plant/pot fixture.
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:50] = 1  # solid "leaf" part
    textured = np.ones((100, 40), dtype=np.uint8)
    textured[::3, ::3] = 0  # sparse holes, none wider than a couple px
    mask[10:90, 50:90] = 0
    mask[10:90, 50:90][textured[:80, :40] == 1] = 1  # "pot" part, holey but connected

    out = _largest_component(mask)
    # The textured region should mostly survive (holes stay holes, but the
    # region itself isn't wiped out).
    assert out[50, 70] == 1


def test_never_adds_pixels_not_in_the_original_mask():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[10:20, 25:35] = 1  # a second nearby blob, close enough to be bridged by closing
    out = _largest_component(mask, close_kernel_px=15)
    # Whatever gets kept must be a subset of the original mask -- closing is
    # only used to judge connectivity, never to invent new foreground area.
    assert np.all(out <= mask)


def test_single_component_mask_passes_through_unchanged():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:45, 5:45] = 1
    out = _largest_component(mask)
    assert np.array_equal(out, mask)


def test_empty_mask_returns_empty():
    mask = np.zeros((30, 30), dtype=np.uint8)
    out = _largest_component(mask)
    assert out.sum() == 0
