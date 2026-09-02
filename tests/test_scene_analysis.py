import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from compositing.scene_analysis import analyze_region, context_ring_mask


def test_flat_color_region_has_matching_mean_and_floored_std():
    rgb = np.full((50, 50, 3), (200, 50, 50), dtype=np.uint8)
    mask = np.ones((50, 50), dtype=bool)
    stats = analyze_region(rgb, mask)
    assert stats.pixel_count == 2500
    # A perfectly flat region has zero true variance -- std is floored to
    # avoid a divide-by-zero later in adaptation's Reinhard transfer.
    assert np.all(stats.std_lab >= 1.0)


def test_two_regions_have_different_mean_lab():
    red = np.full((30, 30, 3), (220, 20, 20), dtype=np.uint8)
    blue = np.full((30, 30, 3), (20, 20, 220), dtype=np.uint8)
    mask = np.ones((30, 30), dtype=bool)
    red_stats = analyze_region(red, mask)
    blue_stats = analyze_region(blue, mask)
    assert not np.allclose(red_stats.mean_lab, blue_stats.mean_lab, atol=2)


def test_analyze_region_raises_on_tiny_mask():
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError):
        analyze_region(rgb, mask)


def test_context_ring_excludes_inner_bbox_and_excluded_mask():
    bbox = (40, 60, 40, 60)  # y1,y2,x1,x2 -- a 20x20 box
    ring = context_ring_mask((150, 150), bbox, margin_frac=0.5)
    # Inside the bbox itself must never be selected.
    assert not ring[50, 50]
    # Somewhere just outside the bbox (within the margin) should be selected.
    assert ring[45, 35]

    exclude = np.zeros((150, 150), dtype=bool)
    exclude[45, 35] = True
    ring2 = context_ring_mask((150, 150), bbox, margin_frac=0.5, exclude_mask=exclude)
    assert not ring2[45, 35]


def test_context_ring_stays_within_canvas_bounds():
    bbox = (0, 10, 0, 10)  # top-left corner, no room above/left
    ring = context_ring_mask((100, 100), bbox, margin_frac=1.0)
    assert ring.shape == (100, 100)
    assert ring.sum() > 0
