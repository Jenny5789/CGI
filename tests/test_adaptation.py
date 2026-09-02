import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from compositing.adaptation import match_appearance


def make_fixture():
    """A cool blue-gray subject (fg) placed on a warm orange-toned
    background (bg) -- the same kind of mismatch as a bear photographed
    under overcast light placed onto a sunset beach photo."""
    canvas = np.full((200, 200, 3), (230, 170, 110), dtype=np.uint8)  # warm bg
    bbox = (80, 140, 80, 140)  # y1,y2,x1,x2 -- 60x60 box
    fg_rgb = canvas.copy()
    fg_alpha = np.zeros((200, 200), dtype=np.float32)
    y1, y2, x1, x2 = bbox
    fg_rgb[y1:y2, x1:x2] = (90, 110, 150)  # cool blue-gray subject
    fg_alpha[y1:y2, x1:x2] = 1.0
    return fg_rgb, fg_alpha, canvas, bbox


def mean_rgb_in_bbox(rgb, bbox):
    y1, y2, x1, x2 = bbox
    return rgb[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32).mean(axis=0)


def test_strength_zero_leaves_subject_unchanged():
    fg_rgb, fg_alpha, bg_rgb, bbox = make_fixture()
    out = match_appearance(fg_rgb, fg_alpha, bg_rgb, bbox, strength=0.0)
    y1, y2, x1, x2 = bbox
    assert np.array_equal(out[y1:y2, x1:x2], fg_rgb[y1:y2, x1:x2])


def test_positive_strength_pulls_subject_color_toward_background():
    fg_rgb, fg_alpha, bg_rgb, bbox = make_fixture()
    out = match_appearance(fg_rgb, fg_alpha, bg_rgb, bbox, strength=0.6)

    original_mean = mean_rgb_in_bbox(fg_rgb, bbox)
    adapted_mean = mean_rgb_in_bbox(out, bbox)
    bg_mean = mean_rgb_in_bbox(bg_rgb, bbox)

    dist_before = np.linalg.norm(original_mean - bg_mean)
    dist_after = np.linalg.norm(adapted_mean - bg_mean)
    assert dist_after < dist_before


def test_background_pixels_outside_alpha_are_untouched():
    fg_rgb, fg_alpha, bg_rgb, bbox = make_fixture()
    out = match_appearance(fg_rgb, fg_alpha, bg_rgb, bbox, strength=0.8)
    # Outside the subject's own alpha, fg_rgb was already == bg_rgb in this
    # fixture, and should stay that way (adaptation only ever touches
    # alpha-covered pixels).
    outside = fg_alpha <= 0.02
    assert np.array_equal(out[outside], fg_rgb[outside])


def test_output_shape_and_dtype_preserved():
    fg_rgb, fg_alpha, bg_rgb, bbox = make_fixture()
    out = match_appearance(fg_rgb, fg_alpha, bg_rgb, bbox)
    assert out.shape == fg_rgb.shape
    assert out.dtype == np.uint8


def test_empty_alpha_returns_copy_without_error():
    fg_rgb, _, bg_rgb, bbox = make_fixture()
    empty_alpha = np.zeros((200, 200), dtype=np.float32)
    out = match_appearance(fg_rgb, empty_alpha, bg_rgb, bbox)
    assert np.array_equal(out, fg_rgb)
