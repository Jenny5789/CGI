"""Integration tests for the SAM2/ViTMatte stages.

These are the two model-backed stages (see compositing/README.md for why
they run in a separate .venv-seg) -- skipped automatically if that
environment or its checkpoints aren't present, rather than failing hard, so
the rest of the suite stays runnable on a machine that hasn't set it up.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEG_READY = (
    os.path.exists(os.path.join(_ROOT, ".venv-seg", "Scripts", "python.exe"))
    and os.path.exists(os.path.join(_ROOT, "checkpoints", "sam2", "sam2.1_hiera_base_plus.pt"))
    and os.path.exists(os.path.join(_ROOT, "checkpoints", "vitmatte"))
)
_BEAR_PATH = os.path.join(_ROOT, "data", "objects", "bear.jpg")

pytestmark = pytest.mark.skipif(
    not _SEG_READY, reason=".venv-seg / SAM2 / ViTMatte checkpoints not set up -- see compositing/README.md"
)


@pytest.fixture(scope="module")
def bear_image():
    if not os.path.exists(_BEAR_PATH):
        pytest.skip(f"test fixture not present: {_BEAR_PATH}")
    img = cv2.imread(_BEAR_PATH)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def test_segment_with_box_prompt_covers_expected_region(bear_image):
    from compositing.segmentation import segment

    h, w = bear_image.shape[:2]
    mask = segment(bear_image, box=(1600, 400, 3600, 3150))

    assert mask.shape == (h, w)
    assert set(np.unique(mask)).issubset({0, 1})
    # The bear occupies a substantial but not overwhelming fraction of the
    # full 4551x3251 photo -- a sanity bound, not a tight match.
    frac = mask.sum() / mask.size
    assert 0.10 < frac < 0.60


def test_segment_requires_exactly_one_prompt(bear_image):
    from compositing.segmentation import segment

    with pytest.raises(ValueError):
        segment(bear_image, point=(100, 100), box=(0, 0, 10, 10))
    with pytest.raises(ValueError):
        segment(bear_image)


def test_matte_produces_genuinely_continuous_alpha(bear_image):
    from compositing.matting import matte
    from compositing.segmentation import segment

    mask = segment(bear_image, box=(1600, 400, 3600, 3150))
    alpha = matte(bear_image, mask)

    assert alpha.shape == mask.shape
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    # The whole point of matting over a binary mask: a real population of
    # strictly-intermediate alpha values at the boundary, not just 0/1.
    intermediate = (alpha > 0.02) & (alpha < 0.98)
    assert intermediate.sum() > 1000
