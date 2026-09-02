"""Stage 6: scene analysis.

Deterministic color statistics only for v1 (per docs/architecture.md:
cheap estimators first, a model only if/when something they can't capture
turns out to matter -- depth/ground-plane reasoning is deferred, not needed
yet). Estimates what compositing.adaptation needs to match the placed
subject's tone to its surroundings: mean/spread of color in LAB space
(L = lightness/exposure, a/b = color temperature and tint) over a region.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneStats:
    mean_lab: np.ndarray  # (3,) float32
    std_lab: np.ndarray   # (3,) float32
    pixel_count: int


def analyze_region(rgb, region_mask) -> SceneStats:
    """rgb: (H, W, 3) uint8. region_mask: (H, W) bool -- which pixels to
    summarize (e.g. the background context around a placement, or the
    solid-interior pixels of an extracted subject)."""
    if region_mask.sum() < 10:
        raise ValueError("region_mask selects too few pixels to get a stable estimate")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[region_mask]
    mean_lab = pixels.mean(axis=0)
    std_lab = pixels.std(axis=0)
    # A flat-color region gives std=0, which would divide-by-zero in
    # adaptation's Reinhard transfer -- floor it instead of letting that
    # stage special-case it.
    std_lab = np.maximum(std_lab, 1.0)
    return SceneStats(mean_lab=mean_lab, std_lab=std_lab, pixel_count=int(region_mask.sum()))


def context_ring_mask(canvas_shape, bbox, margin_frac=0.5, exclude_mask=None):
    """A ring of canvas pixels around `bbox` (the placed subject's own
    footprint), used to sample genuine surrounding-background color/tone --
    not the whole background image, which can have very different lighting
    in different areas (shade vs. direct sun, sky vs. sand).

    bbox: (y1, y2, x1, x2) -- typically compositing.transform's placed bbox.
    exclude_mask: e.g. the subject's own alpha > threshold, so its own
        pixels never leak into the "surrounding background" sample.
    """
    h, w = canvas_shape[:2]
    y1, y2, x1, x2 = bbox
    box_h, box_w = y2 - y1, x2 - x1
    my = max(1, int(box_h * margin_frac))
    mx = max(1, int(box_w * margin_frac))

    outer = np.zeros((h, w), dtype=bool)
    oy1, oy2 = max(0, y1 - my), min(h, y2 + my)
    ox1, ox2 = max(0, x1 - mx), min(w, x2 + mx)
    outer[oy1:oy2, ox1:ox2] = True

    inner = np.zeros((h, w), dtype=bool)
    inner[y1:y2, x1:x2] = True

    ring = outer & ~inner
    if exclude_mask is not None:
        ring &= ~exclude_mask
    return ring
