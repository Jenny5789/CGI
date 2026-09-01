"""
Pixel-preserving composite: paste the real segmented object onto the
background instead of letting AnyDoor's diffusion model regenerate it.

Trades AnyDoor's automatic light/shadow harmonization for 1:1 fidelity to
the source photo (exact orientation, no detail smoothing) -- use this when
identity/texture fidelity matters more than photorealistic re-lighting.

Usage:
    python scripts/direct_composite.py <object_path> <background_path> <save_path> [box_h_frac] [cx_frac] [cy_frac]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_from_photos import segment_object, get_mask_aspect_ratio, make_placement_mask

import cv2
import numpy as np


def feathered_alpha(mask, feather_px=4):
    mask_f = mask.astype(np.float32)
    dist_in = cv2.distanceTransform((mask_f > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(dist_in / feather_px, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)
    return np.clip(alpha, 0, 1)


def match_lighting(obj_rgb, obj_alpha, bg_region):
    # Nudge the object's overall brightness/warmth toward the region of the
    # background it's being placed on, without touching per-pixel detail.
    obj_lab = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_region, cv2.COLOR_RGB2LAB).astype(np.float32)

    m = obj_alpha > 0.5
    if m.sum() < 10:
        return obj_rgb

    for c in range(3):
        obj_mean = obj_lab[:, :, c][m].mean()
        bg_mean = bg_lab[:, :, c].mean()
        shift = (bg_mean - obj_mean) * (0.35 if c == 0 else 0.5)
        obj_lab[:, :, c] = obj_lab[:, :, c] + shift

    obj_lab = np.clip(obj_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(obj_lab, cv2.COLOR_LAB2RGB)


def add_soft_shadow(bg_rgb, box, alpha, strength=0.35):
    y1, y2, x1, x2 = box
    h, w = bg_rgb.shape[:2]
    shadow = np.zeros((h, w), dtype=np.float32)
    shadow_h = max(4, int((y2 - y1) * 0.08))
    ys = min(h, y2 + shadow_h // 2)
    cv2.ellipse(
        shadow,
        (int((x1 + x2) / 2), min(h - 1, y2)),
        (int((x2 - x1) * 0.42), shadow_h),
        0, 0, 360, 1.0, -1,
    )
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=shadow_h * 0.6)
    shadow = shadow * strength
    out = bg_rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= (1 - shadow)
    return np.clip(out, 0, 255).astype(np.uint8)


def seamless_edge_blend(bg_image, obj_resized, mask_resized, x1, y1, box_w, box_h):
    # Poisson blending (cv2.seamlessClone): keeps the object's own interior
    # gradients (fur texture) intact while solving for a seam at the mask
    # boundary that matches the destination's colors -- much less visible
    # than a feathered-alpha edge.
    pad = max(15, int(min(box_w, box_h) * 0.08))
    src = np.zeros((box_h + 2 * pad, box_w + 2 * pad, 3), dtype=np.uint8)
    src[pad:pad + box_h, pad:pad + box_w] = obj_resized
    mask = np.zeros((box_h + 2 * pad, box_w + 2 * pad), dtype=np.uint8)
    mask[pad:pad + box_h, pad:pad + box_w] = (mask_resized * 255).astype(np.uint8)

    cx = x1 + box_w // 2
    cy = y1 + box_h // 2
    h, w = bg_image.shape[:2]
    cx = min(max(cx, pad + 1), w - pad - 1)
    cy = min(max(cy, pad + 1), h - pad - 1)

    bg_bgr = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
    src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.seamlessClone(src_bgr, bg_bgr, mask, (cx, cy), cv2.NORMAL_CLONE)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def main():
    object_path = sys.argv[1]
    bg_path = sys.argv[2]
    save_path = sys.argv[3]
    box_h_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
    cx_frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    cy_frac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.68

    print('Segmenting object...')
    ref_image, ref_mask = segment_object(object_path)

    ys, xs = np.where(ref_mask > 0)
    y1o, y2o, x1o, x2o = ys.min(), ys.max(), xs.min(), xs.max()
    obj_crop = ref_image[y1o:y2o, x1o:x2o]
    mask_crop = ref_mask[y1o:y2o, x1o:x2o]

    bg_image = cv2.imread(bg_path)
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    aspect_wh = get_mask_aspect_ratio(ref_mask)
    tar_mask = make_placement_mask(bg_image, aspect_wh, box_h_frac, cx_frac, cy_frac)
    tys, txs = np.where(tar_mask > 0)
    y1, y2, x1, x2 = tys.min(), tys.max() + 1, txs.min(), txs.max() + 1
    box_h, box_w = y2 - y1, x2 - x1

    print('Compositing...')
    obj_resized = cv2.resize(obj_crop, (box_w, box_h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask_crop, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
    alpha = feathered_alpha(mask_resized, feather_px=max(2, box_w // 150))

    bg_region = bg_image[y1:y2, x1:x2]
    obj_resized = match_lighting(obj_resized, alpha, bg_region)

    out = add_soft_shadow(bg_image, (y1, y2, x1, x2), alpha)
    out = seamless_edge_blend(out, obj_resized, mask_resized, x1, y1, box_w, box_h)

    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print('SAVED', save_path)


if __name__ == '__main__':
    main()
