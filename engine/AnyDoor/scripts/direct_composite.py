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


def soft_alpha_mask(mask_crop_01, box_w, box_h, blur_px_fullres=42):
    # Feather at FULL resolution, then downsize -- not the other way around.
    # Blurring *after* a ~15x downsize (as this used to do) makes a small
    # sigma cover a proportionally huge area of the tiny image, pulling
    # alpha from rows well below (e.g. the wide part of the ear) up into
    # rows near the top (the ear tip) that shouldn't have much alpha yet.
    # That mismatch -- high alpha with no matching object detail -- is what
    # showed up as a translucent rectangular patch above the head. Blurring
    # at full res keeps the feather physically proportional (~10px at full
    # res becomes under 1px after a 15x downsize), then INTER_AREA is the
    # correct filter for a large downsize (no LANCZOS ringing).
    mask_f = (mask_crop_01 * 255).astype(np.float32)
    mask_f = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=blur_px_fullres)
    mask_resized = cv2.resize(mask_f, (box_w, box_h), interpolation=cv2.INTER_AREA)
    return np.clip(mask_resized / 255.0, 0, 1)


def apply_edge_light_bleed(obj_rgb, alpha, bg_region, strength=0.45):
    # Real photographed edges pick up a sliver of ambient/reflected color
    # from whatever's around them -- a hard cutout never does, which is a
    # classic "pasted" tell even with a perfectly soft alpha. Fake it by
    # blending a bit of the background's own color into the object only in
    # a thin band right at the silhouette boundary (alpha ~0.5), tapering to
    # nothing both in the solid interior (alpha~1) and out in clean
    # background (alpha~0) -- interior detail (fur) is left untouched.
    rim_weight = 4.0 * alpha * (1.0 - alpha)  # 0 at alpha=0/1, peaks at alpha=0.5
    rim_weight = (rim_weight * strength)[..., None]
    return obj_rgb.astype(np.float32) * (1 - rim_weight) + bg_region.astype(np.float32) * rim_weight


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


def main():
    object_path = sys.argv[1]
    bg_path = sys.argv[2]
    save_path = sys.argv[3]
    box_h_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.225
    cx_frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    cy_frac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.72

    print('Segmenting object...')
    ref_image, ref_mask = segment_object(object_path)

    ys, xs = np.where(ref_mask > 0)
    y1o, y2o, x1o, x2o = ys.min(), ys.max(), xs.min(), xs.max()
    obj_crop = ref_image[y1o:y2o, x1o:x2o]
    mask_crop = ref_mask[y1o:y2o, x1o:x2o]
    print(f'ref bbox: {x2o-x1o}x{y2o-y1o} (aspect {(x2o-x1o)/(y2o-y1o):.3f})')

    bg_image = cv2.imread(bg_path)
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    bg_h, bg_w = bg_image.shape[:2]
    aspect_wh = get_mask_aspect_ratio(ref_mask)
    tar_mask = make_placement_mask(bg_image, aspect_wh, box_h_frac, cx_frac, cy_frac)
    tys, txs = np.where(tar_mask > 0)
    y1, y2, x1, x2 = tys.min(), tys.max() + 1, txs.min(), txs.max() + 1
    box_h, box_w = y2 - y1, x2 - x1
    print(f'bg size: {bg_w}x{bg_h}')
    print(f'placement box: x[{x1}:{x2}] y[{y1}:{y2}] ({box_w}x{box_h}), '
          f'center=({(x1+x2)//2},{(y1+y2)//2}), frac_center=({cx_frac},{cy_frac})')

    print('Compositing...')
    obj_resized = cv2.resize(obj_crop, (box_w, box_h), interpolation=cv2.INTER_AREA)
    alpha = soft_alpha_mask(mask_crop, box_w, box_h)

    out = add_soft_shadow(bg_image, (y1, y2, x1, x2), alpha)
    region = out[y1:y2, x1:x2].astype(np.float32)
    obj_resized = apply_edge_light_bleed(obj_resized, alpha, region)
    blended = region * (1 - alpha[..., None]) + obj_resized * alpha[..., None]
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print('SAVED', save_path)


if __name__ == '__main__':
    main()
