"""
Compose an arbitrary object photo onto an arbitrary background photo using AnyDoor.

Steps:
1. Auto-segment the foreground object out of the object photo (rembg / U2Net),
   producing a binary ref_mask.
2. Auto-place a bounding box on the background photo (centered, sized to a
   fraction of the background) as tar_mask -- this defines WHERE AnyDoor
   composites the object, not a precise mask.
3. Run AnyDoor's inference_single_image with those two masks.

Usage:
    python scripts/composite_from_photos.py <object_path> <background_path> <save_path> [box_h_frac] [cx_frac] [cy_frac]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from rembg import remove, new_session


def segment_object(object_path):
    """Returns (rgb uint8, alpha uint8 0-255).

    Uses rembg's alpha-matting mode instead of a hard threshold: a plain
    U2Net mask cuts the silhouette at a hard 0/1 boundary, which leaves a
    fringe of background-color-contaminated edge pixels that reads as a
    strong outline once pasted onto a different-colored background. Matting
    estimates a soft, color-corrected alpha over the true (fuzzy) boundary
    instead, which is what actually removes that outline.
    """
    with open(object_path, 'rb') as f:
        input_bytes = f.read()
    session = new_session('u2net')
    output_bytes = remove(
        input_bytes,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=8,
    )
    rgba = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    alpha = rgba[:, :, 3]
    bgr = rgba[:, :, :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, alpha


def get_mask_aspect_ratio(ref_mask):
    ys, xs = np.where(ref_mask > 20)
    box_w = xs.max() - xs.min()
    box_h = ys.max() - ys.min()
    return box_w / box_h


def make_placement_mask(bg_image, aspect_wh, box_h_frac=0.75, cx_frac=0.5, cy_frac=0.62):
    # Placement box mirrors the reference object's own width/height ratio, so
    # AnyDoor doesn't have to squash/stretch the object's silhouette into a
    # mismatched box -- that distortion is what causes it to lose identity
    # and hallucinate a generic replacement instead of the actual object.
    h, w = bg_image.shape[:2]
    box_h = int(h * box_h_frac)
    box_w = int(box_h * aspect_wh)
    if box_w > w:
        box_w = w
        box_h = int(box_w / aspect_wh)
    cx, cy = int(w * cx_frac), int(h * cy_frac)
    x1 = max(0, cx - box_w // 2)
    x2 = min(w, x1 + box_w)
    y1 = max(0, cy - box_h // 2)
    y2 = min(h, y1 + box_h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def main():
    object_path = sys.argv[1]
    bg_path = sys.argv[2]
    save_path = sys.argv[3]
    box_h_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75
    cx_frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    cy_frac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.62

    print('Segmenting object...')
    ref_image, ref_mask = segment_object(object_path)

    bg_image = cv2.imread(bg_path)
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    aspect_wh = get_mask_aspect_ratio(ref_mask)
    tar_mask = make_placement_mask(bg_image, aspect_wh, box_h_frac, cx_frac, cy_frac)

    from run_inference import inference_single_image
    print('Running AnyDoor composite...')
    ref_mask_bin = (ref_mask > 128).astype(np.uint8)
    gen_image = inference_single_image(ref_image, ref_mask_bin, bg_image.copy(), tar_mask)

    cv2.imwrite(save_path, cv2.cvtColor(gen_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print('SAVED', save_path)


if __name__ == '__main__':
    main()
