"""ViTMatte alpha-matting worker.

MUST be run with the isolated .venv-seg interpreter (needs a recent
transformers, which conflicts with the main .venv's transformers pin used by
AnyDoor). Not meant to be imported directly; compositing/matting.py invokes
this as a subprocess.

Builds a trimap from a binary segmentation mask (erode = definite
foreground, dilate-minus-erode = unknown band for ViTMatte to resolve, rest
= definite background) and runs ViTMatte to produce a continuous alpha.

Usage:
    python _matte_worker.py <image_path> <mask_path> <out_alpha_path> [unknown_band_px]
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "vitmatte",
)


def make_trimap(mask_binary, unknown_band_px=15):
    kernel = np.ones((3, 3), np.uint8)
    iterations = max(1, unknown_band_px // 3)
    sure_fg = cv2.erode(mask_binary, kernel, iterations=iterations)
    sure_bg = cv2.erode(1 - mask_binary, kernel, iterations=iterations)
    trimap = np.full(mask_binary.shape, 128, dtype=np.uint8)  # unknown by default
    trimap[sure_fg > 0] = 255
    trimap[sure_bg > 0] = 0
    return trimap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("mask_path")
    parser.add_argument("out_alpha_path")
    parser.add_argument("unknown_band_px", nargs="?", type=int, default=15)
    args = parser.parse_args()

    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = VitMatteImageProcessor.from_pretrained(CHECKPOINT_DIR)
    model = VitMatteForImageMatting.from_pretrained(CHECKPOINT_DIR).to(device).eval()

    image_full = Image.open(args.image_path).convert("RGB")
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    mask_binary = (mask > 127).astype(np.uint8)
    trimap_full = make_trimap(mask_binary, args.unknown_band_px)

    # ViT self-attention is quadratic in pixel count -- feeding a full multi-
    # thousand-pixel photo directly OOMs (confirmed: a 4551x3251 image asked
    # to allocate 76GiB). Work at a bounded resolution, matching how AnyDoor
    # itself only ever operates at 512x512 internally, then upsample the
    # resulting alpha back to the original size.
    h, w = mask.shape
    max_side = 1024
    scale = min(1.0, max_side / max(h, w))
    work_w, work_h = max(32, int(w * scale)), max(32, int(h * scale))
    image = image_full.resize((work_w, work_h), Image.BILINEAR)
    trimap = cv2.resize(trimap_full, (work_w, work_h), interpolation=cv2.INTER_NEAREST)

    inputs = processor(images=image, trimaps=Image.fromarray(trimap), return_tensors="pt").to(device)
    with torch.no_grad():
        alpha = model(**inputs).alphas

    alpha = alpha[0, 0].clamp(0, 1).cpu().numpy()
    alpha = cv2.resize(alpha, (w, h))
    trimap = trimap_full
    # Definite background/foreground regions from the trimap override any
    # model noise there -- ViTMatte's job is the unknown boundary band only.
    alpha = np.where(trimap == 0, 0.0, alpha)
    alpha = np.where(trimap == 255, 1.0, alpha)

    cv2.imwrite(args.out_alpha_path, (alpha * 255).astype(np.uint8))
    print("SAVED", args.out_alpha_path)


if __name__ == "__main__":
    main()
