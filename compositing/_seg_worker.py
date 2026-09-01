"""SAM2 segmentation worker.

MUST be run with the isolated .venv-seg interpreter (SAM2 needs torch>=2.5,
which conflicts with the main .venv's torch==2.0.0 pin required by AnyDoor --
see docs/architecture.md and the incident note in compositing/README.md).
Not meant to be imported directly; compositing/segmentation.py invokes this
as a subprocess so the rest of the compositing package stays venv-agnostic.

Usage:
    python _seg_worker.py <image_path> <out_mask_path> --point x,y
    python _seg_worker.py <image_path> <out_mask_path> --box x1,y1,x2,y2
"""
import argparse
import os

import cv2
import numpy as np
import torch

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "sam2", "sam2.1_hiera_base_plus.pt",
)
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("out_mask_path")
    parser.add_argument("--point", help="x,y foreground point prompt")
    parser.add_argument("--box", help="x1,y1,x2,y2 box prompt")
    args = parser.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(CONFIG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(model)

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    kwargs = {}
    if args.point:
        x, y = map(float, args.point.split(","))
        kwargs["point_coords"] = np.array([[x, y]])
        kwargs["point_labels"] = np.array([1])
    if args.box:
        x1, y1, x2, y2 = map(float, args.box.split(","))
        kwargs["box"] = np.array([x1, y1, x2, y2])
    if not kwargs:
        raise ValueError("must supply --point or --box")

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        masks, scores, _ = predictor.predict(**kwargs, multimask_output=True)

    best = masks[int(np.argmax(scores))]
    cv2.imwrite(args.out_mask_path, (best.astype(np.uint8) * 255))
    print("SAVED", args.out_mask_path)


if __name__ == "__main__":
    main()
