"""Batch regression test across diverse object/background pairs.

The bear+beach pair was ONE scenario, not proof of general correctness
(per docs/architecture.md's testing philosophy: keep it one regression
case among many, not the definition of done). This runs the same 6-stage
chain across genuinely different materials (fur, hair/fabric, rigid
product, thin plant leaves) and environments (indoor, outdoor overcast,
dappled forest light, plain studio) and saves every result for visual
review -- a "golden set" run, meant to be re-run whenever a stage changes.

Each case uses a center point prompt for SAM2 (all fixtures are
roughly-centered "hero shot" stock photos) and a simple centered placement
on its background -- placement quality isn't the point of this script,
generalization of extraction/matting/decontamination/color-adaptation is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from compositing.segmentation import segment
from compositing.matting import matte
from compositing.decontaminate import decontaminate
from compositing.transform import Placement, transform_subject
from compositing.adaptation import match_appearance
from compositing.composite import alpha_composite

OUT_DIR = "data/outputs/batch"
os.makedirs(OUT_DIR, exist_ok=True)

CASES = [
    {"name": "dog_livingroom", "object": "data/objects/dog.jpg", "point": (600, 850), "bg": "data/backgrounds/living_room.jpg"},
    {"name": "person_citystreet", "object": "data/objects/person.jpg", "point": (800, 1300), "bg": "data/backgrounds/city_street.jpg"},
    {"name": "sneaker_studio", "object": "data/objects/sneaker.jpg", "point": (800, 950), "bg": "data/backgrounds/studio_gray.jpg"},
    {"name": "plant_forest", "object": "data/objects/plant.jpg", "point": (1050, 1000), "bg": "data/backgrounds/forest_path.jpg"},
]

for case in CASES:
    print(f"=== {case['name']} ===")
    try:
        photo = cv2.cvtColor(cv2.imread(case["object"]), cv2.COLOR_BGR2RGB)
        bg = cv2.cvtColor(cv2.imread(case["bg"]), cv2.COLOR_BGR2RGB)
        bg_h, bg_w = bg.shape[:2]

        print("  [2] segmenting...")
        mask = segment(photo, point=case["point"])
        print("      mask px:", mask.sum())

        print("  [3] matting...")
        alpha = matte(photo, mask)

        print("  [4] decontaminating + grain reduction...")
        clean_rgb = decontaminate(photo, alpha)
        clean_rgb = cv2.bilateralFilter(clean_rgb, d=5, sigmaColor=35, sigmaSpace=35)

        print("  [5] placing...")
        ys, xs = (alpha > 0.02).nonzero()
        crop_h = ys.max() - ys.min()
        target_h_px = int(bg_h * 0.45)
        scale = target_h_px / crop_h
        placement = Placement(
            position=(bg_w // 2, int(bg_h * 0.85)),
            scale=scale,
            grounding_point=(0.5, 1.0),
        )
        result = transform_subject(clean_rgb, alpha, (bg_w, bg_h), placement)

        print("  [7] adapting color...")
        adapted_rgb = match_appearance(result.rgb, result.alpha, bg, result.bbox, strength=0.5)

        print("  [10] compositing...")
        final = alpha_composite(bg, adapted_rgb, result.alpha)

        out_path = os.path.join(OUT_DIR, f"{case['name']}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        print("  SAVED", out_path)
    except Exception as e:
        print(f"  FAILED: {e}")

print("Done.")
