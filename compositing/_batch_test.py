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
import numpy as np

from compositing.segmentation import segment
from compositing.matting import matte
from compositing.decontaminate import decontaminate
from compositing.transform import Placement, transform_subject
from compositing.adaptation import match_appearance
from compositing.composite import alpha_composite

OUT_DIR = "data/outputs/batch"
os.makedirs(OUT_DIR, exist_ok=True)


def tighten_alpha(alpha, erode_px=2):
    """Shrinks the alpha boundary inward by a couple pixels.

    Cuts off the outermost, lowest-confidence ring of alpha -- exactly the
    pixels most likely to still carry background-color spill that
    decontaminate() can't fully correct when the subject's own color is
    very close to its background's (e.g. a cream-colored puppy against a
    white backdrop: decontaminate()'s math divides by alpha and multiplies
    by (fg - bg) color difference, both of which go unstable/near-zero
    exactly when fg and bg colors nearly match, so contamination there
    isn't something that step can fully undo on its own). Trades a little
    edge softness/fine wisp detail for removing a visible light-colored
    halo -- not a substitute for decontaminate(), a complement to it.
    """
    kernel = np.ones((3, 3), np.uint8)
    alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    eroded = cv2.erode(alpha_u8, kernel, iterations=erode_px)
    return eroded.astype(np.float32) / 255.0

CASES = [
    # target_h_frac: rough real-world-plausible fraction of background
    # height (a puppy shouldn't be as tall as a person; a floating product
    # shot doesn't have "feet" so it's grounded at frame-center instead of
    # frame-bottom). Picking one fixed fraction for every case regardless of
    # what the object actually is (first version of this script) is exactly
    # how a puppy ended up rendered person-sized.
    {"name": "dog_livingroom", "object": "data/objects/dog.jpg", "point": (600, 850),
     "bg": "data/backgrounds/living_room.jpg", "target_h_frac": 0.06, "grounding": (0.5, 1.0), "pos_y_frac": 0.85},
    # A single point on a person's torso is ambiguous (does it mean "the
    # shirt", "the torso", "the whole person"?) -- confirmed by inspecting
    # SAM2's raw candidates: the highest-confidence one only covered ~6% of
    # the frame (torso-only), not the full standing figure. A box anchors
    # the intended extent explicitly; this is standard SAM2 usage, not a
    # person-specific hack.
    {"name": "person_citystreet", "object": "data/objects/person.jpg", "box": (500, 120, 1080, 2400),
     "bg": "data/backgrounds/city_street.jpg", "target_h_frac": 0.75, "grounding": (0.5, 1.0), "pos_y_frac": 0.90},
    {"name": "sneaker_studio", "object": "data/objects/sneaker.jpg", "point": (800, 950),
     "bg": "data/backgrounds/studio_gray.jpg", "target_h_frac": 0.30, "grounding": (0.5, 0.5), "pos_y_frac": 0.5},
    # A single point on this plant's thin leaf was fundamentally unstable --
    # SAM2's own 3 candidates for that point were [80%-of-frame, an 11%
    # candidate that turned out to be a noisy garbage mask, 53%-of-frame] --
    # none usable. A box around the whole plant (leaves+pot), same fix
    # pattern as the person case above, gave a clean result instead.
    # Box widened to the actual bottom of the photo (1998, not 1900) --
    # the previous box cut off part of the pot, so the mask only captured
    # leaves+partial pot. A "complete" object extraction here means the
    # pot comes along too, not just the plant growing out of it.
    {"name": "plant_forest", "object": "data/objects/plant.jpg", "box": (350, 100, 1280, 1998),
     "bg": "data/backgrounds/forest_path.jpg", "target_h_frac": 0.35, "grounding": (0.5, 1.0), "pos_y_frac": 0.88},
]

for case in CASES:
    print(f"=== {case['name']} ===")
    try:
        photo = cv2.cvtColor(cv2.imread(case["object"]), cv2.COLOR_BGR2RGB)
        bg = cv2.cvtColor(cv2.imread(case["bg"]), cv2.COLOR_BGR2RGB)
        bg_h, bg_w = bg.shape[:2]

        print("  [2] segmenting...")
        if "box" in case:
            mask = segment(photo, box=case["box"])
        else:
            mask = segment(photo, point=case["point"])
        print("      mask px:", mask.sum())

        print("  [3] matting...")
        alpha = matte(photo, mask)
        alpha = tighten_alpha(alpha, erode_px=2)

        print("  [4] decontaminating + grain reduction...")
        clean_rgb = decontaminate(photo, alpha)
        clean_rgb = cv2.bilateralFilter(clean_rgb, d=5, sigmaColor=35, sigmaSpace=35)

        print("  [5] placing...")
        ys, xs = (alpha > 0.02).nonzero()
        crop_h = ys.max() - ys.min()
        target_h_px = int(bg_h * case["target_h_frac"])
        scale = target_h_px / crop_h
        placement = Placement(
            position=(bg_w // 2, int(bg_h * case["pos_y_frac"])),
            scale=scale,
            grounding_point=case["grounding"],
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
