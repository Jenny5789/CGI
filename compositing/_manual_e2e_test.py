"""Manual end-to-end smoke test chaining the implemented stages.

Not a permanent pipeline entrypoint (that's pipeline.py, not yet built) --
just wires segmentation -> matting -> decontaminate -> transform ->
adaptation -> composite by hand so we can see where things stand before
physical_integration exists. No shadow/AO correction happens here yet --
that's the known, expected remaining gap.
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

OBJECT_PATH = "data/objects/bear.jpg"
BG_PATH = "data/backgrounds/beach.jpg"
OUT_PATH = "data/outputs/compositing_pkg_e2e_test_v3.png"

print("Loading images...")
photo = cv2.cvtColor(cv2.imread(OBJECT_PATH), cv2.COLOR_BGR2RGB)
bg = cv2.cvtColor(cv2.imread(BG_PATH), cv2.COLOR_BGR2RGB)
bg_h, bg_w = bg.shape[:2]

print("[2] Segmenting (SAM2)...")
mask = segment(photo, box=(1600, 400, 3600, 3150))
print("    mask px:", mask.sum())

print("[3] Matting (ViTMatte)...")
alpha = matte(photo, mask)
print("    alpha range:", alpha.min(), alpha.max())

print("[4] Decontaminating edges...")
clean_rgb = decontaminate(photo, alpha)

print("[4b] Reducing grain (bilateral filter, edge-preserving)...")
# Removes low-amplitude sensor-grain noise while leaving strong local
# gradients -- individual hair-strand edges, eye/nose contours -- intact.
# Not a blur: a normal Gaussian/box blur would soften those edges too and
# was explicitly ruled out for exactly that reason.
clean_rgb = cv2.bilateralFilter(clean_rgb, d=5, sigmaColor=35, sigmaSpace=35)

print("[5] Transforming (place on canvas)...")
ys, xs = (alpha > 0.02).nonzero()
crop_h = ys.max() - ys.min()
crop_w = xs.max() - xs.min()
# Clear sand gap between the two center palm trees on this background,
# located by inspecting a coordinate grid overlaid on beach.jpg -- trunks
# sit at roughly x=100, 190, 310, 400, so x=200-300 is open sand.
gap_center_x, gap_width = 250, 100
feet_y = 690
target_h_px = min(int(bg_h * 0.19), int(gap_width / (crop_w / crop_h)))
scale = target_h_px / crop_h
placement = Placement(
    position=(gap_center_x, feet_y),
    scale=scale,
    rotation_deg=0.0,
    flip_horizontal=False,
    grounding_point=(0.5, 1.0),
)
result = transform_subject(clean_rgb, alpha, (bg_w, bg_h), placement)

print("[7] Adapting color/exposure to the background...")
adapted_rgb = match_appearance(result.rgb, result.alpha, bg, result.bbox, strength=0.5)

print("[10] Compositing...")
final = alpha_composite(bg, adapted_rgb, result.alpha)

cv2.imwrite(OUT_PATH, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
print("SAVED", OUT_PATH)
