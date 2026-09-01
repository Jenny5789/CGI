# General-Purpose Photorealistic Compositing System — Analysis & Architecture Plan

## Context

The bear+beach regression test kept failing in the same visible way (sticker/cut-out silhouette) across several rounds of blind parameter tuning (blur radius 3→28→75→42, seamlessClone tried and reverted, alpha matting tried and reverted). The user correctly identified this as a sign we were patching symptoms rather than fixing an architecture, and asked for a full stop: inspect the actual code, the actual AnyDoor/VACE capabilities, and produce a general-purpose architecture (not bear/beach-specific) before touching any more code. This document is that analysis, grounded in two parallel code-tracing passes over the current pipeline and over AnyDoor/VACE internals.

No code has been changed to produce this document.

---

## A. Current Pipeline (as it actually exists)

Two independent compositing paths exist side by side, both under `engine/AnyDoor/scripts/`:

**`composite_from_photos.py`** (AnyDoor diffusion path):
1. `segment_object()` — rembg + U2Net, hard threshold `alpha>128` → **binary** mask (`:39`). Docstring records alpha-matting was tried and reverted (pymatting's trimap collapsed, erasing the object).
2. `get_mask_aspect_ratio()` / `make_placement_mask()` — deterministic rectangle placement (fraction-of-background math), no AI.
3. `inference_single_image()` (AnyDoor) does everything else — see `process_pairs` trace below.

**`direct_composite.py`** (pixel-paste path, the one being iterated on):
1. Same `segment_object()` binary mask.
2. Tight bbox crop, same deterministic placement math (different default box size/position).
3. `soft_alpha_mask()` — Gaussian-blurs the **binary** mask at full crop resolution (currently sigma=42), then downsizes with `INTER_AREA`.
4. `add_soft_shadow()` — fixed-geometry ellipse under the placement box, blurred, multiplied into background brightness. Not derived from any actual light estimate.
5. `apply_edge_light_bleed()` — blends background color into the object only in a parabolic rim band peaking at alpha=0.5.
6. Straight-alpha blend: `bg*(1-alpha) + fg*alpha`. Verified consistent (no premultiplied-alpha bug).

Neither script does: edge decontamination, relighting, color-temperature/exposure/sharpness/DOF matching, rotation, flip, occlusion, or true alpha matting. Confirmed absent, not just unused.

**AnyDoor's own `process_pairs`** (`run_inference.py:61-138`, invoked only by `composite_from_photos.py`):
- Masks the reference to a white background, crops to bbox, pads 1.2x, resizes to 224x224 → feeds DINOv2 for identity.
- Builds a **Sobel edge map** of the reference (mask-eroded, gradient-thresholded at 50) — this, not raw RGB, is what's pasted into the target bbox as ControlNet's shape hint.
- Placement-box context crop uses **two randomized expansion ratios** (`np.random.randint`-driven, ranges `[1.1,1.2]` and `[1.5,3]`) — non-deterministic per call, internal to AnyDoor, not exposed to callers.
- No segmentation, no matting, no decontamination, no relighting exists inside AnyDoor itself — it assumes a clean caller-supplied mask and does 100% of appearance harmonization implicitly inside one diffusion forward pass (not a separable/inspectable stage).

## B. Root-Cause Analysis

**Confirmed** (from code + this session's own git history):
1. **No true alpha matting** — only a hard binary threshold, softened after the fact by blurring. Blurring a binary mask can approximate softness but destroys any real strand-level structure (fur/hair) that a proper matte would have preserved distinctly per-material.
2. **No edge decontamination** — background-color bleed in the *original* photo's edge pixels (from `bear.jpg`'s own out-of-focus background) is never removed. `apply_edge_light_bleed` does the opposite (deliberately blends the *new* background in) — a stylistic patch, not a fix for source contamination.
3. **Feathering is applied as one uniform blur radius over the entire silhouette**, exactly the anti-pattern the user's spec calls out. The empirical tuning cycle (3px → too hard against textured backgrounds → 75px → foggy on plain sand) is a direct symptom of using one global parameter when the right amount of softness is material- and local-background-dependent.
4. **No relighting / color-temperature / exposure / sharpness / DOF matching exists anywhere.** The object is native-resolution/native-sharpness pixels dropped onto a background with unrelated photographic characteristics. This is likely a *bigger* contributor to "looks pasted" than edge softness alone, and it's 100% unaddressed today.
5. **Shadow is a fixed ellipse**, not derived from any estimated light direction — physically arbitrary.
6. Straight-alpha compositing math itself is correct and consistent (verified, not a bug).

**Suspected, not confirmed** (worth dedicated tests, not fixes yet):
- Sharpness/DOF mismatch may remain the dominant "sticker" cue even after edge and lighting fixes land — needs an isolated before/after test once stage 7 (foreground adaptation) exists.
- The specific bear/bush test background (dark, busy texture) may structurally hide feathering quality regardless of how it's tuned — the architecture must be validated against multiple background types, not just this one.

## C. Gap Analysis vs. Project Goal

| Capability | Status |
|---|---|
| Subject extraction (segmentation) | Present, but binary-only, no material awareness |
| Alpha matting | **Missing** (attempted via rembg's bundled matting, failed, reverted) |
| Edge decontamination | **Missing** entirely |
| Alpha compositing correctness | OK (verified straight-alpha, consistent) |
| Deterministic placement (x/y/scale) | Present |
| Rotation / flip | **Missing** entirely (flip exists as dead/unused code in AnyDoor) |
| Grounding point / depth-layer API | **Missing** |
| Scene analysis (lighting/color-temp/exposure/DOF/grain) | **Missing** entirely |
| Foreground adaptation (relight/color/sharpness match) | **Missing** entirely (tried once as a global color shift, reverted — but never replaced with anything else) |
| Shadows / AO | Present but physically arbitrary (fixed ellipse) |
| Reflections / occlusion / depth layering | **Missing** entirely |
| Final harmonization (geometry-locked) | **Missing** — AnyDoor's harmonization exists but is NOT geometry-locked (it can move/reshape/mirror the subject, confirmed by this session's own regression) |
| Debuggable/inspectable intermediate stages | Partially (debug prints exist, but no structured saved-artifact convention) |

## D. AnyDoor / VACE Analysis

**AnyDoor — reusable:**
- `FrozenDinoV2Encoder` (`ldm/modules/encoders/modules.py:279-315`) is a clean, standalone, frozen identity-embedding module (global CLS token + 256 spatially-local patch tokens, not just one pooled vector). Usable independently of the rest of AnyDoor — e.g. as an identity-fidelity QA metric (cosine-similarity check that harmonization didn't drift the subject), or as an optional conditioning signal for a future generative touch-up stage.
- The deterministic crop/pad/bbox primitives in `datasets/data_utils.py` (`get_bbox_from_mask`, `expand_bbox`, `box2squre`, `pad_to_square`) are solid, reusable geometry building blocks — once stripped of their internal `np.random` calls (which make AnyDoor's own placement non-deterministic, violating the spec's determinism requirement).
- The general pattern — "separate a shape/edge hint from an identity embedding, feed both into one conditioned generative model" — is a reasonable template for the *optional final harmonization stage only*, not for the whole pipeline.

**AnyDoor — should NOT reuse:**
- `ControlLDM`'s single-forward-pass design, where identity + shape + harmonization are entangled and jointly regenerated. This is structurally incompatible with "separate geometry from appearance" and is the confirmed cause of this project's own regression (mirrored orientation, smoothed fur) when `composite_from_photos.py` was tested end-to-end earlier this session.
- `iseg/BaselineModel` — a mask *refiner* requiring a human-drawn coarse mask (`forward(image, coarse_mask)`), not usable as an automatic segmenter.
- Internal randomness in `expand_bbox` — must not carry into a deterministic geometry stage.

**VACE:**
- Confirmed **video-only, end-to-end** (Wan/LTX video diffusion backbones; every script path — `vace_preproccess.py`, `vace_wan_inference.py`, `vace_ltx_inference.py`, `vace_pipeline.py` — terminates in one of those two video models). No standalone single-image generative or harmonization capability exists independent of them.
- Its `annotators/` (`sam.py`, `salient.py`, `depth.py`, `inpainting.py`, `outpainting.py`) are per-frame preprocessing utilities that could in principle be evaluated as standalone single-image tools — but this needs a follow-up read of the actual annotator source (only a directory-listing-level pass was done so far). Notably: if `annotators/sam.py` just wraps the real Segment Anything model, we'd want to depend on SAM2 directly rather than through VACE's wrapper.
- **Neither reference solves**: true material-aware alpha matting, edge decontamination, scene lighting/color/exposure/DOF estimation, or a geometry-locked harmonization stage. These have to be built or sourced elsewhere regardless of which reference we lean on.

## E. Proposed Architecture

Each stage classified by processing type, per the spec's requirement:

| # | Stage | Type | Notes |
|---|---|---|---|
| 1 | Subject detection | Specialized vision model (open-vocab detector) — **deferred**; v1 assumes caller supplies the reference photo/crop directly | Not needed until a "find the subject automatically in a busy photo" use case is required |
| 2 | Segmentation | Specialized vision model: **SAM2** (promptable: point/box) with a salient-object fallback for zero-prompt cases | Deterministic connected-component cleanup wraps the model output (regression-tested against the noise-blob bbox bug hit earlier this session) |
| 3 | Alpha matting | Specialized vision model: a **dedicated matting network** (trimap-free hair/fur-capable matting model) — NOT rembg's bundled `alpha_matting`, confirmed to fail | Must produce genuinely continuous alpha with real sub-pixel structure, not a blurred binary mask |
| 4 | Edge decontamination | Deterministic image processing (foreground-color solve at semi-transparent boundary pixels) | Operates on the (RGB, alpha) pair from stage 3; currently 100% absent |
| 5 | Deterministic subject transform (x/y/scale/rotation/flip/grounding) | Deterministic (affine/perspective warp) | Generalizes AnyDoor's bbox math (de-randomized), extended to rotation/flip which don't exist today |
| 6 | Scene analysis | Hybrid — cheap deterministic estimators (histogram color-temp/exposure, edge-density sharpness/grain) first; a monocular depth model only if/when occlusion or ground-plane reasoning is actually needed | Depth estimation deferred to a later milestone — expensive, not required for the correctness-first pass |
| 7 | Foreground adaptation (color/exposure/sharpness/DOF match) | Deterministic (LAB-space local color-statistics transfer restricted to FG pixels; matched blur/grain injection) | **Highest-priority missing capability** — likely the dominant remaining "pasted" cue once edges are fixed |
| 8 | Physical integration (shadow/AO) | Hybrid — deterministic shadow casting from an *estimated* light direction (not assumed straight-down) | Reflections explicitly out of scope for v1 (rare need, expensive) |
| 9 | Occlusion/depth | Deterministic, config-driven for v1 (explicit layer param vs. pre-supplied scene layer masks) | Full automatic scene-element occlusion deferred |
| 10 | Composite (alpha blend) | Deterministic | Already correct in `direct_composite.py` — reuse the formula as-is |
| 11 | Final harmonization | **Optional**, hybrid/generative, geometry-locked | The only stage where a generative model may help; must run as a light touch-up with the subject's position/scale/identity protected — not AnyDoor's whole-object regeneration, which we've already shown drifts identity |

## F. Implementation Plan

**Structural change**: stop growing this inside `engine/AnyDoor/scripts/` (that directory is a vendored third-party clone). Create a new top-level `compositing/` package, sibling to `engine/`, `reference/`, `ui/`, `data/`:

```
compositing/
  segmentation.py          # Stage 2: SAM2 wrapper + salient fallback, connected-component cleanup
  matting.py                # Stage 3: matting model wrapper
  decontaminate.py          # Stage 4: color decontamination given (RGB, alpha)
  transform.py               # Stage 5: deterministic affine transform; ports/de-randomizes
                             #   get_bbox_from_mask/pad_to_square from
                             #   engine/AnyDoor/datasets/data_utils.py
  scene_analysis.py          # Stage 6: color-temp/exposure/sharpness/grain estimators
  adaptation.py               # Stage 7: LAB color-statistics transfer, sharpness/grain matching
  physical_integration.py     # Stage 8: light-direction estimate, shadow casting, contact AO
  composite.py                 # Stage 10: alpha blend — port the verified formula from direct_composite.py
  harmonize.py                 # Stage 11: optional generative touch-up, off by default
  pipeline.py                   # orchestrates 1-11, exposes composite(...) API, saves
                                 #   intermediate artifacts (mask/matte/decontaminated fg/
                                 #   transformed fg/scene analysis/relit fg/shadow/
                                 #   pre-harmonization composite) when debug_dir is passed
tests/
  test_segmentation.py, test_matting.py, test_decontaminate.py, test_transform.py,
  test_composite.py, test_adaptation.py, test_physical_integration.py,
  test_harmonize.py, fixtures/ (synthetic + real per-stage test images)
```

Public API sketch (finalized during implementation, not fixed here):
```python
compositing.pipeline.composite(
    foreground, background,
    position=(x, y), scale=..., rotation=..., flip_horizontal=...,
    grounding_point=..., depth_layer=..., debug_dir=None,
) -> Image
```

**Migration**: `engine/AnyDoor/scripts/composite_from_photos.py` stays as-is — a useful "AnyDoor full-regeneration" reference mode for comparison. `direct_composite.py`'s already-debugged primitives get *ported* (not copy-pasted) into `transform.py`/`composite.py`, since this session already found and fixed real bugs in them (INTER_AREA vs. LANCZOS4 ringing, full-res-then-downsize blur ordering, connected-component-safe bbox math) — no reason to re-discover those.

**Sequencing** (build order, each independently testable before the next starts):
1. `transform.py` (pure deterministic, no model dependency, fastest to get right and unblocks everything downstream)
2. `segmentation.py` + `matting.py` (the two missing model stages)
3. `decontaminate.py`
4. `composite.py` (port existing verified blend)
5. `adaptation.py` (highest-value missing capability — do this before shadow/AO polish)
6. `physical_integration.py`
7. `scene_analysis.py` (built alongside 5/6 as needed, not necessarily first)
8. `pipeline.py` orchestration + debug-artifact saving
9. `harmonize.py` (optional, last, only after 1-8 are solid — needs the "geometry-locked" guarantee to be independently testable first)

## G. Testing Plan

- **Segmentation**: fixtures spanning high-contrast subject, fine-hair subject, low-contrast subject-vs-background; IoU vs. hand-labeled ground truth; regression test for the connected-component/noise-blob bbox bug already hit this session.
- **Alpha matting**: hair/fur fixture — assert a genuine population of alpha values strictly between 0.05-0.95 at the true boundary (not just a blurred binary step).
- **Edge decontamination**: synthetic ground-truth fixture (known-color object composited onto a strongly colored background, extracted, decontaminated) — assert recovered edge color matches true original within tolerance.
- **Alpha compositing**: unit test blend formula against hand-computed values; round-trip test on a synthetic gradient edge to catch any premultiplied/straight mismatch.
- **Geometric placement**: deterministic code — assert pixel-exact position/scale/rotation/flip against hand-computed expected transforms.
- **Hair/fur fine boundaries**: quantitative high-frequency-energy-retained metric, input crop vs. final composited boundary region.
- **Hard-edge rigid objects**: assert the pipeline does *not* over-feather a product/box photo — edge sharpness metric stays high (proves material-awareness, not one global blur setting).
- **Transparent/translucent objects**: dedicated glass/plastic fixture — assert alpha is not forced to hard 0/1.
- **Lighting adaptation**: same subject onto a warm-sunset vs. cool-overcast background — assert measurable color-temperature shift toward target without hue-category flip.
- **Color/exposure adaptation**: analogous exposure-mismatched fixture pair.
- **Shadows/grounding**: assert shadow/AO direction and density track the estimated (or specified) light direction, not a fixed generic ellipse regardless of scene.
- **Depth/occlusion**: fixture with a known scene element that should occlude the subject at a given depth/layer param.
- **Final harmonization (geometry lock)**: regression test that enabling this stage does not move the subject's centroid/bbox beyond a small tolerance and does not drop DINOv2 identity-embedding cosine similarity below a threshold.
- **Bear+beach**: kept as one regression scenario re-run after each stage lands — explicitly not the definition of done.

---

## Verification

Once implementation begins: each module in `compositing/` gets its own pytest file runnable in isolation (`pytest tests/test_<stage>.py`) against fixtures in `tests/fixtures/`, so a bad result can be attributed to a specific stage without re-running the whole pipeline. `pipeline.py`'s `debug_dir` option lets any end-to-end run be inspected stage-by-stage the same way this session's manual debugging (saving mask/alpha/crop PNGs by hand) was done — just built into the tool instead of ad hoc.
