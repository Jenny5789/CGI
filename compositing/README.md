# compositing/

General-purpose photorealistic compositing pipeline. See
[docs/architecture.md](../docs/architecture.md) for the full design.

## Two virtual environments -- read this before installing anything here

This package spans two **separate, non-interchangeable** Python
environments:

- **`.venv`** (repo root) -- AnyDoor's environment. Pinned to
  `torch==2.0.0+cu118`, `xformers==0.0.18`, `transformers==4.40.0`. Runs
  `compositing.transform`, `compositing.composite`, `compositing.decontaminate`
  (pure numpy/opencv, no heavy deps) as well as everything under
  `engine/AnyDoor/`.
- **`.venv-seg`** (repo root) -- SAM2 + ViTMatte's environment. Needs
  `torch>=2.5`. Runs `compositing/_seg_worker.py` and
  `compositing/_matte_worker.py`.

**Do not `pip install` SAM2, ViTMatte, or a newer transformers into `.venv`.**
This was tried once and it silently force-upgraded `torch` to a CPU-only
2.13, `numpy` to 2.2.6, and `setuptools` to a version that dropped
`pkg_resources` -- breaking AnyDoor's xformers build and pytorch_lightning
entirely. All three had to be pinned back by hand and re-verified against
the AnyDoor smoke test before the environment was trustworthy again. If you
need a new model dependency for a `compositing/` stage, check whether it
needs `.venv`'s or `.venv-seg`'s constraints first, and if neither fits,
this project's dependency footprint has genuinely diverged enough to warrant
a third venv rather than forcing it into one of these two.

`compositing.segmentation` and `compositing.matting` are the bridge: they
run under `.venv`, but call out to `.venv-seg`'s python via `subprocess`
(passing images through temp files) so the public API stays in one place
without either venv having to see the other's packages.

## Setting up `.venv-seg`

```bash
py -3.10 -m venv .venv-seg
.venv-seg/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv-seg/Scripts/python.exe -m pip install "git+https://github.com/facebookresearch/sam2.git" transformers opencv-python pillow
```

Checkpoints (already downloaded into `checkpoints/`, gitignored):
- `checkpoints/sam2/sam2.1_hiera_base_plus.pt` (~324MB, Meta)
- `checkpoints/vitmatte/` (`hustvl/vitmatte-small-composition-1k`, ~103MB, HuggingFace)
