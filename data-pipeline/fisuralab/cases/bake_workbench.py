"""Bake the per-image workbench artifacts for the App (ADR-0016 section 9 workbench, Felipe's spec
2026-07-20): the preprocessing intermediates and the SLIC superpixel grid, on the SAME committed
example images the method ladders run on.

The App workbench has, per selected image, tabs that need baked replay artifacts:
  - Preprocessing: the real classical stage outputs (gray, illumination-flattened, NLM-denoised,
    Hessian ridge response) as committed PNGs, so the reader SEES what each preprocessing step does.
  - SLIC: a grid of superpixel segmentations over (n_segments x compactness); the left-column
    sliders pick the nearest baked variant (replay; the live in-browser SLIC is the BL-013 upgrade).

scikit-image is CPU-only and cheap here, so this bakes in seconds. Outputs are committed (git-as-data):
  data/derived/workbench/<sample_id>/prep_{gray,flatten,denoise,ridge}.png
  data/derived/workbench/<sample_id>/slic_n{N}_c{C}.png   (boundaries overlaid on the image)
  data/derived/workbench/index.json                        (the grid axes + per-sample file map)

    python -m fisuralab.cases.bake_workbench
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io.image_formats import read_image
from ..model.classical import denoise_nlm, flatten_median, ridge_response, to_gray_float

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "data" / "derived" / "bcl_examples" / "artifact.json"
REPO_EXAMPLES = REPO_ROOT / "data" / "examples"
OUT = REPO_ROOT / "data" / "derived" / "workbench"

# SLIC grid axes (the left-column sliders snap to these). Kept small so the bake + repo stay light.
SLIC_N = [80, 150, 300]          # number of superpixels
SLIC_C = [5, 10, 20]             # compactness (colour-vs-space balance)

# Hessian ridge scale-space: the individual sigmas whose per-scale response the viewer sweeps, plus an
# argmax-sigma map (which scale fires strongest per pixel) that shows WHY multi-scale matters.
RIDGE_SIGMAS = [1.0, 2.0, 3.0, 4.0]


def _to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def _write_png(path: Path, arr_u8: np.ndarray) -> None:
    import imageio.v3 as iio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, arr_u8)


def _slic_boundaries(image_rgb: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    """SLIC superpixels; return the boundary overlay (yellow edges) on the image as uint8 RGB."""
    from skimage.segmentation import mark_boundaries, slic  # noqa: PLC0415

    img = image_rgb.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    seg = slic(img, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=-1)
    ov = mark_boundaries(img, seg, color=(1.0, 0.85, 0.1))
    return _to_u8(ov), int(seg.max())


def main() -> None:
    ex = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    samples = ex["samples"]
    OUT.mkdir(parents=True, exist_ok=True)

    index = {
        "schema": "fisura.workbench/v1",
        "slic_grid": {"n_segments": SLIC_N, "compactness": SLIC_C},
        "samples": {},
    }

    for s in samples:
        sid = s["sample_id"]
        img = read_image(REPO_EXAMPLES / s["image_rel"])
        gray = to_gray_float(img)  # HxW float [0,1]
        flat = flatten_median(gray, radius=21)
        den = denoise_nlm(flat, h_factor=0.8)
        ridge = ridge_response(den, method="sato", sigmas=(1.0, 2.0, 3.0, 4.0))
        ridge_norm = (ridge - ridge.min()) / max(1e-6, ridge.max() - ridge.min())

        d = OUT / sid
        _write_png(d / "prep_gray.png", _to_u8(gray))
        _write_png(d / "prep_flatten.png", _to_u8(flat))
        _write_png(d / "prep_denoise.png", _to_u8(den))
        _write_png(d / "prep_ridge.png", _to_u8(ridge_norm))

        slic_files = {}
        real_counts = {}
        for n in SLIC_N:
            for cc in SLIC_C:
                ov, real_n = _slic_boundaries(img, n, cc)
                fn = f"slic_n{n}_c{cc}.png"
                _write_png(d / fn, ov)
                slic_files[f"{n}_{cc}"] = f"workbench/{sid}/{fn}"
                real_counts[f"{n}_{cc}"] = real_n

        # ridge scale-space: per-sigma response + the argmax-sigma map (which scale wins per pixel)
        scale_files = {}
        stack = []
        for sig in RIDGE_SIGMAS:
            r = ridge_response(den, method="sato", sigmas=(sig,))
            stack.append(r)
            fn = f"ridge_s{sig:g}.png"
            _write_png(d / fn, _to_u8(r))
            scale_files[f"{sig:g}"] = f"workbench/{sid}/{fn}"
        arr = np.stack(stack, axis=0)              # (S, H, W)
        argmax = arr.argmax(axis=0).astype(np.float32) / max(1, len(RIDGE_SIGMAS) - 1)
        # tint the winning-scale index blue->red only where a ridge actually responds, else transparent
        strength = arr.max(axis=0)
        h, w = argmax.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0] = (argmax * 235).astype(np.uint8)
        rgba[..., 2] = ((1 - argmax) * 220).astype(np.uint8)
        rgba[..., 3] = (np.clip(strength * 1.6, 0, 1) * 220).astype(np.uint8)
        _write_png(d / "ridge_argmax.png", rgba)
        scale_files["argmax"] = f"workbench/{sid}/ridge_argmax.png"

        index["samples"][sid] = {
            "material": s["material"],
            "size": s["size"],
            "prep": {
                "gray": f"workbench/{sid}/prep_gray.png",
                "flatten": f"workbench/{sid}/prep_flatten.png",
                "denoise": f"workbench/{sid}/prep_denoise.png",
                "ridge": f"workbench/{sid}/prep_ridge.png",
            },
            "slic": slic_files,
            "slic_real_counts": real_counts,
            "scale_space": {"sigmas": RIDGE_SIGMAS, "maps": scale_files},
        }
        print(f"  {sid}: 4 prep + {len(slic_files)} SLIC + {len(RIDGE_SIGMAS)} scales")

    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"-> {OUT / 'index.json'}  ({len(samples)} samples)")


if __name__ == "__main__":
    main()
