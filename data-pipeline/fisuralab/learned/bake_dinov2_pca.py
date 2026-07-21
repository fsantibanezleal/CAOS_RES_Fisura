"""Bake the DINOv2 dense-feature PCA-to-RGB overlays (enrichment shortlist #8).

What it shows: what a foundation model actually "sees" on a crack surface, before any crack-specific
head. DINOv2 emits a 384-dim descriptor per 14x14 patch; projecting those descriptors onto their first
three principal components and mapping them to RGB renders the feature space directly as an image.
Semantically similar regions get similar colours, so the crack separates from the background as a
distinct hue WITHOUT any supervision. That is the honest visual argument for (and limit of) frozen
foundation features: they carry the structure, and the tiny linear probe only has to read it out.

PCA is fit PER IMAGE: a basis shared across the whole example set ends up encoding which image a patch
came from (steel vs concrete vs the noise control dominate the variance) and washes out the within-image
structure this view exists to show. Component signs are fixed deterministically (largest-absolute
loading positive) so the colours are reproducible run to run.

Writes data/derived/dinov2/pca.json + overlays/<sample_id>_pca.png (37x37 upsampled to the image).

    python -m fisuralab.learned.bake_dinov2_pca
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from ..io.image_formats import read_image

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "data" / "derived" / "bcl_examples" / "artifact.json"
REPO_EXAMPLES = REPO_ROOT / "data" / "examples"
OUT = REPO_ROOT / "data" / "derived" / "dinov2"
INPUT = 518          # DINOv2 ViT-S/14 -> 37x37 patch grid
GRID = 37
VIEW = 384           # overlay render size


def _prep(img: np.ndarray) -> np.ndarray:
    from skimage.transform import resize as rz  # noqa: PLC0415

    g = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    if g.ndim == 2:
        g = np.stack([g, g, g], -1)
    g = rz(g[..., :3], (INPUT, INPUT), order=1, preserve_range=True).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return ((g - mean) / std).transpose(2, 0, 1)


def _png(rgb_u8: np.ndarray) -> bytes:
    import imageio.v3 as iio  # noqa: PLC0415

    buf = io.BytesIO()
    iio.imwrite(buf, rgb_u8, extension=".png")
    return buf.getvalue()


def main() -> None:
    import torch  # noqa: PLC0415
    from skimage.transform import resize as rz  # noqa: PLC0415

    from .dinov2_probe import build_dinov2_probe  # noqa: PLC0415

    ex = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    samples = ex["samples"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = build_dinov2_probe().to(device)
    backbone = probe.bb
    backbone.eval()

    # 1) extract patch descriptors for every example
    feats_all = []
    ids = []
    for s in samples:
        img = read_image(REPO_EXAMPLES / s["image_rel"])
        x = torch.from_numpy(_prep(img)[None]).to(device)
        with torch.no_grad():
            f = backbone.forward_features(x)["x_norm_patchtokens"][0]  # (GRID*GRID, 384)
        feats_all.append(f.cpu().numpy().astype(np.float32))
        ids.append(s["sample_id"])

    # 2) PER-IMAGE PCA. A basis shared across all the examples ends up encoding WHICH image a patch
    # came from (steel vs concrete vs noise-control dominate the variance), which washes out the
    # within-image structure. Fitting per image is what makes the crack separate from its own
    # background, which is the point of the view.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    out_samples = []
    var_top3 = []
    for sid, f in zip(ids, feats_all, strict=False):
        mu = f.mean(axis=0, keepdims=True)
        centered = f - mu
        _u, sv, vt = np.linalg.svd(centered, full_matrices=False)
        pcs = vt[:3]
        for i in range(3):  # deterministic sign
            if pcs[i][np.argmax(np.abs(pcs[i]))] < 0:
                pcs[i] = -pcs[i]
        proj = centered @ pcs.T                       # (GRID*GRID, 3)
        lo = np.percentile(proj, 2, axis=0)
        hi = np.percentile(proj, 98, axis=0)
        norm = np.clip((proj - lo) / np.maximum(1e-6, hi - lo), 0, 1)
        rgb = norm.reshape(GRID, GRID, 3)
        big = rz(rgb, (VIEW, VIEW), order=1, preserve_range=True)
        (OUT / "overlays" / f"{sid}_pca.png").write_bytes(_png((big * 255).astype(np.uint8)))
        v = (sv ** 2) / (sv ** 2).sum()
        var_top3.append([round(float(x), 4) for x in v[:3]])
        out_samples.append({"id": sid, "pca": f"dinov2/overlays/{sid}_pca.png", "explained_variance_top3": var_top3[-1]})
        print(f"  {sid}: PCA-RGB baked (var top3 {var_top3[-1]})")

    stack = np.concatenate(feats_all, axis=0)
    var = np.mean(np.array(var_top3), axis=0)
    rec = {
        "schema": "fisura.dinov2pca/v1",
        "backbone": "DINOv2 ViT-S/14 (frozen, ImageNet-free self-supervised)",
        "input": INPUT,
        "patch_grid": GRID,
        "feature_dim": int(stack.shape[1]),
        "explained_variance_top3": [round(float(v), 4) for v in var[:3]],
        "pca_scope": "per image (a shared basis encodes which image a patch came from, not its content)",
        "samples": out_samples,
        "framing": (
            "Each 14x14 patch gets a 384-dim DINOv2 descriptor; the first three principal components of "
            "those descriptors, fit per image, map to RGB. Similar material reads as a similar colour with "
            "NO crack supervision, which is why a 385-parameter linear head on these frozen features is "
            "already competitive. The grid is 37x37, so the view is coarse by construction: it shows the "
            "semantic structure the probe reads out, not a mask."
        ),
    }
    with open(OUT / "pca.json", "w", encoding="utf-8", newline="\n") as fjson:
        json.dump(rec, fjson, ensure_ascii=False, indent=1)
    print(f"explained variance (top 3): {rec['explained_variance_top3']}")
    print(f"-> {OUT / 'pca.json'}")


if __name__ == "__main__":
    main()
