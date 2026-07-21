"""Bake Grad-CAM overlays for the learned segmenters (enrichment shortlist #9).

What it answers: a mask tells you WHERE the model fired; Grad-CAM tells you WHAT EVIDENCE drove it.
For a segmentation head the scalar to explain is the summed crack logit over the predicted region;
its gradient w.r.t. the last encoder feature map, global-average-pooled per channel, weights those
channels, and the ReLU'd weighted sum is the class-activation map (Selvaraju et al., ICCV 2017).

Reading it honestly: a CAM that concentrates on the crack means the decision rests on the crack; a CAM
smeared over background texture means the model is leaning on context and will transfer badly. That
diagnostic is exactly what a mask alone hides, and it is why the enrichment research ranked it.

Target layer: `encoder.layer3` of the ResNet-18 backbones. layer4 is the textbook site but at this
input size it is a 12x12 grid, so its CAM upsamples to a blob; layer3 (24x24) keeps deep semantics
while resolving a thin crack path. U-Net R18 and DeepLabV3+ R18 are covered; SegFormer's transformer stages have no
equivalent single conv site, so it is deliberately excluded rather than faked.

Writes data/derived/gradcam/cam.json + overlays/<sample>_<arch>.png (CAM blended on the image).

    python -m fisuralab.learned.bake_gradcam
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
OUT = REPO_ROOT / "data" / "derived" / "gradcam"
SIZE = 384
ARCHS = ["unet_r18", "deeplabv3p_r18"]  # ResNet-18 encoders: a real layer4 to hook


def _model_root() -> Path:
    from .shards import data_root  # noqa: PLC0415

    root = data_root() / "derived" / "learned" / "checkpoints"
    return root


def _build(arch: str):
    import segmentation_models_pytorch as smp  # noqa: PLC0415

    if arch == "unet_r18":
        return smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    if arch == "deeplabv3p_r18":
        return smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    raise ValueError(arch)


def _prep(img: np.ndarray) -> np.ndarray:
    from skimage.transform import resize as rz  # noqa: PLC0415

    g = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    if g.ndim == 2:
        g = np.stack([g, g, g], -1)
    g = rz(g[..., :3], (SIZE, SIZE), order=1, preserve_range=True).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return ((g - mean) / std).transpose(2, 0, 1)


def _cam_png(base_u8: np.ndarray, cam: np.ndarray) -> bytes:
    """Blend a [0,1] CAM over the base image as a blue->red heat, weighted by the CAM itself."""
    import imageio.v3 as iio  # noqa: PLC0415

    r = (cam * 235).astype(np.float32)
    g = (np.clip(1 - np.abs(cam - 0.5) * 2, 0, 1) * 120).astype(np.float32)
    b = ((1 - cam) * 220).astype(np.float32)
    heat = np.dstack([r, g, b])
    a = np.clip(cam * 1.15, 0, 1)[..., None]
    out = base_u8.astype(np.float32) * (1 - a * 0.75) + heat * (a * 0.75)
    buf = io.BytesIO()
    iio.imwrite(buf, np.clip(out, 0, 255).astype(np.uint8), extension=".png")
    return buf.getvalue()


def main() -> None:
    import torch  # noqa: PLC0415
    from skimage.transform import resize as rz  # noqa: PLC0415

    ex = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    samples = ex["samples"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)

    out_rows = []
    archs_done = []
    for arch in ARCHS:
        ckpt = _model_root() / f"{arch}.pt"
        if not ckpt.exists():
            print(f"  skip {arch}: no checkpoint at {ckpt}")
            continue
        net = _build(arch).to(device)
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state else state["state_dict"])
        net.eval()

        feats: dict = {}
        grads: dict = {}
        # layer3, not layer4: at 384 px input layer4 is a 12x12 grid, so its CAM upsamples to a
        # coarse blob that says little about a 1-5 px wide crack. layer3 (24x24) keeps the standard
        # deep-semantic Grad-CAM site while resolving the crack path.
        target = net.encoder.layer3

        def fwd_hook(_m, _i, o):
            feats["a"] = o

        def bwd_hook(_m, _gi, go):
            grads["g"] = go[0]

        h1 = target.register_forward_hook(fwd_hook)
        h2 = target.register_full_backward_hook(bwd_hook)

        for s in samples:
            img = read_image(REPO_EXAMPLES / s["image_rel"])
            x = torch.from_numpy(_prep(img)[None]).to(device)
            x.requires_grad_(False)
            net.zero_grad(set_to_none=True)
            logits = net(x)                       # (1,1,H,W)
            # the scalar to explain: total crack evidence over the predicted-positive region
            score = (logits * (logits > 0).float()).sum()
            if float(score.detach()) == 0.0:      # nothing predicted: explain the max logit instead
                score = logits.max()
            score.backward()

            a = feats["a"].detach()[0]            # (C,h,w)
            g = grads["g"].detach()[0]            # (C,h,w)
            w = g.mean(dim=(1, 2), keepdim=True)  # channel weights = GAP of gradients
            cam = torch.relu((w * a).sum(dim=0)).cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            cam_big = rz(cam, (SIZE, SIZE), order=1, preserve_range=True).astype(np.float32)

            from skimage.transform import resize as rz2  # noqa: PLC0415

            base = rz2(img[..., :3] if img.ndim == 3 else np.stack([img] * 3, -1),
                       (SIZE, SIZE), order=1, preserve_range=True).astype(np.uint8)
            sid = s["sample_id"]
            (OUT / "overlays" / f"{sid}_{arch}.png").write_bytes(_cam_png(base, cam_big))
            # a scalar honesty metric: how much of the CAM mass falls on the true crack (if GT exists)
            on_crack = None
            note = None
            degenerate = float(cam_big.sum()) <= 0.0
            if degenerate:
                note = "degenerate CAM (all-zero after ReLU): the gradient gave no positive evidence here"
            elif s.get("gt_rle"):
                from ..core.artifact import rle_decode  # noqa: PLC0415

                gt = rle_decode(s["gt_rle"]).astype(bool)
                gt_small = rz2(gt.astype(np.float32), (SIZE, SIZE), order=0, preserve_range=True) > 0.5
                on_crack = round(float(cam_big[gt_small].sum() / float(cam_big.sum())), 4)
            elif not s.get("gt_rle"):
                note = "no pixel ground truth for this sample: the on-crack fraction is undefined"
            out_rows.append({"id": sid, "arch": arch, "cam": f"gradcam/overlays/{sid}_{arch}.png",
                             "cam_mass_on_crack": on_crack, "note": note})
        h1.remove()
        h2.remove()
        archs_done.append(arch)
        print(f"  {arch}: CAMs baked for {len(samples)} images")

    rec = {
        "schema": "fisura.gradcam/v1",
        "method": "Grad-CAM (Selvaraju et al., ICCV 2017) on encoder layer3 (layer4 is a 12x12 grid at this input size, too coarse for thin cracks)",
        "explained_scalar": "summed positive crack logit over the predicted region",
        "archs": archs_done,
        "excluded": {"segformer_b2": "transformer stages have no single conv site; excluded rather than faked",
                     "hrsegnet_b16": "custom architecture, no resnet layer4",
                     "dinov2s14_linear": "frozen features; see the DINOv2 feature-PCA view instead"},
        "samples": out_rows,
        "limitation": (
            "Grad-CAM was designed for whole-image classification. On a 1-5 px wide crack the encoder "
            "grid is coarse (24x24 at layer3 for a 384 px input), so these maps are genuinely sparse: "
            "they localise the evidence to a region, not to the crack outline. Occasional inputs yield a "
            "degenerate all-zero CAM, which is recorded per sample rather than hidden."
        ),
        "framing": (
            "A mask says WHERE the model fired; Grad-CAM says what evidence drove it. CAM mass "
            "concentrated on the crack means the decision rests on the crack; CAM smeared over "
            "background texture means the model leans on context and will transfer badly. The "
            "cam_mass_on_crack fraction quantifies exactly that, per image and per architecture."
        ),
    }
    with open(OUT / "cam.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"-> {OUT / 'cam.json'}  ({len(out_rows)} CAMs)")


if __name__ == "__main__":
    main()
