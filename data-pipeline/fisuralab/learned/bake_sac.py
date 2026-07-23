"""Bake the SAC (Segment Any Crack) overlays + metric card for the foundation-adapter rung.

Loads the norm-only-tuned SAM ViT-B checkpoint (train_sac.py), predicts crack masks on the committed
examples, writes a blended overlay per sample and the per-sample F1 against ground truth, and records
the published reference point (61.22 F1 / 44.13 IoU on OmniCrack30k) alongside this run's own val
F1@2px so the app can state the method's cost/benefit honestly.

Prediction is tiled at the shard crop (512), each tile upsampled to SAM's fixed 1024 grid, the 512
logit downsampled back, and tiles blended with reflect padding, matching training.predict_full but for
the SAM encoder + head. Only the low-res overlay and the metric ship; the checkpoint stays in the vault.

    python -m fisuralab.learned.bake_sac
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
OUT = REPO_ROOT / "data" / "derived" / "sac"
CROP = 512
SAM_INPUT = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], np.float32)


def _checkpoint() -> Path:
    from .shards import data_root  # noqa: PLC0415

    return data_root() / "derived" / "learned" / "checkpoints" / "sac_samvitb.pt"


def _record_path() -> Path:
    from .shards import data_root  # noqa: PLC0415

    return data_root() / "derived" / "learned" / "checkpoints" / "sac_samvitb.json"


def _predict(model, img: np.ndarray, device, tile: int = CROP, overlap: int = 64) -> np.ndarray:
    """Tiled crack-probability map in [0,1]. Each 512 tile is upsampled to 1024 for the SAM encoder."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    rgb = img if img.ndim == 3 else np.stack([img] * 3, -1)
    rgb = rgb[..., :3].astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    h, w = rgb.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    hh, ww = rgb.shape[:2]
    prob = np.zeros((hh, ww), np.float32)
    weight = np.zeros((hh, ww), np.float32)
    step = tile - overlap
    norm = (rgb - PIXEL_MEAN) / PIXEL_STD
    with torch.no_grad():
        for r in range(0, max(1, hh - tile + 1), step):
            for c in range(0, max(1, ww - tile + 1), step):
                patch = norm[r:r + tile, c:c + tile]
                x = torch.from_numpy(np.transpose(patch, (2, 0, 1)))[None].to(device)
                x = F.interpolate(x, size=SAM_INPUT, mode="bilinear", align_corners=False)
                with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                    logits = model(x)                    # (1,1,512,512)
                p = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
                prob[r:r + tile, c:c + tile] += p
                weight[r:r + tile, c:c + tile] += 1.0
    prob = prob / np.maximum(weight, 1e-6)
    return prob[:h, :w]


def _overlay_png(base_u8: np.ndarray, mask: np.ndarray) -> bytes:
    """Blend the predicted mask as a magenta overlay on the image."""
    import imageio.v3 as iio  # noqa: PLC0415

    if base_u8.ndim == 2:
        base_u8 = np.stack([base_u8] * 3, -1)
    out = base_u8[..., :3].astype(np.float32)
    a = mask.astype(np.float32)[..., None]
    tint = np.array([230, 40, 200], np.float32)
    out = out * (1 - a * 0.55) + tint * (a * 0.55)
    buf = io.BytesIO()
    iio.imwrite(buf, np.clip(out, 0, 255).astype(np.uint8), extension=".png")
    return buf.getvalue()


def main() -> None:
    import torch  # noqa: PLC0415

    from ..core.artifact import rle_decode  # noqa: PLC0415
    from ..model.metrics import buffered_prf  # noqa: PLC0415
    from .train_sac import build_sac  # noqa: PLC0415

    ckpt = _checkpoint()
    if not ckpt.exists():
        print(f"  no SAC checkpoint at {ckpt}; run `python -m fisuralab.learned.train_sac` first")
        return

    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    model, n_norm, n_head = build_sac()
    state = torch.load(ckpt, map_location=device)
    # checkpoint holds only the trainable deltas (norm + head); load non-strictly onto the full model
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"loaded SAC deltas ({len(state)} tensors); {len(missing)} frozen params kept from SAM")

    ex = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    samples = ex["samples"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    for s in samples:
        sid = s["sample_id"]
        img = read_image(REPO_EXAMPLES / s["image_rel"])
        prob = _predict(model, img, device)
        mask = prob > 0.5
        base = img[..., :3] if img.ndim == 3 else img
        (OUT / "overlays" / f"{sid}_sac.png").write_bytes(_overlay_png(base, mask))
        f1_2 = f1_5 = None
        if s.get("gt_rle"):
            gt = rle_decode(s["gt_rle"]).astype(bool)
            f1_2 = round(float(buffered_prf(mask, gt, 2)["f1"]), 4)
            f1_5 = round(float(buffered_prf(mask, gt, 5)["f1"]), 4)
        rows.append({"id": sid, "overlay": f"sac/overlays/{sid}_sac.png",
                     "pred_positive_frac": round(float(mask.mean()), 5),
                     "f1_2px": f1_2, "f1_5px": f1_5})
        print(f"  {sid}: pred {mask.mean()*100:.2f}%"
              + (f" F1@2px {f1_2}" if f1_2 is not None else " (no GT)"))

    train_rec = json.loads(_record_path().read_text(encoding="utf-8")) if _record_path().exists() else {}
    rec = {
        "schema": "fisura.sac/v1",
        "method": "Segment Any Crack (SAC): SAM ViT-B, norm-only tuning + light decode head",
        "reference": "Rostami, Chen, Hosseini, 'Segment Any Crack', arXiv:2504.14138 (2025)",
        "published_reference_point": {"dataset": "OmniCrack30k", "f1": 0.6122, "iou": 0.4413},
        "trainable_norm_params": n_norm,
        "trainable_head_params": n_head,
        "frozen_encoder_params": sum(p.numel() for p in model.encoder.parameters()) - n_norm,
        "val_f1_2px": train_rec.get("best_val_f1_2px"),
        "n_train": train_rec.get("n_train"),
        "n_val": train_rec.get("n_val"),
        "train_minutes": train_rec.get("train_minutes"),
        "samples": rows,
        "framing": (
            "The foundation-adapter rung: freeze SAM's 93 M-parameter ViT and tune ONLY its "
            f"{n_norm/1e3:.0f}K normalization parameters plus a {n_head/1e3:.0f}K decode head. That "
            "is under 0.5 percent of the network, yet it adapts a general segmenter to cracks. It shows "
            "the trade the rest of the ladder is measured against: near-frozen foundation weights vs a "
            "small task head, at a fraction of a full fine-tune's cost."
        ),
        "limitation": (
            "SAM was built for prompt-driven, object-scale segmentation; a 1-5 px crack is far from its "
            "training distribution, so the prompt-free head does the heavy lifting and the norm deltas "
            "only re-tune the statistics. The published 61.22 F1 is on OmniCrack30k, a larger and cleaner "
            "benchmark than these few examples; the per-sample numbers here are illustrative, not a "
            "benchmark result."
        ),
    }
    with open(OUT / "sac.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"-> {OUT / 'sac.json'}  ({len(rows)} overlays)")


if __name__ == "__main__":
    main()
