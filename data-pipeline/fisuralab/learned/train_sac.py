"""Segment Any Crack (SAC): foundation-adapter rung of ladder A (dossier 02 section, ref
arXiv:2504.14138).

The published method adapts SAM to crack segmentation by fine-tuning ONLY the normalization
parameters of the frozen ViT image encoder, plus a lightweight decode head, and reports 61.22 percent
F1 / 44.13 percent IoU on OmniCrack30k at a fraction of the cost of a full fine-tune. This module
reproduces that recipe on Fisura's own CrackSeg9k shards so the number is comparable to the rest of
ladder A (same 512 crops, same val F1@2px protocol, same buffered metric).

What trains and what does not (this is the whole point of the method):
  - FROZEN: every ViT block weight, the patch embed, the positional embedding (~93 M params).
  - TRAINED: the LayerNorm affine params inside the encoder (~41.5 K) + the neck LayerNorm2d + a
    small conv decode head on the 256-channel image embedding (~0.4 M). The head is unavoidable:
    SAM's own mask decoder is prompt-driven, and crack segmentation here is prompt-free.

8 GB discipline: SAM ViT-B at 1024 px is heavy, so every block forward is gradient-checkpointed
(recomputed in the backward pass) and the batch is 1 with gradient accumulation. That trades compute
for memory and keeps the peak under 8 GB. Input crops are 512 (the shard size) upsampled to SAM's
fixed 1024 grid; the head output is upsampled back to 512 to score against the mask.

    python -m fisuralab.learned.train_sac --epochs 8 --accum 8

Weights are non-redistributable-adjacent only in that CrackSeg9k is CC BY (redistributable); the SAM
checkpoint is Apache-2.0. The trained head + norm deltas are small and stay in the vault; only the
metric and low-res overlays ship, consistent with the rest of the lab.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..model.metrics import buffered_prf
from .shards import data_root, read_image_mask

SAM_CKPT = Path("E:/_Models/fisura/sam/sam_vit_b_01ec64.pth")
SAM_INPUT = 1024   # SAM ViT-B positional embedding is fixed at 1024
CROP = 512         # shard crop; upsampled to SAM_INPUT for the encoder


def _lazy_torch():
    import torch  # noqa: PLC0415

    torch.backends.cudnn.benchmark = True
    return torch


def _is_norm_param(name: str, p) -> bool:
    """The SAC-trainable set: LayerNorm/LayerNorm2d affine params (1-D weight or bias)."""
    return ("norm" in name.lower()) and p.ndim <= 1


class SacCrackModel:
    """Built lazily so torch import stays inside the training entry point."""


def build_sac(sam_ckpt: Path = SAM_CKPT):
    """Return (model, n_norm_trainable, n_head_trainable). Freezes the ViT, unfreezes norm params,
    attaches a conv decode head, and gradient-checkpoints every block."""
    import torch  # noqa: PLC0415
    from segment_anything import sam_model_registry  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    enc = sam.image_encoder

    # gradient-checkpoint every transformer block: recompute activations in backward to fit 8 GB.
    for blk in enc.blocks:
        _orig = blk.forward

        def _ckpt_forward(x, _orig=_orig):
            if x.requires_grad:
                return checkpoint(_orig, x, use_reentrant=False)
            return _orig(x)

        blk.forward = _ckpt_forward

    class Head(nn.Module):
        """256-channel 64x64 image embedding -> 1-channel crack logit at 512 px. Lightweight, trainable."""

        def __init__(self, cin: int = 256):
            super().__init__()
            self.up = nn.Sequential(
                nn.Conv2d(cin, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 64->128
                nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 128->256
                nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 256->512
                nn.Conv2d(32, 1, 1),
            )

        def forward(self, emb):
            return self.up(emb)

    class Model(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            emb = self.encoder(x)           # (B,256,64,64)
            return self.head(emb)           # (B,1,512,512)

    head = Head(cin=enc.neck[-1].weight.shape[0] if hasattr(enc.neck[-1], "weight") else 256)
    model = Model(enc, head)

    # freeze everything, then unfreeze the norm affine params
    n_norm = 0
    for name, p in model.encoder.named_parameters():
        if _is_norm_param(name, p):
            p.requires_grad_(True)
            n_norm += p.numel()
        else:
            p.requires_grad_(False)
    n_head = sum(p.numel() for p in model.head.parameters())
    _ = torch
    return model, n_norm, n_head


def _to_chw3(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return np.transpose(x, (2, 0, 1))


def _pad_crop(img, mask, rng, train: bool):
    h, w = mask.shape
    ph, pw = max(0, CROP - h), max(0, CROP - w)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)) + (((0, 0),) if img.ndim == 3 else ()), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        h, w = mask.shape
    if train:
        r = int(rng.integers(0, h - CROP + 1))
        c = int(rng.integers(0, w - CROP + 1))
    else:
        r = (h - CROP) // 2
        c = (w - CROP) // 2
    return img[r:r + CROP, c:c + CROP], mask[r:r + CROP, c:c + CROP]


def _augment(img, mask, rng):
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1, :], mask[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        img, mask = np.rot90(img, k, axes=(0, 1)), np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SacDataset:
    """SAM normalizes internally (its own pixel_mean/std) but that lives in sam.forward, not the bare
    image_encoder, so we feed the encoder the SAM-normalized 1024-px tensor ourselves."""

    # SAM's published pixel statistics (RGB, 0-255 scale)
    PIXEL_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    PIXEL_STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, records, train: bool, seed: int):
        self.records = records
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        rng = np.random.default_rng(self.seed * 100_003 + i)
        img, mask = read_image_mask(self.records[i])
        img, mask = _pad_crop(img, mask, rng, self.train)
        if self.train:
            img, mask = _augment(img, mask, rng)
        rgb = img if img.ndim == 3 else np.stack([img] * 3, -1)
        rgb = rgb[..., :3].astype(np.float32)
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = (rgb - self.PIXEL_MEAN) / self.PIXEL_STD
        x = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))[None]          # (1,3,512,512)
        x = F.interpolate(x, size=SAM_INPUT, mode="bilinear", align_corners=False)[0]
        y = torch.from_numpy(mask.astype(np.float32))[None]               # (1,512,512)
        return x, y


def _dice_loss(logits, target, eps: float = 1.0):
    import torch  # noqa: PLC0415

    p = torch.sigmoid(logits.float())
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def train(epochs: int = 8, accum: int = 8, seed: int = 42, limit_train: int | None = None,
          limit_val: int | None = None) -> dict:
    torch = _lazy_torch()
    from torch.utils.data import DataLoader  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    index_path = data_root() / "derived" / "shards" / "crackseg9k" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    train_recs = index["train"][:limit_train] if limit_train else index["train"]
    val_recs = index["val"][:limit_val] if limit_val else index["val"]

    dl_train = DataLoader(SacDataset(train_recs, True, seed), batch_size=1, shuffle=True,
                          num_workers=2, pin_memory=True)
    dl_val = DataLoader(SacDataset(val_recs, False, seed), batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    model, n_norm, n_head = build_sac()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(dl_train)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))

    out_dir = data_root() / "derived" / "learned" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "sac_samvitb.pt"

    print(f"SAC: trainable = {n_norm/1e3:.1f}K norm + {n_head/1e3:.1f}K head "
          f"({sum(p.numel() for p in trainable)/1e3:.1f}K total); {len(train_recs)} train / {len(val_recs)} val")

    best = {"f1_2px": -1.0, "epoch": -1}
    history = []
    n_skipped = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(dl_train):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
            # losses in fp32 outside autocast (the dacl10k lesson: fp16 dice sums underflow to NaN)
            loss = (bce(logits.float(), y.float()) + _dice_loss(logits, y)) / accum
            if not torch.isfinite(loss):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += float(loss.detach()) * accum

        model.eval()
        f1s = []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                    logits = model(x)
                pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0].astype(bool)
                gt = y.numpy()[:, 0].astype(bool)
                for p, g in zip(pred, gt):
                    f1s.append(buffered_prf(p, g, 2)["f1"])
        f1 = float(np.mean(f1s)) if f1s else 0.0
        history.append({"epoch": epoch, "train_loss": running / max(1, len(dl_train)), "val_f1_2px": f1})
        print(f"[sac] epoch {epoch}: loss {history[-1]['train_loss']:.4f} val F1@2px {f1:.4f}"
              + (f" (skipped {n_skipped})" if n_skipped else ""))
        if f1 > best["f1_2px"] and np.isfinite(f1):
            best = {"f1_2px": f1, "epoch": epoch}
            # save ONLY the trainable deltas (norm + head): small, and all that changed
            torch.save({k: v for k, v in model.state_dict().items()
                        if any(k == n for n, p in model.named_parameters() if p.requires_grad)}, ckpt)

    record = {
        "arch": "sac_samvitb",
        "method": "Segment Any Crack (SAC): norm-only tuning of SAM ViT-B + light decode head",
        "reference": "Rostami, Chen, Hosseini, arXiv:2504.14138 (2025)",
        "published_reference_point": {"dataset": "OmniCrack30k", "f1": 0.6122, "iou": 0.4413},
        "seed": seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "trainable_norm_params": n_norm,
        "trainable_head_params": n_head,
        "frozen_encoder_params": sum(p.numel() for p in model.encoder.parameters()) - n_norm,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "crop": CROP,
        "sam_input": SAM_INPUT,
        "batch": 1,
        "accum": accum,
        "loss": "BCE(pos_weight=8) + Dice (fp32)",
        "optimizer": "AdamW 3e-4 cosine",
        "grad_checkpointing": True,
        "amp": device == "cuda",
        "nonfinite_steps_skipped": n_skipped,
        "best_val_f1_2px": best["f1_2px"],
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_minutes": round((time.perf_counter() - t0) / 60.0, 1),
        "checkpoint": str(ckpt),
        "history": history,
    }
    (out_dir / "sac_samvitb.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"-> best val F1@2px {best['f1_2px']:.4f} @ epoch {best['epoch']}; {record['train_minutes']} min")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    a = ap.parse_args()
    train(epochs=a.epochs, accum=a.accum, seed=a.seed, limit_train=a.limit_train, limit_val=a.limit_val)
