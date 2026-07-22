"""Training for ladder A (torch imported lazily; 8 GB discipline from the research).

Recipe (dossier 02 section 9): 512 crops with reflect padding, batch 6 with 2-step gradient
accumulation (effective 12), AdamW 3e-4 cosine, AMP bf16/fp16, BCE + Dice loss, seeded everything,
early stopping on val F1@2px (the stricter protocol), max 12 epochs (the sets are small; minutes
per epoch on the 4070). Deterministic given (arch, seed) up to cuDNN nondeterminism, which is
disabled where torch allows it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..model.metrics import buffered_prf
from .shards import read_image_mask

ARCHS = ("unet_r18", "deeplabv3p_r18", "segformer_b2")
CROP = 512


def _lazy_torch():
    import torch  # noqa: PLC0415

    torch.backends.cudnn.benchmark = True
    return torch


def _device(torch) -> str:
    """'cuda' only when a device is really usable.

    torch.cuda.is_available() can return True while device_count() is 0, which is exactly what
    happens under CUDA_VISIBLE_DEVICES="" on this build. The old check then selected 'cuda', and
    torch.load(map_location='cuda') died with "Attempting to deserialize object on CUDA device 0 but
    torch.cuda.device_count() is 0". Checking the count as well makes CPU-only runs work.
    """
    return "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"


def build_model(arch: str):
    import segmentation_models_pytorch as smp  # noqa: PLC0415

    if arch == "unet_r18":
        return smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
    if arch == "deeplabv3p_r18":
        return smp.DeepLabV3Plus("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
    if arch == "segformer_b2":
        return smp.Segformer("mit_b2", encoder_weights="imagenet", in_channels=3, classes=1)
    if arch.startswith("hrsegnet_b"):
        from .hrsegnet import build_hrsegnet  # noqa: PLC0415

        return _TwoClassAsLogit(build_hrsegnet(base=int(arch.split("_b")[1])))
    raise ValueError(f"unknown arch {arch}")


def _TwoClassAsLogit(net):
    """Wrap the 2-class HrSegNet so it exposes the same 1-channel-logit interface as the SMP models
    (logit = crack-class logit minus background logit; sigmoid of that equals 2-class softmax crack prob)."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, tuple):
                out = out[0]
            return (out[:, 1:2] - out[:, 0:1])

        def load_state_dict(self, sd, **kw):
            # checkpoints are saved from the RAW HrSegNet; delegate so both layouts load
            if any(k.startswith("inner.") for k in sd):
                return super().load_state_dict(sd, **kw)
            return self.inner.load_state_dict(sd, **kw)

    _ = torch
    return Wrap(net)


def _to_chw3(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return np.transpose(x, (2, 0, 1))


def _pad_crop(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, train: bool) -> tuple[np.ndarray, np.ndarray]:
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
    return img[r : r + CROP, c : c + CROP], mask[r : r + CROP, c : c + CROP]


def _augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1, :], mask[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        img, mask = np.rot90(img, k, axes=(0, 1)), np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class CrackDataset:
    def __init__(self, records: list[dict], train: bool, seed: int):
        self.records = records
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        torch = _lazy_torch()
        rng = np.random.default_rng(self.seed * 100_003 + i)
        img, mask = read_image_mask(self.records[i])
        img, mask = _pad_crop(img, mask, rng, self.train)
        if self.train:
            img, mask = _augment(img, mask, rng)
        x = torch.from_numpy(_to_chw3(img))
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y


def _dice_loss(logits, target, eps: float = 1.0):
    torch = _lazy_torch()
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def train_arch(
    arch: str,
    index_path: Path,
    out_dir: Path,
    seed: int = 42,
    max_epochs: int = 12,
    batch: int = 6,
    accum: int = 2,
    patience: int = 3,
    limit_train: int | None = None,
    limit_val: int | None = None,
) -> dict:
    """Train one architecture; returns the training record (also written to out_dir/<arch>.json)."""
    torch = _lazy_torch()
    from torch.utils.data import DataLoader  # noqa: PLC0415

    device = _device(torch)
    torch.manual_seed(seed)
    np.random.seed(seed)

    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    train_recs = index["train"][:limit_train] if limit_train else index["train"]
    val_recs = index["val"][:limit_val] if limit_val else index["val"]

    dl_train = DataLoader(CrackDataset(train_recs, True, seed), batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    dl_val = DataLoader(CrackDataset(val_recs, False, seed), batch_size=batch, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = build_model(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * max(1, len(dl_train)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))

    best = {"f1_2px": -1.0, "epoch": -1}
    history = []
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{arch}.pt"

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(dl_train):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
                loss = (bce(logits, y) + _dice_loss(logits, y)) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += float(loss.detach()) * accum
        # validation with the lab's own buffered metric at the STRICT tolerance
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
        print(f"[{arch}] epoch {epoch}: loss {history[-1]['train_loss']:.4f} val F1@2px {f1:.4f}")
        if f1 > best["f1_2px"]:
            best = {"f1_2px": f1, "epoch": epoch}
            torch.save(model.state_dict(), ckpt)
        elif epoch - best["epoch"] >= patience:
            print(f"[{arch}] early stop at epoch {epoch} (best {best['f1_2px']:.4f} @ {best['epoch']})")
            break

    record = {
        "arch": arch,
        "seed": seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "crop": CROP,
        "batch": batch,
        "accum": accum,
        "loss": "BCE(pos_weight=8) + Dice",
        "optimizer": "AdamW 3e-4 cosine",
        "amp": device == "cuda",
        "best_val_f1_2px": best["f1_2px"],
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_minutes": round((time.perf_counter() - t0) / 60.0, 1),
        "checkpoint": str(ckpt),
        "history": history,
    }
    (out_dir / f"{arch}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def predict_full(arch: str, ckpt: Path, image: np.ndarray, tile: int = 512, overlap: int = 64) -> np.ndarray:
    """Tiled full-image inference (reflect-padded), returning a probability map in [0, 1]."""
    torch = _lazy_torch()
    device = _device(torch)
    model = build_model(arch)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    x = _to_chw3(image)
    _, h, w = x.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    _, hh, ww = x.shape
    prob = np.zeros((hh, ww), dtype=np.float32)
    weight = np.zeros((hh, ww), dtype=np.float32)
    step = tile - overlap
    with torch.no_grad():
        for r in range(0, max(1, hh - tile + 1), step):
            for c in range(0, max(1, ww - tile + 1), step):
                patch = torch.from_numpy(x[:, r : r + tile, c : c + tile][None]).to(device)
                with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                    logits = model(patch)
                p = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
                prob[r : r + tile, c : c + tile] += p
                weight[r : r + tile, c : c + tile] += 1.0
    prob = prob / np.maximum(weight, 1e-6)
    return prob[:h, :w]
