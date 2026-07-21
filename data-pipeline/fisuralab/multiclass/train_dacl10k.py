"""Train a multi-label semantic segmenter on dacl10k (BL-010, GPU local lane, RTX 4070).

Modest-bar architecture from the dossier (section 6.1): SMP with a SegFormer (MiT) or EfficientNet
encoder + FPN/Unet head, 19 sigmoid outputs, BCE + Dice loss, 512-crop training. Target: beat the
WACV 2024 paper baseline of 0.424 mIoU on the test split (macro IoU over the 19 classes).

Everything heavy stays OUTSIDE git (FISURA_DATA_ROOT/derived/multiclass); only the metrics record +
tiny qualitative overlays ship. Torch imported lazily; the module skips without CUDA (CI never trains).

    python -m fisuralab.multiclass.train_dacl10k --arch segformer --epochs 30
    python -m fisuralab.multiclass.train_dacl10k --arch effb4 --epochs 1 --limit-train 200   # smoke
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..io.image_formats import read_image
from ..learned.shards import data_root
from .dacl10k import CLASSES, N_CLASSES, list_split, rasterize

CROP = 512


def _build(arch: str):
    import segmentation_models_pytorch as smp  # noqa: PLC0415

    if arch == "segformer":
        return smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet", in_channels=3, classes=N_CLASSES)
    if arch == "effb4":
        return smp.FPN(encoder_name="tu-efficientnet_b4", encoder_weights="imagenet", in_channels=3, classes=N_CLASSES)
    if arch == "unet_r34":
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=N_CLASSES)
    raise ValueError(f"unknown arch '{arch}' (segformer | effb4 | unet_r34)")


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _dacl_item(img_p, ann_p, train: bool, rng):
    """One (image, mask) sample: read, rasterize, 512-crop, normalize. Torch-free (returns numpy)."""
    img = read_image(img_p)
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)
    img = img[..., :3].astype(np.float32) / 255.0
    m = rasterize(ann_p, out_hw=img.shape[:2])  # (C,H,W)
    H, W = img.shape[:2]
    if H < CROP or W < CROP:
        ph, pw = max(0, CROP - H), max(0, CROP - W)
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
        m = np.pad(m, ((0, 0), (0, ph), (0, pw)))
        H, W = img.shape[:2]
    if train:
        y0 = int(rng.integers(0, H - CROP + 1))
        x0 = int(rng.integers(0, W - CROP + 1))
    else:
        y0, x0 = (H - CROP) // 2, (W - CROP) // 2
    img = img[y0:y0 + CROP, x0:x0 + CROP]
    m = m[:, y0:y0 + CROP, x0:x0 + CROP]
    if train and rng.random() < 0.5:
        img = img[:, ::-1].copy()
        m = m[:, :, ::-1].copy()
    img = (img - _MEAN) / _STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)), np.ascontiguousarray(m).astype(np.float32)


class DaclDataset:
    """A module-level (Windows-spawn-picklable) map-style dataset. It does NOT inherit torch's
    Dataset so the module imports without torch (CI-safe); DataLoader accepts any object with
    __len__/__getitem__, and torch is imported lazily per item."""

    def __init__(self, pairs, train, seed):
        self.pairs = pairs
        self.train = train
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import torch  # noqa: PLC0415

        x, y = _dacl_item(self.pairs[i][0], self.pairs[i][1], self.train, self.rng)
        return torch.from_numpy(x), torch.from_numpy(y)


def _iou_parts(logits, target, thr: float = 0.5):
    """Per-class intersection / union / ground-truth pixel counts for POOLED IoU.

    Pooling these over the whole validation split and dividing ONCE is the benchmark protocol.
    Averaging a per-batch IoU instead (the earlier form here) is a different and systematically lower
    quantity: a class holding a handful of pixels in one batch contributes an equally-weighted
    near-zero term. Quoting that against the paper's 0.424 would not be a like-for-like comparison.
    See eval_dacl10k.py, which recomputes the pooled number from a saved checkpoint.
    """
    import torch  # noqa: PLC0415

    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(0, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(0, 2, 3))
    gt = target.sum(dim=(0, 2, 3))
    return inter.double(), union.double(), gt.double()


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.multiclass.train_dacl10k")
    ap.add_argument("--arch", default="segformer", choices=["segformer", "effb4", "unet_r34"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for dacl10k training (GPU local lane)")
    device = "cuda"
    torch.manual_seed(args.seed)

    train_pairs = list_split("train")
    val_pairs = list_split("validation")
    if not train_pairs:
        raise SystemExit("dacl10k not extracted under the vault; extract dacl10k_v2_devphase.zip first")
    if args.limit_train:
        train_pairs = train_pairs[: args.limit_train]
    if args.limit_val:
        val_pairs = val_pairs[: args.limit_val]

    tl = DataLoader(DaclDataset(train_pairs, True, args.seed), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(DaclDataset(val_pairs, False, args.seed), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    net = _build(args.arch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(tl)))
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    def dice_loss(logits, target, eps=1.0):
        # computed in fp32: under AMP the fp16 sums over 512x512x19 underflow/overflow and the loss
        # goes NaN a few epochs in (observed: NaN from epoch 6 of the first full-data run).
        p = torch.sigmoid(logits.float())
        t = target.float()
        inter = (p * t).sum(dim=(2, 3))
        denom = p.sum(dim=(2, 3)) + t.sum(dim=(2, 3))
        return (1 - (2 * inter + eps) / (denom + eps)).mean()

    print(f"dacl10k {args.arch}: train {len(train_pairs)}, val {len(val_pairs)}, {N_CLASSES} classes")
    out_dir = data_root() / "derived" / "multiclass"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / f"{args.arch}.pt"
    best_miou = -1.0
    best_per_class = None
    best_present = None
    t0 = time.perf_counter()
    n_skipped = 0
    for ep in range(args.epochs):
        net.train()
        run = 0.0
        for bi, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = net(x)
            # losses in fp32 outside autocast (AMP fp16 reductions over 19x512x512 go NaN)
            loss = bce(out.float(), y.float()) + dice_loss(out, y)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue  # skip a bad step rather than poison every weight with NaN
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # the other half of the fix
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.item())
            if bi % 100 == 0:
                print(f"  ep{ep} it{bi}/{len(tl)} loss {loss.item():.4f}")
        # validation macro-IoU
        net.eval()
        inter_sum = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)
        union_sum = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)
        gt_sum = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda"):
                    out = net(x)
                i_, u_, g_ = _iou_parts(out, y)
                inter_sum += i_
                union_sum += u_
                gt_sum += g_
        per_class = (inter_sum / union_sum.clamp(min=1)).cpu().numpy()
        # a class counts if it OCCURS in the validation ground truth, not merely if something fired
        present = gt_sum.cpu().numpy() > 0
        miou = float(per_class[present].mean()) if present.any() else 0.0
        star = ""
        if miou > best_miou and np.isfinite(miou) and miou > 0:
            best_miou = miou
            best_per_class = per_class
            best_present = present
            torch.save(net.state_dict(), best_ckpt)  # keep the BEST epoch, not the last
            star = " *best (saved)"
        print(f"epoch {ep}: train_loss {run / max(1, len(tl)):.4f}  val_mIoU {miou:.4f}{star}")

    per_class = best_per_class
    present = best_present
    miou = best_miou
    ckpt = best_ckpt
    rec = {
        "arch": args.arch,
        "dataset": "dacl10k v2 (CC BY-NC 4.0; local, metrics only)",
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "epochs": args.epochs,
        "val_mIoU": round(miou, 4),
        "per_class_IoU": {CLASSES[i]: round(float(per_class[i]), 4) for i in range(N_CLASSES) if present[i]},
        "baseline_mIoU": 0.424,
        "baseline_source": "Flotzinger et al. WACV 2024 Table 4 (FPN + EfficientNet-B4 + aux loss)",
        "minutes": round((time.perf_counter() - t0) / 60.0, 1),
        "nonfinite_steps_skipped": n_skipped,
        "checkpoint": str(ckpt),
    }
    (out_dir / f"{args.arch}_results.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "per_class_IoU"}, indent=1))
    print(f"-> {out_dir / f'{args.arch}_results.json'}")


if __name__ == "__main__":
    main()
