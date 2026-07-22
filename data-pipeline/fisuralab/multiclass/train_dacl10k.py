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
    # Stay in uint8 until AFTER the crop. dacl10k originals are ~3024x4032, so converting the whole
    # image to float32 here costs about 146 MB per sample and every byte of it is discarded by the
    # 512-crop two lines later. With several workers each holding a prefetched batch that is the
    # difference between comfortable and a MemoryError in a DataLoader worker, which is exactly how
    # this run died once.
    img = img[..., :3]
    H0, W0 = img.shape[:2]                      # true image size, before any padding
    if H0 < CROP or W0 < CROP:
        img = np.pad(img, ((0, max(0, CROP - H0)), (0, max(0, CROP - W0)), (0, 0)))
    H, W = img.shape[:2]
    if train:
        y0 = int(rng.integers(0, H - CROP + 1))
        x0 = int(rng.integers(0, W - CROP + 1))
    else:
        y0, x0 = (H - CROP) // 2, (W - CROP) // 2
    img = img[y0:y0 + CROP, x0:x0 + CROP]
    # rasterize ONLY the crop window, in the unpadded image's coordinate space. Building the full
    # (19, H, W) mask first and slicing it costs ~183 MB per sample and is what exhausted host RAM.
    m = rasterize(ann_p, out_hw=(H0, W0), window=(y0, x0, CROP, CROP))
    if train and rng.random() < 0.5:
        img = img[:, ::-1].copy()
        m = m[:, :, ::-1].copy()
    img = img.astype(np.float32) / 255.0   # now 512x512, so the float copy is ~3 MB, not ~146 MB
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


def class_pos_weight(pairs, n_sample: int = 400, cap: float = 50.0, seed: int = 0):
    """Per-class BCE pos_weight = negative/positive pixel ratio, estimated on a TRAIN-split sample.

    Why this is needed (measured, not assumed): the first full-data run used a plain unweighted BCE
    and 9 of the 19 classes ended at exactly IoU 0 in the pooled evaluation, including Crack itself
    (1.4 percent of labelled pixels). The 10 classes the model did learn averaged 0.32 IoU, so the
    macro mean was destroyed purely by rare classes collapsing to an all-negative prediction, which
    is the textbook failure of unweighted BCE on heavily imbalanced multi-label segmentation.

    Weighting the positive term by the inverse positive rate is the standard remedy. The weight is
    capped because the rarest classes would otherwise draw weights in the thousands and dominate the
    gradient. The estimate is sampled (not a full pass over ~6.9k images) and cached, since it only
    sets a loss constant.
    """
    import numpy as _np  # noqa: PLC0415

    cache = data_root() / "derived" / "multiclass" / f"pos_weight_sqrt_n{n_sample}_s{seed}.json"
    if cache.exists():
        w = json.loads(cache.read_text(encoding="utf-8"))["pos_weight"]
        return _np.array(w, dtype=_np.float32)

    rng = _np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=min(n_sample, len(pairs)), replace=False)
    pos = _np.zeros(N_CLASSES, dtype=_np.float64)
    tot = 0.0
    for k, i in enumerate(idx):
        img_p, ann_p = pairs[i]
        img = read_image(img_p)
        m = rasterize(ann_p, out_hw=img.shape[:2])          # (C,H,W) binary
        pos += m.reshape(N_CLASSES, -1).sum(axis=1)
        tot += m.shape[1] * m.shape[2]
        if k % 100 == 0:
            print(f"  pos_weight scan {k}/{len(idx)}")
    pos = _np.maximum(pos, 1.0)                              # never divide by zero
    # sqrt of the neg/pos ratio, not the raw ratio. Measured rates span 0.094 percent
    # (Restformwork) to 7.02 percent (Weathering), i.e. raw ratios from 13 to 1060. Using the raw
    # ratio would hand the rarest classes a weight in the hundreds, which buys recall by destroying
    # precision (and the IoU with it); a flat cap low enough to be safe instead collapses 14 of the
    # 19 classes onto the same value and throws the ordering away. The square root keeps the
    # ordering with a well-behaved 3.6x-to-32.6x spread, and Dice already carries per-class scale
    # invariance alongside it.
    w = _np.clip(_np.sqrt((tot - pos) / pos), 1.0, cap).astype(_np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "n_sample": int(len(idx)), "cap": cap, "seed": seed,
        "positive_pixel_rate": {CLASSES[i]: float(pos[i] / tot) for i in range(N_CLASSES)},
        "pos_weight": [float(v) for v in w],
    }, indent=1), encoding="utf-8")
    return w


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
    ap.add_argument("--pos-weight", action="store_true",
                    help="per-class BCE positive weighting (fixes the rare-class collapse to IoU 0)")
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last saved training state if one exists")
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
    pw = None
    if args.pos_weight:
        pw_np = class_pos_weight(train_pairs, cap=args.pos_weight_cap, seed=args.seed)
        pw = torch.from_numpy(pw_np).to(device).view(N_CLASSES, 1, 1)
        top = sorted(zip(CLASSES, pw_np, strict=False), key=lambda kv: -kv[1])[:5]
        print("pos_weight (top 5): " + ", ".join(f"{c} {v:.1f}" for c, v in top))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    # bf16, not fp16. The fp16 forward overflowed to inf once the pos_weighted loss pushed the logits
    # up, every step then failed the isfinite check, and the run silently stopped learning while still
    # occupying the GPU. bf16 carries fp32's exponent range so it cannot overflow that way, and it
    # needs no GradScaler (kept disabled rather than removed so the call sites stay unchanged).
    AMP_DTYPE = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Resumable by design, not by luxury. This run was killed three times mid-training by host memory
    # pressure from other work on the machine, with no Python traceback, and each kill threw away
    # every completed epoch. A full training state written once per epoch turns that from hours lost
    # into minutes lost. The state file sits next to the best-metric checkpoint and holds the
    # optimizer and scheduler too, so a resumed run is not silently restarting the LR schedule.
    state_path = data_root() / "derived" / "multiclass" / f"{args.arch}_trainstate.pt"
    start_epoch = 0

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

    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch = int(st["epoch"]) + 1
        best_miou = float(st["best_miou"])
        best_per_class = st.get("best_per_class")
        best_present = st.get("best_present")
        n_skipped = int(st.get("n_skipped", 0))
        print(f"resumed from epoch {st['epoch']} (best mIoU {best_miou:.4f}); continuing at {start_epoch}")

    for ep in range(start_epoch, args.epochs):
        net.train()
        run = 0.0
        ep_skipped = 0
        ep_stepped = 0
        for bi, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                out = net(x)
            # losses in fp32 outside autocast (AMP fp16 reductions over 19x512x512 go NaN)
            loss = bce(out.float(), y.float()) + dice_loss(out, y)
            if not torch.isfinite(loss):
                n_skipped += 1
                ep_skipped += 1
                if ep_skipped in (1, 10, 100) or ep_skipped % 500 == 0:
                    print(f"  ep{ep} it{bi}: NON-FINITE loss, step skipped ({ep_skipped} this epoch)")
                continue  # skip a bad step rather than poison every weight with NaN
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # the other half of the fix
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.item())
            ep_stepped += 1
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
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
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
        skip_frac = ep_skipped / max(1, ep_skipped + ep_stepped)
        print(f"epoch {ep}: train_loss {run / max(1, ep_stepped):.4f}  val_mIoU {miou:.4f}"
              f"  steps {ep_stepped}/{ep_skipped + ep_stepped}{star}")
        if skip_frac > 0.25:
            raise SystemExit(
                f"ABORT epoch {ep}: {skip_frac:.0%} of steps had a non-finite loss, so the model is "
                f"barely updating. Failing loudly instead of holding the GPU for hours while the "
                f"metric quietly re-scores stale weights."
            )
        torch.save({"epoch": ep, "model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_miou": best_miou, "best_per_class": best_per_class,
                    "best_present": best_present, "n_skipped": n_skipped}, state_path)

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
        "pos_weight": bool(args.pos_weight),
        "pos_weight_cap": args.pos_weight_cap if args.pos_weight else None,
        "val_mIoU": round(miou, 4),
        "val_mIoU_protocol": "pooled (dataset-level) macro IoU over classes present in the val ground truth",
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
