"""Evaluate a trained dacl10k checkpoint with the BENCHMARK-COMPARABLE pooled IoU.

Why this exists (a metric-comparability fix, not a second opinion): the in-training validation metric
in train_dacl10k.py accumulates a per-BATCH IoU and averages those across batches. The dacl10k
benchmark, like every standard segmentation protocol, pools intersection and union over the WHOLE
validation set first and divides once:

    IoU_c = sum_over_dataset(pred_c AND gt_c) / sum_over_dataset(pred_c OR gt_c)
    mIoU  = mean over classes present in the validation ground truth

Batch-averaging is a different and systematically lower quantity (a class with few pixels in one batch
contributes an equally-weighted near-zero term), so quoting it against the paper's 0.424 would be an
invalid comparison. This module recomputes the pooled number from the saved best checkpoint in a
single validation pass, so no retraining is needed.

It also sweeps the decision threshold. With 19 independent sigmoid outputs and a heavy class
imbalance, a fixed 0.5 leaves rare classes never predicted (IoU exactly 0), which drags the macro
mean down for a reason that is calibration, not segmentation quality. The sweep is reported
explicitly as tuned-on-val so the headline number is never mistaken for a held-out result.

    python -m fisuralab.multiclass.eval_dacl10k --arch effb4
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..learned.shards import data_root
from .dacl10k import CLASSES, N_CLASSES, list_split
from .train_dacl10k import DaclDataset, _build

THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.multiclass.eval_dacl10k")
    ap.add_argument("--arch", default="effb4", choices=["segformer", "effb4", "unet_r34"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-val", type=int, default=None)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA: this evaluation is the GPU lane")

    out_dir = data_root() / "derived" / "multiclass"
    ckpt = out_dir / f"{args.arch}.pt"   # the trainer saves the BEST epoch here
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}; train first")

    val_pairs = list_split("validation")
    if args.limit_val:
        val_pairs = val_pairs[: args.limit_val]
    vl = DataLoader(DaclDataset(val_pairs, False, 0), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    device = "cuda"
    net = _build(args.arch).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    # pooled accumulators, one row per threshold
    inter = torch.zeros(len(THRESHOLDS), N_CLASSES, dtype=torch.float64, device=device)
    union = torch.zeros(len(THRESHOLDS), N_CLASSES, dtype=torch.float64, device=device)
    gt_pix = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, (x, y) in enumerate(vl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = net(x)
            prob = torch.sigmoid(out.float())
            gt_pix += y.sum(dim=(0, 2, 3)).double()
            for ti, thr in enumerate(THRESHOLDS):
                pred = (prob > thr).float()
                inter[ti] += (pred * y).sum(dim=(0, 2, 3)).double()
                union[ti] += ((pred + y) > 0).float().sum(dim=(0, 2, 3)).double()
            if bi % 50 == 0:
                print(f"  val it{bi}/{len(vl)}")

    gt_pix_np = gt_pix.cpu().numpy()
    present = gt_pix_np > 0  # classes that actually occur in the validation ground truth
    results = {}
    for ti, thr in enumerate(THRESHOLDS):
        iou = (inter[ti] / union[ti].clamp(min=1)).cpu().numpy()
        miou = float(iou[present].mean()) if present.any() else 0.0
        results[f"{thr:.2f}"] = {
            "mIoU_pooled": round(miou, 4),
            "per_class_IoU": {CLASSES[i]: round(float(iou[i]), 4) for i in range(N_CLASSES) if present[i]},
            "n_classes_zero": int((iou[present] == 0).sum()),
        }
        print(f"thr {thr:.2f}: pooled mIoU {miou:.4f}  ({results[f'{thr:.2f}']['n_classes_zero']} classes at 0)")

    best_thr = max(results, key=lambda k: results[k]["mIoU_pooled"])
    rec = {
        "arch": args.arch,
        "checkpoint": str(ckpt),
        "protocol": (
            "pooled (dataset-level) IoU: intersection and union summed over the whole validation split "
            "per class, divided once, then macro-averaged over classes present in the validation ground "
            "truth. This is the benchmark-comparable definition; the in-training metric averaged "
            "per-batch IoU and is systematically lower."
        ),
        "n_val": len(val_pairs),
        "thresholds": results,
        "best_threshold": float(best_thr),
        "mIoU_pooled_at_best_threshold": results[best_thr]["mIoU_pooled"],
        "mIoU_pooled_at_0.50": results["0.50"]["mIoU_pooled"],
        "threshold_caveat": (
            "The best threshold is selected ON the validation split, so the headline number is "
            "tuned-on-val, not held-out. The 0.50 column is the untuned reference."
        ),
        "class_pixel_counts": {CLASSES[i]: int(gt_pix_np[i]) for i in range(N_CLASSES) if present[i]},
        "baseline_mIoU": 0.424,
        "baseline_source": "Flotzinger et al. WACV 2024 Table 4 (FPN + EfficientNet-B4 + aux loss)",
        "eval_minutes": round((time.perf_counter() - t0) / 60.0, 1),
    }
    p = out_dir / f"{args.arch}_eval_pooled.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"\nbest threshold {best_thr}: pooled mIoU {rec['mIoU_pooled_at_best_threshold']} "
          f"(at 0.50: {rec['mIoU_pooled_at_0.50']}) vs baseline 0.424")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
