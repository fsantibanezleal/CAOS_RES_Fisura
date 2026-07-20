"""Train a Faster R-CNN detector on CODEBRIM (BL-010 detection rung, GPU local lane).

Dossier section 6.2: the clean permissive baseline is torchvision Faster R-CNN ResNet50-FPN V2
(BSD-3-Clause, COCO-pretrained, fine-tuned on CODEBRIM). 5 defect classes + background. Reports
mAP@0.5 and mAP@0.5:0.95 on the published test split; the honest bar is the ertis-research YOLOv5x
number (mAP@0.5 = 0.357). CODEBRIM is a bespoke NC license: images + weights stay local, metrics ship.

Images are 4608x3456; we read them from the zip and downscale to a fixed short side for 8 GB VRAM.
Torch imported lazily; the module skips without CUDA (CI never trains).

    python -m fisuralab.multiclass.train_codebrim --epochs 12
    python -m fisuralab.multiclass.train_codebrim --epochs 1 --limit-train 40   # smoke
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile

import numpy as np

from ..learned.shards import data_root
from .codebrim import DEFECT_INDEX, DEFECTS, codebrim_zip, parse_annotations, split_records

LONG_SIDE = 1024  # downscale target (keeps aspect; 4608 -> 1024 fits 8 GB with FPN)


def _read_image_from_zip(zip_path: str, member: str) -> np.ndarray:
    import imageio.v3 as iio  # noqa: PLC0415

    # Open + close a fresh handle per read. The 8 GB CODEBRIM zip corrupts on any reused/shared
    # handle across reads or worker processes ("Overlapped entries" / "Bad magic number"); a
    # short-lived handle per image is the only reliable path. imread from the raw bytes.
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    arr = iio.imread(io.BytesIO(data))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], -1)
    return arr[..., :3]


class CodebrimDetDataset:
    """Module-level picklable detection dataset: reads a downscaled image + scaled boxes from the zip.
    Does not inherit torch Dataset (CI-safe import); torch imported lazily per item."""

    def __init__(self, records, zip_path):
        self.records = records
        self.zip_path = str(zip_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch  # noqa: PLC0415
        from skimage.transform import resize as rz  # noqa: PLC0415

        r = self.records[i]
        img = _read_image_from_zip(self.zip_path, r["image_member"]).astype(np.float32) / 255.0
        H, W = img.shape[:2]
        scale = LONG_SIDE / max(H, W)
        nh, nw = int(round(H * scale)), int(round(W * scale))
        img = rz(img, (nh, nw), order=1, preserve_range=True).astype(np.float32)
        boxes = np.array([b["xyxy"] for b in r["boxes"]], np.float32) * scale
        labels = np.array([DEFECT_INDEX[b["label"]] for b in r["boxes"]], np.int64)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)}
        return x, target


def _collate(batch):
    return tuple(zip(*batch, strict=False))


def _build_model():
    import torchvision  # noqa: PLC0415

    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: PLC0415

    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, len(DEFECTS) + 1)  # +1 background
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ba = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ba[None, :] - inter + 1e-9)


def _ap_at(preds: list[dict], iou_thr: float) -> float:
    """A simple VOC-style mAP over the defect classes at one IoU threshold (macro over classes)."""
    aps = []
    for ci in range(1, len(DEFECTS) + 1):
        scores, matches, n_gt = [], [], 0
        for p in preds:
            gt = p["gt_boxes"][p["gt_labels"] == ci]
            n_gt += len(gt)
            pi = p["labels"] == ci
            pb, ps = p["boxes"][pi], p["scores"][pi]
            order = np.argsort(-ps)
            used = np.zeros(len(gt), bool)
            for j in order:
                scores.append(ps[j])
                if len(gt) == 0:
                    matches.append(0)
                    continue
                iou = _iou(pb[j:j + 1], gt)[0]
                k = int(np.argmax(iou))
                if iou[k] >= iou_thr and not used[k]:
                    used[k] = True
                    matches.append(1)
                else:
                    matches.append(0)
        if n_gt == 0:
            continue
        if not scores:
            aps.append(0.0)
            continue
        o = np.argsort(-np.array(scores))
        m = np.array(matches)[o]
        tp = np.cumsum(m)
        fp = np.cumsum(1 - m)
        rec = tp / n_gt
        prec = tp / (tp + fp + 1e-9)
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            p = prec[rec >= r].max() if (rec >= r).any() else 0.0
            ap += p / 11
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.multiclass.train_codebrim")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for CODEBRIM detection training")
    device = "cuda"
    torch.manual_seed(args.seed)

    recs = parse_annotations()
    if not recs:
        raise SystemExit("CODEBRIM zip not found under the vault")
    sp = split_records(recs, seed=args.seed)
    train, test = sp["train"], sp["test"]
    if args.limit_train:
        train = train[: args.limit_train]
    zp = codebrim_zip()
    tl = DataLoader(CodebrimDetDataset(train, zp), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=_collate, pin_memory=True)

    net = _build_model().to(device)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(tl)))

    print(f"CODEBRIM Faster R-CNN: train {len(train)}, test {len(test)}, {len(DEFECTS)} defect classes")
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        net.train()
        run = 0.0
        for bi, (imgs, targets) in enumerate(tl):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = net(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.item())
            if bi % 40 == 0:
                print(f"  ep{ep} it{bi}/{len(tl)} loss {loss.item():.4f}")
        print(f"epoch {ep}: train_loss {run / max(1, len(tl)):.4f}")

    # evaluate mAP on the test split
    net.eval()
    preds = []
    dl_test = DataLoader(CodebrimDetDataset(test, zp), batch_size=1, num_workers=args.workers, collate_fn=_collate)
    with torch.no_grad():
        for imgs, targets in dl_test:
            out = net([imgs[0].to(device)])[0]
            preds.append({
                "boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy(),
                "labels": out["labels"].cpu().numpy(),
                "gt_boxes": targets[0]["boxes"].numpy(), "gt_labels": targets[0]["labels"].numpy(),
            })
    map50 = _ap_at(preds, 0.5)
    map5095 = float(np.mean([_ap_at(preds, thr) for thr in np.arange(0.5, 1.0, 0.05)]))

    out_dir = data_root() / "derived" / "multiclass"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "codebrim_fasterrcnn.pt"
    torch.save(net.state_dict(), ckpt)
    rec = {
        "arch": "fasterrcnn_resnet50_fpn_v2",
        "dataset": "CODEBRIM (custom NC license; images + weights local, metrics only)",
        "classes": DEFECTS,
        "n_train": len(train), "n_test": len(test), "epochs": args.epochs,
        "mAP_50": round(map50, 4), "mAP_50_95": round(map5095, 4),
        "baseline_yolov5x_mAP50": 0.357, "baseline_source": "ertis-research/ConcreteDamageDetection (YOLOv5x + aug)",
        "minutes": round((time.perf_counter() - t0) / 60.0, 1), "checkpoint": str(ckpt),
    }
    (out_dir / "codebrim_results.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "checkpoint"}, indent=1))
    print(f"-> {out_dir / 'codebrim_results.json'}")


if __name__ == "__main__":
    main()
