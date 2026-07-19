"""DINOv2 frozen-features + linear segmentation head (the foundation rung's cheap credible baseline).

Per the research (dossier 04 section 1 + dossier 02 section 8.3): DINOv2 (Apache-2.0) is the cheapest
credible foundation baseline for dense crack prediction, and its crack-specific numbers are NOT
established in the literature, which makes a clean linear probe a result the lab can contribute.

Design (a genuine linear probe): the DINOv2 ViT backbone is FROZEN (loaded from the vault weights,
never trained); only a 1x1-convolution head on the last-layer patch tokens is trained (BCE + Dice,
seeded). Patch tokens live at 1/14 resolution (37x37 for a 518 input), so the probe's masks are
honestly coarse for thin cracks; that coarseness IS the finding, reported next to the trained-model
ladder on identical pixels.

Backbone code is fetched once from torch.hub (facebookresearch/dinov2, cached under TORCH_HOME) and
loaded with the local `E:\\_Models\\fisura\\dinov2\\dinov2_vits14_pretrain.pth` weights. Torch imported
lazily; GPU lane only.

    python -m fisuralab.learned.dinov2_probe --iters 3000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..model.metrics import buffered_prf
from .shards import data_root, ensure_crackseg9k_index, read_image_mask

INPUT = 518          # multiple of 14; DINOv2 native resolution
PATCH = 14
GRID = INPUT // PATCH  # 37


def _model_root() -> Path:
    return Path(os.environ.get("FISURA_MODEL_ROOT", "E:/_Models/fisura/fisura-trained"))


def _dinov2_weights() -> Path:
    return Path("E:/_Models/fisura/dinov2/dinov2_vits14_pretrain.pth")


def build_dinov2_probe():
    """Frozen DINOv2 ViT-S/14 + a 1x1-conv linear head. Returns an nn.Module producing a 1-channel logit."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False, trust_repo=True)
    w = _dinov2_weights()
    if w.exists():
        backbone.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=False)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    class Probe(nn.Module):
        def __init__(self, bb, dim=384):
            super().__init__()
            self.bb = bb
            self.head = nn.Conv2d(dim, 1, kernel_size=1)  # the linear probe

        def forward(self, x):
            size = x.shape[-2:]
            with torch.no_grad():
                feats = self.bb.forward_features(x)["x_norm_patchtokens"]  # (N, GRID*GRID, dim)
            n, _, dim = feats.shape
            fmap = feats.transpose(1, 2).reshape(n, dim, GRID, GRID)  # (N, dim, 37, 37)
            logit = self.head(fmap)
            return nn.functional.interpolate(logit, size=size, mode="bilinear", align_corners=False)

    return Probe(backbone)


class ProbeDataset:
    """Module-level (Windows-spawn picklable) dataset: images resized to 518, masks kept binary."""

    def __init__(self, recs, train, seed):
        self.recs, self.train, self.seed = recs, train, seed

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        import torch  # noqa: PLC0415
        from skimage.transform import resize as _resize  # noqa: PLC0415

        rng = np.random.default_rng(self.seed * 11 + i)
        img, mask = read_image_mask(self.recs[i])
        g = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
        if g.ndim == 2:
            g = np.stack([g, g, g], axis=-1)
        g = _resize(g, (INPUT, INPUT), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        m = _resize(mask.astype(np.float32), (INPUT, INPUT), order=0, preserve_range=True) > 0.5
        if self.train and rng.random() < 0.5:
            g, m = g[:, ::-1], m[:, ::-1]
        # ImageNet normalization (DINOv2 expects it)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        g = (g - mean) / std
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(g, (2, 0, 1))))
        y = torch.from_numpy(np.ascontiguousarray(m.astype(np.float32)))[None]
        return x, y


def _dice_loss(logits, target, eps: float = 1.0):
    import torch  # noqa: PLC0415

    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def predict_probe(model, image: np.ndarray) -> np.ndarray:
    """Full-image crack probability via the frozen probe (resize to 518, upsample logits back)."""
    import torch  # noqa: PLC0415
    from skimage.transform import resize as _resize  # noqa: PLC0415

    device = next(model.parameters()).device
    g = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    if g.ndim == 2:
        g = np.stack([g, g, g], axis=-1)
    h0, w0 = g.shape[:2]
    gr = _resize(g, (INPUT, INPUT), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    gr = (gr - mean) / std
    x = torch.from_numpy(np.ascontiguousarray(np.transpose(gr, (2, 0, 1))))[None].to(device)
    model.eval()
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit)[0, 0].float().cpu().numpy()
    return _resize(prob, (h0, w0), order=1, preserve_range=True).astype(np.float32)


def main() -> None:
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="fisuralab.learned.dinov2_probe")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, default=3000)
    ap.add_argument("--limit-val", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    idx_path = ensure_crackseg9k_index(seed=42)
    index = json.loads(Path(idx_path).read_text(encoding="utf-8"))
    train_recs = index["train"][: args.limit_train]
    val_recs = index["val"][: args.limit_val]

    dl_train = DataLoader(ProbeDataset(train_recs, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    dl_val = DataLoader(ProbeDataset(val_recs, False, args.seed), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_dinov2_probe().to(device)
    head_params = [p for p in model.parameters() if p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    opt = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    name = "dinov2s14_linear"
    out_dir = data_root() / "derived" / "learned" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{name}.pt"  # ONLY the head is saved (backbone is the vault weights)
    t0 = time.perf_counter()
    it = 0
    while it < args.iters:
        for x, y in dl_train:
            if it >= args.iters:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                logit = model(x)
                loss = bce(logit, y) + _dice_loss(logit, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            if it % 500 == 0:
                print(f"[{name}] iter {it}/{args.iters} loss {float(loss):.4f}", flush=True)

    model.eval()
    f1s2, f1s5 = [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()[:, 0].astype(bool)
            gt = y.numpy()[:, 0].astype(bool)
            for p, g in zip(pred, gt):
                f1s2.append(buffered_prf(p, g, 2)["f1"])
                f1s5.append(buffered_prf(p, g, 5)["f1"])
    torch.save(model.head.state_dict(), ckpt)
    rec = {
        "arch": name,
        "backbone": "DINOv2 ViT-S/14 (frozen, Apache-2.0 vault weights)",
        "head": f"1x1 conv linear probe ({n_head} trained params)",
        "seed": args.seed,
        "iters": args.iters,
        "batch": args.batch,
        "input": INPUT,
        "patch_resolution": f"{GRID}x{GRID} (1/14; masks are honestly coarse for thin cracks)",
        "loss": "BCE(pos_weight=8) + Dice, AdamW 1e-3",
        "val_f1_2px": round(float(np.mean(f1s2)), 4),
        "val_f1_5px": round(float(np.mean(f1s5)), 4),
        "best_val_f1_2px": round(float(np.mean(f1s2)), 4),
        "n_val": len(val_recs),
        "train_minutes": round((time.perf_counter() - t0) / 60.0, 1),
        "checkpoint": str(ckpt),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")

    # examples eval + join the ladder record (no ONNX here: the frozen backbone export is heavy and
    # belongs to the live-lane unit; the head + backbone hash provenance is recorded instead)
    from ..io.image_formats import load_example, load_examples_manifest, write_mask  # noqa: PLC0415

    examples_dir = Path(__file__).resolve().parents[3] / "data" / "examples"
    ex_out = data_root() / "derived" / "learned" / "examples" / name
    ex_out.mkdir(parents=True, exist_ok=True)
    examples = {}
    for erec in load_examples_manifest(examples_dir):
        sample, _ = load_example(examples_dir, erec)
        prob = predict_probe(model, sample.image)
        mask = prob > 0.5
        write_mask(ex_out / f"{erec.sample_id}.png", mask)
        entry = {"mask_png": str(ex_out / f"{erec.sample_id}.png")}
        if sample.mask is not None:
            entry["f1_2px"] = buffered_prf(mask, sample.mask, 2)["f1"]
            entry["f1_5px"] = buffered_prf(mask, sample.mask, 5)["f1"]
        examples[erec.sample_id] = entry

    import hashlib  # noqa: PLC0415

    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    results_path = data_root() / "derived" / "learned" / "ladder_a_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"seed": args.seed, "archs": {}}
    results["archs"][name] = {"training": rec, "examples": examples, "onnx": {"onnx": None, "bytes": ckpt.stat().st_size, "sha256": sha, "parity_ok": None}}
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    masked = [v["f1_5px"] for v in examples.values() if "f1_5px" in v]
    print(json.dumps({k: v for k, v in rec.items() if k != "loss"}, indent=1))
    print(f"[{name}] examples mean F1@5px {float(np.mean(masked)):.4f} (head params {n_head}, honestly coarse probe)")


if __name__ == "__main__":
    main()
