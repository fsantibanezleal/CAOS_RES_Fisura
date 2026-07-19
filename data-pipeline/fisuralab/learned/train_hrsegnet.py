"""Train the HrSegNet reimplementation with the published recipe (ladder B runner).

    python -m fisuralab.learned.train_hrsegnet --base 16 --iters 20000

Recipe per the paper (dossier 02 section 1.7): 400x400 crops, SGD momentum 0.9, weight decay 5e-4,
poly LR (initial 0.01, power 0.9), from scratch, warm-up 2000 iters, loss = CE + 0.5 x two
auxiliary CEs, brightness/contrast/saturation jitter (+-0.5), horizontal flip, random resize
0.5x-2.0x + random crop. Published reference at 100k iters batch 32: B16 78.43 / B32 79.70 /
B48 80.32 2-class mIoU on refined CrackSeg9k (900-val split). The first bake uses an honestly
recorded shorter budget; the manifest carries iters/batch so nothing pretends to be the full run.
Reports BOTH the paper's 2-class mIoU protocol and the lab's dual-tolerance F1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..model.metrics import buffered_prf
from .export_onnx import export
from .hrsegnet import build_hrsegnet
from .shards import data_root, ensure_crackseg9k_index, read_image_mask


class HrSegDataset:
    """Module-level (Windows-spawn picklable) dataset with the paper's augmentation."""

    def __init__(self, recs, train, seed):
        self.recs, self.train, self.seed = recs, train, seed

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        rng = np.random.default_rng(self.seed * 7 + i)
        img, mask = read_image_mask(self.recs[i])
        g = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
        if g.ndim == 2:
            g = np.stack([g, g, g], axis=-1)
        m = mask
        if self.train:
            s = float(rng.uniform(0.5, 2.0))
            from skimage.transform import resize as _resize  # noqa: PLC0415

            hh, ww = max(8, int(g.shape[0] * s)), max(8, int(g.shape[1] * s))
            g = _resize(g, (hh, ww), order=1, preserve_range=True, anti_aliasing=s < 1).astype(np.float32)
            m = _resize(m.astype(np.float32), (hh, ww), order=0, preserve_range=True) > 0.5
            ph, pw = max(0, 400 - g.shape[0]), max(0, 400 - g.shape[1])
            if ph or pw:
                g = np.pad(g, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
            r = int(rng.integers(0, g.shape[0] - 400 + 1))
            c = int(rng.integers(0, g.shape[1] - 400 + 1))
            g, m = g[r : r + 400, c : c + 400], m[r : r + 400, c : c + 400]
            if rng.random() < 0.5:
                g, m = g[:, ::-1], m[:, ::-1]
            jitter = 1.0 + rng.uniform(-0.5, 0.5)
            g = np.clip(g * jitter, 0, 1)
        else:
            ph, pw = max(0, 400 - g.shape[0]), max(0, 400 - g.shape[1])
            if ph or pw:
                g = np.pad(g, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                m = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
            g, m = g[:400, :400], m[:400, :400]
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(g, (2, 0, 1))))
        y = torch.from_numpy(np.ascontiguousarray(m.astype(np.int64)))
        return x, y


def _miou_2class(pred: np.ndarray, gt: np.ndarray) -> float:
    ious = []
    for cls in (False, True):
        p, g = pred == cls, gt == cls
        union = np.logical_or(p, g).sum()
        ious.append(1.0 if union == 0 else np.logical_and(p, g).sum() / union)
    return float(np.mean(ious))


def main() -> None:
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="fisuralab.learned.train_hrsegnet")
    ap.add_argument("--base", type=int, default=16, choices=(16, 32, 48))
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-val", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0)  # 0 = in-process (Windows-safe default)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    idx_path = ensure_crackseg9k_index(seed=42)
    index = json.loads(Path(idx_path).read_text(encoding="utf-8"))
    train_recs, val_recs = index["train"], index["val"][: args.limit_val]


    dl_train = DataLoader(HrSegDataset(train_recs, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    dl_val = DataLoader(HrSegDataset(val_recs, False, args.seed), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_hrsegnet(base=args.base).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    warmup = 2000

    def lr_at(it: int) -> float:
        if it < warmup:
            return 0.01 * (it + 1) / warmup
        return 0.01 * (1 - (it - warmup) / max(1, args.iters - warmup)) ** 0.9

    out_dir = data_root() / "derived" / "learned" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"hrsegnet_b{args.base}"
    ckpt = out_dir / f"{name}.pt"
    t0 = time.perf_counter()
    it = 0
    model.train()
    while it < args.iters:
        for x, y in dl_train:
            if it >= args.iters:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                out, a1, a2 = model(x)
                loss = ce(out, y) + 0.5 * (ce(a1, y) + ce(a2, y))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            if it % 1000 == 0:
                print(f"[{name}] iter {it}/{args.iters} loss {float(loss):.4f} lr {lr_at(it):.5f}")

    # validation: paper protocol (2-class mIoU) + the lab's dual-tolerance F1
    model.eval()
    mious, f1s2, f1s5 = [], [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(x)
            pred = out.argmax(1).cpu().numpy().astype(bool)
            gt = y.numpy().astype(bool)
            for p, g in zip(pred, gt):
                mious.append(_miou_2class(p, g))
                f1s2.append(buffered_prf(p, g, 2)["f1"])
                f1s5.append(buffered_prf(p, g, 5)["f1"])
    torch.save(model.state_dict(), ckpt)
    rec = {
        "arch": name,
        "base": args.base,
        "seed": args.seed,
        "iters": args.iters,
        "batch": args.batch,
        "recipe": "paper: SGD 0.01 poly 0.9 wd 5e-4, warmup 2000, CE + 0.5x2 aux CE, 400 crops, jitter/flip/resize",
        "published_reference": {"B16_miou_100k_b32": 78.43, "B32": 79.70, "B48": 80.32},
        "val_miou_2class": round(float(np.mean(mious)), 4),
        "val_f1_2px": round(float(np.mean(f1s2)), 4),
        "val_f1_5px": round(float(np.mean(f1s5)), 4),
        "n_val": len(val_recs),
        "train_minutes": round((time.perf_counter() - t0) / 60.0, 1),
        "checkpoint": str(ckpt),
        "best_val_f1_2px": round(float(np.mean(f1s2)), 4),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    # join the ladder record: examples evaluation + ONNX export, so learned_replay replays this arch too
    from .run_ladder_a import evaluate_on_examples  # noqa: PLC0415

    base_dir = data_root() / "derived" / "learned"
    examples = evaluate_on_examples(name, ckpt, base_dir / "examples" / name)
    exp = export(name, ckpt)
    results_path = base_dir / "ladder_a_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"seed": args.seed, "archs": {}}
    results["archs"][name] = {"training": rec, "examples": examples, "onnx": exp}
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    masked = [v["f1_5px"] for v in examples.values() if "f1_5px" in v]
    print(json.dumps({k: v for k, v in rec.items() if k not in ("recipe", "published_reference")}, indent=1))
    print(f"[{name}] examples mean F1@5px {float(np.mean(masked)):.4f} | onnx {exp['bytes']/1e6:.1f} MB parity {exp['parity_ok']}")


if __name__ == "__main__":
    main()
