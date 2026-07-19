"""Ladder-A runner (GPU, local): index the vault, train the three architectures, evaluate on the
val split AND on the committed examples, export ONNX, persist everything the pipeline case
``learned_on_examples`` replays.

    python -m fisuralab.learned.run_ladder_a            # full run (all three archs)
    python -m fisuralab.learned.run_ladder_a --archs unet_r18 --limit-train 800

Outputs (all OUTSIDE git):
- FISURA_DATA_ROOT/derived/learned/checkpoints/<arch>.pt + <arch>.json (training record)
- FISURA_DATA_ROOT/derived/learned/examples/<arch>/<sample_id>.png (predicted masks on the examples)
- FISURA_DATA_ROOT/derived/learned/ladder_a_results.json (the record the pipeline case consumes)
- FISURA_MODEL_ROOT/<arch>.onnx + export records
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..io.image_formats import load_example, load_examples_manifest, write_mask
from ..model.metrics import buffered_prf
from .export_onnx import export
from .shards import data_root, ensure_crackseg9k_index
from .training import ARCHS, predict_full, train_arch

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"


def evaluate_on_examples(arch: str, ckpt: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample = {}
    for rec in load_examples_manifest(EXAMPLES_DIR):
        sample, _ = load_example(EXAMPLES_DIR, rec)
        prob = predict_full(arch, ckpt, sample.image)
        mask = prob > 0.5
        write_mask(out_dir / f"{rec.sample_id}.png", mask)
        entry: dict = {"mask_png": str(out_dir / f"{rec.sample_id}.png")}
        if sample.mask is not None:
            entry["f1_2px"] = buffered_prf(mask, sample.mask, 2)["f1"]
            entry["f1_5px"] = buffered_prf(mask, sample.mask, 5)["f1"]
        per_sample[rec.sample_id] = entry
    return per_sample


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.learned.run_ladder_a")
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    args = ap.parse_args()

    idx = ensure_crackseg9k_index(seed=args.seed)
    base = data_root() / "derived" / "learned"
    ckpt_dir = base / "checkpoints"
    results: dict = {"seed": args.seed, "index": str(idx), "archs": {}}
    results_path = base / "ladder_a_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    for arch in args.archs:
        print(f"=== {arch} ===")
        rec_path = ckpt_dir / f"{arch}.json"
        if rec_path.exists() and Path(json.loads(rec_path.read_text(encoding="utf-8"))["checkpoint"]).exists():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            print(f"[{arch}] resume: checkpoint exists (best val F1@2px {rec['best_val_f1_2px']:.4f}); skipping training")
        else:
            rec = train_arch(
                arch, idx, ckpt_dir, seed=args.seed, max_epochs=args.max_epochs,
                limit_train=args.limit_train, limit_val=args.limit_val,
            )
        examples = evaluate_on_examples(arch, Path(rec["checkpoint"]), base / "examples" / arch)
        exp = export(arch, Path(rec["checkpoint"]))
        results["archs"][arch] = {"training": rec, "examples": examples, "onnx": exp}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        masked = [v["f1_5px"] for v in examples.values() if "f1_5px" in v]
        print(f"[{arch}] val F1@2px {rec['best_val_f1_2px']:.4f} | examples mean F1@5px {np.mean(masked):.4f} | onnx {exp['bytes']/1e6:.1f} MB parity {exp['parity_ok']}")

    print(f"ladder A results -> {results_path}")


if __name__ == "__main__":
    main()
