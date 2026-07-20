"""The concrete-transfer study (the anomaly track's contribution): fit PatchCore on UNCRACKED concrete
patches only, score cracked vs held-out uncracked, report image-level AUROC.

Dossier 04 section 2 flagged that NO published head-to-head of industrial anomaly detection on civil
surfaces exists; this run measures it. Fit set = uncracked SDNET2018 concrete patches (walls, decks,
pavements); test = a held-out balanced split of cracked + uncracked. SDNET2018 is CC BY 4.0, but only
METRICS + a few overlays from redistributable committed imagery ship; the raw fit imagery stays local.

    python -m fisuralab.anomaly.run_concrete_transfer --n-fit 400 --n-test 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..io.image_formats import read_image
from ..learned.shards import data_root
from .patchcore import PatchCore, auroc


def _sdnet_root() -> Path:
    return data_root() / "raw" / "sdnet2018" / "sdnet2018"


def _list(subdir: str) -> list[Path]:
    d = _sdnet_root() / subdir
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def collect(seed: int, n_fit: int, n_test: int):
    """Uncracked fit set + a balanced cracked/uncracked test set, disjoint, across D/P/W structures."""
    rng = np.random.default_rng(seed)
    cracked = _list("W/CW") + _list("D/CD") + _list("P/CP")
    uncracked = _list("W/UW") + _list("D/UD") + _list("P/UP")
    if not cracked or not uncracked:
        raise FileNotFoundError("SDNET2018 cracked/uncracked folders not found under the vault")
    rng.shuffle(cracked)
    rng.shuffle(uncracked)
    n_test_each = n_test // 2
    fit = uncracked[:n_fit]
    test_unc = uncracked[n_fit : n_fit + n_test_each]
    test_crk = cracked[:n_test_each]
    return fit, test_unc, test_crk


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.anomaly.run_concrete_transfer")
    ap.add_argument("--n-fit", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coreset-cap", type=int, default=8000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    fit_paths, test_unc, test_crk = collect(args.seed, args.n_fit, args.n_test)
    print(f"fit {len(fit_paths)} uncracked; test {len(test_crk)} cracked + {len(test_unc)} uncracked")

    fit_imgs = [read_image(p) for p in fit_paths]
    pc = PatchCore(coreset_cap=args.coreset_cap, seed=args.seed)
    bank = pc.fit(fit_imgs)
    print(f"memory bank: {bank['memory_patches']} / {bank['bank_pool']} pool (of {bank['bank_patches_total']} total; dim {bank['feature_dim']})")

    test_paths = test_crk + test_unc
    labels = np.array([1] * len(test_crk) + [0] * len(test_unc))
    scores, _maps = pc.score([read_image(p) for p in test_paths])
    a = auroc(scores, labels)
    # a simple balanced-accuracy at the median-score threshold
    thr = float(np.median(scores))
    pred = (scores > thr).astype(int)
    tpr = float(((pred == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum()))
    tnr = float(((pred == 0) & (labels == 0)).sum() / max(1, (labels == 0).sum()))

    rec = {
        "method": "PatchCore (in-repo reimplementation: WideResNet50 layer2+3, greedy coreset, kNN)",
        "study": "concrete-transfer: fit on uncracked SDNET2018 concrete, score cracked vs uncracked",
        "dataset": "SDNET2018 (CC BY 4.0; metrics only, raw imagery local)",
        "seed": args.seed,
        "n_fit_uncracked": len(fit_paths),
        "n_test_cracked": len(test_crk),
        "n_test_uncracked": len(test_unc),
        "memory_bank": bank,
        "image_auroc": round(float(a), 4),
        "threshold_median": thr,
        "tpr_at_median": round(tpr, 4),
        "tnr_at_median": round(tnr, 4),
        "score_hist": _hist(scores, labels),
        "minutes": round((time.perf_counter() - t0) / 60.0, 1),
    }
    out = data_root() / "derived" / "anomaly"
    out.mkdir(parents=True, exist_ok=True)
    (out / "concrete_transfer.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("score_hist", "memory_bank")}, indent=1))
    print(f"-> {out / 'concrete_transfer.json'}")


def _hist(scores: np.ndarray, labels: np.ndarray, bins: int = 24) -> dict:
    lo, hi = float(scores.min()), float(scores.max())
    edges = np.linspace(lo, hi, bins + 1)
    hc, _ = np.histogram(scores[labels == 1], bins=edges)
    hu, _ = np.histogram(scores[labels == 0], bins=edges)
    centers = ((edges[:-1] + edges[1:]) / 2).round(4).tolist()
    return {"centers": centers, "cracked": hc.tolist(), "uncracked": hu.tolist()}


if __name__ == "__main__":
    main()
