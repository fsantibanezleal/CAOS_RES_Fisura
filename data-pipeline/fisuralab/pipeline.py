"""The offline pipeline orchestrator + CLI (ADR-0057). Runs the named stages per case, applies CONTRACT 1,
writes the compact artifact + manifest (CONTRACT 2) and a flat index.json.

    python -m fisuralab.pipeline                    # all cases
    python -m fisuralab.pipeline synthetic_battery  # one case
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from . import registry
from .core.manifest import build_index
from .io.formats import write_json
from .stages import evaluate, export, feature_extraction, infer, preprocess, train

# data-pipeline/fisuralab/pipeline.py -> parents[2] = repo root (works under `pip install -e .` too)
REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED = REPO_ROOT / "data" / "derived"
MANIFESTS = DERIVED / "manifests"

STAGES = ("preprocess", "feature_extraction", "train", "infer", "evaluate", "export")


def precompute(case_id: str, seed: int | None = None) -> dict:
    case = registry.get_case(case_id)
    seed = case.seed if seed is None else seed
    if getattr(case, "engine", "classical") == "learned_replay":
        from .stages import learned_replay  # noqa: PLC0415

        return learned_replay.run(case=case, seed=seed, derived_dir=str(DERIVED), manifests_dir=str(MANIFESTS))
    t0 = time.perf_counter()

    samples, flags = preprocess.run(case)
    _responses = feature_extraction.run(samples, case.ladder)  # shared S3 responses (overlap with run_level is deliberate: stage products are inspectable)
    rf, train_prov = train.run(samples, seed=seed)
    inferred = infer.run(samples, case.ladder, rf)
    case_metrics, per_sample = evaluate.run(case, samples, inferred)
    run_ms = (time.perf_counter() - t0) * 1000.0

    params = {"ladder": dataclasses.asdict(case.ladder), "train": train_prov}
    return export.run(
        case=case,
        params=params,
        samples=samples,
        inferred=inferred,
        per_sample=per_sample,
        case_metrics=case_metrics,
        flags=flags,
        seed=seed,
        run_ms=run_ms,
        derived_dir=str(DERIVED),
        manifests_dir=str(MANIFESTS),
    )


def run_all(seed: int | None = None) -> list[dict]:
    entries = []
    for c in registry.list_cases():
        precompute(c.id, seed=seed)
        entries.append({"case_id": c.id, "category": c.category, "manifest_path": f"manifests/{c.id}.json"})
    write_json(MANIFESTS / "index.json", build_index(entries))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.pipeline")
    ap.add_argument("case", nargs="?", default="all", help="a case id, or 'all'")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.case == "all":
        entries = run_all(args.seed)
        print(f"precomputed {len(entries)} cases -> {DERIVED}")
        for e in entries:
            print(f"  {e['case_id']:22s} [{e['category']}]")
        print(f"index -> {MANIFESTS / 'index.json'}")
    else:
        m = precompute(args.case, args.seed)
        print(
            f"precomputed {args.case}: lane={m['lane']} bytes={m['artifact']['bytes']} "
            f"metrics={m['metrics']} -> {DERIVED / m['artifact']['path']}"
        )


if __name__ == "__main__":
    main()
