"""Stage 1, preprocess: load the case's samples through CONTRACT 1 (S0 ingest of the stage graph).

Sources: the committed examples (via the attribution manifest) or the synthetic battery (generated,
exact ground truth). Every sample passes ``validate_sample``; a hard violation aborts the case (bad
data never flows), soft flags are carried into the case manifest.
"""
from __future__ import annotations

from pathlib import Path

from ..io.image_contract import ImageSample, validate_sample
from ..io.image_formats import load_example, load_examples_manifest
from ..model.synthetic import battery

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"


def run(case) -> tuple[list[ImageSample], list[dict]]:
    """Returns (validated samples, contract flags [{sample_id, flags}])."""
    samples: list[ImageSample] = []
    flags: list[dict] = []

    if case.source == "examples":
        for rec in load_examples_manifest(EXAMPLES_DIR):
            sample, result = load_example(EXAMPLES_DIR, rec)
            if not result.ok:
                raise ValueError(f"CONTRACT 1 rejected committed example {rec.sample_id}: {result.errors}")
            if result.flags:
                flags.append({"sample_id": rec.sample_id, "flags": result.flags})
            sample.image_rel = rec.file  # type: ignore[attr-defined]  # pointer for redistributable overlays
            if case.mm_per_px_demo is not None and sample.mask is not None and sample.mask.any():
                sample.mm_per_px = case.mm_per_px_demo
                flags.append({"sample_id": rec.sample_id, "flags": [
                    f"mm_per_px={case.mm_per_px_demo} is a DEMONSTRATION scale (no real calibration exists for this source imagery)"
                ]})
            samples.append(sample)
        return samples, flags

    if case.source == "synthetic_subpx":
        from ..model.synthetic import battery_subpx

        for i, spec in enumerate(battery_subpx(seed=case.seed + 100)):
            sample = ImageSample(
                image=spec.image,
                mask=spec.mask,
                mm_per_px=case.mm_per_px_demo,
                material="synthetic",
                source="fisuralab-synthetic",
                license_tag="cc0",
                sample_id=f"subpx-{i:02d}-w{spec.params['width_px']}",
            )
            sample.params = spec.params  # type: ignore[attr-defined]
            sample.true_width_px = spec.width_px  # type: ignore[attr-defined]
            result = validate_sample(sample)
            if not result.ok:
                raise ValueError(f"CONTRACT 1 rejected {sample.sample_id}: {result.errors}")
            if result.flags:
                flags.append({"sample_id": sample.sample_id, "flags": result.flags})
            samples.append(sample)
        return samples, flags

    if case.source == "synthetic":
        for i, spec in enumerate(battery(seed=case.seed)):
            sample = ImageSample(
                image=spec.image,
                mask=spec.mask,
                mm_per_px=None,
                material="synthetic",
                source="fisuralab-synthetic",
                license_tag="cc0",  # generated in-repo; freely redistributable
                sample_id=f"synth-{i:02d}-{spec.params['kind']}",
            )
            sample.params = spec.params  # type: ignore[attr-defined]  # generator provenance, exported
            sample.true_width_px = spec.width_px  # type: ignore[attr-defined]
            result = validate_sample(sample)
            if not result.ok:
                raise ValueError(f"CONTRACT 1 rejected synthetic sample {sample.sample_id}: {result.errors}")
            if result.flags:
                flags.append({"sample_id": sample.sample_id, "flags": result.flags})
            samples.append(sample)
        return samples, flags

    raise ValueError(f"unknown case source '{case.source}'")
