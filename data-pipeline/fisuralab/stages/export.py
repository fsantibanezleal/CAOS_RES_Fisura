"""Stage 6, export: CONTRACT 2. Compact artifact (RLE masks + metrics + geometry) + manifest + overlays.

Overlay PNGs are written ONLY for redistributable imagery (the license boundary enforced in code):
the committed examples (CC0 / CC BY) and the in-repo synthetic specimens. Four representative
synthetic specimens get overlays (artifact budget); every sample's masks and metrics are always in
the artifact JSON.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.artifact import build_artifact, rle_encode
from ..core.gate import classify_lane
from ..core.manifest import build_case_manifest
from ..io.formats import write_json
from ..io.image_contract import ImageSample, is_redistributable
from ..io.image_formats import write_mask
from ..model.classical import LEVELS, LevelResult, to_gray_float

OVERLAY_SYNTH_IDS = ("synth-02", "synth-04", "synth-08", "synth-11", "subpx-")  # bars, wavy, joint trap, width bench


def _overlay_png(path: Path, gray: np.ndarray, mask: np.ndarray) -> None:
    import imageio.v3 as iio

    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb[mask] = (0.35 * rgb[mask] + 0.65 * np.array([230, 57, 70])).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, rgb)


def run(
    *,
    case,
    params: dict,
    samples: list[ImageSample],
    inferred: list[dict[str, LevelResult]],
    per_sample: list[dict],
    case_metrics: dict,
    flags: list[dict],
    seed: int,
    run_ms: float,
    derived_dir: str,
    manifests_dir: str,
) -> dict:
    derived = Path(derived_dir) / case.id
    payload_samples: list[dict] = []

    for s, levels, rec in zip(samples, inferred, per_sample):
        gray = to_gray_float(s.image)
        redistributable = is_redistributable(s.license_tag)
        want_overlays = redistributable and (
            s.source != "fisuralab-synthetic" or any(s.sample_id.startswith(p) for p in OVERLAY_SYNTH_IDS)
        )
        entry: dict = {
            "sample_id": s.sample_id,
            "source": s.source,
            "license_tag": s.license_tag,
            "material": s.material,
            "size": list(gray.shape),
            "mm_per_px": s.mm_per_px,
            "image_rel": getattr(s, "image_rel", None),
            "synthetic_params": getattr(s, "params", None),
            "gt_rle": rle_encode(s.mask) if s.mask is not None else None,
            "levels": {},
            "geometry_level": rec["geometry_level"],
            "geometry": rec["geometry"],
            "width_validation": rec.get("width_validation"),
            "width_mm": rec.get("width_mm"),
            "severity": rec.get("severity"),
        }
        if want_overlays:
            base = derived / "overlays" / s.sample_id
            _overlay_png(base.with_name(base.name + "_image.png"), gray, np.zeros_like(gray, dtype=bool))
            if s.mask is not None:
                write_mask(base.with_name(base.name + "_gt.png"), s.mask)
            entry["overlays_rel"] = f"{case.id}/overlays/{s.sample_id}"
        for level in LEVELS:
            if level not in levels:
                continue
            res = levels[level]
            lentry: dict = {
                "mask_rle": rle_encode(res.mask),
                "notes": res.notes,
                "segmentation": rec["levels"].get(level, {}).get("segmentation"),
            }
            if want_overlays:
                _overlay_png(derived / "overlays" / f"{s.sample_id}_{level}.png", gray, res.mask)
            entry["levels"][level] = lentry
        payload_samples.append(entry)

    artifact = build_artifact(case_id=case.id, samples=payload_samples)
    artifact_rel = f"{case.id}/artifact.json"
    artifact_bytes = write_json(Path(derived_dir) / artifact_rel, artifact)

    gate = classify_lane(
        pure_python=True,
        wheels={"numpy", "scipy", "scikit-image", "scikit-learn", "imageio"},
        run_ms=run_ms,
        trace_bytes=artifact_bytes,
    )
    manifest = build_case_manifest(
        case=case,
        params=params,
        seed=seed,
        artifact_rel=artifact_rel,
        artifact_bytes=artifact_bytes,
        gate=gate,
        flags=flags,
        metrics=case_metrics,
    )
    write_json(Path(manifests_dir) / f"{case.id}.json", manifest)
    return manifest
