"""Bake the ``learned_on_examples`` case from the persisted ladder-A results (torch-free).

The GPU runner (``fisuralab.learned.run_ladder_a``) trains and predicts OUTSIDE the pipeline; this
stage replays its persisted outputs (predicted masks on the committed examples + val metrics +
ONNX records) through the SAME dual-tolerance harness and CONTRACT 2 as every other case, so CI
and the frontend need no torch. Fails with a clear instruction when the runner has not been run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.artifact import build_artifact, rle_encode
from ..core.gate import classify_lane
from ..core.manifest import build_case_manifest
from ..io.formats import write_json
from ..io.image_formats import load_example, load_examples_manifest, read_mask
from ..learned.shards import data_root
from ..model.classical import to_gray_float
from ..model.geometry import measure, width_stats
from ..model.metrics import evaluate_mask, restrict_to_fov

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"


def _overlay_png(path: Path, gray: np.ndarray, mask: np.ndarray) -> None:
    import imageio.v3 as iio  # noqa: PLC0415

    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb[mask] = (0.35 * rgb[mask] + 0.65 * np.array([230, 57, 70])).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, rgb)


def run(*, case, seed: int, derived_dir: str, manifests_dir: str) -> dict:
    results_path = data_root() / "derived" / "learned" / "ladder_a_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found: run `python -m fisuralab.learned.run_ladder_a` (GPU, local) "
            "before baking the learned_on_examples case"
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    archs = sorted(results["archs"].keys())
    if not archs:
        raise ValueError("ladder_a_results.json contains no trained architectures")

    derived = Path(derived_dir) / case.id
    payload_samples: list[dict] = []
    flags: list[dict] = [{"sample_id": "-", "flags": [
        "learned masks replayed from the local GPU run recorded in ladder_a_results.json (seed, recipe, ONNX hashes in this manifest's metrics)"
    ]}]

    for rec in load_examples_manifest(EXAMPLES_DIR):
        sample, _ = load_example(EXAMPLES_DIR, rec)
        gray = to_gray_float(sample.image)
        entry: dict = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "license_tag": sample.license_tag,
            "material": sample.material,
            "size": list(gray.shape),
            "mm_per_px": None,
            "image_rel": rec.file,
            "synthetic_params": None,
            "gt_rle": rle_encode(sample.mask) if sample.mask is not None else None,
            "levels": {},
            "width_validation": None,
            "width_mm": None,
            "severity": None,
        }
        base = derived / "overlays" / sample.sample_id
        _overlay_png(base.with_name(base.name + "_image.png"), gray, np.zeros_like(gray, dtype=bool))
        entry["overlays_rel"] = f"{case.id}/overlays/{sample.sample_id}"
        best_mask = None
        for arch in archs:
            png = results["archs"][arch]["examples"].get(sample.sample_id, {}).get("mask_png")
            if not png or not Path(png).exists():
                continue
            mask = restrict_to_fov(read_mask(png), sample.fov)  # drop any response outside the retina disc
            if arch.startswith("dinov2"):
                note = f"{arch}: DINOv2 frozen features + linear head (518 resize, 1/14-resolution probe, coarse by design)"
            elif arch.startswith("hrsegnet"):
                note = f"{arch}: in-repo HrSegNet reimplementation trained on CrackSeg9k (seed {results['seed']})"
            else:
                note = f"{arch}: SMP model trained on CrackSeg9k (seed {results['seed']}), tiled 512 inference"
            lentry: dict = {
                "mask_rle": rle_encode(mask),
                "notes": [note],
                "segmentation": evaluate_mask(mask, sample.mask) if sample.mask is not None else None,
            }
            _overlay_png(derived / "overlays" / f"{sample.sample_id}_{arch}.png", gray, mask)
            entry["levels"][arch] = lentry
            best_mask = mask if best_mask is None else best_mask
        geom = measure(best_mask if best_mask is not None else np.zeros_like(gray, dtype=bool))
        entry["geometry_level"] = archs[0]
        entry["geometry"] = {
            "length_px": geom.length_px,
            "n_branches": geom.n_branches,
            "n_endpoints": geom.n_endpoints,
            "orientation_hist": geom.orientation_hist.tolist(),
            "width": width_stats(geom),
        }
        payload_samples.append(entry)

    artifact = build_artifact(case_id=case.id, samples=payload_samples)
    artifact_rel = f"{case.id}/artifact.json"
    artifact_bytes = write_json(Path(derived_dir) / artifact_rel, artifact)

    # case metrics: per-arch val scores + example means + ONNX provenance
    metrics: dict = {"protocol": "buffered P/R/F1 at 2 px AND 5 px tolerances; strict IoU; no thinning/NMS"}
    for arch in archs:
        a = results["archs"][arch]
        metrics[f"{arch}_val_f1_2px"] = round(a["training"]["best_val_f1_2px"], 4)
        ex = [v["f1_5px"] for v in a["examples"].values() if "f1_5px" in v]
        if ex:
            metrics[f"{arch}_examples_mean_f1_5px"] = round(float(np.mean(ex)), 4)
        metrics[f"{arch}_onnx_sha256"] = a["onnx"]["sha256"][:16]
        metrics[f"{arch}_train_minutes"] = a["training"]["train_minutes"]
    best_val = max(results["archs"].items(), key=lambda kv: kv[1]["training"]["best_val_f1_2px"])
    metrics["best_arch_val"] = best_val[0]
    metrics["best_val_f1_2px"] = round(best_val[1]["training"]["best_val_f1_2px"], 4)

    band = case.expected_band
    if band:
        val = metrics.get(band["metric"])
        if val is None or not (band["min"] <= val <= band["max"]):
            raise AssertionError(f"{case.id}: {band['metric']}={val} outside [{band['min']}, {band['max']}]")

    gate = classify_lane(
        pure_python=False,  # the TRAINING is torch; the replay itself is static
        wheels={"torch", "segmentation-models-pytorch"},
        run_ms=10_000.0,
        trace_bytes=artifact_bytes,
    )
    manifest = build_case_manifest(
        case=case,
        params={"ladder_a": {a: results["archs"][a]["training"] | {"history": "see ladder_a_results.json"} for a in archs}},
        seed=seed,
        artifact_rel=artifact_rel,
        artifact_bytes=artifact_bytes,
        gate=gate,
        flags=flags,
        metrics=metrics,
        engine_model=f"learned track ({', '.join(archs)}), replayed from the local GPU runs",
    )
    write_json(Path(manifests_dir) / f"{case.id}.json", manifest)
    return manifest
