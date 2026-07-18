"""CONTRACT 2 artifact payload for image cases (schema fisura.artifact/v1) + mask RLE codec.

The compact committed artifact of a classical-ladder run: per sample, the ground truth and each
ladder level's mask as row-major run-length encodings, the dual-tolerance metrics, the geometry
summary, and pointers to the small committed overlay PNGs (only for redistributable imagery).
Masks are stored as RLE (alternating run lengths of 0s and 1s over the flattened row-major mask,
starting with the 0-run) so the artifact stays JSON, diffable and tiny; the TypeScript mirror
decodes the same format in the browser.
"""
from __future__ import annotations

import numpy as np

ARTIFACT_SCHEMA = "fisura.artifact/v1"


def rle_encode(mask: np.ndarray) -> dict:
    """Row-major RLE: {'shape': [h, w], 'runs': [len0, len1, len0, ...]} starting with a 0-run."""
    m = np.asarray(mask, dtype=bool).ravel()
    h, w = mask.shape
    if m.size == 0:
        return {"shape": [h, w], "runs": []}
    changes = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    bounds = np.concatenate(([0], changes, [m.size]))
    runs = np.diff(bounds).tolist()
    if m[0]:  # convention: runs start with the 0-run
        runs = [0] + runs
    return {"shape": [h, w], "runs": runs}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["shape"]
    out = np.zeros(h * w, dtype=bool)
    pos = 0
    val = False
    for run in rle["runs"]:
        if val:
            out[pos : pos + run] = True
        pos += run
        val = not val
    return out.reshape(h, w)


def build_artifact(*, case_id: str, samples: list[dict]) -> dict:
    """samples: prepared per-sample payload dicts (see stages/export.py)."""
    return {
        "schema": ARTIFACT_SCHEMA,
        "case_id": case_id,
        "n_samples": len(samples),
        "samples": samples,
    }
