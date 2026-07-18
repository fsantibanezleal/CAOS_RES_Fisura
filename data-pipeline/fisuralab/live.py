"""LIVE lane entrypoint: analyze ONE user-provided image in the browser (Pyodide) or any host.

Pure model/ calls (numpy + scipy + scikit-image, the gate's Pyodide-safe wheel set). Input is a
contract-legal array (the browser decodes the user's photo to grayscale or RGB); output is a plain
dict: the chosen ladder level's mask as RLE, notes, and the geometry summary (physical widths only
when the user provided a scale; never invented). L5 is offline-only (needs the trained forest).
"""
from __future__ import annotations

import numpy as np

from .core.artifact import rle_encode
from .io.image_contract import ImageSample, validate_sample
from .model.classical import LadderParams, run_level
from .model.geometry import measure, width_stats

LIVE_LEVELS = ("L0", "L1", "L2", "L3", "L4")


def run_live(image: np.ndarray, level: str = "L3", mm_per_px: float | None = None) -> dict:
    if level not in LIVE_LEVELS:
        raise ValueError(f"live lane supports {LIVE_LEVELS}; '{level}' is offline-only")
    sample = ImageSample(image=image, mm_per_px=mm_per_px, material="other", source="user-live", license_tag="unknown", sample_id="live")
    result = validate_sample(sample)
    if not result.ok:
        return {"ok": False, "errors": result.errors}
    res = run_level(image, level, params=LadderParams())
    geom = measure(res.mask)
    ws = width_stats(geom)
    out = {
        "ok": True,
        "level": level,
        "flags": result.flags,
        "mask_rle": rle_encode(res.mask),
        "notes": res.notes,
        "geometry": {
            "length_px": geom.length_px,
            "n_branches": geom.n_branches,
            "n_endpoints": geom.n_endpoints,
            "width_px": ws,
        },
    }
    if mm_per_px is not None:
        out["geometry"]["length_mm"] = geom.length_px * mm_per_px
        out["geometry"]["width_mm_median"] = (ws["edt_median"] * mm_per_px) if ws.get("edt_median") is not None else None
    return out
