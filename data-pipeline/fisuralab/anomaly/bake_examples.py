"""Bake the Beyond-SOTA anomaly overlay for the shared example images (the App's Beyond-SOTA tab).

The concrete-transfer study (run_concrete_transfer) reports aggregate AUROC over SDNET2018. For the
App workbench, the Beyond-SOTA method must be VISIBLE on the SAME six example images the classical and
learned tracks run on: a PatchCore memory bank fit on uncracked concrete, scored on each example, with
the per-patch anomaly heatmap upsampled to the image and a threshold-derived anomaly mask (RLE).

This writes an artifact shaped like the others (levels keyed by the single method `patchcore`, each with
mask_rle + notes + an anomaly readout) plus a committed heatmap PNG per sample (redistributable: the
heat is derived, the base image is already committed under CC0/CC-BY). SDNET2018 fit imagery stays local.

    python -m fisuralab.anomaly.bake_examples
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..core.artifact import rle_encode
from ..io.image_formats import read_image
from .patchcore import PatchCore
from .run_concrete_transfer import collect

# The committed examples + the App-facing derived artifacts live in the REPO tree (git-as-data),
# not in the local heavy data vault (data_root() = E:/_Datos). Only the uncracked FIT imagery for
# the memory bank comes from the vault (via run_concrete_transfer.collect).
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "data" / "derived" / "bcl_examples" / "artifact.json"
OUT_ART = REPO_ROOT / "data" / "derived" / "anomaly_examples"
REPO_EXAMPLES = REPO_ROOT / "data" / "examples"


def _upsample(map_grid: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a (gh,gw) patch map to (h,w) with numpy (no scipy dep)."""
    h, w = hw
    gh, gw = map_grid.shape
    yi = (np.arange(h) * gh / h).astype(int).clip(0, gh - 1)
    xi = (np.arange(w) * gw / w).astype(int).clip(0, gw - 1)
    return map_grid[yi][:, xi]


def _heat_png(norm: np.ndarray) -> bytes:
    """Encode a [0,1] heat field as a PNG (blue->red), no matplotlib. Returns PNG bytes."""
    import zlib
    import struct

    h, w = norm.shape
    # simple perceptual ramp: low = transparent blue, high = opaque red
    r = (norm * 235).astype(np.uint8)
    g = (np.clip(1 - np.abs(norm - 0.5) * 2, 0, 1) * 120).astype(np.uint8)
    b = ((1 - norm) * 220).astype(np.uint8)
    a = (np.clip(norm * 1.4, 0, 1) * 210).astype(np.uint8)
    rgba = np.dstack([r, g, b, a]).astype(np.uint8)
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main() -> None:
    t0 = time.perf_counter()
    ex = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    samples = ex["samples"]

    # Fit the memory bank on uncracked concrete (same source as the transfer study), so the App's
    # Beyond-SOTA tab shows a bank that never saw a crack, scored on the example images.
    fit_paths, _tu, _tc = collect(seed=42, n_fit=200, n_test=2)
    print(f"fit PatchCore on {len(fit_paths)} uncracked concrete patches")
    pc = PatchCore(coreset_cap=6000, seed=42)
    bank = pc.fit([read_image(p) for p in fit_paths])
    print(f"memory bank: {bank['memory_patches']} patches, grid {bank['grid']}")

    OUT_ART.mkdir(parents=True, exist_ok=True)
    (OUT_ART / "overlays").mkdir(parents=True, exist_ok=True)

    imgs, ids, sizes = [], [], []
    for s in samples:
        img = read_image(REPO_EXAMPLES / s["image_rel"])
        imgs.append(img)
        ids.append(s["sample_id"])
        sizes.append(s["size"])
    scores, maps = pc.score(imgs)

    # a shared normalisation so heat is comparable across the 6 examples (min..max over all patches)
    lo = float(min(m.min() for m in maps))
    hi = float(max(m.max() for m in maps))
    rng = max(1e-6, hi - lo)

    out_samples = []
    for s, sid, size, score, mp in zip(samples, ids, sizes, scores, maps):
        h, w = size
        up = _upsample(mp, (h, w))
        norm = (up - lo) / rng
        # anomaly mask = the high-anomaly region (top of the shared scale); honest, thresholded at 0.6
        mask = (norm >= 0.6).astype(np.uint8)
        heat_png = _heat_png(norm)
        (OUT_ART / "overlays" / f"{sid}_heat.png").write_bytes(heat_png)
        img_norm = float((score - lo) / rng)
        out_samples.append({
            "sample_id": sid,
            "source": s["source"],
            "license_tag": s["license_tag"],
            "material": s["material"],
            "size": size,
            "mm_per_px": s.get("mm_per_px"),
            "image_rel": s["image_rel"],
            "synthetic_params": None,
            "gt_rle": s.get("gt_rle"),
            "levels": {
                "patchcore": {
                    "mask_rle": rle_encode(mask),
                    "notes": [
                        f"PatchCore memory bank ({bank['memory_patches']} patches) fit on uncracked concrete only; "
                        f"image anomaly score {score:.3f} (normalised {img_norm:.2f} on the shared 6-image scale). "
                        "Mask = region above 0.6 of the shared anomaly scale. The heat overlay is the per-patch "
                        "nearest-memory distance upsampled to the image."
                    ],
                    "segmentation": None,
                    "anomaly_score": round(float(score), 4),
                    "anomaly_score_norm": round(img_norm, 4),
                },
            },
            "geometry_level": "patchcore",
            "geometry": s["geometry"],
            "width_validation": None,
            "overlays_rel": f"anomaly_examples/overlays/{sid}",
            "heat_rel": f"anomaly_examples/overlays/{sid}_heat.png",
        })

    art = {
        "schema": "fisura.artifact/v1",
        "case_id": "anomaly_examples",
        "n_samples": len(out_samples),
        "samples": out_samples,
    }
    with open(OUT_ART / "artifact.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(art, f, ensure_ascii=False, indent=1)

    meta = {
        "method": "PatchCore (in-repo): WideResNet50 layer2+3, greedy coreset, kNN, fit on uncracked concrete",
        "memory_bank": bank,
        "shared_scale": {"lo": lo, "hi": hi},
        "scores": {sid: round(float(sc), 4) for sid, sc in zip(ids, scores)},
        "minutes": round((time.perf_counter() - t0) / 60.0, 1),
    }
    with open(OUT_ART / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta["scores"], indent=1))
    print(f"-> {OUT_ART / 'artifact.json'}  (+ {len(out_samples)} heat PNGs)")


if __name__ == "__main__":
    main()
