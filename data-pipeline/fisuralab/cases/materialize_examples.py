"""Materialize the synthetic battery (and any extra source set) into data/examples.

Why this exists: the App originally offered the synthetic battery only a subset of the workbench
tabs, because the workbench, learned, anomaly and enrichment artifacts had never been baked on
synthetic cases. That was the wrong fix. There is no reason a trained segmenter, a superpixel
decomposition or a skeleton measurement cannot run on a generated crack; the artifacts simply had
not been produced. Withholding the tabs hid a gap instead of closing it.

Every downstream bake in this repo reads `data/examples` through the attribution manifest
(`load_examples_manifest`). So the fix is to write the generated cases out as ordinary example
files with ordinary manifest entries, after which every existing bake covers them with no
special-casing at all.

The battery is deterministic in its seed, so this is reproducible rather than a data dump: the same
seed regenerates byte-identical images.

    python -m fisuralab.cases.materialize_examples --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "data" / "examples"
SYNTH_DIR = EXAMPLES / "synthetic"


def _write_png(path: Path, arr: np.ndarray) -> None:
    import imageio.v3 as iio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, arr, extension=".png")


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.cases.materialize_examples")
    ap.add_argument("--seed", type=int, default=42, help="must match the synthetic_battery case seed")
    args = ap.parse_args()

    from ..model.synthetic import battery  # noqa: PLC0415

    manifest_path = EXAMPLES / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in entries if e.get("source") != "fisuralab-synthetic"]  # idempotent

    added = []
    for i, spec in enumerate(battery(seed=args.seed)):
        kind = spec.params["kind"]
        sid = f"synth-{i:02d}-{kind}"
        img_rel = f"synthetic/{sid}.png"
        mask_rel = f"synthetic/{sid}_mask.png"

        img8 = np.clip(spec.image * 255.0, 0, 255).astype(np.uint8)
        _write_png(EXAMPLES / img_rel, img8)
        # the mask is exact by construction; store it as a plain binary PNG like the real examples
        _write_png(EXAMPLES / mask_rel, (np.asarray(spec.mask) > 0).astype(np.uint8) * 255)

        entries.append({
            "sample_id": sid,
            "file": img_rel,
            "mask": mask_rel,
            "material": "synthetic",
            "source": "fisuralab-synthetic",
            "license_tag": "cc0",
            "mm_per_px": None,
            "url": None,
            "citation": (
                "Generated in-repo by fisuralab.model.synthetic.battery "
                f"(seed {args.seed}, {kind}, deterministic). Ground truth is exact by construction."
            ),
            "label": "uncracked" if kind == "uncracked" else "cracked",
        })
        added.append(sid)
        print(f"  {sid}: {img8.shape} image + exact mask")

    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"-> {len(added)} synthetic examples materialized; manifest now holds {len(entries)} records")


if __name__ == "__main__":
    main()
