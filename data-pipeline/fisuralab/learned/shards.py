"""Offline shard preparation: vault datasets to fixed 512-crop training shards (dossier discipline).

CrackSeg9k layout (extracted volumes): images/<name>.jpg + masks/<name>.png pairs (400 x 400 in the
source; crops are taken at up to 400 with padding to 512 handled by the dataset transform). Shards
are written OUTSIDE git (FISURA_DATA_ROOT/derived/shards/<set>) as paired PNGs with a positive-pixel
index JSON so the sampler can balance crack/background crops without re-scanning 4k images per epoch.
Deterministic in seed; idempotent (skips when the index exists).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..io.image_formats import read_image, read_mask


def data_root() -> Path:
    root = os.environ.get("FISURA_DATA_ROOT")
    if root:
        return Path(root)
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("FISURA_DATA_ROOT="):
                return Path(line.split("=", 1)[1].strip())
    return Path(__file__).resolve().parents[3] / "data" / "raw-local"


def find_crackseg9k_pairs(root: Path | None = None) -> list[tuple[Path, Path]]:
    """Locate image/mask pairs in the extracted CrackSeg9k volumes (searched, not hardcoded)."""
    base = (root or data_root()) / "raw" / "crackseg9k" / "extracted"
    if not base.exists():
        return []
    img_dirs = [d for d in base.rglob("*") if d.is_dir() and d.name.lower() in ("images", "image", "img")]
    pairs: list[tuple[Path, Path]] = []
    for idir in img_dirs:
        for mname in ("masks", "mask", "labels", "label"):
            mdir = idir.parent / mname
            if mdir.exists():
                for img in sorted(idir.iterdir()):
                    if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        continue
                    for ext in (".png", ".jpg", ".bmp"):
                        m = mdir / (img.stem + ext)
                        if m.exists():
                            pairs.append((img, m))
                            break
                break
    return pairs


def prepare_split(
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    seed: int = 42,
    val_fraction: float = 0.15,
    limit: int | None = None,
) -> dict:
    """Deterministic train/val split + positive-fraction index. Images are copied as-is (the crop
    to 512 with padding happens in the torch dataset); the index records mask positive fractions."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    if limit:
        order = order[:limit]
    n_val = max(1, int(len(order) * val_fraction))
    val_idx = set(order[:n_val].tolist())

    index = {"seed": seed, "train": [], "val": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(order.tolist()):
        img_p, mask_p = pairs[i]
        split = "val" if i in val_idx else "train"
        mask = read_mask(mask_p)
        rec = {
            "image": str(img_p),
            "mask": str(mask_p),
            "pos_fraction": float(mask.mean()),
            "hw": list(mask.shape),
        }
        index[split].append(rec)
        if k % 500 == 0:
            print(f"  indexed {k}/{len(order)}")
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return {"train": len(index["train"]), "val": len(index["val"]), "index": str(out_dir / "index.json")}


def ensure_crackseg9k_index(seed: int = 42, limit: int | None = None) -> Path:
    out = data_root() / "derived" / "shards" / "crackseg9k"
    idx = out / "index.json"
    if idx.exists():
        return idx
    pairs = find_crackseg9k_pairs()
    if not pairs:
        raise FileNotFoundError(
            "CrackSeg9k pairs not found under the vault; run scripts/fetch-data and extract the volumes"
        )
    print(f"CrackSeg9k: {len(pairs)} image/mask pairs; indexing...")
    prepare_split(pairs, out, seed=seed, limit=limit)
    return idx


def read_image_mask(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    return read_image(rec["image"]), read_mask(rec["mask"])
