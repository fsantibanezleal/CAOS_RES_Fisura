"""dacl10k multi-label semantic segmentation: the data layer (BL-010, from the verified dossier
`wip/fisura/research/BL-010-multiclass-research-2026-07-19.md`).

dacl10k v2 (Flotzinger et al., WACV 2024, arXiv:2309.00460) is 9,920 bridge-inspection images with 19
OVERLAPPING damage + object classes annotated as LabelMe polygons. It is MULTI-LABEL semantic
segmentation: one pixel can carry several class labels, so the target is a (C, H, W) stack of binary
masks, trained with per-class sigmoid + BCE/Dice, NOT a softmax over mutually-exclusive classes.

License: CC BY-NC 4.0 (verified). Training + metrics publication are fine; the raw images are NOT
redistributed and trained dacl10k weights carry a parallel NC notice, never MIT. Data stays in the
local vault (E:/_Datos/fisura/raw/dacl10k/extracted); only metrics + tiny qualitative crops ship.

The 19 classes are the toolkit TARGET_LIST verbatim (dossier section 1.3): 13 damage + 6 object.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The canonical 19-class order (dacl10k v2 toolkit Dacl10kDataset.TARGET_LIST).
CLASSES: list[str] = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket",
    "Hollowareas", "Cavity", "Spalling", "Graffiti", "Weathering",
    "Restformwork", "ExposedRebars", "Bearing", "EJoint", "Drainage",
    "PEquipment", "JTape", "WConccor",
]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# 13 damage classes first, then 6 object classes (dossier section 1.3 grouping).
DAMAGE_CLASSES = set(CLASSES[:13])


def dacl_root() -> Path:
    from ..learned.shards import data_root  # noqa: PLC0415

    return data_root() / "raw" / "dacl10k" / "extracted" / "dacl10k_v2_devphase"


def list_split(split: str) -> list[tuple[Path, Path]]:
    """Return (image, annotation-json) pairs for a split (train | validation)."""
    root = dacl_root()
    img_dir = root / "images" / split
    ann_dir = root / "annotations" / split
    if not ann_dir.exists():
        return []
    pairs = []
    for ann in sorted(ann_dir.glob("*.json")):
        img = img_dir / (ann.stem + ".jpg")
        if img.exists():
            pairs.append((img, ann))
    return pairs


def rasterize(ann_path: Path, out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Polygons -> a (C, H, W) uint8 multi-label mask (1 where any polygon of class c covers the pixel).

    Rasterized at the annotation's native resolution unless out_hw is given (then scaled). Uses
    skimage.draw.polygon (no OpenCV dependency needed here)."""
    from skimage.draw import polygon as sk_polygon  # noqa: PLC0415

    a = json.loads(ann_path.read_text(encoding="utf-8"))
    H, W = int(a["imageHeight"]), int(a["imageWidth"])
    mask = np.zeros((N_CLASSES, H, W), dtype=np.uint8)
    for sh in a.get("shapes", []):
        lbl = sh.get("label")
        ci = CLASS_INDEX.get(lbl)
        if ci is None:
            continue
        pts = np.asarray(sh.get("points", []), dtype=np.float64)
        if len(pts) < 3:
            continue
        rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=(H, W))
        mask[ci, rr, cc] = 1
    if out_hw is not None and (H, W) != out_hw:
        from skimage.transform import resize as _resize  # noqa: PLC0415

        oh, ow = out_hw
        out = np.zeros((N_CLASSES, oh, ow), dtype=np.uint8)
        for c in range(N_CLASSES):
            if mask[c].any():
                out[c] = (_resize(mask[c], (oh, ow), order=0, preserve_range=True) > 0.5).astype(np.uint8)
        return out
    return mask


def class_frequencies(split: str = "train", limit: int | None = None) -> dict:
    """Per-class image-presence and pixel-fraction over a split (for the imbalance report)."""
    pairs = list_split(split)
    if limit:
        pairs = pairs[:limit]
    img_count = np.zeros(N_CLASSES, dtype=np.int64)
    pix_frac = np.zeros(N_CLASSES, dtype=np.float64)
    n = 0
    for _img, ann in pairs:
        m = rasterize(ann, out_hw=(256, 256))  # downsampled for the frequency scan
        present = m.reshape(N_CLASSES, -1).any(axis=1)
        img_count += present.astype(np.int64)
        pix_frac += m.reshape(N_CLASSES, -1).mean(axis=1)
        n += 1
    return {
        "n_images": n,
        "class_image_count": {CLASSES[i]: int(img_count[i]) for i in range(N_CLASSES)},
        "class_pixel_fraction": {CLASSES[i]: round(float(pix_frac[i] / max(1, n)), 5) for i in range(N_CLASSES)},
    }
