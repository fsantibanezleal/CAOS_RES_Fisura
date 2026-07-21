"""CODEBRIM concrete-defect detection: the data layer (BL-010 detection rung, from the dossier
section 2).

CODEBRIM (Mundt et al., CVPR 2019, Zenodo 10.5281/zenodo.2620293) is 1,590 high-res bridge images
with native Pascal-VOC bounding boxes carrying MULTI-LABEL defect flags. Verified classes (dossier
section 2.3): 5 defects + Background = Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain,
Background. NOT vegetation (that is S2DS). Boxes ship natively; no mask derivation needed.

License: a bespoke non-commercial license (dossier section 2.2). Clause 5 extends NC to trained
models, so CODEBRIM weights ship under a parallel NC notice, never MIT; the raw images stay local.
For detection we collapse the multi-label box flags to the dominant defect (single-label detection),
the standard CODEBRIM detection protocol; the multi-label classification task is a separate rung.

Boxes are read from the EXTRACTED tree, not from the zip. The published CODEBRIM archives carry a
Zip64 offset defect: every local-header offset in the central directory points past the real 4 GB
boundary, so Python's spec-compliant `zipfile` fails every image read with "Bad magic number for file
header" (7-Zip, which is tolerant of the defect, verifies the same archive as "Everything is Ok").
The archives are therefore extracted once with 7-Zip into the vault and read as plain files:

    7z x CODEBRIM_original_images.zip -o E:\\_Datos\\fisura\\raw\\codebrim\\extracted

A deterministic 70/20/10 split by image is published, seeded.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFECTS = ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
DEFECT_INDEX = {c: i + 1 for i, c in enumerate(DEFECTS)}  # 0 reserved for background in torchvision


def codebrim_root() -> Path:
    """The 7-Zip-extracted CODEBRIM tree in the vault (see the module docstring on the Zip64 defect)."""
    from ..learned.shards import data_root  # noqa: PLC0415

    return data_root() / "raw" / "codebrim" / "extracted"


def _box_label(defect_el: ET.Element) -> str | None:
    """The dominant single defect for a box (multi-label -> the first flagged defect; skip pure background)."""
    for c in DEFECTS:
        el = defect_el.find(c)
        if el is not None and (el.text or "0").strip() == "1":
            return c
    return None


def parse_annotations() -> list[dict]:
    """Read every annotation XML from the extracted tree into
    {image_path, width, height, boxes:[{xyxy,label}]}. image_path is an absolute path on disk."""
    root_dir = codebrim_root()
    if not root_dir.exists():
        return []
    imgs = {p.name: p for p in root_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    recs = []
    for xml_p in root_dir.rglob("*.xml"):
        try:
            root = ET.fromstring(xml_p.read_bytes())
        except ET.ParseError:
            continue
        fn = (root.findtext("filename") or "").strip()
        img_path = imgs.get(fn)
        if img_path is None:
            continue
        size = root.find("size")
        if size is None:
            continue
        W = int(size.findtext("width"))
        H = int(size.findtext("height"))
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            defect = obj.find("Defect")
            if bb is None or defect is None:
                continue
            lbl = _box_label(defect)
            if lbl is None:
                continue
            x0 = float(bb.findtext("xmin"))
            y0 = float(bb.findtext("ymin"))
            x1 = float(bb.findtext("xmax"))
            y1 = float(bb.findtext("ymax"))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append({"xyxy": [x0, y0, x1, y1], "label": lbl})
        if boxes:
            recs.append({"image_path": str(img_path), "width": W, "height": H, "boxes": boxes})
    return recs


def split_records(recs: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Deterministic 70/20/10 image split (dossier: CODEBRIM has no predefined split)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    n = len(order)
    n_tr = int(n * 0.7)
    n_va = int(n * 0.2)
    idx = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    return {k: [recs[i] for i in v] for k, v in idx.items()}


def class_counts(recs: list[dict]) -> dict:
    from collections import Counter  # noqa: PLC0415

    c = Counter(b["label"] for r in recs for b in r["boxes"])
    return {"n_images": len(recs), "n_boxes": sum(len(r["boxes"]) for r in recs), "per_class": dict(c)}
