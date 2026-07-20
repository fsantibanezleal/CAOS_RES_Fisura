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

Boxes are read directly from the XML inside the zip (the images are 4608x3456; we do not extract all
8 GB). A deterministic 70/20/10 split by image is published, seeded.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

DEFECTS = ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
DEFECT_INDEX = {c: i + 1 for i, c in enumerate(DEFECTS)}  # 0 reserved for background in torchvision


def codebrim_zip() -> Path:
    from ..learned.shards import data_root  # noqa: PLC0415

    return data_root() / "raw" / "codebrim" / "CODEBRIM_original_images.zip"


def _box_label(defect_el: ET.Element) -> str | None:
    """The dominant single defect for a box (multi-label -> the first flagged defect; skip pure background)."""
    for c in DEFECTS:
        el = defect_el.find(c)
        if el is not None and (el.text or "0").strip() == "1":
            return c
    return None


def parse_annotations() -> list[dict]:
    """Read every XML from the zip into {image_member, width, height, boxes:[{xyxy,label}]}.

    image_member is the path inside the zip to the JPG (read lazily during training)."""
    zp = codebrim_zip()
    if not zp.exists():
        return []
    recs = []
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        img_by_name = {Path(n).name: n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        for n in names:
            if not (n.endswith(".xml") and "annotation" in n.lower()):
                continue
            try:
                root = ET.fromstring(z.read(n))
            except ET.ParseError:
                continue
            fn = (root.findtext("filename") or "").strip()
            member = img_by_name.get(fn)
            if member is None:
                continue
            size = root.find("size")
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
                recs.append({"image_member": member, "width": W, "height": H, "boxes": boxes})
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
