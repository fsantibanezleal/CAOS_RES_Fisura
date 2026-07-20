"""Multi-class track tests. The training/inference paths need torch + the vault, so they skip in CI;
the class list, the torch-free rasterizer contract, and the committed dacl10k artifact validate everywhere."""
import json
from pathlib import Path

import numpy as np

from fisuralab.multiclass.dacl10k import CLASSES, DAMAGE_CLASSES, N_CLASSES
from fisuralab.multiclass.codebrim import DEFECTS, DEFECT_INDEX


def test_dacl10k_class_list():
    # 19 classes, 13 damage + 6 object (dossier section 1.3, toolkit TARGET_LIST verbatim)
    assert N_CLASSES == 19
    assert len(CLASSES) == 19
    assert len(DAMAGE_CLASSES) == 13
    assert "Crack" in CLASSES and "Bearing" in CLASSES
    assert "Vegetation" not in CLASSES  # vegetation is S2DS, not dacl10k (dossier correction)


def test_codebrim_classes():
    # 5 defects (dossier: NOT vegetation), background is index 0 in the detector
    assert DEFECTS == ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
    assert DEFECT_INDEX["Crack"] == 1  # 0 reserved for background
    assert "Vegetation" not in DEFECTS


def test_committed_dacl10k_artifact_is_coherent():
    p = Path(__file__).resolve().parents[1] / "data" / "derived" / "multiclass" / "dacl10k.json"
    if not p.exists():
        return  # artifact optional until the track ships; skip cleanly
    d = json.loads(p.read_text(encoding="utf-8"))
    assert len(d["classes"]) == 19
    assert len(d["palette"]) == 19
    assert 0.0 <= d["val_mIoU"] <= 1.0
    assert d["baseline_mIoU"] == 0.424
    # every per-class IoU is a valid fraction
    for _c, v in d.get("per_class_IoU", {}).items():
        assert 0.0 <= v <= 1.0
    # only metrics + overlay pointers ship, no raw image paths leaked
    assert "E:" not in json.dumps(d) and "/raw/" not in json.dumps(d)


def test_rasterize_contract_shape():
    # rasterize returns a (19, H, W) uint8 multi-label stack; verify the shape contract without the vault
    # by constructing a minimal LabelMe-style annotation and rasterizing it.
    from fisuralab.multiclass.dacl10k import rasterize  # noqa: PLC0415
    import tempfile

    ann = {
        "imageHeight": 32, "imageWidth": 32,
        "shapes": [
            {"label": "Crack", "shape_type": "polygon", "points": [[2, 2], [20, 2], [20, 20], [2, 20]]},
            {"label": "Rust", "shape_type": "polygon", "points": [[10, 10], [30, 10], [30, 30], [10, 30]]},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ann, f)
        path = Path(f.name)
    m = rasterize(path)
    path.unlink()
    assert m.shape == (19, 32, 32)
    assert m.dtype == np.uint8
    assert m[CLASSES.index("Crack")].any()
    assert m[CLASSES.index("Rust")].any()
    # overlap region carries BOTH labels (multi-label, not mutually exclusive)
    assert m[CLASSES.index("Crack"), 15, 15] == 1 and m[CLASSES.index("Rust"), 15, 15] == 1
