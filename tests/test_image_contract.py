"""CONTRACT 1 (image domain): validation paths + the committed examples must pass the gate."""
from pathlib import Path

import numpy as np

from fisuralab.io.image_contract import (
    ImageSample,
    is_redistributable,
    validate_sample,
)
from fisuralab.io.image_formats import load_example, load_examples_manifest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "examples"


def _good_sample(**over):
    rng = np.random.default_rng(7)
    base = dict(
        image=rng.integers(0, 255, size=(64, 80), dtype=np.uint8),
        mask=None,
        mm_per_px=0.1,
        material="concrete",
        source="unit-test",
        license_tag="cc0",
        sample_id="t",
    )
    base.update(over)
    return ImageSample(**base)


def test_valid_grayscale_and_rgb_pass():
    assert validate_sample(_good_sample()).ok
    rgb = np.random.default_rng(1).integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    assert validate_sample(_good_sample(image=rgb)).ok


def test_float_image_range_enforced():
    ok = _good_sample(image=np.random.default_rng(2).random((64, 64), dtype=np.float32))
    assert validate_sample(ok).ok
    bad = _good_sample(image=np.full((64, 64), 2.0, dtype=np.float32))
    res = validate_sample(bad)
    assert not res.ok and any("outside [0, 1]" in e for e in res.errors)


def test_bad_shapes_dtypes_rejected():
    assert not validate_sample(_good_sample(image=np.zeros((10, 10), dtype=np.uint8))).ok  # too small
    assert not validate_sample(_good_sample(image=np.zeros((64, 64, 4), dtype=np.uint8))).ok  # 4 channels
    assert not validate_sample(_good_sample(image=np.zeros((64, 64), dtype=np.int32))).ok  # wrong dtype


def test_mask_rules():
    img = np.random.default_rng(3).integers(0, 255, size=(64, 64), dtype=np.uint8)
    good_mask = np.zeros((64, 64), dtype=np.uint8)
    good_mask[30:34, 10:50] = 255
    assert validate_sample(_good_sample(image=img, mask=good_mask)).ok
    # mismatched shape rejects
    assert not validate_sample(_good_sample(image=img, mask=np.zeros((32, 32), dtype=np.uint8))).ok
    # non-binary integer mask rejects
    tri = np.zeros((64, 64), dtype=np.uint8)
    tri[0, 0] = 7
    assert not validate_sample(_good_sample(image=img, mask=tri)).ok
    # suspicious coverage flags but does not reject
    heavy = np.ones((64, 64), dtype=np.uint8)
    res = validate_sample(_good_sample(image=img, mask=heavy))
    assert res.ok and any("coverage" in f for f in res.flags)


def test_scale_material_source_license_rules():
    assert not validate_sample(_good_sample(mm_per_px=1000.0)).ok
    assert not validate_sample(_good_sample(material="wood")).ok
    assert not validate_sample(_good_sample(source="")).ok
    assert not validate_sample(_good_sample(license_tag="gpl")).ok
    assert validate_sample(_good_sample(mm_per_px=None)).ok  # scale is optional, never invented


def test_redistribution_boundary():
    assert is_redistributable("cc0") and is_redistributable("cc-by")
    for tag in ("cc-by-nc", "cc-by-nc-sa", "academic", "competition", "unknown"):
        assert not is_redistributable(tag)


def test_committed_examples_pass_contract():
    records = load_examples_manifest(EXAMPLES_DIR)
    assert len(records) >= 4, "the curated example set must exist (BCL + SDNET2018 at minimum)"
    for rec in records:
        assert is_redistributable(rec.license_tag), f"{rec.sample_id}: only redistributable licenses may be committed"
        sample, result = load_example(EXAMPLES_DIR, rec)
        assert result.ok, f"{rec.sample_id}: {result.errors}"
        assert sample.image.shape[0] >= 32
