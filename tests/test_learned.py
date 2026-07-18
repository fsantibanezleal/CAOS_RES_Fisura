"""Learned-track tests. CI has no torch and no vault: everything GPU-dependent skips cleanly and
the committed learned artifact (baked locally) is validated by the generic contract tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from fisuralab.learned.shards import data_root, find_crackseg9k_pairs, prepare_split


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def test_prepare_split_is_deterministic(tmp_path):
    # a fake tree with 6 image/mask pairs
    import imageio.v3 as iio

    idir = tmp_path / "set" / "images"
    mdir = tmp_path / "set" / "masks"
    idir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(6):
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        mask = (rng.random((64, 64)) > 0.9).astype(np.uint8) * 255
        iio.imwrite(idir / f"s{i}.jpg", img)
        iio.imwrite(mdir / f"s{i}.png", mask)
        pairs.append((idir / f"s{i}.jpg", mdir / f"s{i}.png"))
    a = prepare_split(pairs, tmp_path / "outA", seed=7)
    b = prepare_split(pairs, tmp_path / "outB", seed=7)
    ia = json.loads((tmp_path / "outA" / "index.json").read_text(encoding="utf-8"))
    ib = json.loads((tmp_path / "outB" / "index.json").read_text(encoding="utf-8"))
    assert a["train"] == b["train"] and a["val"] == b["val"]
    assert [r["image"] for r in ia["train"]] == [r["image"] for r in ib["train"]]
    assert all(0.0 <= r["pos_fraction"] <= 1.0 for r in ia["train"] + ia["val"])


@pytest.mark.skipif(not _torch_available(), reason="torch not installed (GPU lane is local-only)")
def test_build_models_construct():
    from fisuralab.learned.training import ARCHS, build_model

    for arch in ARCHS:
        m = build_model(arch)
        assert sum(p.numel() for p in m.parameters()) > 1e6


@pytest.mark.skipif(
    not (data_root() / "derived" / "learned" / "ladder_a_results.json").exists(),
    reason="ladder-A results not present (GPU run is local-only)",
)
def test_learned_replay_bakes_and_scores():
    from fisuralab import pipeline

    m = pipeline.precompute("learned_on_examples")
    assert m["schema"].startswith("fisura.manifest/")
    assert m["metrics"]["best_val_f1_2px"] >= 0.45
    artifact = json.loads((pipeline.DERIVED / m["artifact"]["path"]).read_text(encoding="utf-8"))
    assert artifact["n_samples"] == 6
    s0 = artifact["samples"][0]
    assert len(s0["levels"]) >= 1  # at least one trained arch replayed


def test_vault_pairs_lookup_never_crashes():
    # returns [] on machines without the vault; never raises
    pairs = find_crackseg9k_pairs(Path("Z:/definitely-missing"))
    assert pairs == []
