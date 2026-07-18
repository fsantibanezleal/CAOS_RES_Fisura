"""Unit tests for the analytic core: RLE codec, buffered metrics, width estimators on exact ground
truth, the ridge ladder on clean synthetic bars, and the live-lane entrypoint. These are the
synthetic-battery regression gates from the research (a scikit-image upgrade that shifts the ridge
stack must fail HERE, loudly)."""
import numpy as np

from fisuralab.core.artifact import rle_decode, rle_encode
from fisuralab.live import run_live
from fisuralab.model.classical import LadderParams, run_level
from fisuralab.model.geometry import measure, width_stats
from fisuralab.model.metrics import buffered_prf, evaluate_mask
from fisuralab.model.synthetic import straight_bar, uncracked


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(5):
        mask = rng.random((37, 53)) > 0.7
        assert np.array_equal(rle_decode(rle_encode(mask)), mask)
    empty = np.zeros((8, 8), dtype=bool)
    assert np.array_equal(rle_decode(rle_encode(empty)), empty)
    full = np.ones((8, 8), dtype=bool)
    assert np.array_equal(rle_decode(rle_encode(full)), full)


def test_buffered_metrics_tolerance_semantics():
    gt = np.zeros((64, 64), dtype=bool)
    gt[30, 10:50] = True
    pred = np.zeros_like(gt)
    pred[33, 10:50] = True  # 3 px off the ground truth
    m2 = buffered_prf(pred, gt, 2)
    m5 = buffered_prf(pred, gt, 5)
    assert m2["f1"] < 0.1, "a 3 px offset must fail the 2 px tolerance"
    assert m5["f1"] > 0.95, "a 3 px offset must pass the 5 px tolerance"
    full = evaluate_mask(pred, gt)
    assert full["tol2px"]["f1"] < 0.1 < full["tol5px"]["f1"]
    assert "protocol" in full


def test_width_estimators_on_exact_ground_truth():
    for width in (3.0, 5.0, 9.0):
        spec = straight_bar(width_px=width, angle_deg=30.0, seed=3)
        geom = measure(spec.mask)
        ws = width_stats(geom)
        assert ws["edt_median"] is not None
        assert abs(ws["edt_median"] - width) <= 1.0, f"EDT width {ws['edt_median']} vs true {width}"
        if ws["profile_median"] is not None:
            assert abs(ws["profile_median"] - width) <= 1.2, f"profile width {ws['profile_median']} vs true {width}"


def test_ridge_l3_detects_clean_bar():
    spec = straight_bar(width_px=5.0, angle_deg=30.0, seed=4)
    res = run_level(spec.image, "L3", LadderParams(sigmas=(1.0, 2.0, 3.0, 4.5)))
    m = buffered_prf(res.mask, spec.mask, 5)
    assert m["f1"] >= 0.8, f"L3 on a clean 5 px bar should be strong at 5 px tolerance, got {m}"


def test_l0_floor_is_honestly_weak_on_texture():
    spec = uncracked(joint=False, seed=5)
    res = run_level(spec.image, "L0")
    # global Otsu on textured raw imagery fires everywhere: the floor exists to be visibly bad
    assert res.mask.mean() > 0.05


def test_live_entrypoint_smoke():
    spec = straight_bar(width_px=4.0, angle_deg=45.0, seed=6)
    out = run_live((np.clip(spec.image, 0, 1) * 255).astype(np.uint8), level="L3", mm_per_px=0.2)
    assert out["ok"] is True
    assert out["geometry"]["length_px"] > 50
    assert out["geometry"]["width_mm_median"] is None or out["geometry"]["width_mm_median"] > 0
    assert "mask_rle" in out
