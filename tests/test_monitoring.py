"""Monitoring track tests (BL-011). The registration + differential-mapping engine is pure CPU
(scikit-image + numpy), so a small synthetic case runs everywhere; the committed artifact is validated too."""
import json
from pathlib import Path

import numpy as np

from fisuralab.monitoring.registration import differential_map


def test_differential_map_detects_growth():
    # a vertical crack that WIDENS and EXTENDS between two masks in the same frame (no pose change)
    h = w = 64
    m1 = np.zeros((h, w), bool)
    m2 = np.zeros((h, w), bool)
    m1[10:40, 30:32] = True           # epoch 1: thin, length 30
    m2[10:50, 29:33] = True           # epoch 2: wider (4 px) and longer (length 40)
    d = differential_map(m1, m2, mm_per_px=0.1)
    assert d["unit"] == "mm"
    assert d["width_median_ep2"] > d["width_median_ep1"]   # widened
    assert d["width_delta_median"] > 0
    assert d["length_ep2_px"] > d["length_ep1_px"]         # extended
    assert d["new_branch_px"] > 0                          # new pixels present
    assert d["grew"] is True


def test_differential_map_no_change():
    m = np.zeros((48, 48), bool)
    m[5:40, 20:23] = True
    d = differential_map(m, m.copy(), mm_per_px=None)
    assert d["unit"] == "px"
    assert d["width_delta_median"] == 0.0
    assert d["length_delta_px"] == 0
    assert d["new_branch_px"] == 0
    assert d["grew"] is False


def test_committed_monitoring_artifact_is_coherent():
    p = Path(__file__).resolve().parents[1] / "data" / "derived" / "monitoring" / "growth.json"
    if not p.exists():
        return  # optional until the track ships
    g = json.loads(p.read_text(encoding="utf-8"))
    assert g["registration"]["inliers"] > 0
    m = g["measured"]
    gt = g["ground_truth"]
    # the pipeline should recover the true width growth within a small tolerance
    assert abs(m["width_delta_median"] - gt["true_width_median_delta_mm"]) < 0.05
    assert m["grew"] is True
    # only metrics + overlay pointers, no leaked local paths
    assert "E:" not in json.dumps(g) and "/raw/" not in json.dumps(g)
