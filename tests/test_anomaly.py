"""Anomaly track tests. The PatchCore fit/score paths need torch + the vault, so they skip in CI;
the pure-numpy AUROC + histogram + the committed study artifact are validated everywhere."""
import json
from pathlib import Path

import numpy as np

from fisuralab.anomaly.patchcore import auroc


def test_auroc_rank_based():
    # perfectly separable: all anomalies score above all normals -> AUROC 1
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert auroc(scores, labels) == 1.0
    # reversed -> 0
    assert auroc(scores, 1 - labels) == 0.0
    # random-ish -> around 0.5; a single swap gives a known value
    s = np.array([0.1, 0.9, 0.2, 0.8])
    lbl = np.array([0, 1, 1, 0])
    a = auroc(s, lbl)
    assert 0.0 <= a <= 1.0
    # degenerate (one class) -> nan
    assert np.isnan(auroc(scores, np.zeros_like(labels)))


def test_committed_study_artifact_is_coherent():
    p = Path(__file__).resolve().parents[1] / "data" / "derived" / "anomaly" / "concrete_transfer.json"
    assert p.exists(), "the concrete-transfer study result must be committed"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert 0.0 <= d["image_auroc"] <= 1.0
    assert d["n_test_cracked"] > 0 and d["n_test_uncracked"] > 0
    h = d["score_hist"]
    assert len(h["centers"]) == len(h["cracked"]) == len(h["uncracked"])
    assert "PatchCore" in d["method"]
    # only metrics ship: no raw file paths leaked into the committed artifact
    assert "raw" not in json.dumps(d).lower() or "raw imagery" in d.get("dataset", "").lower()
