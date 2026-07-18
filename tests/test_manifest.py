"""CONTRACT 2 (artifact) tests: the manifest points to a real artifact with the recorded byte size, the lane
verdict is consistent with the gate, and the artifact parses with round-trippable RLE masks."""
import json

from fisuralab import pipeline
from fisuralab.core.artifact import rle_decode


def test_manifest_matches_artifact_and_gate(battery_manifest):
    m = battery_manifest
    artifact_path = pipeline.DERIVED / m["artifact"]["path"]
    assert artifact_path.exists(), "manifest points to a non-existent artifact"
    assert artifact_path.stat().st_size == m["artifact"]["bytes"], "manifest byte size drifted from the artifact"
    assert m["schema"].startswith("fisura.manifest/")
    assert m["lane"] in ("live", "precompute")
    assert m["gate"]["lane"] == m["lane"], "manifest lane disagrees with the gate verdict"

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema"].startswith("fisura.artifact/")
    assert artifact["n_samples"] == len(artifact["samples"]) > 0
    s0 = artifact["samples"][0]
    mask = rle_decode(s0["levels"]["L3"]["mask_rle"])
    assert list(mask.shape) == s0["size"]
    # every masked sample carries dual-tolerance metrics with the protocol attached
    seg = s0["levels"]["L3"]["segmentation"]
    assert seg is not None and "tol2px" in seg and "tol5px" in seg and "protocol" in seg
