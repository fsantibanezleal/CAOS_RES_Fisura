"""Pipeline smoke + determinism: cases regenerate deterministically, the expected bands hold (the
version-pinning regression gate), the controls tell the honest ladder story, and the index exists."""
import json

from fisuralab import pipeline, registry


def test_synthetic_battery_deterministic_and_in_band(battery_manifest):
    b = pipeline.precompute("synthetic_battery")
    assert battery_manifest["artifact"]["bytes"] == b["artifact"]["bytes"], "same seed must reproduce the identical artifact"
    # the expected band is checked inside evaluate (raises out-of-band); assert the metrics exist too
    assert "mean_f1_L3_tol5_clean_bars" in battery_manifest["metrics"]
    assert "width_edt_abs_error_median_px" in battery_manifest["metrics"]


def test_uncracked_control_tells_the_ladder_story(battery_manifest):
    artifact = json.loads((pipeline.DERIVED / battery_manifest["artifact"]["path"]).read_text(encoding="utf-8"))
    controls = [
        s for s in artifact["samples"]
        if s["synthetic_params"] and s["synthetic_params"]["kind"] == "uncracked" and not s["synthetic_params"]["joint"]
    ]
    assert controls, "the battery must include an uncracked control"
    for s in controls:
        size = s["size"][0] * s["size"][1]
        l0 = s["levels"]["L0"]["segmentation"]["tol5px"]["n_pred"] / size
        l3 = s["levels"]["L3"]["segmentation"]["tol5px"]["n_pred"] / size
        # the honest exhibit: the L0 floor fires on most of the texture; L3 reduces false positives
        # by an order of magnitude but does NOT reach zero (percentile hysteresis marks texture maxima)
        assert l0 > 0.30, f"L0 should fire broadly on texture (got {l0:.1%})"
        assert l3 < 0.10, f"L3 should stay below 10 percent on the uncracked control (got {l3:.1%})"


def test_bcl_examples_runs_and_reports(bcl_manifest):
    assert "mean_f1_L4_tol5" in bcl_manifest["metrics"]
    assert bcl_manifest["schema"].startswith("fisura.manifest/")


def test_index_inventories_all_cases(battery_manifest, bcl_manifest):
    entries = [
        {"case_id": c.id, "category": c.category, "manifest_path": f"manifests/{c.id}.json"}
        for c in registry.list_cases()
    ]
    from fisuralab.core.manifest import build_index
    from fisuralab.io.formats import write_json

    write_json(pipeline.MANIFESTS / "index.json", build_index(entries))
    idx = json.loads((pipeline.MANIFESTS / "index.json").read_text(encoding="utf-8"))
    assert idx["n_cases"] == len(entries) >= 2
    assert idx["schema"].startswith("fisura.index/")
