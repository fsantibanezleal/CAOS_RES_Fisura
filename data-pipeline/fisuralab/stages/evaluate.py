"""Stage 5, evaluate: dual-tolerance metrics per level + width validation + the expected-band check.

Every masked sample is scored at BOTH 2 px and 5 px tolerances (the protocol travels with every
number). Synthetic samples with exact known widths additionally validate the two width estimators
(the battery is the regression gate for the ridge stack and the geometry). The case's
expected_band is checked here: out-of-band means the pinned stack regressed, and the pipeline
fails loudly instead of committing drifted artifacts.
"""
from __future__ import annotations

import numpy as np

from ..io.image_contract import ImageSample
from ..model.classical import LevelResult
from ..model.geometry import measure, width_stats
from ..model.metrics import evaluate_mask


def run(case, samples: list[ImageSample], inferred: list[dict[str, LevelResult]]) -> tuple[dict, list[dict]]:
    """Returns (case_metrics, per_sample_records)."""
    per_sample: list[dict] = []
    for s, levels in zip(samples, inferred):
        rec: dict = {"sample_id": s.sample_id, "levels": {}}
        for level, res in levels.items():
            entry: dict = {"notes": res.notes}
            if s.mask is not None:
                entry["segmentation"] = evaluate_mask(res.mask, s.mask)
            rec["levels"][level] = entry
        # geometry on the quantification level (L4; falls back to the best available)
        geo_level = "L4" if "L4" in levels else max(levels)
        geom = measure(levels[geo_level].mask)
        rec["geometry_level"] = geo_level
        rec["geometry"] = {
            "length_px": geom.length_px,
            "n_branches": geom.n_branches,
            "n_endpoints": geom.n_endpoints,
            "orientation_hist": geom.orientation_hist.tolist(),
            "width": width_stats(geom),
        }
        # width validation against exact synthetic truth
        true_w = getattr(s, "true_width_px", None)
        if true_w is not None and len(true_w) > 0 and s.mask is not None and s.mask.any():
            gt_geom = measure(s.mask)
            ws = width_stats(gt_geom)
            rec["width_validation"] = {
                "true_width_px": float(np.median(true_w)),
                "edt_on_gt_median": ws["edt_median"],
                "profile_on_gt_median": ws["profile_median"],
                "edt_abs_error": abs(ws["edt_median"] - float(np.median(true_w))) if ws["edt_median"] is not None else None,
                "profile_abs_error": abs(ws["profile_median"] - float(np.median(true_w))) if ws["profile_median"] is not None else None,
            }
        per_sample.append(rec)

    case_metrics = _case_rollup(case, samples, per_sample)
    _check_band(case, case_metrics)
    return case_metrics, per_sample


def _mean_f1(per_sample: list[dict], sample_ids: list[str], level: str, tol: str) -> float | None:
    vals = []
    for rec in per_sample:
        if rec["sample_id"] not in sample_ids:
            continue
        seg = rec["levels"].get(level, {}).get("segmentation")
        if seg:
            vals.append(seg[tol]["f1"])
    return float(np.mean(vals)) if vals else None


def _case_rollup(case, samples: list[ImageSample], per_sample: list[dict]) -> dict:
    masked_ids = [s.sample_id for s in samples if s.mask is not None and s.mask.any()]
    out: dict = {"protocol": "buffered P/R/F1 at 2 px AND 5 px tolerances; strict IoU; no thinning/NMS"}
    for level in ("L0", "L1", "L2", "L3", "L4", "L5"):
        for tol in ("tol2px", "tol5px"):
            v = _mean_f1(per_sample, masked_ids, level, tol)
            if v is not None:
                out[f"mean_f1_{level}_{tol[3:]}"] = round(v, 4)
    if case.id == "synthetic_battery":
        clean = [r["sample_id"] for r in per_sample if "straight_bar" in r["sample_id"] and r["sample_id"] in masked_ids]
        # exclude the deliberately low-contrast bar (the last straight_bar in the battery order)
        if len(clean) > 1:
            clean = clean[:-1]
        v = _mean_f1(per_sample, clean, "L3", "tol5px")
        if v is not None:
            out["mean_f1_L3_tol5_clean_bars"] = round(v, 4)
        errs = [r["width_validation"]["edt_abs_error"] for r in per_sample if "width_validation" in r and r["width_validation"]["edt_abs_error"] is not None]
        if errs:
            out["width_edt_abs_error_median_px"] = round(float(np.median(errs)), 3)
    if case.id == "bcl_examples":
        v = _mean_f1(per_sample, masked_ids, "L4", "tol5px")
        if v is not None:
            out["mean_f1_L4_tol5"] = round(v, 4)
    return out


def _check_band(case, case_metrics: dict) -> None:
    band = case.expected_band
    if not band:
        return
    metric = band["metric"]
    val = case_metrics.get(metric)
    if val is None:
        raise AssertionError(f"{case.id}: expected-band metric '{metric}' was not produced")
    if not (band["min"] <= val <= band["max"]):
        raise AssertionError(
            f"{case.id}: {metric}={val} outside the expected band [{band['min']}, {band['max']}]; "
            "the pinned stack regressed (or the band needs a deliberate recalibration)"
        )
