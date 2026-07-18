"""The classical-track cases (the first two of the validated 16-case matrix).

- ``bcl_examples``: the committed CC0/CC BY example set (BCL patches with pixel masks, concrete +
  steel + an uncracked control, plus two SDNET2018 patches without masks) run through the full
  ladder. Real imagery, redistributable, so overlays ship in the artifact.
- ``synthetic_battery``: generated cracks with EXACT ground truth (widths 2..9 px, two angles, a
  tapering wavy crack, a low-contrast bar, uncracked and joint-trap controls). The regression gate
  for the ridge stack and the width estimators; also the honest floor/ceiling exhibit.

expected_band values are calibrated to the measured behaviour of the pinned stack (scikit-image
0.26) on these exact inputs; the evaluate stage checks them and CI fails if a version bump moves
results out of band (the dossier's version-pinning discipline made executable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..model.classical import LadderParams


@dataclass
class Case:
    id: str
    category: str
    title: str
    real_or_synthetic: str            # "real" | "synthetic"
    source: str                       # "examples" | "synthetic" | "synthetic_subpx"
    expected_band: dict = field(default_factory=dict)
    ladder: LadderParams = field(default_factory=LadderParams)
    seed: int = 42
    # Demonstration scale injected on masked samples (mm per px), for the severity-context case.
    # Explicitly labelled a demonstration in the artifact; the lab never invents a real scale.
    mm_per_px_demo: float | None = None


CASES: list[Case] = [
    Case(
        id="bcl_examples",
        category="classical-segmentation",
        title="Committed examples: BCL patches (+ SDNET controls) through the ladder",
        real_or_synthetic="real",
        source="examples",
        # Band over the 3 masked BCL crack patches, L4, 5 px tolerance, mean F1.
        # Calibrated on the pinned stack; a collapse below the floor fails evaluate.
        expected_band={"metric": "mean_f1_L4_tol5", "min": 0.35, "max": 1.0},
        ladder=LadderParams(sigmas=(1.0, 1.5, 2.0, 3.0), flatten_radius=15, tophat_length=15),
        seed=42,
    ),
    Case(
        id="synthetic_battery",
        category="quantification-validation",
        title="Synthetic crack battery: exact ground truth for ladder + width validation",
        real_or_synthetic="synthetic",
        source="synthetic",
        # Band over clean straight bars (widths 3/5/9, both angles), L3, 5 px tolerance, mean F1.
        expected_band={"metric": "mean_f1_L3_tol5_clean_bars", "min": 0.80, "max": 1.0},
        ladder=LadderParams(sigmas=(1.0, 2.0, 3.0, 4.5)),
        seed=42,
    ),
    Case(
        id="width_bench",
        category="quantification-validation",
        title="Width bench: sub-pixel estimators vs exact truth, calibrated to mm",
        real_or_synthetic="synthetic",
        source="synthetic_subpx",
        # Band: intensity-domain sub-pixel estimator, median absolute error on widths >= 2.5 px.
        expected_band={"metric": "subpx_intensity_fwhm_abs_error_median_px", "min": 0.0, "max": 0.5},
        ladder=LadderParams(sigmas=(0.8, 1.2, 2.0, 3.0)),
        seed=42,
        mm_per_px_demo=0.20,
    ),
    Case(
        id="severity_grading",
        category="quantification-validation",
        title="Severity context: measured widths in mm against ACI 224R-01 and EC2 Table 7.1N bands",
        real_or_synthetic="real",
        source="examples",
        # Band: the case must produce a severity record for every masked crack sample.
        expected_band={"metric": "n_severity_records", "min": 3, "max": 6},
        ladder=LadderParams(sigmas=(1.0, 1.5, 2.0, 3.0), flatten_radius=15, tophat_length=15),
        seed=42,
        mm_per_px_demo=0.10,
    ),
]
