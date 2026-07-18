"""Quantification flagship tests: calibration (reference scale + DLT homography), severity banding
with first-class caveats, the intensity-domain sub-pixel estimator against exact optical truth, and
the two quantification cases end to end."""
import json

import numpy as np
import pytest

from fisuralab import pipeline
from fisuralab.model.calibration import (
    apply_homography,
    homography_dlt,
    scale_from_reference,
    width_mm_via_homography,
)
from fisuralab.model.geometry import intensity_width_stats, measure
from fisuralab.model.severity import band_widths
from fisuralab.model.synthetic import straight_bar


def test_scale_from_reference():
    assert scale_from_reference((0, 0), (0, 100), 50.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        scale_from_reference((0, 0), (0, 1), 50.0)  # degenerate points
    with pytest.raises(ValueError):
        scale_from_reference((0, 0), (0, 100), 1e6)  # absurd length


def test_homography_roundtrip_and_local_width():
    # a known projective map: image corners of a 100 mm square photographed obliquely
    plane = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    image = np.array([[10, 12], [220, 30], [200, 210], [25, 190]], dtype=float)
    H = homography_dlt(image, plane)
    mapped = apply_homography(H, image)
    assert np.allclose(mapped, plane, atol=1e-6)
    # a 10 px width near the first corner maps to a physically plausible mm width under this view
    w_mm = width_mm_via_homography(H, point=(12, 10), tangent=0.0, width_px=10.0)
    assert 2.0 < w_mm < 20.0


def test_severity_banding_logic_and_caveats():
    rec = band_widths(0.25, 0.35)
    by = {(r["source"], r["exposure"]): r for r in rec["bands"]}
    dry = by[("ACI 224R-01", "dry air or protective membrane")]
    assert dry["median_within"] and dry["p95_within"]  # 0.25/0.35 <= 0.41
    water = by[("ACI 224R-01", "water-retaining structures (excluding nonpressure pipes)")]
    assert not water["median_within"] and not water["p95_within"]  # > 0.10
    moist = by[("ACI 224R-01", "humidity, moist air, soil")]
    assert moist["median_within"] and not moist["p95_within"]  # 0.25 <= 0.30 < 0.35
    assert len(rec["caveats"]) == 2 and "not a structural safety verdict" in rec["framing"]
    assert any("UNVERIFIED-primary" in c for c in rec["caveats"])


def test_intensity_subpixel_recovers_optical_fwhm():
    for width in (2.5, 4.0):
        spec = straight_bar(width_px=width, angle_deg=35.0, contrast=0.35, noise=0.015, blotch=0.02, seed=7)
        geom = measure(spec.mask)
        iw = intensity_width_stats((spec.image), geom)
        true_fwhm = width + 2.355 * spec.params["softness"]
        assert iw["intensity_median"] is not None
        assert abs(iw["intensity_median"] - true_fwhm) <= 0.5, (
            f"width {width}: intensity {iw['intensity_median']:.2f} vs optical FWHM {true_fwhm:.2f}"
        )


def test_width_bench_case(width_bench_manifest):
    m = width_bench_manifest
    assert m["metrics"]["subpx_intensity_fwhm_abs_error_median_px"] <= 0.5
    artifact = json.loads((pipeline.DERIVED / m["artifact"]["path"]).read_text(encoding="utf-8"))
    s0 = artifact["samples"][0]
    assert s0["width_mm"] is not None and s0["width_mm"]["mm_per_px"] == 0.20


def test_severity_case(severity_manifest):
    m = severity_manifest
    assert m["metrics"]["n_severity_records"] == 3
    artifact = json.loads((pipeline.DERIVED / m["artifact"]["path"]).read_text(encoding="utf-8"))
    graded = [s for s in artifact["samples"] if s.get("severity")]
    assert len(graded) == 3
    for s in graded:
        sev = s["severity"]
        assert len(sev["bands"]) >= 7 and len(sev["caveats"]) == 2
        assert "not a structural safety verdict" in sev["framing"]
    # the demonstration scale is flagged in the manifest, never silent
    assert any("DEMONSTRATION scale" in f for entry in m["flags"] for f in entry["flags"])
