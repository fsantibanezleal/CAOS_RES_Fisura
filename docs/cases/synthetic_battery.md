# Case `synthetic_battery`: exact ground truth for the ladder and the width estimators

Category `quantification-validation` · synthetic · seed 42 · 12 specimens

## Why it exists

Generated cracks are the only inputs whose centerline, mask and width are known EXACTLY (by
construction), which buys three things at once:

1. **A regression gate for the classical stack.** The scikit-image ridge filters carry documented
   version sensitivity (paper deviations and a sigma-scaling regression across releases; see the
   framework card). The pinned stack must reproduce this battery's scores; `evaluate` fails the
   pipeline out-of-band, and `tests/test_stages_and_geometry.py` fails a bare upgrade.
2. **Width-estimator accuracy with a true answer.** Both estimators run on the exact masks: the
   inscribed-circle width (2 times the distance transform at the skeleton) and the orthogonal
   profile width with sub-pixel boundary interpolation. Measured on the pinned stack: median
   absolute error 0.172 px across the battery.
3. **The honest floor/ceiling exhibit.** The battery includes what breaks naive methods: a
   low-contrast bar, an uncracked textured control, and a straight dark JOINT (the classic false
   positive; joints are straight, cracks meander).

## Specimens

Widths 2, 3, 5 and 9 px at two orientations (straight bars); a meandering crack with tapering
width; a low-contrast bar; two negative controls (plain texture, texture + joint). Background is a
generated concrete-like texture (base level + broadband noise + smooth blotches). Dark-crack
convention throughout.

## Measured behaviour (pinned stack, seed 42; buffered P/R/F1, protocol printed)

- Clean straight bars, L3 at 5 px tolerance: mean F1 0.952 (expected band 0.80 to 1.0).
- Full-battery means run from L0 0.92 to L2 1.00 to L4 0.94 at 5 px; at 2 px the same masks read
  8 to 15 points lower: the protocol effect, visible on one chart in the App.
- The uncracked control fires on about half the texture at L0 and about 6 percent at L3: the ladder
  reduces false positives by an order of magnitude and does NOT reach zero (percentile hysteresis
  always marks some texture maxima). This is asserted in tests, not just narrated.
- Caveat carried from the honest reading: synthetic backgrounds flatter L0/L1 (their smoothness is
  kinder than real texture); the real-imagery ceiling lives in `bcl_examples`.

## Artifact

`data/derived/synthetic_battery/artifact.json` (RLE masks per level, dual-tolerance metrics,
geometry, width validation) + overlays for four representative specimens; manifest under
`data/derived/manifests/`. Everything regenerates with
`python -m fisuralab.pipeline synthetic_battery`.
