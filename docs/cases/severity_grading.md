# Case `severity_grading`: measured widths in mm against published guidance bands

Category `quantification-validation` · real imagery (BCL, CC0) · seed 42 · 3 masked crack patches ·
DEMONSTRATION scale 0.10 mm/px (flagged in the manifest; no real calibration exists for the source
imagery)

## What it does

Runs the ladder on the three masked BCL crack patches, measures widths on the L4 mask, converts to
mm with the demonstration scale, and compares the median and 95th-percentile widths against the two
published reference tables the research verified:

- **ACI 224R-01** guide table of tolerable crack widths at the tensile face (0.41 mm dry air down
  to 0.10 mm for water-retaining structures), carrying the document's own warning that width is not
  always a reliable indicator of corrosion or deterioration.
- **EN 1992-1-1:2004 Table 7.1N** recommended limiting calculated widths (0.4 mm X0/XC1 for
  appearance; 0.3 mm XC2 to XC4 and the chloride/seawater classes), carrying the National-Annex
  dependence and the UNVERIFIED-primary flag on the XC1 row (secondary sources disagree; both
  readings are shown until the standard text is checked).

## The framing (binding, embedded in every output)

These are serviceability reference bands for calculated design widths, used strictly as CONTEXT
for measured widths: the artifact's severity record states "reference bands for context; not a
structural safety verdict" and ships both caveats verbatim. The workbench renders the full band
table with within/exceeds badges for median and p95 separately (a crack can sit within a band at
the median and exceed it at p95; both facts are shown, never collapsed).

## Why a demonstration scale

Percentile-vs-band comparison is only meaningful in mm, and the BCL source publishes no
calibration. Rather than fake precision, the case declares 0.10 mm/px as a demonstration (a
plausible close-up macro scale), flags it in the manifest, and exists to exercise the REAL code
path end to end: mask, widths, calibration, banding, caveats. Bring-your-own-image use with a real
reference-object or homography calibration produces honest mm from the same functions.

## Artifact

`data/derived/severity_grading/artifact.json` (+ overlays). Regenerate with
`python -m fisuralab.pipeline severity_grading`.
