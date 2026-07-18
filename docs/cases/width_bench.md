# Case `width_bench`: sub-pixel width estimators against exact truth, calibrated to mm

Category `quantification-validation` · synthetic · seed 42 · 4 soft-edged bars (widths 1.5, 2.5,
4.0, 6.0 px) · demonstration scale 0.20 mm/px

## The central lesson: width has more than one definition

The bench runs THREE estimators on every specimen and validates each against its own true value:

1. **Inscribed-circle width** (2 x distance transform at the skeleton) and **orthogonal-profile
   width** measure the BINARY MASK: their truth is the generated plateau width.
2. **Intensity-domain sub-pixel width** reads half-depth crossings of the grayscale profile with
   linear interpolation: it measures the OPTICAL full width at half maximum. For the generator's
   plateau + Gaussian-shoulder edge profile the FWHM identity gives
   `optical = mask + 2.355 * softness`, so the two definitions differ by the edge physics, not by
   an error. The bench surfaces exactly that gap (about 1.9 px at softness 0.8), the trap a naive
   "width" comparison falls into.

## Measured accuracy (pinned stack, seed 42)

- Intensity estimator vs optical FWHM: median absolute error **0.007 px** on widths of 2.5 px and
  above (0.012 px including the 1.5 px bar). Expected band: at most 0.5 px; out-of-band fails the
  pipeline.
- Mask estimators vs mask width: the same 0.17 px class measured in `synthetic_battery`.
- With the demonstration scale (0.20 mm/px) the artifact carries every width in mm, exercising the
  same code path a calibrated field image would use.

## Calibration context

The scale here is DECLARED by the generator. For real imagery the lab implements two hardware-free
calibrations (`model/calibration.py`): a known reference object (two points + a known length) and a
pure-numpy DLT homography for oblique planar views (metric rectification from 4 or more
correspondences), with the standoff route (GSD = pZ/f) documented in the calibration guide. Scale
is never invented: no scale, no mm output.

## Artifact

`data/derived/width_bench/artifact.json` + overlays for all four specimens. Regenerate with
`python -m fisuralab.pipeline width_bench`.
