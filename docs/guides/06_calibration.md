# Guide, calibrating pixels to millimetres

Physical-width output exists ONLY when a scale is provided; the pipeline never invents one
(CONTRACT 1 treats `mm_per_px` as optional, and mm fields simply do not appear without it). Three
routes, in increasing generality (`fisuralab/model/calibration.py`):

## 1. Known reference object (local scale)

Place an object of known size in the crack plane (a crack gauge card, a coin, a ruler), identify
two image points spanning a known length, and call:

```python
from fisuralab.model.calibration import scale_from_reference
mm_per_px = scale_from_reference((r0, c0), (r1, c1), length_mm=25.0)
```

Valid locally; error grows with perspective obliquity and surface curvature. Degenerate point pairs
and absurd lengths are rejected.

## 2. Planar homography (oblique views of flat elements)

Photograph a printed checkerboard or any 4+ known metric points on the SAME plane as the crack,
estimate the image-to-plane homography (pure-numpy DLT with Hartley normalization), and measure in
plane coordinates; the spatially varying scale of an oblique view is handled correctly:

```python
from fisuralab.model.calibration import homography_dlt, width_mm_via_homography
H = homography_dlt(image_pts, plane_pts_mm)          # (N,2) each, N >= 4
w_mm = width_mm_via_homography(H, point=(r, c), tangent=theta, width_px=w_px)
```

The width endpoints are mapped through H at the measurement location, so each width uses the local
scale.

## 3. Standoff geometry (documented; requires known camera parameters)

With pixel pitch p, focal length f and standoff distance Z (laser rangefinder or measured), the
ground sampling distance is GSD = p Z / f and a fronto-parallel width reads w_mm = w_px x GSD.
The lab documents this route (it is how UAV systems with laser modules obtain per-frame scale, per
the research) but ships no function for it: it would require inventing Z, f or p when they are not
actually known, which the contract forbids.

## Accuracy expectations (from the verified research record)

Published systems with an in-image metric reference and sub-pixel edges validate to roughly 0.05 to
0.2 mm absolute width error at close range; UAV standoff imagery bottoms out near 0.1 to 0.25 mm
minimum measurable width. Below those floors the limit is ground sampling distance, not the
algorithm. The `width_bench` case measures this lab's estimator errors in px against exact truth;
your mm accuracy is that error times your calibrated scale, plus your calibration error.
