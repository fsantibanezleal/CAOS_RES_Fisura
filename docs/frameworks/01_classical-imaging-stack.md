# Framework card: the classical imaging stack

The research-chosen engines of the classical track, pinned in `data-pipeline/requirements.txt` and
used for real by `fisuralab.model.classical` + `fisuralab.model.geometry`. No hand-rolled toy
substitutes: every primitive maps to a maintained library call, introspected and pinned.

## What and why

| Engine | Pin | Role | License |
|---|---|---|---|
| scikit-image | 0.26.0 | ridge filters (sato, frangi, meijering), thresholds (otsu, sauvola, hysteresis), morphology (black top-hat, diameter/area opening, remove-small), skeletonize/medial axis, minimal paths (route_through_array), texture features (LBP, GLCM, HOG) | BSD-3 |
| SciPy | 1.18.0 | ndimage convolutions, Euclidean distance transform, zoom, binary dilation | BSD-3 |
| scikit-learn | 1.9.0 | the L5 patch random forest | BSD-3 |
| imageio | 2.37.2 | PNG/JPG IO at the contract boundary | BSD-2 |
| PyWavelets | 1.9.0 | noise sigma estimation (NLM); the wavelet enhance option | MIT |
| numpy | 2.4.6 | everything | BSD-3 |

The whole default ladder is permissive (BSD/MIT) and Pyodide-safe by design: the browser live lane
imports the same `model/` modules. License-gated or native extras from the research (BM3D
non-commercial, DIPlib path openings, OpenCV specials) enter only as optional offline plugins,
never in the core.

## The version-pinning discipline (binding)

The research documented paper deviations and version-to-version behaviour changes in the
scikit-image ridge-filter family, and excluded `skimage.filters.hessian` outright. Consequences,
implemented here:

1. scikit-image is pinned EXACTLY (0.26.0); an upgrade is a deliberate act.
2. The synthetic battery (case `synthetic_battery`) is the regression gate: known bars at known
   widths must keep their scores (`expected_band` fails the pipeline; unit tests fail the suite).
3. `skimage.filters.hessian` is not called anywhere; sato is the default ridge, frangi and
   meijering are selectable alternatives.

## The staged engine in one paragraph

`run_level` composes pure primitives per ladder level: median-subtract illumination flattening
(recentred at 0.5, deliberately without min-max renormalization), non-local-means denoising with
estimated sigma, multiscale ridge or oriented black-top-hat enhancement, percentile hysteresis
binarization, connected-component shape rules (permissive on large components), endpoint
minimal-path bridging through the response cost map, skeleton topology (Lee), and the dual width
estimators (inscribed circle vs orthogonal profiles; disagreement is a per-point quality flag,
junction neighbourhoods excluded). Every function is deterministic; every case is a pure function
of (params, seed).

## Install and run

```bash
./scripts/setup.sh                                  # .venv-pipeline with the pins above
.venv-pipeline/bin/python -m fisuralab.pipeline     # bake all cases
.venv-pipeline/bin/python -m pytest tests -q        # incl. the synthetic regression gates
```

## Runnable example

```python
from fisuralab.model.classical import LadderParams, run_level
from fisuralab.model.geometry import measure, width_stats
from fisuralab.model.synthetic import straight_bar

spec = straight_bar(width_px=5.0, angle_deg=30.0, seed=4)
res = run_level(spec.image, "L3", LadderParams(sigmas=(1.0, 2.0, 3.0, 4.5)))
print(width_stats(measure(res.mask)))   # width medians land near 5 px
```
