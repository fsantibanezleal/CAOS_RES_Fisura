# Changelog

All notable changes to this product. Format: `X.XX.XXX` (display), see `fisuralab.__version__`. Keep `0.x`
while on mock/synthetic data. Tag every release.

## [0.06.000], 2026-07-18

### Added
- Learned ladder A, trained in-repo on the local RTX 4070 (the research recipe: 512 reflect-padded
  crops, flip/rot90, batch 6 x 2 accumulation, AdamW 3e-4 cosine, AMP, BCE(pos_weight 8) + Dice,
  early stop on val F1 at the STRICT 2 px protocol, seeded): SMP U-Net resnet18 (val F1@2px 0.752,
  11.6 min), DeepLabV3+ resnet18 (0.720, 13.0 min) and SegFormer mit_b2 (0.766, 24.5 min) on a
  3,000-image CrackSeg9k subset (9,159 pairs indexed; the full-data rerun is the same command
  without the limit and re-bakes the case).
- Case `learned_on_examples`: the three trained architectures replayed on the SAME committed
  patches as the classical ladder, same harness, same workbench. Measured: examples mean F1@5px
  0.681 (DeepLabV3+) / 0.653 (SegFormer) / 0.486 (U-Net) vs the classical ladder's 0.455: the
  learned advantage on hard real patches, quantified on identical pixels.
- ONNX exports with CPU-parity verification and sha256 recorded in the manifest (49 to 99 MB, in
  the local model vault; the live-lane unit later commits the one compact browser model).
- The GPU lane (`data-pipeline/requirements-gpu.txt`, torch 2.11 cu128 + SMP + onnxscript) is
  local-only: all torch imports are lazy, training tests skip without CUDA, and the pipeline bakes
  the learned case torch-free from the persisted run record (CI unchanged).

## [0.05.000], 2026-07-18

### Added
- The quantification flagship: pixel-to-mm calibration (known reference object + pure-numpy DLT
  homography with Hartley normalization and local width mapping; standoff GSD route documented,
  never computed from invented parameters), and severity CONTEXT from published guidance (ACI
  224R-01 tolerable-width table and EN 1992-1-1 Table 7.1N as data with their caveats first-class:
  the ACI corrosion warning, the National-Annex dependence, the flagged XC1-row ambiguity; every
  output states "reference bands, not a structural safety verdict").
- Intensity-domain sub-pixel width estimator (half-depth crossings with linear interpolation),
  validated against the correct optical definition: on the soft-edge generator the FWHM identity
  is optical = mask + 2.355 x softness, and the estimator lands within 0.007 px median of it.
- Cases `width_bench` (three estimators, each vs its own true definition; mm via a demonstration
  scale) and `severity_grading` (BCL mm percentiles vs the bands; demonstration scale explicitly
  flagged in the manifest), both band-gated.
- Workbench: mm KPI row, the estimator-definition validation table, and the severity band table
  with per-band within/exceeds badges for median and p95 plus the caveats rendered verbatim.
- Docs: width-bench and severity case docs, the calibration guide (06).

## [0.04.000], 2026-07-18

### Added
- The classical engine (the anchor slice): staged S0-S8 pipeline composed into ladder L0-L5
  (global-Otsu floor; flatten + Sauvola; oriented black-top-hat + hysteresis; NLM + multiscale
  ridge sato/frangi/meijering; endpoint minimal-path bridging; LBP+GLCM+HOG random-forest fusion),
  all numpy/scipy/scikit-image, Pyodide-safe by design.
- The quantification core: dual width estimators (inscribed-circle 2xEDT at the skeleton vs
  orthogonal profiles with sub-pixel crossings; disagreement as a per-point quality flag, junction
  neighbourhoods excluded), skeleton length, orientation histograms, branch/endpoint topology.
- The dual-tolerance evaluation harness: buffered P/R/F1 at BOTH 2 px and 5 px with the protocol
  string attached to every record, plus strict IoU.
- The synthetic validation battery (12 specimens, exact ground truth): the version-pinning
  regression gate (expected bands fail the pipeline; unit tests fail a bare scikit-image upgrade).
  Measured: L3 clean-bar mean F1@5px 0.952; width estimator median error 0.172 px.
- Two shipped cases with committed artifacts (RLE masks + metrics + geometry + overlays):
  `bcl_examples` (real CC0/CC BY imagery; honest per-patch spread, mean L4 F1@5px 0.455) and
  `synthetic_battery`; CONTRACT 2 schemas fisura.manifest/artifact v1 with the TS mirror + RLE
  decoder; the live-lane entrypoint (`live.py`) over the same model code.
- The App workbench: case + sample selectors, the L0-L5 variant bar, Field canvas with client-side
  RLE overlays (prediction/ground-truth/overlap), interactive uPlot charts (dual-protocol F1 per
  level, orientation histogram), width KPI tiles, deep bilingual Context write-ups.
- Docs: classical-stack framework card (pins + the version-sensitivity discipline), case docs with
  measured behaviour, case taxonomy README.

### Changed
- Percentile hysteresis + shape rules recalibrated on real imagery (flatten without contrast
  renormalization; permissive rules on large components); BCL example masks inverted once at
  curation from the source's black-on-white convention (recorded in the case doc).
- CI pipeline smoke now regenerates `synthetic_battery`.

### Removed
- The archetype's SIR example engine, its cases, baked artifacts and the `.template-source`
  sentinel: the template-residue guard is now ARMED and CI-enforced.

## [0.03.000], 2026-07-18

### Added
- The real CONTRACT 1 for the image domain (`io/image_contract.py`): image + optional binary mask +
  optional mm-per-px scale + material/source/license metadata, with hard-reject rules, soft flags
  (near-constant image, suspicious mask coverage, tiny masks) and the redistribution boundary
  (`is_redistributable`). Numpy-only core so the browser live lane reuses the exact validation.
- Standard-format IO (`io/image_formats.py`): PNG/JPG readers, mask IO, float conversion, and the
  committed-examples manifest loader.
- Curated committed example set (`data/examples/`): 4 Bridge Crack Library patches with pixel masks
  (CC0; concrete, steel, and an uncracked control) and 2 SDNET2018 patches (CC BY 4.0; cracked and
  uncracked), each attributed in `manifest.json`; CI validates every example through CONTRACT 1.
- `scripts/fetch-data.ps1` + `.sh`: idempotent acquisition of every direct-download dataset the lab
  uses (Dataverse, S3, Zenodo, Mendeley API, GitHub, Kaggle mirrors, Drive), rooted at
  `FISURA_DATA_ROOT`; gated sets documented as manual steps.
- `data/README.md` rewritten as the dataset registry: license, retrieval and redistribution ruling
  per source; the bring-your-own-data guide updated to the image contract.
- Tests: contract validation paths + committed examples through the gate (7 new tests).

## [0.02.000], 2026-07-18

### Added
- Base frontend shell wired on the shared design system (`@fasl-work/caos-app-shell` 0.3): six pages
  (App, Introduction, Methodology, Implementation, Experiments, Benchmark), EN/ES i18n, light/dark
  theming, per-panel error boundary, Pages SPA 404 redirect shim, display version derived from the
  manifest.
- Real page content transcribed from the verified research dossiers: the seven method tracks, the
  16-case matrix with honest planned statuses, KaTeX method equations (inscribed-circle width,
  Paris-Erdogan, ZNCC), the published-anchor benchmark tables with per-row primary citations, and
  the evaluation-protocol duality (2 px vs 5 px tolerance) documented up front.
- Citation spine (data/citations.ts) with verified DOIs and arXiv identifiers only.
- Screenshot verification: all six pages captured light + dark (plus ES sanity), zero console errors.

### Removed
- SIR frontend residue (the example App view, its chart component, the Pyodide stub wired to the example engine); the SIR
  pipeline engine itself remains until the classical-ladder unit replaces it.

## [0.01.000], 2026-07-18

### Added
- Initial instantiation from the CAOS product-repo template (ADR-0057).
- Offline `data-pipeline/` (`fisuralab`): the two data contracts (ingestion + artifact), the named staged
  pipeline (preprocess → feature_extraction → train → infer → evaluate → export), the seeded RNG, the compact
  trace, the manifest, and the measured live-vs-precompute gate.
- EXAMPLE engine: a deterministic SIR epidemic (numpy-only, Pyodide-safe), **replace with the product's
  research-chosen SOTA engine**.
- Cases-by-category registry (4 regimes + 1 degenerate control); a live-lane entrypoint (`live.py`); tests for
  both contracts + pipeline determinism.
