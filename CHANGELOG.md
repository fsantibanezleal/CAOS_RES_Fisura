# Changelog

All notable changes to this product. Format: `X.XX.XXX` (display), see `fisuralab.__version__`. Keep `0.x`
while on mock/synthetic data. Tag every release.

## [0.14.000], 2026-07-20

### Added (BL-010 multi-class damage track: dacl10k 19-class segmentation)
- The App's binary crack ladder gains real bridge-inspection grading. New `fisuralab.multiclass`
  module (from the verified dossier `wip/fisura/research/BL-010-multiclass-research-2026-07-19.md`):
  - `dacl10k.py`: the 19-class list (13 damage + 6 object, toolkit TARGET_LIST verbatim; NOT vegetation,
    which the dossier corrected as an S2DS class) + a LabelMe-polygon rasterizer to a (19, H, W)
    multi-label mask (a pixel can carry several classes) + the split lister.
  - `train_dacl10k.py`: an SMP EfficientNet-B4 + FPN multi-label segmenter (sigmoid + BCE + Dice,
    512-crop, macro-mIoU), GPU local lane. Trained 12 epochs on 2,500 images: measured **val mIoU
    0.117** (peaked 0.122). Below the 0.424 WACV 2024 paper baseline, the honest cost of the reduced
    budget (2,500 of 6,935 images on an 8 GB laptop GPU), stated rather than hidden; a full-data rerun
    is one flag away.
  - `bake_dacl10k.py`: writes the committed metrics artifact (`data/derived/multiclass/dacl10k.json`)
    + five low-resolution class-coloured overlays. dacl10k is CC BY-NC 4.0: the raw images and trained
    weights stay local; only metrics + tiny transformative low-res derivatives ship.
- New **Multi-class** page: the measured mIoU vs the 0.424 baseline, a class-coloured prediction viewer
  over real inspection crops with a per-image present-classes legend, the per-class IoU chart (the
  honest imbalance: Graffiti 0.38 down to thin classes near 0), and the license/honesty framing.
- CODEBRIM box-detection code (`codebrim.py` VOC parser, `train_codebrim.py` Faster R-CNN, `bake_codebrim.py`)
  is complete and lint-clean but NOT shipped: the local `CODEBRIM_original_images.zip` is truncated
  (~284 MB short, every image entry past the 4 GB Zip64 mark is unreadable), a download failure that
  needs a re-download. It ships as the next rung; the Multi-class page names it as such.
- `tests/test_multiclass.py`: the 19-class + 5-defect lists, the rasterizer multi-label contract, and
  the committed-artifact coherence (all CI-safe, torch-free).

## [0.13.000], 2026-07-20

### Changed (ADR-0016 section 6 compliance: the five content pages now use Tabs/SubTabs)
- The content pages were flat single-scroll walls. They now honor the shell structure: **Methodology**
  is a top-level Tabs (The seven tracks / Evaluation protocol) with a **vertical SubTabs rail** of the
  seven method tracks (1 Classical through 7 Monitoring), each carrying its own prose + KaTeX + Refs.
  **Benchmark** is a SubTabs of the four published records (Classical / Protocol trap / Learned / Anomaly).
  **Introduction** is four Tabs (What Fisura is / The whole ladder / Masks become numbers / Honesty as
  method). **Implementation** is three Tabs (Three lanes / Data & storage / License & quality). All prose,
  equations, citations and SVG figures are preserved verbatim, only reorganized.
- **Footer compacted** to the ADR-0016 section 2 bar: provenance and disclaimer are now single crisp
  clauses (was two long paragraphs).

### Fixed
- **Orientation rose was vertically flipped** relative to the image: the wedge math used a y-up convention
  while the tangent angle is `arctan2(row, col)` (row-down, image coordinates). The rose now grows y
  downward so a crack running down in the image reads down in the rose; the 90 degree label moved to the bottom.

### Added
- **Overlay legends** under every coloured image overlay (accessibility, dataviz rule: identity is never
  colour-alone): the prediction/segmentation tabs, the Summary matrix, and the skeleton graph each carry a
  colour to label key (prediction / ground truth / overlap; anomaly heat gradient + flagged-region outline;
  branch / junction / endpoint). New `render/OverlayLegend.tsx`.
- **Hessian ridge scale-space viewer** in the Preprocessing tab (research shortlist #7): sweep the per-sigma
  ridge response and the argmax-sigma map (which scale fires strongest per pixel, red = wide crack, blue =
  fine texture), so the reader sees why the ridge filter is multi-scale. Baked by `bake_workbench` (per-sigma
  PNGs + the argmax map).

## [0.12.000], 2026-07-20

### Added (workbench enrichment: Quantification + Metrics tabs, from the enrichment research shortlist)
- Deep online research (`wip/fisura/research/app-enrichment-viz-methods-2026-07-20.md`, 12-item primary-source
  shortlist) surfaced the highest-ROI rich visualizations. This lands the first two as new workbench tabs.
- **Quantification tab**: the crack turned into engineering quantities on the image, the skeleton graph
  overlay (branch polylines + junction/endpoint nodes by degree, hoverable), the width-along-arc-length
  profile w(s) (inscribed-circle width traced down the centerline, the crack-opening-displacement view the
  severity bands read against), and a length-weighted orientation rose (dominant crack direction).
- **Metrics tab**: the honest scoring, F1-vs-tolerance sweep (0..8 px) for all five learned models on the
  selected image (the tolerance axis that makes the same method read 0.85 or 0.23), a TP/FP/FN confusion
  table at 2 px, and the ensemble-disagreement uncertainty (per-pixel stdev across the models, free).
- `fisuralab.cases.bake_enrichment` computes these per committed image from the GT/reference crack + the
  per-model masks (scikit-image/scipy only, all from the existing geometry engine): `data/derived/enrichment/`.
  New `render/SkeletonOverlay.tsx` (SVG graph on the image) + `render/RoseDiagram.tsx` (themed polar SVG).

## [0.11.000], 2026-07-20

### Changed (App rebuilt into a per-image interactive WORKBENCH, Felipe's spec 2026-07-20)
- The App is no longer method-per-tab. It is a per-case workbench: the LEFT COLUMN is the selection +
  parameter area (case source: Prebaked / Pretrained / Upload-your-own; the image thumbnail strip; and the
  parameters that drive the tabs, including the SLIC superpixel-count and compactness sliders), and the TABS
  walk the pipeline ON the selected image, left to right: Overview, Preprocessing, Semantic segmentation,
  SLIC, Classical, SOTA, Beyond SOTA, Summary. Every method is VISIBLE ON THE IMAGE; nothing is a bare table.
- Preprocessing tab shows the REAL classical stages applied to the image (grayscale, illumination-flattened,
  NLM-denoised, Hessian ridge response), baked as committed PNGs from `fisuralab.model.classical`.
- SLIC tab renders scikit-image superpixel boundaries over the image and REACTS to the left-column sliders
  (a baked grid over n_segments x compactness; the slider snaps to the nearest baked variant, replay lane).
- Summary tab is the full comparison matrix: every method applied to the image at once, the winner (highest
  F1@2px) starred, plus a cross-method F1 ranking at both tolerance protocols.

### Added
- `fisuralab.cases.bake_workbench`: bakes the preprocessing intermediates + the SLIC grid per example image
  (`data/derived/workbench/`), plus the workbench index. `render/MethodTile.tsx` (compact prediction-on-image
  tile) and the workbench loader in `api/artifacts.ts`.

## [0.10.001], 2026-07-20

### Changed (Experiments rebuilt from a card grid to a real protocol page, ADR-0017 section 2)
- The Experiments page was a grid of status cards (the exact "info-box cards" ADR-0017 forbids). It is
  now prose + tabs separating the distinct experimental questions: The metric (buffered P/R/F1 with the
  exact equation and the dual 2px/5px convention), Leakage-safe protocol (split by physical surface, with
  a hand-authored theme-aware SVG that draws the forbidden random-patch split STRUCK OUT plus the
  tolerance-protocol band), Datasets (a real table with per-set license + redistribution mode + live/planned
  status), Coverage matrix (the 7 tracks x 16 cases with real per-track live counts), and Results so far
  (per-architecture mean F1 at 2px and 5px read LIVE from the committed learned-on-examples artifact, not
  typed into the page: DeepLabV3+ 0.504, SegFormer 0.472, HrSegNet 0.445, DINOv2 probe 0.440, U-Net 0.335).
- New `svg/tech/exp-protocol.svg` (theme-aware, CSS-var tokens, zero hardcoded hex); per-section `<Refs>`.

## [0.10.000], 2026-07-20

### Changed (App workbench rebuilt to the ADR-0017 section 3 structure)
- The App is re-architected so the methods are VISIBLE ON THE IMAGE, not in a table. The left column
  is now parametrization only: a data-source segmented control (Synthetic knobs / Real cases / Your
  own image), the sample picker, the per-method parameters (classical rung, SOTA model, overlay and
  anomaly-heat opacity, ground-truth toggle), and a live diagnosis read-out that updates with every
  control (classical best F1 vs SOTA best F1 vs anomaly score, plus a one-line verdict of whether the
  learned model earns its complexity on this sample).
- The top tabs are the METHODS in ladder order: Classical (L0-L5), SOTA (learned: U-Net, DeepLabV3+,
  SegFormer-B2, HrSegNet, DINOv2 linear probe), Beyond SOTA (PatchCore anomaly), Quantification,
  Context. Every method tab renders its prediction ON the same image with a value read-out; sweeping
  the classical rung or the SOTA model in the sidebar changes the overlay and the F1 live. This is the
  "what does each rung buy on the same pixels" comparison the Introduction promised, now realized in
  the interactive surface instead of stranded as a Benchmark table.

### Added
- The Beyond-SOTA anomaly output is now visible per image: `fisuralab.anomaly.bake_examples` fits a
  PatchCore memory bank on uncracked concrete only and scores the six shared example images, baking a
  per-image anomaly heatmap (warm = far from the healthy-concrete memory bank) plus a thresholded
  anomaly region and the honest anomaly score. The heat overlays are committed (derived, redistributable);
  the SDNET2018 fit imagery stays local. `HeatCanvas` blends the heat over the base image with the GT
  crack and the anomaly-region outline. The tab carries the honest framing (0.72 AUROC, a screen not a
  detector).

## [0.09.000], 2026-07-19

### Added
- Anomaly track: PatchCore (Roth et al., CVPR 2022) reimplemented faithfully in-repo (torchvision
  wide_resnet50_2 layer2+3 locally-aware patch features, greedy k-center coreset with a JL random
  projection on the GPU, kNN memory-bank scoring). Reimplemented rather than via anomalib because
  anomalib's dependency tree conflicts with torch 2.11; the real algorithm, not a substitute
  (same choice as HrSegNet).
- The concrete-transfer study (the dossier's flagged missing head-to-head): PatchCore fit on
  UNCRACKED SDNET2018 concrete only, then scoring cracked vs held-out uncracked. Measured image
  AUROC 0.720 (TPR/TNR 66/66 at the median threshold) over 300 uncracked fit + 100/100 test
  patches: modest-but-real transfer, far below the 0.996 the same method reaches on industrial
  MVTec AD. The committed study artifact carries metrics + the score histogram only; SDNET2018
  imagery stays local (CC BY 4.0).
- The Benchmark page renders the measured concrete-transfer result live (AUROC + TPR/TNR KPIs +
  the cracked-vs-uncracked score-distribution chart + the honest framing). Framework card 03
  (anomaly stack); torchvision pinned in the GPU lane.

## [0.08.000], 2026-07-19

### Added
- Foundation rung: DINOv2 (Apache-2.0) frozen-features + linear probe. A frozen ViT-S/14 backbone
  (loaded from the vault weights; architecture code cached from torch.hub) with a single 1x1-conv
  head on the last-layer patch tokens (the only trained weights: 385 parameters, BCE + Dice,
  AdamW, seeded). Trained on the CrackSeg9k subset; reaches CrackSeg9k val F1@2px 0.671 / F1@5px
  0.735 and examples mean F1@5px 0.620, competitive with the fully-trained SMP models and 17 points
  above the classical ladder while training essentially nothing. The masks are honestly coarse
  (1/14 patch resolution), which the 2 px number and the overlay make visible; the crack-specific
  probe number is a lab contribution the literature had not established.
- The DINOv2 probe joins `learned_on_examples` as a 5th architecture in the workbench; framework
  card 02 gains a DINOv2 section; per-architecture replay notes and the case engine label are now
  method-accurate (SMP tiled 512, HrSegNet reimplementation, DINOv2 518-resize probe).

## [0.07.000], 2026-07-19

### Added
- Learned ladder B: HrSegNet reimplemented in PyTorch in-repo from the published paper (two parallel
  paths, a high-resolution path kept at 1/4 resolution plus an auxiliary semantic-guidance path fused
  by elementwise sum; head = transposed conv + bilinear; loss = CE + 0.5 x two auxiliary CEs; the
  paper's SGD/poly/from-scratch recipe with brightness/flip/random-resize augmentation). The
  reference implementation is Apache-2.0 PaddlePaddle; this is a clean PyTorch re-expression, nothing
  vendored.
- HrSegNet-B16 trained on the 4070 (4,000-iteration budget, honestly recorded vs the paper's 100k):
  val F1@2px 0.547 / mIoU 0.654, examples mean F1@5px 0.540, joining the `learned_on_examples`
  workbench as a 4th architecture next to the SMP ladder A. It beats the classical ladder's 0.455 on
  the hard committed patches at a fraction of the parameters.
- Framework card 02 (the learned segmentation stack: SMP ladder A + the HrSegNet reimplementation +
  the CrackFormer-II reference-only handling).
- CrackFormer-II is cited as a published reference on the Benchmark page (no-license upstream,
  unverified Drive weights): never vendored, never quoted per-dataset without the paper's tables.

### Changed
- `deploy-pages.yml` builds the SPA from the committed artifacts (git-as-data) instead of
  regenerating in CI: the learned cases are baked locally on the GPU and cannot regenerate in a
  torch-free CI runner; drift + the classical smoke stay enforced in `ci.yml`.
- Added `frontend/public/CNAME` (fisura.fasl-work.com) for the Pages custom domain.
- `write_json` newline pinning + the GPU lane (`requirements-gpu.txt`, torch 2.11 cu128) carried from
  the learned track; all torch imports lazy, so CI stays pure-Python.

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
