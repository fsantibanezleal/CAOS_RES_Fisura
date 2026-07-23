# Changelog

All notable changes to this product. Format: `X.XX.XXX` (display), see `fisuralab.__version__`. Keep `0.x`
while on mock/synthetic data. Tag every release.

## [0.23.000], 2026-07-22

### Fixed (fundus FOV: exclude the region, do not just recolour it)
The v0.22.000 FOV handling recoloured outside the disc and still scored the whole frame, so detectors
kept firing on the retina-to-fill transition and that firing was still counted (Felipe caught the rim
arc persisting in the learned overlays). A first attempt filled outside the ERODED disc, putting the
transition exactly on the FOV boundary, so masking the prediction to the FOV still kept it.

Now the disc edge and the FOV boundary are separated: the image is filled outside the FULL disc (so
the transition is at the disc edge), the FOV is the disc eroded 4 percent inward, and every predicted
mask is intersected with the FOV before it is scored, stored or drawn (ImageSample.fov +
metrics.restrict_to_fov, applied in infer, learned_replay, run_ladder_a, bake_sac). The transition band
falls outside the FOV and is genuinely excluded, verified in the overlays.

Final FOV-restricted means, F1@2px retina / concrete:

| rung | retina | concrete |
|---|---|---|
| classical RF fusion (L5) | 0.723 | 0.361 |
| classical ridge (L3) | 0.692 | 0.317 |
| SAM adapter (SAC) | 0.182 | 0.634 |
| DINOv2 frozen | 0.086 | 0.031 |
| SegFormer / DeepLabV3+ / U-Net | 0.07 / 0.01 / 0.00 | 0.47 / 0.50 / 0.33 |

Conclusion unchanged: hand-designed curvilinear filters transfer strongly (better on retinas than the
concrete they were borrowed for), the SAM foundation prior weakly, small concrete-fitted nets not at
all. Supersedes the v0.22.000 fundus figures.

## [0.22.000], 2026-07-22

### Fixed (fundus: mask to the field of view; the disc rim was a confound)
A FIVES fundus image is a bright retina disc on black, so the disc RIM is a strong closed curvilinear
edge that every crack detector fires on, the ridge filters especially. Scoring over the whole frame
confounded "found vessels" with "found the edge of the photograph". The fundus cases are now masked to
the field of view (Otsu on luminance, largest component, fill holes, erode 2 percent so the rim falls
outside), the outside flattened to the median retina colour and the ground truth zeroed there, as
retinal-vessel evaluation requires. The result is CLEANER, since the rim had been adding false
positives:

| rung | mean F1 retina | same rung concrete |
|---|---|---|
| classical RF fusion (L5) | 0.717 | 0.361 |
| classical ridge (L3) | 0.661 | 0.317 |
| SAM adapter (SAC, 0.45 percent tuned) | 0.332 | 0.634 |
| DINOv2 frozen | 0.086 | 0.030 |
| SegFormer / DeepLabV3+ / U-Net | 0.03 / 0.01 / 0.00 | 0.47 / 0.50 / 0.34 |

What transfers is the generality of the prior: hand-designed curvilinear filters best (better on
retinas than on the concrete they were borrowed for), a SAM foundation prior moderately, small
concrete-fitted networks not at all. Supersedes the v0.21.000 fundus figures, which were unmasked and
partly single-image.

### Added (SAC wired into the App)
- The foundation-adapter rung, baked in v0.21.000, now renders on the SOTA tab: the per-image SAC
  overlay plus val F1 0.750, 0.47 percent of the network trained, vs the published 0.612 on
  OmniCrack30k.

### Fixed
- The SAC train/bake device guard selected cuda when `is_available()` is True but `device_count()` is
  0 (under `CUDA_VISIBLE_DEVICES=""`); it now checks the count, matching the learned lane.

## [0.21.000], 2026-07-22

### Added (fundus images as first-class cases: the lab's own thesis, tested)
- Four FIVES retinal fundus images (normal, diabetic retinopathy, glaucoma, AMD) run through the
  ENTIRE ladder unchanged. If a crack is just a thin, branching, low-contrast curvilinear structure,
  the ladder should behave on a retina the way it behaves on concrete. Measuring that beats asserting
  it, and the answer is more interesting than the assertion was.
- CC BY 4.0, verified on the primary source (Jin K. et al., Sci Data 9:475, 2022), which is why these
  may ship while the non-commercial crack datasets stay local. CONTRACT 1 gains `retina`.
- Downscaled 2048 -> 768 so vessels sit at 3-8 px, the regime the ladder and its 2/5 px tolerances
  are tuned for.

### THE RESULT: the hand-designed filter transfers, the learned appearance model does not
Mean F1 at 2 px over the four retinas (CORRECTED 2026-07-22: an earlier draft of this entry quoted
single-image figures from fives-100_d, which overstated the effect; these are the means):

| rung | mean F1 on retina | same rung on concrete |
|---|---|---|
| classical ridge (L3, sato/frangi/meijering) | 0.645 | 0.317 |
| classical RF fusion (L5) | 0.684 | 0.369 |
| DINOv2 frozen + linear | 0.095 | 0.030 |
| SegFormer-B2 | 0.058 | 0.472 |
| DeepLabV3+ | 0.010 | 0.503 |
| U-Net R18 | 0.001 | 0.338 |

The vessel filters, published for vessels, score HIGHER on retinas than on the cracks they were
borrowed for. Every network trained on concrete collapses below 0.10 on the retina, from 0.34-0.50 on
concrete: they learned the look of concrete, not curvilinear structure. DINOv2 is the mirror image,
worst of the learned tier on concrete and best on the retina, because it was fitted to neither. The
clean monotonic story is NOT true per image (SegFormer beats DINOv2 on fives-151_n); the honest result
is the split between hand-designed and learned, on the means.

### Added (SAC foundation-adapter rung)
- SAM ViT-B, normalization parameters only (36.9K) + a 387.8K decode head = 424.6K trainable, 0.45
  percent of the network. Best val F1@2px **0.7500**, matching a fully trained U-Net (0.7523) and
  beating DeepLabV3+ (0.7199). 7786 train / 1373 val, 508 min.

### Fixed (dacl10k: the rare-class collapse is gone)
- pooled mIoU **0.1441 -> 0.2975** (untuned, thr 0.50), classes at exactly IoU 0 **9 of 19 -> 0 of 19**,
  gap to the 0.424 WACV baseline 2.94x -> 1.43x. No change to the model or the training budget: only
  per-class positive weighting, and bf16 so that weighting could actually run.
- The fp16 forward overflowed to inf once pos-weighting raised the logits, so every step failed the
  non-finite check and hit a `continue` placed BEFORE the progress print and the loss accumulator. A
  fully diverged run therefore looked quiet and cheap while occupying the GPU for hours. Skips are now
  printed, the epoch line reports steps taken/total, and an epoch over 25 percent skipped ABORTS.
- Training state is written every epoch with `--resume`.

### Changed (every source gets every tab)
- Case source is a DATA axis with three real sources: Real samples (6), Synthetic (12), Fundus (4).
  All 10 tabs on every one. Degradation is an 11th for synthetic only, because it needs a controlled
  width sweep against exact ground truth, which is a different thing from withholding a tab that applies.
- Synthetic and fundus are materialized into `data/examples`, so every existing bake covers them with
  no special-casing.

### Fixed (memory, and a silent metric)
- `rasterize(window=...)` draws polygons straight into the 512 crop: 183 MB -> 5 MB per sample.
  `_dacl_item` stays uint8 until after the crop: 146 MB -> 3 MB. Both verified bit-identical.
- `predict_full` selects cuda only when `device_count() > 0`; `is_available()` alone returns True under
  `CUDA_VISIBLE_DEVICES=""` and killed CPU-side runs.
- DINOv2 is wired into the prediction path (518 tile, head-only checkpoint), so the rung now covers all
  22 cases instead of the original 6.
- The workbench sidebar drops out of sticky when the layout stacks below 940 px; it was overlapping the
  method image at narrow viewports.

## [0.20.000], 2026-07-21

### Changed (the case source is a DATA axis, not a method axis)
- The App's case-source selector offered `Prebaked` / `Pretrained` / `Upload`. The first two loaded the
  SAME artifact and rendered the SAME tabs, differing only by one sentence of hint copy: dead UI. The
  axis was also wrong, since prebaked-vs-pretrained is a method distinction and that is exactly what the
  tabs already do (Classical / SOTA / Beyond SOTA), so wiring it would only have duplicated them.
- The source now selects the DATA the workbench runs on: **Real samples / Synthetic / Upload**.
- The `synthetic_battery` artifact, baked back in v0.03 and left unreachable behind a literal
  `void SYNTH_SLUG`, is now wired. Overlay paths are derived from the shared naming convention rather
  than re-baking, and only the samples whose overlays exist are offered as thumbnails.
- Synthetic exposes only the tabs backed by real data (Overview, Classical, Degradation). The learned,
  anomaly, SLIC and enrichment artifacts were never baked on synthetic cases, so those tabs are withheld
  rather than rendered empty.

### Added (Degradation tab)
- Classical-ladder F1@2px against crack width, 9 px down to 2 px at fixed contrast and angle, over all 12
  battery cases. The ground truth is exact by construction, so a drop measures the METHOD and not the
  labelling.
- The panel states the two things a reader would otherwise read as defects: the ordering INVERTS (Otsu
  and Sauvola pinned at F1 1.0 while minimal-path bridging and RF fusion score below, because a generated
  crack is precisely the case a global threshold solves perfectly), and fewer curves are visible than
  methods because L3 and L4 land on identical values at every width when there is no gap to bridge.

### Fixed (dacl10k validation metric was not benchmark-comparable)
- `_macro_iou` averaged a per-BATCH IoU across batches. The dacl10k benchmark, like every standard
  segmentation protocol, pools intersection and union over the whole split and divides once. The shipped
  "0.117 vs the 0.424 baseline" comparison was therefore never like-for-like, and batch-averaging is
  systematically lower.
- Replaced by `_iou_parts` (per-class inter/union/gt counts, float64, pooled globally; a class counts as
  present if it OCCURS in the val ground truth, not merely if something fired).
- New `fisuralab.multiclass.eval_dacl10k` recomputes the pooled number from a saved checkpoint in a
  single validation pass, so an in-flight run needs no restart, and sweeps the decision threshold.
  Measured on the full-data best checkpoint: **0.0912** batch-averaged, **0.1441** pooled untuned,
  **0.1685** pooled at a val-tuned threshold. The tuned-on-val caveat ships next to the untuned column.

### Fixed (rare classes collapsing to exactly IoU 0)
- The pooled per-class breakdown shows 19 classes present but only 10 non-zero, and those 10 average
  **0.3201** IoU. The 9 zeros are precisely the rarest classes (ExposedRebars at 0.48 percent of labelled
  pixels, Rockpocket 0.81, Crack 1.37). Crack scoring 0 is untenable for this lab.
- Diagnosis: unweighted BCE collapsing rare classes to an all-negative prediction, a calibration failure
  rather than a capacity or budget limit.
- `class_pos_weight()` adds per-class positive weighting as `sqrt(neg/pos)`, estimated on a cached sample
  of the TRAIN split (never val, which would leak). Raw ratios span 13x to 1060x and would buy recall by
  destroying precision; a flat safe cap instead collapses 14 of 19 classes onto one value. sqrt keeps the
  ordering at a 3.6x-to-32.6x spread. Opt-in via `--pos-weight`; setting and protocol recorded in results.

### Added (SAC foundation-adapter scaffold)
- `fisuralab.learned.train_sac` + `bake_sac`: SAM ViT-B with ONLY its normalization parameters tuned
  (36.9K of 93.7M) plus a 387.8K decode head, under 0.5 percent of the network, after Rostami, Chen,
  Hosseini, arXiv:2504.14138. Gradient-checkpointed blocks and batch-1 accumulation fit the 8 GB card;
  fp32 losses, gradient clipping and non-finite step skipping carry the dacl10k NaN lesson forward.

### Notes
- A uPlot default-time-scale bug was caught in browser review: crack widths rendered as clock times
  (":02.000", "12/31/69 9:00pm") on an axis the build reported as fine.

## [0.19.000], 2026-07-21

### Added (enrichment #9: Grad-CAM, the evidence behind the mask)
- The SOTA tab gains a three-way view switch (mask / DINOv2 features / Grad-CAM). A mask says WHERE a
  model fired; Grad-CAM (Selvaraju et al., ICCV 2017) says what evidence drove it: gradients of the
  summed crack logit weight the encoder feature channels, and the ReLU'd weighted sum is the map.
- `fisuralab.learned.bake_gradcam` bakes CAMs for the two ResNet-18-encoder segmenters (U-Net,
  DeepLabV3+) and records `cam_mass_on_crack`, the fraction of CAM mass landing on the true crack, so
  the view carries a number and not just a picture (U-Net 0.256, DeepLabV3+ 0.239 on the clearest
  concrete crack; 0.0 on the uncracked noise control, which is the right answer).
- Honest limits, recorded per sample rather than hidden: the CAM site is `encoder.layer3`, because
  `layer4` is a 12x12 grid at this input size and upsamples to a meaningless blob; Grad-CAM was built
  for whole-image classification, so on a 1-5 px crack these maps localise evidence to a region, not to
  an outline; samples with no pixel ground truth carry an "undefined" note, and the one degenerate
  all-zero CAM is labelled as such. SegFormer, HrSegNet and the DINOv2 probe are excluded with a stated
  reason each (no single conv site / custom architecture / frozen features) rather than faked.

### Changed (CODEBRIM unblocked)
- The published CODEBRIM archives were NOT a truncated download. They carry a **Zip64 local-header
  offset defect**: every offset in the central directory points past the real 4 GB boundary, so Python's
  `zipfile` fails every image read with "Bad magic number" while 7-Zip verifies the same archive as
  "Everything is Ok". All four archives failing identically is what ruled out four bad downloads.
  The data layer now reads the 7-Zip-extracted tree from disk (`codebrim_root()`, records carry
  `image_path`). Verified: 1,022 annotated images, 5,181 boxes across the 5 defect classes,
  deterministic 70/20/10 split. Detector training is queued behind the dacl10k run (one 8 GB GPU).

## [0.18.000], 2026-07-20

### Added (enrichment #8: the DINOv2 dense-feature PCA-to-RGB view)
- A "DINOv2 features" toggle in the SOTA tab renders what the FROZEN foundation model encodes, with no
  crack supervision at all: each 14 px patch becomes a 384-dim descriptor and its first three principal
  components map to RGB. The crack separates as its own hue against the concrete texture, which is the
  visual argument for why a 385-parameter linear head on these features is already competitive with
  fully trained segmenters.
- `fisuralab.learned.bake_dinov2_pca` extracts the patch descriptors and bakes the overlays.
  PCA is fit PER IMAGE: a basis shared across the example set encodes which image a patch came from
  (steel vs concrete vs the noise control dominate the variance) and washed the within-image structure
  out entirely; per-image fitting roughly doubles the explained variance (PC1 0.11 to 0.20) and is what
  makes the crack visible. Component signs are pinned so the colours are reproducible run to run.

### Fixed (dacl10k full-data training stability)
- The first full-data run diverged: `train_loss` went NaN at epoch 6 and every later epoch scored 0.000
  mIoU. Cause: AMP fp16 reductions over the 19x512x512 Dice term underflow. The losses are now computed
  in fp32 outside `autocast`, gradients are clipped at norm 1.0, non-finite steps are skipped rather than
  poisoning the weights, and the best-checkpoint guard rejects non-finite/zero scores. Verified stable on
  a smoke run (zero skipped steps, mIoU climbing); the full 6,935-image run was relaunched.

## [0.17.000], 2026-07-20

### Added (BL-013 live lane: bring-your-own-image, in-browser inference)
- The App's Upload mode is now real. Drop a crack photo and the compact HrSegNet segmenter runs
  ENTIRELY in the browser via onnxruntime-web (wasm); the image never leaves the device. New
  `lib/liveEngine.ts` (session + preprocess + segment) and `pages/workbench/LiveLane.tsx` (dropzone,
  crack-probability overlay, inference-time + backend + crack-fraction read-outs, an optional
  millimetres-per-pixel scale, honest framing).
- The browser-shippable model is HrSegNet (0.2 MB, committed under `data/derived/models/`); the
  47-95 MB SOTA models stay in the offline lane. onnxruntime-web configured for the static Pages host:
  single-thread wasm (no COOP/COEP on Pages), wasm + loader-mjs paths pinned to the app base (so
  nothing 404s under the subpath), WebGPU deliberately off (the tiny model runs in ~55 ms on wasm and
  the jsep glue does not resolve on the static host).
- Correctness catch: HrSegNet was trained on img/255 with NO ImageNet mean/std normalization; applying
  mean/std over-segmented to 52 percent of the image. Fixed the browser preprocessing to match the
  training convention (verified against the ONNX in Python), which drops the prediction to a clean
  0.2 percent conservative crack mask, matching the baked model's precision.

## [0.16.000], 2026-07-20

### Added (BL-012 DIC: 2D digital image correlation, deformation + crack-opening)
- New `fisuralab.dic` module (dossier 04 section 5), CPU-only (numpy + scipy):
  - `correlation.py`: the real subset-based 2D DIC. Each reference subset is located in the deformed
    image by maximizing the zero-normalized cross-correlation (ZNCC, invariant to affine intensity
    changes), integer search then sub-pixel quadratic-peak refinement; strain from LOCAL polynomial
    fits of the displacement field, never raw pointwise differentiation. Implemented in-repo (muDIC is
    global-FE, py2DIC is GPL) so the real algorithm stays MIT-clean.
  - `bake_dic.py`: a virtual experiment. A synthetic speckle image is deformed by a KNOWN field (1 percent
    uniform stretch + a crack-opening jump of 2.5 px), so the engine validates against exact ground truth:
    it recovers the crack opening as 2.52 px, the strain as 1.00 percent, mean displacement error 0.026 px.
    The crack reads as the displacement discontinuity. The same field on natural concrete-like texture is
    3.9x less accurate, matching the DIC literature (natural texture costs ~3x painted speckle).
- The temporal/deformation track now lives under one **Monitoring** nav item with two tabs (Growth
  monitoring + Deformation DIC), keeping the header compact (ADR-0016 section 6). The DIC tab shows the
  ZNCC equation, the u-displacement + strain fields (crack as a discontinuity), and the honest
  speckle-vs-texture + planar-specimen scope.
- `tests/test_dic.py`: the engine recovers a known translation and a known strain, plus committed-artifact
  coherence (CI-safe).

## [0.15.000], 2026-07-20

### Added (BL-011 monitoring: two-epoch registration + differential crack mapping)
- Detection becomes prognosis. New `fisuralab.monitoring` module (dossier 04 section 4), CPU-only
  (scikit-image + numpy):
  - `registration.py`: ORB features + RANSAC homography to register two surveys into a common frame,
    then differential crack mapping (skeleton + inscribed-circle width per epoch) reporting per-branch
    width deltas + new-branch pixels + tip extension, never raw pixel differences (which lighting
    dominates).
  - `bake_growth.py`: a synthetic two-epoch case with a crack that GROWS between epoch 1 and epoch 2
    under a camera-pose change, with EXACT ground truth by construction (the only honest way to validate
    a monitoring pipeline). The pipeline recovers the pose from features and measures the growth: median
    width +0.095 mm vs a ground-truth +0.095 mm (absolute error 0.000 mm), tip extension 63 px vs 64 px,
    993 ORB inliers.
- New **Monitoring** page: the change map (green = crack in both surveys, red = new growth in epoch 2),
  a view switcher (change / epoch 1 / epoch 2 raw pose / epoch 2 registered), the measured-vs-ground-truth
  callout, the epoch-comparison table, and the honest scope note (Fisura measures and publishes the
  crack-length history a(N); it does not certify remaining life, Paris-law being out of an optical
  concrete scope).
- `tests/test_monitoring.py`: growth detection, no-change, and committed-artifact coherence (CI-safe).

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
