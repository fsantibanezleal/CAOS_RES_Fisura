# Framework card: the anomaly-detection stack

The anomaly track's engine and the study it contributes. The research (dossier 04 section 2) named
anomalib (Intel, Apache-2.0) as the reference framework and PatchCore (Roth et al., CVPR 2022) as
the memory-bank method to run, and flagged a real gap: NO published head-to-head of industrial
anomaly detection on civil (concrete) surfaces exists. That gap is what this track measures.

## Engine: PatchCore, reimplemented in-repo

`fisuralab.anomaly.patchcore` is a faithful in-repo re-expression of the PatchCore algorithm, NOT a
toy substitute:

- Backbone: torchvision `wide_resnet50_2` (ImageNet, BSD-licensed weights), frozen; locally-aware
  patch features from `layer2` + `layer3`, each 3x3 average-pooled, `layer3` bilinearly upsampled to
  `layer2`'s grid, channel-concatenated (PatchCore's mid-level feature design).
- Memory bank: patch features from the GOOD (defect-free) images, reduced by a greedy k-center
  coreset. The coreset uses a Johnson-Lindenstrauss random projection to 128 dims and runs the
  iterative farthest-point selection on the GPU (PatchCore's own speed trick; a naive CPU k-center
  over 300k high-dim patches is infeasible, which the first run confirmed).
- Scoring: each test patch's anomaly score is its nearest-neighbour distance to the memory bank; the
  image score is the max patch distance; the per-patch distances upsample to an anomaly heatmap.

Why reimplement rather than `pip install anomalib`: anomalib pins an older torch and pulls a heavy
Lightning/OpenVINO tree that conflicts with the learned track's torch 2.11. Reimplementing the real
algorithm (the same choice made for HrSegNet) keeps the repo MIT-clean, self-contained, and
dependency-safe. Fit is training-free (build the bank); scoring is nearest-neighbour.

## The contribution: the concrete-transfer study

`fisuralab.anomaly.run_concrete_transfer` fits the memory bank on UNCRACKED SDNET2018 concrete
patches (walls, decks, pavements) and scores a held-out balanced split of cracked vs uncracked,
reporting image-level AUROC (rank-based, no sklearn) plus TPR/TNR at the median-score threshold and
the cracked-vs-uncracked score distributions. This is the head-to-head the literature lacks: does an
industrial memory-bank anomaly detector, fit on good concrete only, separate cracked from uncracked
without ever seeing a crack? SDNET2018 is CC BY 4.0, but only the METRICS + histogram (and overlays
from redistributable committed imagery) ship; the raw fit imagery stays local.

The honest framing (dossier 04): industrial AD saturated MVTec AD (PatchCore 99.6 AUROC) but MVTec
AD 2 caps SOTA below 60 percent AU-PRO, and normal-variance assumptions are strained on uncontrolled
civil imagery. The measured concrete-transfer AUROC is reported as the evidence, not assumed.

## Run

```bash
python -m fisuralab.anomaly.run_concrete_transfer --n-fit 300 --n-test 200
```

Writes `data/derived/anomaly/concrete_transfer.json` (metrics + score histogram); the Benchmark page
renders the AUROC and the score distributions. Future rung (this track): PaDiM / FastFlow /
EfficientAD baselines and the VisA/KolektorSDD2 industrial reference (metrics only).
