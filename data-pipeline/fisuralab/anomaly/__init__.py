"""The anomaly track: a faithful in-repo PatchCore reimplementation + the concrete-transfer study.

PatchCore is reimplemented from Roth et al. 2022 (CVPR) with torchvision features rather than via
anomalib (whose dependency tree conflicts with torch 2.11); the real algorithm, not a substitute.
Torch/torchvision are the local GPU lane, imported lazily."""
