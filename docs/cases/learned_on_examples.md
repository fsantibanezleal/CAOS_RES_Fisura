# Case `learned_on_examples`: ladder A replayed on the committed examples

Category `learned-segmentation` · real imagery (the committed example set) · learned masks baked
from the local GPU run (`fisuralab.learned.run_ladder_a`)

## What it is

The first learned rung: three architectures from the research verdict (SMP U-Net resnet18,
DeepLabV3+ resnet18, SegFormer mit_b2; all MIT-licensed stacks) trained on CrackSeg9k shards from
the local vault and replayed on the SAME six committed patches the classical ladder runs on, so the
workbench shows classical-vs-learned on identical pixels. CrackSeg9k imagery itself never enters
the repo (its CC0 label conflicts with component licenses); only masks over OUR CC0/CC BY examples
plus metrics are committed.

## Training provenance (recorded in the manifest, reproducible by one command)

Recipe from the research: 512 reflect-padded crops, flip/rot90 augmentation, batch 6 with 2-step
gradient accumulation, AdamW 3e-4 cosine, AMP, BCE(pos_weight 8) + Dice, early stop on val F1 at
the STRICT 2 px protocol, seeded end to end. The manifest records n_train/n_val, epochs, minutes,
best val F1@2px per architecture, and the sha256 of each ONNX export (vault:
`FISURA_MODEL_ROOT`). The first baked run trains on a 3,000-image subset (recorded as such) to
keep the build-day loop honest; the full-9k overnight rerun is the same command without
`--limit-train` and simply re-bakes this case.

## How to reproduce

```bash
# GPU lane (local): torch cu128 + requirements-gpu, then
python -m fisuralab.learned.run_ladder_a            # trains, evaluates, exports ONNX
python -m fisuralab.pipeline learned_on_examples     # bakes the replay artifact (torch-free)
```

## Measured results (2026-07-18/19; 3,000-image CrackSeg9k subset, seed 42)

| Architecture | Track | CrackSeg9k val F1@2px | Examples mean F1@5px | Train minutes | Export |
|---|---|---|---|---|---|
| SegFormer mit_b2 | ladder A (SMP) | **0.766** | 0.653 | 24.5 | ONNX 99.4 MB, parity OK |
| U-Net resnet18 | ladder A (SMP) | 0.752 | 0.486 | 11.6 | ONNX 56 MB, parity OK |
| DeepLabV3+ resnet18 | ladder A (SMP) | 0.720 | **0.681** | 13.0 | ONNX 49.3 MB, parity OK |
| HrSegNet-B16 | ladder B (reimpl) | 0.547 (mIoU 0.654) | 0.540 | 46.6 | ONNX, parity OK |
| DINOv2 ViT-S/14 + linear probe | foundation (frozen) | 0.671 (F1@5px 0.735) | 0.620 | ~ | head-only (385 params) |

HrSegNet-B16 is the in-repo PyTorch reimplementation (ladder B), trained from scratch per the paper
recipe (no ImageNet pretraining) at an honestly-recorded 4,000-iteration budget: the paper reaches
78.43 mIoU at 100k iterations, so 0.654 mIoU here is the expected short-budget point, not a failure,
and the manifest records the exact iters/batch so nothing pretends to be the full run. It already
beats the classical ladder's 0.455 mean on the hard committed patches.

The DINOv2 linear probe (foundation rung) is the striking result: a FROZEN DINOv2 ViT-S/14 backbone
with a **385-parameter** 1x1-conv head (the only trained weights) reaches examples mean F1@5px 0.620,
competitive with the fully-trained SMP models and 17 points above the classical ladder, while
training essentially nothing. This is exactly the "cheapest credible foundation baseline" the
research predicted, and its crack-specific number is a lab contribution (the literature had not
established it). The honest caveat: the probe's masks are COARSE (patch features live at 1/14
resolution), so it scores well at the 5 px tolerance but the 2 px number and the visual overlay show
the thin-crack limitation of a pure linear probe on patch tokens.

The research's honest pattern, now measured on identical pixels: every learned architecture beats
the classical ladder's 0.455 mean (DeepLabV3+ by 22 points), the best-on-val model is not the
best-on-examples model (the plain transfer lesson), a tiny from-scratch specialist (HrSegNet-B16)
is competitive with the pretrained SMP models, and 385 frozen-feature parameters (DINOv2) rival
them all. The classical ladder still wins on transparency and CPU cost; all of it lives in the
same workbench.

## CrackFormer-II: cited reference, not vendored

CrackFormer-II (Liu et al., T-ITS 2023; ODS 0.869-0.914) is the accuracy-oriented transformer in
the research verdict. Its repository ships no license file and its per-dataset weights sit on
unverified Google Drive links, so Fisura cites it as a published reference on the Benchmark page
rather than vendoring its code or quoting an unverified per-dataset cell. If a license-clean
re-implementation is later warranted it becomes its own unit.

## Reading the workbench

The variant bar shows the four architectures instead of L0-L5; the Charts view scores them at
both tolerances against the same ground truth as the classical case.
