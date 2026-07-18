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

## Reading the workbench

The variant bar shows the three architectures instead of L0-L5; the Charts view scores them at
both tolerances against the same ground truth as the classical case. Expected honest pattern per
the research: the learned models generalize far better on the hard patches (the wide diffuse crack,
the scratched steel) than the classical ladder, at the price of GPU training and opaque failure
modes; the numbers on this page are the evidence either way.
