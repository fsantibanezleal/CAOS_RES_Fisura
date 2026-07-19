# Framework card: the learned segmentation stack

The learned track's engines, pinned in `data-pipeline/requirements-gpu.txt` (the LOCAL GPU lane,
never installed by CI) and used for real by `fisuralab.learned`. Torch is imported lazily
everywhere, so the pipeline bakes the learned cases torch-free from the persisted run record and CI
stays a pure-Python job.

## What and why

| Engine | Pin | Role | License |
|---|---|---|---|
| PyTorch | 2.11.0 + cu128 | training + inference on the RTX 4070 (8 GB) | BSD-3 |
| torchvision | 0.26.0 + cu128 | ops (nms etc.) the SMP encoders need | BSD-3 |
| segmentation-models-pytorch | 0.5.0 | U-Net / DeepLabV3+ / SegFormer with timm encoders | MIT |
| onnx + onnxruntime | 1.20 / 1.23 | export graph + CPU-parity verification | MIT / MIT |
| onnxscript | 0.7.1 | the torch 2.11 dynamo exporter dependency (legacy path used) | Apache-2.0 |

Every model in the ladder is a permissive stack (MIT/BSD). The one crack-specialized architecture
in the track, HrSegNet, is REIMPLEMENTED in-repo from the published description (below), so nothing
license-encumbered is vendored.

## Ladder A: strong generic segmenters (SMP)

- U-Net (resnet18), DeepLabV3+ (resnet18), SegFormer (mit_b2), ImageNet-pretrained encoders.
- Recipe (dossier 02 section 9, the 8 GB discipline): 512 reflect-padded crops, flip/rot90,
  batch 6 with 2-step gradient accumulation (effective 12), AdamW 3e-4 cosine, AMP, BCE(pos_weight
  8) + Dice, early stop on val F1 at the strict 2 px protocol, seeded end to end.
- Run: `python -m fisuralab.learned.run_ladder_a` (resumable; the first bake trains on a 3,000-image
  CrackSeg9k subset, the full-data rerun drops `--limit-train`).

## Ladder B: HrSegNet, reimplemented in PyTorch

Li, Ma, Liu, Cheng, "Real-time high-resolution neural network with semantic guidance for crack
segmentation", Automation in Construction 156:105112 (2023), arXiv:2307.00270. The reference
implementation (CHDyshli/HrSegNet4CrackSegmentation) is Apache-2.0 but PaddlePaddle;
`fisuralab.learned.hrsegnet` is an in-repo PyTorch re-expression of the published architecture:

- A high-resolution path kept at 1/4 input resolution throughout (three HrSeg blocks, each three
  Conv-BN-ReLU at constant base width B in {16, 32, 48}); the design principle is to never drop to
  low resolution and recover, so thin-crack detail survives.
- An auxiliary semantic-guidance path (progressively downsampled context) fused into the
  high-resolution path at each stage by elementwise summation + ReLU (guidance, not concatenation).
- Head: 3x3 transposed convolution (stride 2) then bilinear to full; two output channels.
- Loss (training): primary cross-entropy + 0.5 x two auxiliary cross-entropies on intermediate
  heads. Recipe: SGD momentum 0.9, weight decay 5e-4, poly LR (initial 0.01, power 0.9), warm-up
  2,000 iters, from scratch, 400x400 crops, brightness jitter + flip + random resize 0.5x-2.0x.
- Published reference (refined CrackSeg9k, 2-class mIoU): B16 78.43 / B32 79.70 / B48 80.32; the
  lab reports BOTH that 2-class mIoU and its own dual-tolerance F1, and the manifest records the
  actual iteration budget so a shorter-than-paper first bake never pretends to be the full run.
- Run: `python -m fisuralab.learned.train_hrsegnet --base 16 --iters 12000` (joins the same
  results record and workbench as ladder A via a 2-class-to-logit wrapper).

## CrackFormer-II: reference only (no-license upstream)

CrackFormer-II (Liu et al., IEEE T-ITS 24(9):9240-9252, 2023) is the accuracy-oriented transformer
in the research verdict, reporting ODS 0.869-0.914 on its four benchmarks. Its repository ships NO
license file and its per-dataset weights sit on Google Drive links the research pass could not
individually verify. Fisura therefore treats it as a CITED PUBLISHED REFERENCE on the Benchmark
page (a comparison bar), not a runnable rung: vendoring its code into this MIT repo is forbidden,
and quoting a per-dataset number requires the paper's PDF tables (the abstract does not state the
dataset-to-score mapping, flagged UNVERIFIED). If a license-clean re-implementation is later
warranted, it becomes its own unit.

## Foundation probe: DINOv2 frozen features + linear head

DINOv2 (Meta, Apache-2.0; Oquab et al. 2023) is the cheapest credible foundation baseline for dense
crack prediction, and the research flagged that its crack-specific numbers are NOT established, so a
clean linear probe is a result the lab can contribute (`fisuralab.learned.dinov2_probe`):

- The ViT-S/14 backbone is loaded from the vault weights (`E:\\_Models\\fisura\\dinov2\\
  dinov2_vits14_pretrain.pth`; the architecture code is fetched once from torch.hub,
  `facebookresearch/dinov2`, cached under `TORCH_HOME`) and FROZEN: `requires_grad_(False)` on all
  22M backbone parameters.
- A single 1x1-convolution head on the last-layer patch tokens is the ONLY thing trained (a genuine
  linear probe): last-layer `x_norm_patchtokens` at a 518-input give a 37x37x384 feature map, the
  head maps 384->1, bilinearly upsampled to full resolution. Trained with BCE(pos_weight 8) + Dice,
  AdamW 1e-3, seeded; ImageNet normalization on the input as DINOv2 expects.
- The probe's masks are honestly COARSE for thin cracks (patch tokens live at 1/14 resolution); that
  coarseness is the finding, reported next to the trained-model ladder on identical pixels, not
  hidden. No xFormers is required (the backbone warns and runs without it).
- The checkpoint saves ONLY the head (the backbone is the vault weights); the record carries the
  head sha256 and the backbone provenance. Full-backbone ONNX export is heavy and belongs to the
  live-lane unit, so the probe records its provenance instead of an export at this stage.
- Run: `python -m fisuralab.learned.dinov2_probe --iters 3000` (joins the same learned workbench).

## ONNX

Every trained SMP/HrSegNet model exports to ONNX via the legacy exporter (`dynamo=False`, reliable
for SMP and the HrSegNet graph on torch 2.11) with CPU-parity verification and a sha256 recorded in
the case manifest; exports live in the local model vault (`FISURA_MODEL_ROOT`). The live-lane unit
selects the one compact model the browser ships. The DINOv2 probe is the exception noted above
(head-only checkpoint; backbone export deferred).
