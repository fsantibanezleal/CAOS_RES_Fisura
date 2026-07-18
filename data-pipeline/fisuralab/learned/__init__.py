"""The learned track (OFFLINE lane only; torch imported lazily, never by model/ or live).

Ladder A per the research: SMP U-Net (resnet18), DeepLabV3+ (resnet18) and SegFormer (mit_b2),
trained on vault shards with the 8 GB discipline (AMP, gradient accumulation, seeded, early-stop
on val F1@2px) and evaluated by the SAME dual-tolerance harness as the classical ladder. ONNX
exports go to the local model vault with hashes recorded in the manifests.
"""
