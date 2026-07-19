# Frameworks

One card per research-chosen engine/library, **the deep research, made binding**. Every engine the pipeline uses
gets a card here AND an exact pin in the matching `requirements-*.txt`. No hand-rolled toy substitute for a SOTA
engine the research prescribed.

- [00, card TEMPLATE](frameworks/00_TEMPLATE.md), copy per engine to `frameworks/<NN>_<tool>/<tool>.md`
- [01, the classical imaging stack](frameworks/01_classical-imaging-stack.md), scikit-image / SciPy / scikit-learn, the pinned classical ladder + the version-sensitivity discipline.
- [02, the learned segmentation stack](frameworks/02_learned-segmentation-stack.md), PyTorch + segmentation-models-pytorch (ladder A) + the in-repo HrSegNet reimplementation (ladder B) + the DINOv2 frozen-features linear probe, ONNX export; CrackFormer-II handled as a cited reference.
- [03, the anomaly-detection stack](frameworks/03_anomaly-detection-stack.md), a faithful in-repo PatchCore reimplementation (torchvision features + greedy coreset + kNN) + the concrete-transfer study (the head-to-head the literature lacks).

*(Further engine cards land with their units: anomalib, SAM/DINOv2 adapters, muDIC.)*
