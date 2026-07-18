"""Stage 3, train: the L5 patch classifier (LBP + GLCM + HOG, random forest).

Trained deterministically on the MASKED samples of the case at hand (the committed examples in CI;
the fetched vault datasets in full local runs, same function). The trained model is used in-process
by infer; vault-scale runs may persist it locally (models/ is git-ignored for binaries by the base
guards, and the manifest records the training provenance instead).
"""
from __future__ import annotations

from ..io.image_contract import ImageSample
from ..model.classical import train_patch_rf


def run(samples: list[ImageSample], seed: int):
    """Returns (rf, provenance dict). rf is None when no masked samples exist (L5 then unavailable)."""
    labelled = [s for s in samples if s.mask is not None]
    if not labelled:
        return None, {"trained": False, "reason": "no masked samples in this case"}
    rf = train_patch_rf([s.image for s in labelled], [s.mask for s in labelled], seed=seed)
    if rf is None:
        return None, {"trained": False, "reason": "single-class patch labels; L5 unavailable"}
    return rf, {
        "trained": True,
        "n_samples": len(labelled),
        "features": "LBP(8,1)+LBP(16,2)+GLCM(contrast,homogeneity,energy,correlation)+HOG",
        "model": "RandomForestClassifier(n_estimators=120, min_samples_leaf=2)",
        "seed": seed,
    }
