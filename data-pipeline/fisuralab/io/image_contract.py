"""CONTRACT 1 (image domain): the bring-your-own-data gate for Fisura.

Every image sample entering the pipeline (offline stages, live lane, examples) passes through
``validate_sample``. Hard violations REJECT the sample with explicit reasons; plausible-but-suspicious
properties are FLAGGED, never silently coerced (ADR-0057 contract discipline).

This module is numpy-only ON PURPOSE: the same validation core must run in the offline lane and in
the browser live lane (Pyodide), so file IO lives separately in ``image_formats``.

Schema (documented for humans in data/README.md):
- image: numpy array, H x W (grayscale) or H x W x 3 (RGB); dtype uint8, or float32/float64 in [0, 1].
- size bounds: MIN_SIDE <= H, W <= MAX_SIDE.
- mask (optional): H x W, bool, or integer {0, 1} / {0, 255}; same spatial shape as the image.
- mm_per_px (optional): physical scale, in MM_PER_PX_RANGE (1 micron to 5 cm per pixel); required
  for any physical-width output, never invented.
- material: one of MATERIALS (controlled vocabulary; drives the case taxonomy).
- source: free-form dataset/provenance identifier (non-empty).
- license_tag: one of LICENSE_TAGS (drives what may be redistributed; see data/README.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_SIDE = 32
MAX_SIDE = 8192
MM_PER_PX_RANGE = (0.001, 50.0)

MATERIALS = (
    "concrete",
    "asphalt",
    "masonry",
    "stone",
    "steel",
    "ceramic",
    "synthetic",
    # Not a built material, and deliberately so. The lab's thesis is that a crack is a thin,
    # branching, low-contrast curvilinear structure, and that most of the tooling was invented for
    # some other instance of that shape: the ridge rung calls sato/frangi/meijering, published for
    # vessels and for neurites. Retinal fundus images are therefore carried as first-class cases, so
    # the claim is tested by running the ladder on them rather than asserted in prose.
    "retina",
    "other",
)

LICENSE_TAGS = (
    "cc0",          # public-domain dedication: samples and derivatives may ship in the repo
    "cc-by",        # attribution required: samples and derivatives may ship with credit
    "cc-by-sa",     # share-alike: derivatives ship ONLY under the same license, in a marked area
    "cc-by-nc",     # non-commercial: LOCAL ONLY, never committed
    "cc-by-nc-sa",  # non-commercial share-alike: LOCAL ONLY, never committed
    "academic",     # academic/cite-only terms: LOCAL ONLY, never committed
    "competition",  # competition rules (e.g. Kaggle): LOCAL ONLY, never committed
    "unknown",      # no explicit license found: treated as LOCAL ONLY
)

# License tags whose imagery may be committed to this public MIT repository.
REDISTRIBUTABLE_TAGS = ("cc0", "cc-by", "cc-by-sa")

# Soft-flag thresholds (suspicious, not fatal).
FLAG_MASK_COVERAGE = 0.5      # cracks are thin; a mask covering >50 percent is suspicious
FLAG_MASK_MIN_PIXELS = 10     # a positive mask with fewer than 10 px is suspicious
FLAG_CONSTANT_STD = 1e-6      # a (near-)constant image carries no signal


@dataclass
class ImageSample:
    """One validated unit of work: image + optional mask + metadata."""

    image: np.ndarray
    mask: np.ndarray | None = None
    mm_per_px: float | None = None
    material: str = "other"
    source: str = ""
    license_tag: str = "unknown"
    sample_id: str = ""
    flags: list[str] = field(default_factory=list)
    # Region of interest: where analysis is valid. None means the whole image (the usual case). For a
    # fundus photograph it is the retina disc, eroded off the rim, so a method's response on the
    # circular retina-to-surround edge is EXCLUDED rather than scored as a crack. A prediction is
    # intersected with this before it is measured, stored or drawn.
    fov: np.ndarray | None = None


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _validate_image(img: np.ndarray, errors: list[str], flags: list[str]) -> None:
    if not isinstance(img, np.ndarray):
        errors.append("image: not a numpy array")
        return
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w = img.shape[:2]
    else:
        errors.append(f"image: shape {img.shape} is neither HxW nor HxWx3")
        return
    if h < MIN_SIDE or w < MIN_SIDE:
        errors.append(f"image: {h}x{w} below minimum side {MIN_SIDE}")
    if h > MAX_SIDE or w > MAX_SIDE:
        errors.append(f"image: {h}x{w} above maximum side {MAX_SIDE}")
    if img.dtype == np.uint8:
        pass
    elif img.dtype in (np.float32, np.float64):
        finite = np.isfinite(img)
        if not finite.all():
            errors.append("image: non-finite values in float image")
        else:
            mn, mx = float(img.min()), float(img.max())
            if mn < 0.0 or mx > 1.0:
                errors.append(f"image: float range [{mn:.3g}, {mx:.3g}] outside [0, 1]")
    else:
        errors.append(f"image: dtype {img.dtype} not in (uint8, float32, float64)")
    if img.size and float(np.std(np.asarray(img, dtype=np.float64))) < FLAG_CONSTANT_STD:
        flags.append("image: near-constant (no texture signal)")


def _validate_mask(mask: np.ndarray, image: np.ndarray, errors: list[str], flags: list[str]) -> None:
    if not isinstance(mask, np.ndarray):
        errors.append("mask: not a numpy array")
        return
    if mask.ndim != 2:
        errors.append(f"mask: shape {mask.shape} is not HxW")
        return
    if isinstance(image, np.ndarray) and image.ndim >= 2 and mask.shape != image.shape[:2]:
        errors.append(f"mask: shape {mask.shape} does not match image {image.shape[:2]}")
    if mask.dtype == np.bool_:
        binary = mask
    elif np.issubdtype(mask.dtype, np.integer):
        values = np.unique(mask)
        if not (set(values.tolist()) <= {0, 1} or set(values.tolist()) <= {0, 255}):
            errors.append("mask: integer values are not binary ({0,1} or {0,255})")
            return
        binary = mask > 0
    else:
        errors.append(f"mask: dtype {mask.dtype} is neither bool nor integer")
        return
    positive = int(binary.sum())
    if positive == 0:
        flags.append("mask: empty (no positive pixels); valid for uncracked samples")
    elif positive < FLAG_MASK_MIN_PIXELS:
        flags.append(f"mask: only {positive} positive px (below {FLAG_MASK_MIN_PIXELS})")
    if binary.size and positive / binary.size > FLAG_MASK_COVERAGE:
        flags.append(
            f"mask: coverage {positive / binary.size:.0%} above {FLAG_MASK_COVERAGE:.0%} (cracks are thin; check polarity)"
        )


def validate_sample(sample: ImageSample) -> ValidationResult:
    """Apply CONTRACT 1. Returns ok=False with reasons on any hard violation; soft issues as flags."""
    errors: list[str] = []
    flags: list[str] = []

    _validate_image(sample.image, errors, flags)
    if sample.mask is not None:
        _validate_mask(sample.mask, sample.image, errors, flags)

    if sample.mm_per_px is not None:
        lo, hi = MM_PER_PX_RANGE
        if not (isinstance(sample.mm_per_px, (int, float)) and np.isfinite(sample.mm_per_px)):
            errors.append("mm_per_px: not a finite number")
        elif not (lo <= float(sample.mm_per_px) <= hi):
            errors.append(f"mm_per_px: {sample.mm_per_px} outside sanity range [{lo}, {hi}]")

    if sample.material not in MATERIALS:
        errors.append(f"material: '{sample.material}' not in {MATERIALS}")
    if not sample.source:
        errors.append("source: empty (provenance is mandatory)")
    if sample.license_tag not in LICENSE_TAGS:
        errors.append(f"license_tag: '{sample.license_tag}' not in {LICENSE_TAGS}")

    result = ValidationResult(ok=not errors, errors=errors, flags=flags)
    sample.flags = list(flags)
    return result


def is_redistributable(license_tag: str) -> bool:
    """Whether imagery under this tag may be committed to the public repository."""
    return license_tag in REDISTRIBUTABLE_TAGS
