"""Standard-format IO for the image domain (offline lane).

Readers/writers around CONTRACT 1 (``image_contract``): PNG/JPG images and PNG masks via imageio,
plus the committed-examples manifest loader. The validation core stays numpy-only in
``image_contract``; only this module touches the filesystem, so the browser live lane can reuse the
contract without any IO dependency.

The examples manifest (``data/examples/manifest.json``) is the machine-readable attribution record:
one entry per committed sample with file, mask, source, license_tag, url, citation, material and
optional mm_per_px. The contract test iterates it, so an example that stops passing CONTRACT 1
fails CI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .image_contract import ImageSample, ValidationResult, validate_sample


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as uint8, HxW (kept grayscale) or HxWx3 (alpha dropped, 16-bit scaled)."""
    arr = iio.imread(path)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint16:
        arr = (arr / 257).astype(np.uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(np.asarray(arr, dtype=np.float64), 0, 255).astype(np.uint8)
    return arr


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask as bool HxW (any channel collapse by max; nonzero means positive)."""
    arr = iio.imread(path)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return arr > 0


def to_float01(image: np.ndarray) -> np.ndarray:
    """uint8 image to float32 in [0, 1] (contract-legal float form)."""
    if image.dtype == np.uint8:
        return (image.astype(np.float32)) / 255.0
    return image.astype(np.float32)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    iio.imwrite(path, (mask.astype(np.uint8)) * 255)


@dataclass
class ExampleRecord:
    sample_id: str
    file: str
    source: str
    license_tag: str
    url: str
    citation: str
    material: str
    mask: str | None = None
    mm_per_px: float | None = None
    label: str | None = None  # classification-style label where the source defines one
    fov: str | None = None    # region-of-interest mask (the retina disc for fundus); None = whole image


def load_examples_manifest(examples_dir: str | Path) -> list[ExampleRecord]:
    root = Path(examples_dir)
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return [ExampleRecord(**e) for e in entries]


def load_example(examples_dir: str | Path, rec: ExampleRecord) -> tuple[ImageSample, ValidationResult]:
    """Load one committed example through CONTRACT 1."""
    root = Path(examples_dir)
    sample = ImageSample(
        image=read_image(root / rec.file),
        mask=read_mask(root / rec.mask) if rec.mask else None,
        mm_per_px=rec.mm_per_px,
        material=rec.material,
        source=rec.source,
        license_tag=rec.license_tag,
        sample_id=rec.sample_id,
        fov=read_mask(root / rec.fov) if rec.fov else None,
    )
    return sample, validate_sample(sample)
