"""Synthetic crack generators with EXACT ground truth (pure numpy, Pyodide-safe).

The validation battery of the classical ladder: cracks of known centerline, width and contrast on
controlled backgrounds. Because the geometry is generated, ground-truth masks, centerlines and
widths are exact, which is what makes the battery a regression gate for the ridge-filter stack
(the scikit-image version-pinning discipline from the research: validate on synthetic bars at
every upgrade) and the reference for width-estimator accuracy.

Conventions: images are float32 in [0, 1]; cracks are DARK on a brighter background (the field
convention for concrete/pavement imagery); masks are bool with True = crack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticCrack:
    """One generated specimen: image, exact mask, centerline points and the true width map."""

    image: np.ndarray          # float32 HxW in [0, 1]
    mask: np.ndarray           # bool HxW, True = crack
    centerline: np.ndarray     # (N, 2) float array of (row, col) points along the path
    width_px: np.ndarray       # (N,) true full width at each centerline point
    params: dict


def _texture(rng: np.random.Generator, shape: tuple[int, int], base: float, noise: float, blotch: float) -> np.ndarray:
    """Concrete-like background: base level + broadband noise + smooth blotches (no scipy needed)."""
    h, w = shape
    img = np.full(shape, base, dtype=np.float64)
    img += rng.normal(0.0, noise, size=shape)
    # smooth blotches: upsample a coarse random grid bilinearly
    coarse = rng.normal(0.0, blotch, size=(max(2, h // 32), max(2, w // 32)))
    ys = np.linspace(0, coarse.shape[0] - 1, h)
    xs = np.linspace(0, coarse.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, coarse.shape[0] - 1)
    x1 = np.minimum(x0 + 1, coarse.shape[1] - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    img += (
        coarse[np.ix_(y0, x0)] * (1 - fy) * (1 - fx)
        + coarse[np.ix_(y1, x0)] * fy * (1 - fx)
        + coarse[np.ix_(y0, x1)] * (1 - fy) * fx
        + coarse[np.ix_(y1, x1)] * fy * fx
    )
    return img


def _render_crack(shape: tuple[int, int], centerline: np.ndarray, width_px: np.ndarray, depth: float, softness: float) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a dark crack of varying width along a centerline.

    Returns (darkening field float64 HxW in [0, depth], exact mask bool). The profile across the
    crack is a smooth plateau: full darkening inside half-width, Gaussian shoulder outside
    (softness in px), which mimics the penumbra of real crack edges while keeping the EXACT
    binary mask at the half-width boundary.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dark = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    # distance to the polyline, computed segment-wise (vectorized per segment)
    for i in range(len(centerline) - 1):
        p, q = centerline[i], centerline[i + 1]
        wp, wq = width_px[i], width_px[i + 1]
        d = q - p
        seg_len2 = float(d @ d)
        if seg_len2 == 0:
            continue
        t = ((yy - p[0]) * d[0] + (xx - p[1]) * d[1]) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        dist = np.hypot(yy - (p[0] + t * d[0]), xx - (p[1] + t * d[1]))
        half = (wp + t * (wq - wp)) / 2.0
        inside = dist <= half
        mask |= inside
        shoulder = np.exp(-0.5 * ((dist - half) / max(softness, 1e-6)) ** 2)
        profile = np.where(inside, 1.0, shoulder)
        dark = np.maximum(dark, profile)
    return dark * depth, mask


def straight_bar(
    size: int = 256,
    width_px: float = 3.0,
    angle_deg: float = 30.0,
    contrast: float = 0.35,
    noise: float = 0.02,
    blotch: float = 0.03,
    seed: int = 0,
) -> SyntheticCrack:
    """A straight dark bar of constant width crossing the image at a given angle."""
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    c = size / 2.0
    reach = size  # long enough to cross
    p0 = np.array([c - reach * np.sin(theta), c - reach * np.cos(theta)])
    p1 = np.array([c + reach * np.sin(theta), c + reach * np.cos(theta)])
    n = 64
    ts = np.linspace(0.0, 1.0, n)
    center = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    keep = (center[:, 0] > -width_px) & (center[:, 0] < size + width_px) & (center[:, 1] > -width_px) & (center[:, 1] < size + width_px)
    center = center[keep]
    widths = np.full(len(center), float(width_px))
    bg = _texture(rng, (size, size), base=0.62, noise=noise, blotch=blotch)
    dark, mask = _render_crack((size, size), center, widths, depth=contrast, softness=0.8)
    img = np.clip(bg - dark, 0.0, 1.0).astype(np.float32)
    return SyntheticCrack(img, mask, center, widths, params=dict(
        kind="straight_bar", size=size, width_px=width_px, angle_deg=angle_deg,
        contrast=contrast, noise=noise, blotch=blotch, seed=seed,
    ))


def wavy_crack(
    size: int = 256,
    width_px: float = 3.0,
    width_taper: float = 0.5,
    amplitude: float = 22.0,
    cycles: float = 1.6,
    contrast: float = 0.3,
    noise: float = 0.025,
    blotch: float = 0.04,
    seed: int = 1,
) -> SyntheticCrack:
    """A meandering crack with linearly tapering width (endpoint width = width_px * width_taper)."""
    rng = np.random.default_rng(seed)
    n = 96
    ts = np.linspace(0.0, 1.0, n)
    rows = size * (0.15 + 0.7 * ts)
    cols = size / 2.0 + amplitude * np.sin(2 * np.pi * cycles * ts) + rng.normal(0, 1.0, n).cumsum() * 0.35
    center = np.stack([rows, np.clip(cols, 2, size - 3)], axis=1)
    widths = width_px * (1.0 + (width_taper - 1.0) * ts)
    bg = _texture(rng, (size, size), base=0.58, noise=noise, blotch=blotch)
    dark, mask = _render_crack((size, size), center, widths, depth=contrast, softness=0.9)
    img = np.clip(bg - dark, 0.0, 1.0).astype(np.float32)
    return SyntheticCrack(img, mask, center, widths, params=dict(
        kind="wavy_crack", size=size, width_px=width_px, width_taper=width_taper,
        amplitude=amplitude, cycles=cycles, contrast=contrast, noise=noise, blotch=blotch, seed=seed,
    ))


def uncracked(
    size: int = 256,
    noise: float = 0.02,
    blotch: float = 0.05,
    joint: bool = False,
    seed: int = 2,
) -> SyntheticCrack:
    """A negative control: textured surface, optionally with a straight dark JOINT (a classic
    false-positive trap: joints are straight and long; cracks meander)."""
    rng = np.random.default_rng(seed)
    bg = _texture(rng, (size, size), base=0.6, noise=noise, blotch=blotch)
    mask = np.zeros((size, size), dtype=bool)
    if joint:
        col = size // 2
        bg[:, col - 2 : col + 2] -= 0.18  # a formwork joint: dark, perfectly straight, NOT a crack
    img = np.clip(bg, 0.0, 1.0).astype(np.float32)
    center = np.zeros((0, 2))
    widths = np.zeros(0)
    return SyntheticCrack(img, mask, center, widths, params=dict(
        kind="uncracked", size=size, noise=noise, blotch=blotch, joint=joint, seed=seed,
    ))


def battery(seed: int = 0) -> list[SyntheticCrack]:
    """The standard validation battery: widths 2/3/5/9 px at two angles, a tapering wavy crack,
    a low-contrast bar, an uncracked control and a joint trap. Deterministic in `seed`."""
    out: list[SyntheticCrack] = []
    k = 0
    for width in (2.0, 3.0, 5.0, 9.0):
        for angle in (20.0, 65.0):
            out.append(straight_bar(width_px=width, angle_deg=angle, seed=seed + k))
            k += 1
    out.append(wavy_crack(seed=seed + k))
    k += 1
    out.append(straight_bar(width_px=3.0, angle_deg=45.0, contrast=0.12, seed=seed + k))
    k += 1
    out.append(uncracked(joint=False, seed=seed + k))
    k += 1
    out.append(uncracked(joint=True, seed=seed + k))
    k += 1
    return out
