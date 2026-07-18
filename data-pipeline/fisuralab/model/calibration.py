"""Pixel-to-millimetre calibration (pure numpy, Pyodide-safe).

The two calibrations that need no special hardware, from the research:

1. Known reference object: two image points spanning a known physical length give a local scale
   mm_per_px = length_mm / length_px. Valid locally; error grows with perspective obliquity.
2. Planar homography (DLT): 4+ correspondences between image points and metric plane coordinates
   (e.g. the corners of a printed board of known square size) give H mapping image to the metric
   plane; measurements are taken IN plane coordinates, which handles oblique views of flat
   elements correctly (the scale varies across the image and H carries that variation).

For standoff setups the ground sampling distance closes the triangle: GSD = p * Z / f (pixel pitch
p, distance Z, focal length f), and a fronto-parallel width reads w_mm = w_px * GSD. Documented in
the calibration guide; not a function here because the lab never invents Z, f or p.
"""
from __future__ import annotations

import numpy as np


def scale_from_reference(p0: tuple[float, float], p1: tuple[float, float], length_mm: float) -> float:
    """mm-per-px from two image points spanning a known physical length. Rejects degenerate input."""
    d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if d < 2.0:
        raise ValueError("reference points are closer than 2 px; scale would be meaningless")
    if not (0.01 <= length_mm <= 10000.0):
        raise ValueError(f"reference length {length_mm} mm outside sanity range [0.01, 10000]")
    return length_mm / d


def homography_dlt(image_pts: np.ndarray, plane_pts_mm: np.ndarray) -> np.ndarray:
    """Direct linear transform: H (3x3) with plane_mm ~ H @ image (homogeneous). Needs >= 4 points.

    Points are Hartley-normalized (translation to centroid, mean distance sqrt(2)) before the SVD,
    the standard conditioning step; H is denormalized and scaled so H[2,2] = 1.
    """
    img = np.asarray(image_pts, dtype=np.float64)
    pl = np.asarray(plane_pts_mm, dtype=np.float64)
    if img.shape != pl.shape or img.ndim != 2 or img.shape[1] != 2 or img.shape[0] < 4:
        raise ValueError("need matching (N,2) arrays with N >= 4")

    def normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = pts.mean(axis=0)
        d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2.0) / max(d, 1e-12)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
        ph = np.column_stack([pts, np.ones(len(pts))]) @ T.T
        return ph, T

    a_h, Ta = normalize(img)
    b_h, Tb = normalize(pl)
    rows = []
    for (x, y, _), (u, v, _) in zip(a_h, b_h):
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (H[2,2] ~ 0); check the correspondences")
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N,2) image points to metric plane coordinates (mm)."""
    p = np.asarray(pts, dtype=np.float64)
    ph = np.column_stack([p, np.ones(len(p))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def width_mm_via_homography(H: np.ndarray, point: tuple[float, float], tangent: float, width_px: float) -> float:
    """Convert a local pixel width to mm by mapping the two width endpoints through H.

    The endpoints are placed half a width apart along the crack NORMAL at the point, so the local,
    spatially varying scale of the oblique view is applied where the measurement was taken.
    """
    n = np.array([np.cos(tangent), -np.sin(tangent)])  # normal (row, col) as in geometry
    half = width_px / 2.0
    a = np.array(point) + half * n
    b = np.array(point) - half * n
    # geometry uses (row, col); homographies conventionally take (x, y) = (col, row)
    ab = apply_homography(H, np.array([[a[1], a[0]], [b[1], b[0]]]))
    return float(np.hypot(*(ab[0] - ab[1])))
