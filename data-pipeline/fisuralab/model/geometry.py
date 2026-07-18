"""Crack geometry: skeleton, dual width estimators, length and orientation (numpy + scipy + skimage).

The quantification core (the lab's flagship direction), implementing the two independent width
estimators from the research:

1. Inscribed-circle width: w(s) = 2 D(s) at each skeleton pixel, with D the Euclidean distance
   transform of the mask interior. One pass, overestimates at branch points and bends.
2. Orthogonal-profile width: cast the normal to the local skeleton tangent and measure the
   distance between the two mask boundary crossings, with linear sub-pixel interpolation.

Their per-point disagreement is a quality flag, and junction neighbourhoods are excluded from
width statistics (both rules from the dossier). All functions are deterministic and Pyodide-safe
(numpy, scipy.ndimage, skimage.morphology only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import medial_axis, skeletonize


@dataclass
class WidthProfile:
    """Per-skeleton-point width measurements (px; convert with mm_per_px at presentation)."""

    points: np.ndarray        # (N, 2) int (row, col) skeleton pixels, junction-excluded
    width_edt: np.ndarray     # (N,) inscribed-circle width, 2 * EDT
    width_profile: np.ndarray  # (N,) orthogonal-profile width (NaN where the cast failed)
    disagreement: np.ndarray  # (N,) |edt - profile| where both defined, else NaN


@dataclass
class CrackGeometry:
    skeleton: np.ndarray      # bool HxW
    length_px: float          # total skeleton arc length (4/8-connected step lengths)
    n_branches: int           # skeleton pixels with 3+ neighbours (junction count proxy)
    n_endpoints: int
    orientation_hist: np.ndarray  # (18,) histogram of local tangent angles, 10-degree bins [0, 180)
    widths: WidthProfile


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=int)
    k[1, 1] = 0
    return ndi.convolve(skel.astype(int), k, mode="constant")


def skeleton_and_edt(mask: np.ndarray, method: str = "lee") -> tuple[np.ndarray, np.ndarray]:
    """Skeletonize the mask (default Lee, per the research) and return the EDT of the interior."""
    mask = mask.astype(bool)
    if method == "medial":
        skel, dist = medial_axis(mask, return_distance=True)
        return skel, dist
    skel = skeletonize(mask, method=method)
    dist = ndi.distance_transform_edt(mask)
    return skel, dist


def _local_tangents(skel: np.ndarray, window: int = 5) -> dict[tuple[int, int], float]:
    """Tangent angle (radians, [0, pi)) per skeleton pixel by PCA over a local window."""
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return {}
    tangents: dict[tuple[int, int], float] = {}
    pt_set = set(map(tuple, pts))
    half = window // 2
    for r, c in pts:
        neigh = [
            (rr, cc)
            for rr in range(r - half, r + half + 1)
            for cc in range(c - half, c + half + 1)
            if (rr, cc) in pt_set
        ]
        arr = np.asarray(neigh, dtype=np.float64)
        arr -= arr.mean(axis=0)
        if len(arr) < 2:
            tangents[(r, c)] = 0.0
            continue
        cov = arr.T @ arr
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, int(np.argmax(evals))]  # (dr, dc) of the dominant direction
        ang = float(np.arctan2(v[0], v[1])) % np.pi
        tangents[(r, c)] = ang
    return tangents


def _profile_width(mask: np.ndarray, point: tuple[int, int], tangent: float, max_half: float) -> float:
    """Orthogonal-profile width at one skeleton point with linear sub-pixel boundary crossing."""
    normal = np.array([np.cos(tangent), -np.sin(tangent)])  # perpendicular to (sin, cos)
    step = 0.25
    ts = np.arange(step, max_half + step, step)
    h, w = mask.shape

    def crossing(sign: float) -> float:
        prev_inside = True
        prev_t = 0.0
        for t in ts:
            rr = point[0] + sign * t * normal[0]
            cc = point[1] + sign * t * normal[1]
            if not (0 <= rr < h - 1 and 0 <= cc < w - 1):
                return prev_t if not prev_inside else float("nan")
            r0, c0 = int(rr), int(cc)
            fr, fc = rr - r0, cc - c0
            val = (
                mask[r0, c0] * (1 - fr) * (1 - fc)
                + mask[r0 + 1, c0] * fr * (1 - fc)
                + mask[r0, c0 + 1] * (1 - fr) * fc
                + mask[r0 + 1, c0 + 1] * fr * fc
            )
            inside = val >= 0.5
            if not inside:
                # linear interpolation between prev_t (inside) and t (outside)
                return prev_t + (t - prev_t) * 0.5 if prev_inside else prev_t
            prev_inside, prev_t = inside, t
        return float("nan")

    a = crossing(+1.0)
    b = crossing(-1.0)
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return float(a + b)


def measure(mask: np.ndarray, junction_radius: int = 3, skeleton_method: str = "lee") -> CrackGeometry:
    """Full geometry pass: skeleton, dual widths (junction-excluded), length, orientation, topology."""
    mask = mask.astype(bool)
    skel, dist = skeleton_and_edt(mask, method=skeleton_method)
    ncount = _neighbor_count(skel)
    junctions = skel & (ncount >= 3)
    endpoints = skel & (ncount == 1)

    # exclude a junction neighbourhood from width statistics (dossier rule)
    if junctions.any() and junction_radius > 0:
        excl = ndi.binary_dilation(junctions, iterations=junction_radius)
    else:
        excl = np.zeros_like(skel)
    width_pts_mask = skel & ~excl

    pts = np.argwhere(width_pts_mask)
    tangents = _local_tangents(skel)
    max_half = float(dist.max()) * 2.5 + 3.0

    w_edt = np.array([2.0 * dist[r, c] for r, c in pts], dtype=np.float64)
    w_prof = np.array(
        [_profile_width(mask, (int(r), int(c)), tangents.get((int(r), int(c)), 0.0), max_half) for r, c in pts],
        dtype=np.float64,
    )
    both = ~np.isnan(w_prof)
    disagreement = np.where(both, np.abs(w_edt - w_prof), np.nan)

    # length: sum of inter-pixel steps (1 for 4-neighbours, sqrt(2) for diagonals), counted once
    length = 0.0
    pt_set = set(map(tuple, np.argwhere(skel)))
    for r, c in pt_set:
        for dr, dc, dl in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
            if (r + dr, c + dc) in pt_set:
                length += dl

    angles = np.array([tangents[(int(r), int(c))] for r, c in pts], dtype=np.float64) if len(pts) else np.zeros(0)
    hist, _ = np.histogram(np.rad2deg(angles) % 180.0, bins=18, range=(0.0, 180.0))

    return CrackGeometry(
        skeleton=skel,
        length_px=float(length),
        n_branches=int(junctions.sum()),
        n_endpoints=int(endpoints.sum()),
        orientation_hist=hist,
        widths=WidthProfile(points=pts, width_edt=w_edt, width_profile=w_prof, disagreement=disagreement),
    )


def _bilinear(img: np.ndarray, rr: float, cc: float) -> float:
    r0, c0 = int(rr), int(cc)
    fr, fc = rr - r0, cc - c0
    return float(
        img[r0, c0] * (1 - fr) * (1 - fc)
        + img[r0 + 1, c0] * fr * (1 - fc)
        + img[r0, c0 + 1] * (1 - fr) * fc
        + img[r0 + 1, c0 + 1] * fr * fc
    )


def profile_width_intensity(
    gray: np.ndarray,
    point: tuple[int, int],
    tangent: float,
    max_half: float = 12.0,
    bg_margin: float = 3.0,
) -> float:
    """Sub-pixel crack width from the INTENSITY profile (half-depth crossings, linear interpolation).

    Casts the normal to the local tangent on the grayscale image, estimates local background as the
    median intensity beyond (half-width + margin) on both sides, takes the profile minimum as the
    crack floor, and finds the two half-depth crossings with linear interpolation between samples.
    Genuinely sub-pixel because it reads the image signal, not a binary mask; degrades when depth
    is under about 3 times the noise sigma (returns NaN when no meaningful depth exists).
    """
    normal = np.array([np.cos(tangent), -np.sin(tangent)])
    step = 0.25
    ts = np.arange(-max_half, max_half + step, step)
    h, w = gray.shape
    samples = []
    for t in ts:
        rr = point[0] + t * normal[0]
        cc = point[1] + t * normal[1]
        if not (0 <= rr < h - 1 and 0 <= cc < w - 1):
            return float("nan")
        samples.append(_bilinear(gray, rr, cc))
    prof = np.asarray(samples)
    center = len(ts) // 2
    floor_idx = int(np.argmin(prof[center - int(2 / step) : center + int(2 / step) + 1])) + center - int(2 / step)
    floor = prof[floor_idx]
    tails = np.concatenate([prof[: int(bg_margin / step)], prof[-int(bg_margin / step) :]])
    bg = float(np.median(tails))
    depth = bg - floor
    if depth < 0.03:  # no meaningful crack signal on this profile
        return float("nan")
    half_level = bg - depth / 2.0

    def crossing(direction: int) -> float:
        i = floor_idx
        while 0 < i < len(prof) - 1:
            j = i + direction
            if prof[j] >= half_level:
                # linear interpolation between i (below) and j (at/above)
                frac = (half_level - prof[i]) / max(prof[j] - prof[i], 1e-12)
                return abs((ts[i] + frac * (ts[j] - ts[i])))
            i = j
        return float("nan")

    a = crossing(+1)
    b = crossing(-1)
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return float(a + b)


def intensity_width_stats(gray: np.ndarray, geom: CrackGeometry) -> dict:
    """Intensity-domain sub-pixel widths over the junction-excluded skeleton points (NaN-aware)."""
    tangents = _local_tangents(geom.skeleton)
    vals = []
    for r, c in geom.widths.points:
        wv = profile_width_intensity(gray, (int(r), int(c)), tangents.get((int(r), int(c)), 0.0))
        if not np.isnan(wv):
            vals.append(wv)
    if not vals:
        return {"n_points": 0, "intensity_median": None, "intensity_p95": None}
    arr = np.asarray(vals)
    return {
        "n_points": int(len(arr)),
        "intensity_median": float(np.median(arr)),
        "intensity_p95": float(np.percentile(arr, 95)),
    }


def width_stats(geom: CrackGeometry) -> dict:
    """Robust summary of the dual width estimators (px), NaN-aware; empty-safe."""
    w1, w2 = geom.widths.width_edt, geom.widths.width_profile
    out: dict = {"n_points": int(len(w1))}
    if len(w1) == 0:
        return out | {"edt_median": None, "edt_p95": None, "profile_median": None, "profile_p95": None, "disagreement_median": None}
    out["edt_median"] = float(np.median(w1))
    out["edt_p95"] = float(np.percentile(w1, 95))
    valid = ~np.isnan(w2)
    out["profile_median"] = float(np.median(w2[valid])) if valid.any() else None
    out["profile_p95"] = float(np.percentile(w2[valid], 95)) if valid.any() else None
    d = geom.widths.disagreement
    dv = d[~np.isnan(d)]
    out["disagreement_median"] = float(np.median(dv)) if len(dv) else None
    return out
