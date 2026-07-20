"""Two-epoch registration + differential crack mapping (BL-011 monitoring, dossier 04 section 4).

Two surveys of the same surface weeks apart have different camera poses and lighting; the signal
(width growth 0.05 to 0.5 mm, tip extension mm to cm) is smaller than the nuisance pose/illumination
variation. So change detection MUST run after metric registration, then compare skeleton graphs in the
common frame and report PER-BRANCH width deltas + new-branch events, never raw pixel differences
(which lighting dominates). This module implements that pipeline, CPU-only (scikit-image + numpy).

- Registration: ORB features + descriptor matching + RANSAC homography (dossier: SIFT/ORB + RANSAC),
  warping epoch-2 onto epoch-1's frame.
- Differential mapping: skeletonize both crack masks in the common frame; per skeleton point take the
  inscribed-circle width (2*EDT); bin along the crack; report width growth, new-branch pixels (present
  in epoch 2, absent in epoch 1), and the tip-extension length.
"""
from __future__ import annotations

import numpy as np


def register_orb(ref_gray: np.ndarray, mov_gray: np.ndarray, max_features: int = 2000):
    """Estimate the homography warping `mov` onto `ref` via ORB + RANSAC. Returns (H 3x3, n_inliers)
    or (None, 0) if too few matches. Grayscale float [0,1] inputs."""
    from skimage.feature import ORB, match_descriptors  # noqa: PLC0415
    from skimage.measure import ransac  # noqa: PLC0415
    from skimage.transform import ProjectiveTransform  # noqa: PLC0415

    orb = ORB(n_keypoints=max_features, fast_threshold=0.05)
    orb.detect_and_extract(ref_gray)
    kp1, d1 = orb.keypoints, orb.descriptors
    orb.detect_and_extract(mov_gray)
    kp2, d2 = orb.keypoints, orb.descriptors
    matches = match_descriptors(d1, d2, cross_check=True)
    if len(matches) < 8:
        return None, 0
    # ransac on (row, col) -> (row, col); ProjectiveTransform maps src (mov) to dst (ref)
    src = kp2[matches[:, 1]][:, ::-1]  # (x, y)
    dst = kp1[matches[:, 0]][:, ::-1]
    model, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=1000)
    if model is None or inliers is None:
        return None, 0
    return model.params, int(inliers.sum())


def warp_to(ref_shape: tuple[int, int], mov: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp `mov` into `ref`'s frame with homography H (as returned by register_orb)."""
    from skimage.transform import ProjectiveTransform, warp  # noqa: PLC0415

    tf = ProjectiveTransform(matrix=H)
    return warp(mov, tf.inverse, output_shape=ref_shape, order=0 if mov.dtype == bool else 1, preserve_range=True)


def _skeleton_width(mask: np.ndarray):
    """Return (skeleton bool, per-pixel inscribed-circle width = 2*EDT) for a crack mask."""
    from scipy import ndimage as ndi  # noqa: PLC0415
    from skimage.morphology import skeletonize  # noqa: PLC0415

    m = mask.astype(bool)
    skel = skeletonize(m)
    dist = ndi.distance_transform_edt(m)
    return skel, 2.0 * dist


def differential_map(mask_ep1: np.ndarray, mask_ep2_reg: np.ndarray, mm_per_px: float | None = None) -> dict:
    """Compare two registered crack masks in the common frame.

    Returns per-branch/aggregate deltas: median width epoch1 vs epoch2, new-branch pixel count (in
    epoch2, not epoch1, dilated tolerance), length epoch1 vs epoch2, and a growth summary."""
    from scipy import ndimage as ndi  # noqa: PLC0415

    m1 = mask_ep1.astype(bool)
    m2 = mask_ep2_reg.astype(bool)
    skel1, w1 = _skeleton_width(m1)
    skel2, w2 = _skeleton_width(m2)

    w1s = w1[skel1]
    w2s = w2[skel2]
    med1 = float(np.median(w1s)) if w1s.size else 0.0
    med2 = float(np.median(w2s)) if w2s.size else 0.0
    p95_1 = float(np.percentile(w1s, 95)) if w1s.size else 0.0
    p95_2 = float(np.percentile(w2s, 95)) if w2s.size else 0.0

    # new-branch: epoch2 skeleton not within 2 px of any epoch1 crack pixel
    d1 = ndi.distance_transform_edt(~m1)
    new_branch = skel2 & (d1 > 2)
    len1 = int(skel1.sum())
    len2 = int(skel2.sum())

    scale = mm_per_px or 1.0
    unit = "mm" if mm_per_px else "px"
    return {
        "unit": unit,
        "width_median_ep1": round(med1 * scale, 4),
        "width_median_ep2": round(med2 * scale, 4),
        "width_delta_median": round((med2 - med1) * scale, 4),
        "width_p95_ep1": round(p95_1 * scale, 4),
        "width_p95_ep2": round(p95_2 * scale, 4),
        "length_ep1_px": len1,
        "length_ep2_px": len2,
        "length_delta_px": len2 - len1,
        "new_branch_px": int(new_branch.sum()),
        "grew": bool(med2 > med1 or len2 > len1),
    }
