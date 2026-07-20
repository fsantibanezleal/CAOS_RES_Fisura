"""Bake the two-epoch monitoring case (BL-011): a synthetic crack that GROWS between epoch 1 and
epoch 2 under a camera-pose change, registered and differentially mapped with EXACT ground truth.

Why synthetic: real registered inspection pairs are gated and their true growth is unknown. A
generated pair is the only place the growth (width delta, tip extension, new branches) is known by
construction, so it validates the registration + differential-mapping pipeline honestly (the same
role the synthetic battery plays for the classical ladder). Dossier 04 section 4: report per-branch
width deltas + new-branch events after metric registration, never raw pixel differences.

Writes data/derived/monitoring/growth.json + overlays (epoch1, epoch2-raw, epoch2-registered, the
change map) as committed PNGs. All CPU (scikit-image + numpy).

    python -m fisuralab.monitoring.bake_growth
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from ..learned.shards import data_root  # noqa: F401  (keeps import surface consistent)
from .registration import differential_map, register_orb, warp_to

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "data" / "derived" / "monitoring"
SIZE = 320
MM_PER_PX = 0.08  # a plausible close-range GSD so widths read in mm


def _crack_polyline(t_end: float, width: float, wobble: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A crack mask + its centerline, growing from the top to fraction t_end of the height, with a
    given half-width and lateral wobble. Returns (mask bool, centerline int (N,2) row,col)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((SIZE, SIZE), bool)
    ys = np.arange(int(0.1 * SIZE), int(t_end * SIZE))
    cx = SIZE * 0.42
    center = []
    phase = rng.uniform(0, 6.28)
    for y in ys:
        cx += wobble * np.sin(y / 22.0 + phase) * 0.5 + rng.normal(0, 0.3)
        xi = int(round(cx))
        center.append((y, xi))
        hw = int(round(width))
        mask[y, max(0, xi - hw):min(SIZE, xi + hw + 1)] = True
    return mask, np.array(center, int)


def _texture_base(seed: int) -> np.ndarray:
    """A concrete-like grayscale texture so ORB has real features to register on."""
    from scipy import ndimage as ndi  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    g = rng.normal(0.55, 0.12, (SIZE, SIZE)).astype(np.float32)
    g = ndi.gaussian_filter(g, 1.2)
    # a few darker blobs + specks give distinctive keypoints
    for _ in range(40):
        y, x = rng.integers(0, SIZE, 2)
        r = rng.integers(3, 9)
        yy, xx = np.ogrid[:SIZE, :SIZE]
        g[(yy - y) ** 2 + (xx - x) ** 2 < r * r] -= rng.uniform(0.05, 0.15)
    return np.clip(g, 0, 1)


def _render(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Darken the crack onto the texture -> grayscale float image."""
    img = base.copy()
    img[mask] = np.clip(img[mask] - 0.35, 0, 1)
    return img


def _homography_pose(shift: float, rot_deg: float, scale: float) -> np.ndarray:
    """A mild pose change (translation + small rotation + scale) as a 3x3 homography about the center."""
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    cx = cy = SIZE / 2
    # T(-center) -> R*S -> T(center) + shift
    R = np.array([[scale * c, -scale * s, 0], [scale * s, scale * c, 0], [0, 0, 1]], float)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    T2 = np.array([[1, 0, cx + shift], [0, 1, cy + shift * 0.6], [0, 0, 1]], float)
    return T2 @ R @ T1


def _png(arr01_or_rgb: np.ndarray) -> bytes:
    import imageio.v3 as iio  # noqa: PLC0415

    a = arr01_or_rgb
    if a.ndim == 2:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    iio.imwrite(buf, a, extension=".png")
    return buf.getvalue()


def _change_overlay(base: np.ndarray, m1: np.ndarray, m2reg: np.ndarray) -> np.ndarray:
    """Green = crack in both epochs, red = new in epoch 2 (growth), on the epoch-1 texture."""
    rgb = np.stack([base, base, base], -1) * 255.0
    both = m1 & m2reg
    grew = m2reg & ~m1
    rgb[both] = [70, 200, 110]
    rgb[grew] = [230, 60, 60]
    return rgb


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)

    base1 = _texture_base(seed=1)
    # epoch 1: shorter, thinner crack
    m1, _c1 = _crack_polyline(t_end=0.62, width=1.5, wobble=2.0, seed=10)
    img1 = _render(base1, m1)

    # epoch 2: longer + wider crack (growth), rendered on the SAME texture then posed (new camera view)
    m2, _c2 = _crack_polyline(t_end=0.80, width=2.6, wobble=2.0, seed=10)  # same seed -> same path, extended
    img2_flat = _render(base1, m2)
    H_pose = _homography_pose(shift=9.0, rot_deg=3.5, scale=1.03)
    img2 = warp_to((SIZE, SIZE), img2_flat, np.linalg.inv(H_pose))  # epoch-2 as seen from the new pose
    m2_posed = warp_to((SIZE, SIZE), m2.astype(np.uint8), np.linalg.inv(H_pose)) > 0.5

    # register epoch-2 back onto epoch-1's frame (the real pipeline: recover the pose from features)
    H_est, n_inliers = register_orb(img1.astype(np.float32), img2.astype(np.float32))
    if H_est is None:
        raise SystemExit("registration failed (too few ORB matches) - retune the synthetic texture")
    m2_reg = warp_to((SIZE, SIZE), m2_posed.astype(np.uint8), H_est) > 0.5

    diff = differential_map(m1, m2_reg, mm_per_px=MM_PER_PX)

    # ground truth deltas (known by construction, in the epoch-1 flat frame)
    from .registration import _skeleton_width  # noqa: PLC0415

    sk1, w1 = _skeleton_width(m1)
    sk2, w2 = _skeleton_width(m2)  # true epoch-2 in the flat frame
    gt = {
        "true_width_median_delta_mm": round((float(np.median(w2[sk2])) - float(np.median(w1[sk1]))) * MM_PER_PX, 4),
        "true_length_delta_px": int(sk2.sum() - sk1.sum()),
    }

    (OUT / "overlays" / "epoch1.png").write_bytes(_png(img1))
    (OUT / "overlays" / "epoch2_raw.png").write_bytes(_png(img2))
    (OUT / "overlays" / "epoch2_registered.png").write_bytes(_png(warp_to((SIZE, SIZE), img2.astype(np.float32), H_est)))
    (OUT / "overlays" / "change.png").write_bytes(_png(_change_overlay(base1, m1, m2_reg)))

    rec = {
        "schema": "fisura.monitoring/v1",
        "case": "synthetic two-epoch growth (exact ground truth)",
        "mm_per_px": MM_PER_PX,
        "registration": {"method": "ORB + RANSAC homography", "inliers": n_inliers},
        "measured": diff,
        "ground_truth": gt,
        "overlays": {
            "epoch1": "monitoring/overlays/epoch1.png",
            "epoch2_raw": "monitoring/overlays/epoch2_raw.png",
            "epoch2_registered": "monitoring/overlays/epoch2_registered.png",
            "change": "monitoring/overlays/change.png",
        },
        "framing": (
            "Change detection runs AFTER metric registration; the reported signal is per-branch width "
            "growth and new-branch pixels, not raw pixel differences (which lighting dominates). "
            "Synthetic pair with exact ground truth validates the pipeline; real registered pairs are the field goal."
        ),
    }
    with open(OUT / "growth.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(json.dumps({"inliers": n_inliers, "measured": diff, "ground_truth": gt}, indent=1))
    print(f"-> {OUT / 'growth.json'}")


if __name__ == "__main__":
    main()
