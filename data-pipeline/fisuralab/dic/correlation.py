"""2D Digital Image Correlation (BL-012 DIC track, dossier 04 section 5), implemented in-repo.

The real subset-based method (Pan et al., MST 2009): a reference subset of (2M+1)^2 pixels around each
measurement point is located in the deformed image by maximizing the zero-normalized cross-correlation

    C_ZNCC = sum_i (f_i - fbar)(g_i - gbar) / (sqrt(sum (f_i-fbar)^2) sqrt(sum (g_i-gbar)^2))

invariant to affine intensity changes g = a f + b. Integer-pixel search then sub-pixel refinement by a
local quadratic fit of the correlation surface. Strain is taken from a local polynomial fit of the
displacement field, NEVER raw pointwise differentiation (dossier rule).

Implemented rather than via muDIC/py2DIC: muDIC is global-FE (different method) and py2DIC is GPL; the
real subset-ZNCC algorithm is compact and MIT-clean here. Pure numpy + scipy, CPU. This is the engine
the virtual experiment (bake_dic) validates against a known deformation field with exact ground truth.
"""
from __future__ import annotations

import numpy as np


def _zncc_score(f: np.ndarray, g: np.ndarray) -> float:
    fm, gm = f.mean(), g.mean()
    fc, gc = f - fm, g - gm
    denom = np.sqrt((fc * fc).sum()) * np.sqrt((gc * gc).sum())
    if denom < 1e-9:
        return -1.0
    return float((fc * gc).sum() / denom)


def _subpixel_peak(scores: np.ndarray, cy: int, cx: int) -> tuple[float, float]:
    """Quadratic sub-pixel refinement of a 2D correlation-score peak at integer (cy, cx)."""
    h, w = scores.shape
    dy = dx = 0.0
    if 0 < cy < h - 1:
        a, b, c = scores[cy - 1, cx], scores[cy, cx], scores[cy + 1, cx]
        d = a - 2 * b + c
        if abs(d) > 1e-9:
            dy = 0.5 * (a - c) / d
    if 0 < cx < w - 1:
        a, b, c = scores[cy, cx - 1], scores[cy, cx], scores[cy, cx + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-9:
            dx = 0.5 * (a - c) / d
    return float(np.clip(dy, -1, 1)), float(np.clip(dx, -1, 1))


def dic_field(ref: np.ndarray, defo: np.ndarray, subset: int = 21, step: int = 12, search: int = 12):
    """Compute a dense-ish displacement field by subset ZNCC over a grid.

    ref, defo: HxW grayscale float [0,1]. subset = (2M+1) window; step = grid spacing; search =
    integer search radius. Returns (ys, xs, u, v) where u=horizontal (col), v=vertical (row)
    displacement at each grid point (NaN where the subset/search leaves the frame)."""
    H, W = ref.shape
    m = subset // 2
    ys = np.arange(m + search, H - m - search, step)
    xs = np.arange(m + search, W - m - search, step)
    u = np.full((len(ys), len(xs)), np.nan, np.float32)
    v = np.full((len(ys), len(xs)), np.nan, np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            f = ref[y - m:y + m + 1, x - m:x + m + 1]
            scores = np.full((2 * search + 1, 2 * search + 1), -1.0, np.float32)
            for sy in range(-search, search + 1):
                for sx in range(-search, search + 1):
                    g = defo[y + sy - m:y + sy + m + 1, x + sx - m:x + sx + m + 1]
                    if g.shape == f.shape:
                        scores[sy + search, sx + search] = _zncc_score(f, g)
            cy, cx = np.unravel_index(int(np.argmax(scores)), scores.shape)
            dy, dx = _subpixel_peak(scores, cy, cx)
            v[iy, ix] = (cy - search) + dy
            u[iy, ix] = (cx - search) + dx
    return ys, xs, u, v


def strain_from_field(xs: np.ndarray, ys: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Small-strain e_xx, e_yy, gamma_xy from LOCAL polynomial (least-squares plane) fits of the
    displacement field over a 3x3 grid neighbourhood, never raw pointwise differentiation."""
    ny, nx = u.shape
    exx = np.full_like(u, np.nan)
    eyy = np.full_like(u, np.nan)
    gxy = np.full_like(u, np.nan)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            win_u = u[iy - 1:iy + 2, ix - 1:ix + 2]
            win_v = v[iy - 1:iy + 2, ix - 1:ix + 2]
            if np.isnan(win_u).any() or np.isnan(win_v).any():
                continue
            # plane fit u = a + b*X + c*Y over the local 3x3 (X,Y in px)
            gx, gy = np.meshgrid(np.arange(-1, 2) * dx, np.arange(-1, 2) * dy)
            A = np.column_stack([np.ones(9), gx.ravel(), gy.ravel()])
            cu, *_ = np.linalg.lstsq(A, win_u.ravel(), rcond=None)
            cv, *_ = np.linalg.lstsq(A, win_v.ravel(), rcond=None)
            u_x, u_y = cu[1], cu[2]
            v_x, v_y = cv[1], cv[2]
            exx[iy, ix] = u_x
            eyy[iy, ix] = v_y
            gxy[iy, ix] = u_y + v_x
    return exx, eyy, gxy
