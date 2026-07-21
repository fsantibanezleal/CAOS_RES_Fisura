"""Bake the DIC virtual experiment (BL-012, dossier 04 section 5.2/5.3): a synthetic speckle image
deformed by a KNOWN displacement field, so the DIC engine is validated against exact ground truth,
plus the speckle-vs-natural-texture accuracy comparison the literature says is worth a factor of ~3.

Two knowns imposed (dossier 5.1 crack relevance): a uniform horizontal stretch epsilon_xx = 0.01
(1 percent) PLUS a crack-opening displacement (a jump in u across a vertical line = the crack).
DIC recovers both; the crack reads as the displacement discontinuity. Then the same deformation is
applied to a natural-concrete-like texture and the accuracy loss is measured on the same field.

Writes data/derived/dic/dic.json + overlays (speckle ref, u-field, e_xx field, COD profile as PNGs).
All CPU (numpy + scipy + skimage). muDIC/py2DIC not used (method/license mismatch); in-repo ZNCC.

    python -m fisuralab.dic.bake_dic
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .correlation import dic_field, strain_from_field

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "data" / "derived" / "dic"
SIZE = 260
STRAIN = 0.01     # 1 percent uniform stretch (known)
COD = 2.5         # crack-opening displacement in px (known jump in u across the crack line)
CRACK_X = SIZE // 2


def _speckle(seed: int, density: float = 0.35, blob: float = 2.0) -> np.ndarray:
    """A random high-contrast isotropic speckle pattern (the DIC gold standard)."""
    from scipy import ndimage as ndi  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n = int(density * SIZE * SIZE / (np.pi * blob * blob))
    img = np.zeros((SIZE, SIZE), np.float32)
    ys = rng.integers(0, SIZE, n)
    xs = rng.integers(0, SIZE, n)
    img[ys, xs] = 1.0
    img = ndi.gaussian_filter(img, blob)
    img = (img - img.min()) / max(1e-6, img.max() - img.min())
    return img


def _concrete_texture(seed: int) -> np.ndarray:
    """A natural-concrete-like texture: LOW intensity gradient (weak, smooth large-scale variation)
    with a faint quasi-periodic component, which makes subsets ambiguous to match. The DIC literature
    reports natural texture error ~3x painted speckle, and the cause is exactly this low mean-intensity-
    gradient + self-similarity, versus the high-contrast random isotropic speckle above."""
    from scipy import ndimage as ndi  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    # dominant smooth low-contrast field (weak gradients -> poor sub-pixel conditioning)
    g = ndi.gaussian_filter(rng.normal(0.5, 0.05, (SIZE, SIZE)).astype(np.float32), 6.0)
    # a faint near-periodic aggregate pattern -> self-similar subsets (ambiguous matches)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    g = g + 0.03 * np.sin(xx / 7.0) * np.sin(yy / 7.0)
    # very fine sensor-like noise that does NOT aid matching (uncorrelated between the two frames path)
    g = g + rng.normal(0, 0.01, (SIZE, SIZE))
    g = (g - g.min()) / max(1e-6, g.max() - g.min())
    return np.clip(g * 0.6 + 0.2, 0, 1)  # compress into a low-contrast mid range


def _warp_known(img: np.ndarray) -> np.ndarray:
    """Apply the known field: u(x) = STRAIN*(x - cx) + COD*[x > CRACK_X]; v = 0. Backward-map sample."""
    from scipy.ndimage import map_coordinates  # noqa: PLC0415

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    cx = SIZE / 2
    u = STRAIN * (xx - cx) + COD * (xx > CRACK_X).astype(np.float64)
    # deformed(x) = ref(x - u): sample the reference at the pre-image
    src_x = xx - u
    src_y = yy
    return map_coordinates(img, [src_y, src_x], order=1, mode="reflect").astype(np.float32)


def _known_u(xs: np.ndarray) -> np.ndarray:
    cx = SIZE / 2
    return STRAIN * (xs - cx) + COD * (xs > CRACK_X).astype(np.float64)


def _field_png(field: np.ndarray, lo: float | None = None, hi: float | None = None) -> bytes:
    """Encode a scalar field as a blue->red PNG (nan -> transparent)."""
    import imageio.v3 as iio  # noqa: PLC0415
    from skimage.transform import resize as rz  # noqa: PLC0415

    f = rz(field, (SIZE, SIZE), order=0, preserve_range=True)
    m = np.isnan(f)
    lo = np.nanmin(f) if lo is None else lo
    hi = np.nanmax(f) if hi is None else hi
    norm = np.clip((f - lo) / max(1e-9, hi - lo), 0, 1)
    norm = np.where(m, 0.0, norm)  # zero NaNs before the uint8 cast (transparent anyway)
    r = (norm * 235).astype(np.uint8)
    g = (np.clip(1 - np.abs(norm - 0.5) * 2, 0, 1) * 120).astype(np.uint8)
    b = ((1 - norm) * 220).astype(np.uint8)
    a = np.where(m, 0, 235).astype(np.uint8)
    rgba = np.dstack([r, g, b, a])
    buf = io.BytesIO()
    iio.imwrite(buf, rgba, extension=".png")
    return buf.getvalue()


def _gray_png(img: np.ndarray) -> bytes:
    import imageio.v3 as iio  # noqa: PLC0415

    buf = io.BytesIO()
    iio.imwrite(buf, (np.clip(img, 0, 1) * 255).astype(np.uint8), extension=".png")
    return buf.getvalue()


def _run(ref: np.ndarray):
    defo = _warp_known(ref)
    ys, xs, u, v = dic_field(ref, defo, subset=21, step=10, search=8)
    exx, _eyy, _gxy = strain_from_field(xs, ys, u, v)
    # accuracy vs the known field (exclude the crack column, where the jump is a discontinuity)
    ku = np.array([_known_u(np.array([x]))[0] for x in xs])
    away = np.abs(xs - CRACK_X) > 15
    u_err = np.nanmean(np.abs(u[:, away] - ku[None, away]))
    exx_valid = exx[:, away]
    exx_mean = float(np.nanmean(exx_valid))
    return defo, ys, xs, u, v, exx, float(u_err), exx_mean, ku


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)

    ref_sp = _speckle(seed=3)
    defo_sp, ys, xs, u_sp, _v, exx_sp, u_err_sp, exx_mean_sp, ku = _run(ref_sp)

    ref_tx = _concrete_texture(seed=5)
    _defo_tx, _ys2, _xs2, u_tx, _v2, _exx_tx, u_err_tx, _exx_mean_tx, _ku2 = _run(ref_tx)

    # COD profile: mean measured u across the crack line vs away from it (the displacement jump)
    left = np.abs(xs - (CRACK_X - 20)).argmin()
    right = np.abs(xs - (CRACK_X + 20)).argmin()
    cod_measured = float(np.nanmean(u_sp[:, right]) - np.nanmean(u_sp[:, left]) - STRAIN * 40)

    (OUT / "overlays" / "ref_speckle.png").write_bytes(_gray_png(ref_sp))
    (OUT / "overlays" / "u_field.png").write_bytes(_field_png(u_sp))
    (OUT / "overlays" / "exx_field.png").write_bytes(_field_png(exx_sp, lo=-0.005, hi=0.02))
    (OUT / "overlays" / "ref_texture.png").write_bytes(_gray_png(ref_tx))

    rec = {
        "schema": "fisura.dic/v1",
        "method": "in-repo 2D subset DIC (ZNCC, sub-pixel quadratic peak, local-polynomial strain)",
        "known_field": {"uniform_strain_exx": STRAIN, "crack_opening_px": COD, "crack_x": CRACK_X},
        "speckle": {
            "u_mae_px": round(u_err_sp, 4),
            "measured_mean_exx": round(exx_mean_sp, 5),
            "measured_cod_px": round(cod_measured, 3),
        },
        "natural_texture": {"u_mae_px": round(u_err_tx, 4)},
        "texture_vs_speckle_error_ratio": round(u_err_tx / max(1e-6, u_err_sp), 2),
        "overlays": {
            "ref_speckle": "dic/overlays/ref_speckle.png",
            "u_field": "dic/overlays/u_field.png",
            "exx_field": "dic/overlays/exx_field.png",
            "ref_texture": "dic/overlays/ref_texture.png",
        },
        "framing": (
            "A synthetic speckle image deformed by a KNOWN field (1 percent stretch + a crack-opening "
            "jump) validates the DIC engine against exact ground truth. The crack reads as the "
            "displacement discontinuity. Natural concrete texture is measured on the same field to show "
            "the accuracy cost (the literature reports ~3x worse than painted speckle). 2D DIC assumes a "
            "planar specimen, perpendicular camera axis, negligible out-of-plane motion."
        ),
    }
    with open(OUT / "dic.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rec[k] for k in ("speckle", "natural_texture", "texture_vs_speckle_error_ratio")}, indent=1))
    print(f"-> {OUT / 'dic.json'}")


if __name__ == "__main__":
    main()
