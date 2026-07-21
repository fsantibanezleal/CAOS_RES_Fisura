"""DIC track tests (BL-012). The subset-ZNCC engine is pure CPU (numpy + scipy), so a small
known-deformation case runs everywhere; the committed virtual-experiment artifact is validated too."""
import json
from pathlib import Path

import numpy as np

from fisuralab.dic.correlation import dic_field, strain_from_field


def _speckle(size=120, seed=0):
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    ys = rng.integers(0, size, 900)
    xs = rng.integers(0, size, 900)
    img[ys, xs] = 1.0
    img = ndi.gaussian_filter(img, 1.8)
    return (img - img.min()) / max(1e-6, img.max() - img.min())


def test_dic_recovers_uniform_translation():
    # a pure integer translation of 3 px in x, 0 in y: DIC must recover u approx 3, v approx 0
    ref = _speckle(seed=1)
    defo = np.roll(ref, 3, axis=1)
    ys, xs, u, v = dic_field(ref, defo, subset=21, step=15, search=6)
    assert abs(float(np.nanmedian(u)) - 3.0) < 0.4
    assert abs(float(np.nanmedian(v))) < 0.4


def test_dic_recovers_uniform_strain():
    # a 1 percent horizontal stretch about the center; measured e_xx should be near 0.01
    from scipy.ndimage import map_coordinates

    ref = _speckle(size=160, seed=2)
    size = ref.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = size / 2
    u = 0.01 * (xx - cx)
    defo = map_coordinates(ref, [yy, xx - u], order=1, mode="reflect").astype(np.float32)
    ys, xs, uu, vv = dic_field(ref, defo, subset=21, step=12, search=6)
    exx, _eyy, _gxy = strain_from_field(xs, ys, uu, vv)
    assert abs(float(np.nanmean(exx)) - 0.01) < 0.004


def test_committed_dic_artifact_is_coherent():
    p = Path(__file__).resolve().parents[1] / "data" / "derived" / "dic" / "dic.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    # the engine recovers the known crack opening within ~0.5 px on speckle
    assert abs(d["speckle"]["measured_cod_px"] - d["known_field"]["crack_opening_px"]) < 0.6
    # the known 1 percent strain within 0.3 percentage points
    assert abs(d["speckle"]["measured_mean_exx"] - d["known_field"]["uniform_strain_exx"]) < 0.003
    # natural texture is measurably worse than speckle (the literature's factor)
    assert d["texture_vs_speckle_error_ratio"] > 1.5
    assert "E:" not in json.dumps(d) and "/raw/" not in json.dumps(d)
