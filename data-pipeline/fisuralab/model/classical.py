"""The classical crack-detection primitives and the L0-L5 ladder (numpy + scipy + scikit-image).

Every primitive here is Pyodide-safe on purpose (the browser live lane reuses this module); the
offline stages may add native extras (DIPlib path openings, OpenCV) behind graceful fallbacks.
Function-per-primitive with explicit parameters, exactly the pinned API surface from the research
(scikit-image 0.26 signatures); the ladder composes them into six named levels:

- L0 global-otsu-on-raw: the honest floor (expected near-useless on textured surfaces).
- L1 flatten + Sauvola local threshold + area filtering ("document binarization transplanted").
- L2 flatten + oriented black-top-hat bank + percentile hysteresis + shape rules.
- L3 NLM + multiscale ridge (sato | frangi | meijering) + diameter opening + hysteresis + shape rules.
- L4 L3 + endpoint minimal-path bridging (the level that feeds geometry/quantification).
- L5 L4 fused with a patch-feature classifier (LBP + GLCM + HOG, random forest), multiplicatively.

The scikit-image ridge filters carry documented version sensitivity (the dossier's issue-class
6436/3783/7711): the synthetic-bar battery in tests/ is the regression gate; `skimage.filters.hessian`
is excluded by decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, restoration
from skimage.graph import route_through_array


# ---------------------------------------------------------------------------
# S1 flatten (illumination) and S2 denoise
# ---------------------------------------------------------------------------

def to_gray_float(image: np.ndarray) -> np.ndarray:
    """Contract-legal image (uint8 or float [0,1], gray or RGB) to float32 grayscale [0, 1]."""
    img = image.astype(np.float32)
    if image.dtype == np.uint8:
        img = img / 255.0
    if img.ndim == 3:
        img = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def flatten_median(image: np.ndarray, radius: int = 21) -> np.ndarray:
    """Median-subtract illumination flattening, recentred at 0.5 WITHOUT contrast renormalization.

    The median footprint must exceed the widest crack so cracks stay in the dark residual. The
    residual keeps its physical contrast (a min-max stretch here would amplify texture noise and
    compress the crack signal, the failure observed on the real BCL patches).
    """
    bg = filters.median(image, footprint=morphology.disk(radius))
    return np.clip(0.5 + (image - bg), 0.0, 1.0).astype(np.float32)


def clahe(image: np.ndarray, clip_limit: float = 0.02) -> np.ndarray:
    return exposure.equalize_adapthist(image, clip_limit=clip_limit).astype(np.float32)


def denoise_nlm(image: np.ndarray, h_factor: float = 0.8) -> np.ndarray:
    """Non-local means with sigma estimated from the image (dossier defaults)."""
    sigma = float(restoration.estimate_sigma(image))
    return restoration.denoise_nl_means(
        image, patch_size=7, patch_distance=11, h=h_factor * max(sigma, 1e-6), fast_mode=True, sigma=sigma
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# S3 enhance: curvilinear responses (dark cracks -> bright response in [0, 1])
# ---------------------------------------------------------------------------

def ridge_response(image: np.ndarray, method: str = "sato", sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)) -> np.ndarray:
    """Multiscale Hessian ridge response for DARK ridges. sigmas span half-widths of the target cracks."""
    if method == "sato":
        resp = filters.sato(image, sigmas=sigmas, black_ridges=True)
    elif method == "frangi":
        resp = filters.frangi(image, sigmas=sigmas, alpha=0.5, beta=0.5, black_ridges=True)
    elif method == "meijering":
        resp = filters.meijering(image, sigmas=sigmas, black_ridges=True)
    else:
        raise ValueError(f"unknown ridge method '{method}' (sato | frangi | meijering)")
    return _norm01(resp)


def _line_footprint(length: int, angle_deg: float) -> np.ndarray:
    """A 1-px-thick line structuring element of a given length and orientation."""
    theta = np.deg2rad(angle_deg)
    half = length // 2
    ts = np.arange(-half, half + 1)
    rr = np.round(ts * np.sin(theta)).astype(int)
    cc = np.round(ts * np.cos(theta)).astype(int)
    r0, c0 = rr.min(), cc.min()
    se = np.zeros((rr.max() - r0 + 1, cc.max() - c0 + 1), dtype=bool)
    se[rr - r0, cc - c0] = True
    return se


def oriented_tophat(image: np.ndarray, length: int = 21, n_angles: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Oriented black-top-hat bank: max response over line SEs + argmax orientation map (radians)."""
    responses = []
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    for a in angles:
        responses.append(morphology.black_tophat(image, footprint=_line_footprint(length, a)))
    stack = np.stack(responses)
    best = stack.argmax(axis=0)
    resp = stack.max(axis=0)
    return _norm01(resp), np.deg2rad(angles)[best]


def matched_filter(image: np.ndarray, width_px: float = 3.0, length: int = 15, n_angles: int = 12) -> np.ndarray:
    """Zero-mean inverted-Gaussian line kernels (the classic matched-filter bank), max over angles."""
    sigma = max(width_px / 2.355, 0.6)  # FWHM to sigma
    half = length // 2
    ys, xs = np.mgrid[-half : half + 1, -half : half + 1]
    best = np.full(image.shape, -np.inf, dtype=np.float64)
    for a in np.linspace(0.0, np.pi, n_angles, endpoint=False):
        u = xs * np.cos(a) + ys * np.sin(a)   # along the line
        v = -xs * np.sin(a) + ys * np.cos(a)  # across the line
        kernel = np.where(np.abs(u) <= half, np.exp(-0.5 * (v / sigma) ** 2), 0.0)
        kernel = -(kernel - kernel.mean())    # dark line -> positive response, zero-mean
        best = np.maximum(best, ndi.convolve(image.astype(np.float64), kernel, mode="nearest"))
    return _norm01(best)


# ---------------------------------------------------------------------------
# S4 binarize
# ---------------------------------------------------------------------------

def binarize_otsu_raw(image: np.ndarray) -> np.ndarray:
    """The L0 floor: global Otsu on the RAW image, dark side positive."""
    t = filters.threshold_otsu(image)
    return image < t


def binarize_sauvola(image: np.ndarray, window_size: int = 31, k: float = 0.25) -> np.ndarray:
    """Local Sauvola on the (flattened) image; dark-crack convention (below threshold = positive)."""
    t = filters.threshold_sauvola(image, window_size=window_size, k=k)
    return image < t


def binarize_hysteresis(response: np.ndarray, low_pct: float = 92.0, high_pct: float = 98.5) -> np.ndarray:
    """Percentile hysteresis on a response map (the default from L2 up)."""
    low = np.percentile(response, low_pct)
    high = np.percentile(response, high_pct)
    if high <= low:
        high = low + 1e-6
    return filters.apply_hysteresis_threshold(response, low, high)


# ---------------------------------------------------------------------------
# S5 structure filter
# ---------------------------------------------------------------------------

@dataclass
class ShapeRules:
    min_area: int = 20
    min_eccentricity: float = 0.85   # cracks are elongated; blobs are not (applied to SMALL components)
    max_extent: float = 0.6          # thin structures fill little of their bbox


def filter_structures(mask: np.ndarray, rules: ShapeRules = ShapeRules()) -> np.ndarray:
    """Connected-component shape rules: drop small round/dense blobs, keep elongated structures.

    Deliberately permissive on LARGE components (a long fragmented crack region survives even when
    curved); the elongation and extent rules bite only where speckle lives (small components).
    """
    labels = measure.label(mask, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area < rules.min_area:
            continue
        small = region.area < 8 * rules.min_area
        if small and region.eccentricity < rules.min_eccentricity:
            continue
        if small and region.extent > rules.max_extent:
            continue
        out[labels == region.label] = True
    return out


def diameter_open_response(response: np.ndarray, diameter: int = 9) -> np.ndarray:
    """Grayscale diameter opening of the response (crack-friendly despeckle before binarize)."""
    r8 = (np.clip(response, 0, 1) * 255).astype(np.uint8)
    opened = morphology.diameter_opening(r8, diameter_threshold=diameter)
    residue = r8 - opened
    # cracks are thin: they DISAPPEAR under the opening, so the residue is the crack signal
    return _norm01(residue.astype(np.float32))


# ---------------------------------------------------------------------------
# S6 link: endpoint minimal-path bridging
# ---------------------------------------------------------------------------

def bridge_endpoints(mask: np.ndarray, response: np.ndarray, max_gap: int = 20, max_pairs: int = 40) -> np.ndarray:
    """Bridge nearby skeleton endpoints along minimal-cost paths through the response.

    Cost = 1 - response (bright response is cheap). Only endpoint pairs from DIFFERENT components
    within max_gap are bridged, cheapest-first, capped at max_pairs (determinism + cost control).
    """
    from skimage.morphology import skeletonize as _skel

    skel = _skel(mask)
    k = np.ones((3, 3), dtype=int)
    k[1, 1] = 0
    ncount = ndi.convolve(skel.astype(int), k, mode="constant")
    endpoints = np.argwhere(skel & (ncount == 1))
    if len(endpoints) < 2:
        return mask
    labels = measure.label(mask, connectivity=2)
    cost = 1.0 - np.clip(response, 0.0, 1.0) + 1e-3
    out = mask.copy()
    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p, q = endpoints[i], endpoints[j]
            if labels[p[0], p[1]] == labels[q[0], q[1]]:
                continue
            d = float(np.hypot(*(p - q)))
            if d <= max_gap:
                pairs.append((d, tuple(p), tuple(q)))
    pairs.sort(key=lambda x: x[0])
    for _, p, q in pairs[:max_pairs]:
        try:
            path, _cost = route_through_array(cost, p, q, fully_connected=True, geometric=True)
        except ValueError:
            continue
        for r, c in path:
            out[r, c] = True
    return out


# ---------------------------------------------------------------------------
# L5 patch-feature classifier (LBP + GLCM + HOG -> random forest)
# ---------------------------------------------------------------------------

PATCH = 32


def patch_features(gray: np.ndarray) -> np.ndarray:
    """The dossier's shallow descriptor for one PATCH x PATCH grayscale patch in [0, 1]."""
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    lbp1 = feature.local_binary_pattern(u8, P=8, R=1, method="uniform")
    h1, _ = np.histogram(lbp1, bins=10, range=(0, 10), density=True)
    lbp2 = feature.local_binary_pattern(u8, P=16, R=2, method="uniform")
    h2, _ = np.histogram(lbp2, bins=18, range=(0, 18), density=True)
    glcm = feature.graycomatrix(u8 // 16, distances=(1, 2, 4), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), levels=16, symmetric=True, normed=True)
    props = np.concatenate([feature.graycoprops(glcm, p).ravel() for p in ("contrast", "homogeneity", "energy", "correlation")])
    hog_vec = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    return np.concatenate([h1, h2, props, hog_vec]).astype(np.float64)


def train_patch_rf(images: list[np.ndarray], masks: list[np.ndarray], seed: int = 0, stride: int = 16):
    """Train the L5 random forest on labelled patches (a patch is positive if its mask has crack px).

    Deterministic in `seed`. Small by design so the CI pipeline smoke can train on the committed
    examples; vault-scale training uses the same function over the fetched datasets.
    """
    from sklearn.ensemble import RandomForestClassifier

    X, y = [], []
    for img, msk in zip(images, masks):
        gray = to_gray_float(img)
        for r in range(0, gray.shape[0] - PATCH + 1, stride):
            for c in range(0, gray.shape[1] - PATCH + 1, stride):
                X.append(patch_features(gray[r : r + PATCH, c : c + PATCH]))
                # positive iff the patch carries a meaningful crack fraction (0.5 percent), not a stray pixel
                y.append(1 if msk[r : r + PATCH, c : c + PATCH].mean() >= 0.005 else 0)
    if len(set(y)) < 2:
        return None  # single-class training data: L5 is honestly unavailable
    clf = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1, min_samples_leaf=2)
    clf.fit(np.asarray(X), np.asarray(y))
    return clf


def patch_probability_map(gray: np.ndarray, clf, stride: int = 16) -> np.ndarray:
    """Dense-ish crack probability by sliding the patch classifier and bilinearly upsampling."""
    rows = range(0, gray.shape[0] - PATCH + 1, stride)
    cols = range(0, gray.shape[1] - PATCH + 1, stride)
    grid = np.zeros((len(list(rows)), len(list(cols))), dtype=np.float64)
    pos_col = int(np.argmax(clf.classes_ == 1))
    for i, r in enumerate(range(0, gray.shape[0] - PATCH + 1, stride)):
        feats = [patch_features(gray[r : r + PATCH, c : c + PATCH]) for c in range(0, gray.shape[1] - PATCH + 1, stride)]
        grid[i, :] = clf.predict_proba(np.asarray(feats))[:, pos_col]
    zoom = (gray.shape[0] / grid.shape[0], gray.shape[1] / grid.shape[1])
    return np.clip(ndi.zoom(grid, zoom, order=1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")


@dataclass
class LadderParams:
    ridge_method: str = "sato"
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    flatten_radius: int = 21
    sauvola_window: int = 31
    sauvola_k: float = 0.25
    tophat_length: int = 21
    hyst_low_pct: float = 92.0
    hyst_high_pct: float = 98.5
    max_gap: int = 20
    rules: ShapeRules = field(default_factory=ShapeRules)


@dataclass
class LevelResult:
    level: str
    response: np.ndarray  # float32 [0, 1] (the S3 response driving the level; L0/L1 use inverted image)
    mask: np.ndarray      # bool
    notes: list[str] = field(default_factory=list)


def run_level(image: np.ndarray, level: str, params: LadderParams = LadderParams(), rf=None) -> LevelResult:
    """Run one ladder level end to end on a contract-legal image. Deterministic, stateless."""
    gray = to_gray_float(image)

    if level == "L0":
        mask = binarize_otsu_raw(gray)
        return LevelResult("L0", _norm01(1.0 - gray), mask, ["global Otsu on raw; the honest floor"])

    flat = flatten_median(gray, radius=params.flatten_radius)

    if level == "L1":
        mask = binarize_sauvola(flat, window_size=params.sauvola_window, k=params.sauvola_k)
        mask = morphology.remove_small_objects(mask, max_size=params.rules.min_area - 1)
        mask = filter_structures(mask, params.rules)
        return LevelResult("L1", _norm01(1.0 - flat), mask, ["flatten + Sauvola + shape rules"])

    if level == "L2":
        resp, _ori = oriented_tophat(flat, length=params.tophat_length)
        mask = binarize_hysteresis(resp, params.hyst_low_pct, params.hyst_high_pct)
        mask = filter_structures(mask, params.rules)
        return LevelResult("L2", resp, mask, ["oriented black-top-hat bank + hysteresis + shape rules"])

    # L3+: denoise + ridge (the diameter opening stays available as an S5 option but is NOT in the
    # default chain: on real low-contrast patches it wipes the crack response, measured on BCL)
    den = denoise_nlm(flat)
    resp = ridge_response(den, method=params.ridge_method, sigmas=params.sigmas)
    if level == "L3":
        mask = binarize_hysteresis(resp, params.hyst_low_pct, params.hyst_high_pct)
        mask = filter_structures(mask, params.rules)
        return LevelResult("L3", resp, mask, [f"NLM + {params.ridge_method} ridge + hysteresis + shape rules"])

    if level == "L4":
        mask = binarize_hysteresis(resp, params.hyst_low_pct, params.hyst_high_pct)
        mask = filter_structures(mask, params.rules)
        mask = bridge_endpoints(mask, resp, max_gap=params.max_gap)
        return LevelResult("L4", resp, mask, ["L3 + endpoint minimal-path bridging (feeds geometry)"])

    if level == "L5":
        if rf is None:
            raise ValueError("L5 requires a trained patch classifier (rf)")
        prob = patch_probability_map(gray, rf)
        fused = _norm01(resp * (0.25 + 0.75 * prob))
        mask = binarize_hysteresis(fused, params.hyst_low_pct, params.hyst_high_pct)
        mask = filter_structures(mask, params.rules)
        mask = bridge_endpoints(mask, fused, max_gap=params.max_gap)
        return LevelResult("L5", fused, mask, ["L4 response fused multiplicatively with the LBP+GLCM+HOG random-forest probability"])

    raise ValueError(f"unknown ladder level '{level}' (expected one of {LEVELS})")


def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)
