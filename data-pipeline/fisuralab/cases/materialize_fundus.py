"""Materialize a small FIVES fundus subset into data/examples, as first-class analysis cases.

Why fundus images belong in a crack lab: they are the control that tests the lab's own thesis. If a
crack really is just a thin, branching, low-contrast curvilinear structure on a textured background,
then the whole ladder, from the 1979 global threshold to a segmenter trained only on concrete, should
behave on a retina the way it behaves on concrete. Running it and publishing what happens is a
stronger claim than asserting the analogy in prose. The rungs that transfer and the rungs that do not
are both informative: the ridge filters here were literally published for vessels, while the learned
tier has never seen a retina.

Downscaled to 768 px on purpose. FIVES originals are 2048x2048 where vessels run roughly 8-20 px
wide; at 768 they land around 3-8 px, which is the regime the crack ladder is tuned for and the
regime its 2 px / 5 px tolerances were chosen for. Comparing at native resolution would confound
"the method failed" with "the structure is a different size than anything else in the lab".

Dataset: FIVES, Jin K. et al., Scientific Data 9:475 (2022), doi 10.1038/s41597-022-01564-3,
figshare doi 10.6084/m9.figshare.19688169.v1. Licence CC BY 4.0, verified on the primary source,
which permits redistribution with attribution: that is why these four images may be committed here
while the non-commercial crack datasets stay in the vault.

    python -m fisuralab.cases.materialize_fundus --src E:/_Temp/fives --size 768
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "data" / "examples"
FUNDUS_DIR = EXAMPLES / "fundus"

CONDITION = {"N": "normal", "D": "diabetic retinopathy", "G": "glaucoma", "A": "age-related macular degeneration"}
CITATION = (
    "Jin K., Huang X., Zhou J., Li Y., Yan Y., Sun Y., Zhang Q., Wang Y., Ye J. (2022). FIVES: A Fundus "
    "Image Dataset for Artificial Intelligence based Vessel Segmentation. Scientific Data 9, 475. "
    "doi:10.1038/s41597-022-01564-3. Data: figshare doi:10.6084/m9.figshare.19688169.v1, CC BY 4.0."
)
URL = "https://doi.org/10.1038/s41597-022-01564-3"


def _resize(a: np.ndarray, size: int, order: int) -> np.ndarray:
    from skimage.transform import resize as rz  # noqa: PLC0415

    out = rz(a, (size, size) + a.shape[2:], order=order, preserve_range=True, anti_aliasing=order > 0)
    return out


def _disc_and_fov(rgb: np.ndarray, erode_frac: float = 0.04):
    """Return (disc, fov): the full retina disc, and the disc eroded inward for the analysis ROI.

    A fundus image is a bright retina disc on a black surround, so the disc RIM is a strong closed
    curvilinear edge that every crack detector fires on. Two separate boundaries are needed, and
    conflating them was the earlier bug. The DISC is where the image is filled (outside it goes to a
    flat colour), so the retina-to-fill transition sits at the disc edge. The FOV is the disc eroded
    inward, and analysis is restricted to it. Because the FOV boundary is well INSIDE the disc edge,
    the transition (and the band of rim firing around it) falls OUTSIDE the FOV and is excluded, rather
    than sitting exactly on the FOV boundary where masking would still keep it.

    The disc is found by Otsu on the luminance (a bright optic disc pushes a relative threshold high
    enough to drop a dim retina, so Otsu, capped low, is used, with a near-black fallback), keeping the
    largest connected component and filling holes.
    """
    from scipy import ndimage as ndi  # noqa: PLC0415
    from skimage.filters import threshold_otsu  # noqa: PLC0415

    lum = rgb.mean(axis=2)
    try:
        thr = float(threshold_otsu(lum))
    except ValueError:
        thr = 20.0
    thr = min(thr, 40.0)
    m = lum > thr
    if m.mean() < 0.35:                          # fallback: the surround is genuinely near-black
        m = lum > 18.0
    lbl, n = ndi.label(m)
    if n > 1:                                     # keep the largest blob = the disc, drop stray flecks
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        m = lbl == (1 + int(np.argmax(sizes)))
    disc = ndi.binary_fill_holes(m)
    r = max(1, int(erode_frac * rgb.shape[1]))
    fov = ndi.binary_erosion(disc, iterations=r)
    return disc, fov


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.cases.materialize_fundus")
    ap.add_argument("--src", default="E:/_Temp/fives", help="folder holding orig/ and gt/ subfolders")
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()

    import imageio.v3 as iio  # noqa: PLC0415

    src = Path(args.src)
    orig_dir, gt_dir = src / "orig", src / "gt"
    if not orig_dir.exists():
        raise SystemExit(f"no extracted originals at {orig_dir}")

    manifest_path = EXAMPLES / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in entries if e.get("source") != "fives"]  # idempotent

    FUNDUS_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for img_p in sorted(orig_dir.glob("*.png")):
        stem = img_p.stem                       # e.g. 100_D
        cond_key = stem.split("_")[-1]
        gt_p = gt_dir / img_p.name
        if not gt_p.exists():
            print(f"  skip {stem}: no ground truth")
            continue

        img = iio.imread(img_p)
        if img.ndim == 3:
            img = img[..., :3]
        gt = iio.imread(gt_p)
        if gt.ndim == 3:
            gt = gt[..., 0]

        img_s = np.clip(_resize(img.astype(np.float32), args.size, 1), 0, 255).astype(np.float32)
        # nearest for the mask: a vessel is a few pixels wide after downscaling and interpolating a
        # binary annotation would thin or erase the finest branches before any method has seen them
        gt_s = (_resize((gt > 127).astype(np.float32), args.size, 0) > 0.5)

        # The FOV mask is the retina disc, eroded off the rim. It is SAVED and travels with the sample,
        # so every downstream method intersects its prediction with it before scoring, storing or
        # drawing: a response on the retina-to-surround edge is then excluded from the analysis rather
        # than counted as a crack (filling the outside with a colour, as an earlier version did, does
        # not help, since the detectors still fire on the new retina-to-fill transition and that firing
        # was still being scored). Flattening the outside to the median retina colour is kept only so
        # the displayed image has no harsh rim; it is cosmetic now, not the mechanism.
        disc, fov = _disc_and_fov(img_s)
        # fill outside the FULL disc: the transition is now at the disc edge, which is 4 percent
        # OUTSIDE the FOV, so rim firing lands outside the ROI and is excluded rather than kept on the
        # boundary
        med = np.median(img_s[disc], axis=0) if disc.any() else np.array([0, 0, 0], np.float32)
        img_s[~disc] = med
        gt_s = gt_s & fov
        fov_frac = float(fov.mean())
        img_s = np.clip(img_s, 0, 255).astype(np.uint8)
        gt_s = gt_s.astype(np.uint8) * 255

        sid = f"fives-{stem.lower()}"
        img_rel = f"fundus/{sid}.png"
        mask_rel = f"fundus/{sid}_mask.png"
        fov_rel = f"fundus/{sid}_fov.png"
        iio.imwrite(EXAMPLES / img_rel, img_s, extension=".png")
        iio.imwrite(EXAMPLES / mask_rel, gt_s, extension=".png")
        iio.imwrite(EXAMPLES / fov_rel, fov.astype(np.uint8) * 255, extension=".png")

        entries.append({
            "sample_id": sid,
            "file": img_rel,
            "mask": mask_rel,
            "fov": fov_rel,
            "material": "retina",
            "source": "fives",
            "license_tag": "cc-by",
            "mm_per_px": None,
            "url": URL,
            "citation": CITATION + f" Condition: {CONDITION.get(cond_key, cond_key)}. "
                                    f"Downscaled from 2048 to {args.size} px for this lab.",
            "label": "cracked",   # the lab's binary vocabulary: 'has the curvilinear structure'
        })
        added += 1
        print(f"  {sid}: {img_s.shape} + mask, {CONDITION.get(cond_key, cond_key)}, "
              f"{100.0 * (gt_s > 0).mean():.2f}% vessel pixels, FOV {100.0 * fov_frac:.1f}% of frame")

    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"-> {added} fundus examples; manifest now holds {len(entries)} records")


if __name__ == "__main__":
    main()
