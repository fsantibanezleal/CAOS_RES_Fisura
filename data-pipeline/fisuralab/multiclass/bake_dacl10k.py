"""Bake the committed dacl10k artifacts for the App multi-class case (BL-010).

License discipline (dossier section 1.2): dacl10k is CC BY-NC 4.0, so the raw images are NOT
redistributed. What ships is: (a) the metrics record (per-class IoU + mIoU vs the 0.424 baseline),
and (b) a few SMALL qualitative overlays rendered as class-coloured masks on DOWNSCALED crops, which
are transformative low-resolution derivatives used for academic illustration under the citation
requirement. The full-resolution imagery + trained weights stay local under the vault.

Reads the trained checkpoint (FISURA_DATA_ROOT/derived/multiclass/<arch>.pt), runs it on a handful of
validation images, and writes:
  data/derived/multiclass/dacl10k.json          (metrics + per-class IoU + the class palette + samples)
  data/derived/multiclass/overlays/<id>.png     (class-coloured prediction on a 384px crop)

    python -m fisuralab.multiclass.bake_dacl10k --arch effb4 --n-samples 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..io.image_formats import read_image
from ..learned.shards import data_root
from .dacl10k import CLASSES, DAMAGE_CLASSES, N_CLASSES, list_split, rasterize
from .train_dacl10k import _build

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "data" / "derived" / "multiclass"
VIEW = 384  # downscaled overlay size (license-safe low-res derivative)

# a distinct colour per class (19); damage warm, object cool. Kept in sync with the frontend palette.
PALETTE = [
    [230, 57, 70], [244, 106, 60], [255, 159, 28], [255, 200, 87], [214, 40, 40],
    [156, 102, 68], [190, 120, 90], [138, 80, 60], [233, 79, 55], [204, 51, 139],
    [150, 111, 51], [120, 90, 70], [176, 48, 96],
    [46, 134, 193], [52, 152, 219], [26, 188, 156], [72, 201, 176], [93, 173, 226], [40, 116, 166],
]


def _norm(img: np.ndarray) -> np.ndarray:
    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD = np.array([0.229, 0.224, 0.225], np.float32)
    g = img[..., :3].astype(np.float32) / 255.0
    return ((g - _MEAN) / _STD).transpose(2, 0, 1)


def _overlay_png(base_rgb: np.ndarray, mask: np.ndarray) -> bytes:
    """Blend the class-coloured multi-label mask over the (downscaled) base image; return PNG bytes."""
    import imageio.v3 as iio  # noqa: PLC0415

    out = base_rgb.astype(np.float32).copy()
    for c in range(N_CLASSES):
        if not mask[c].any():
            continue
        col = np.array(PALETTE[c], np.float32)
        sel = mask[c] > 0
        out[sel] = 0.45 * out[sel] + 0.55 * col
    import io  # noqa: PLC0415

    buf = io.BytesIO()
    iio.imwrite(buf, np.clip(out, 0, 255).astype(np.uint8), extension=".png")
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.multiclass.bake_dacl10k")
    ap.add_argument("--arch", default="effb4")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from skimage.transform import resize as _resize  # noqa: PLC0415

    res_path = data_root() / "derived" / "multiclass" / f"{args.arch}_results.json"
    ckpt = data_root() / "derived" / "multiclass" / f"{args.arch}.pt"
    if not res_path.exists() or not ckpt.exists():
        raise SystemExit(f"train first: missing {res_path} or {ckpt}")
    results = json.loads(res_path.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build(args.arch).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    val = list_split("validation")
    rng = np.random.default_rng(args.seed)
    pick = [val[i] for i in rng.choice(len(val), size=min(args.n_samples, len(val)), replace=False)]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    samples = []
    for img_p, ann_p in pick:
        img = read_image(img_p)
        if img.ndim == 2:
            img = np.stack([img, img, img], -1)
        H, W = img.shape[:2]
        # infer at 512 then upsample logits to native for the overlay; downscale to VIEW for shipping
        from skimage.transform import resize as rz  # noqa: PLC0415

        inp = rz(img[..., :3], (512, 512), order=1, preserve_range=True).astype(np.float32)
        x = torch.from_numpy(_norm(inp)[None]).to(device)
        with torch.no_grad():
            logits = net(x)[0].cpu().numpy()
        pred = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.uint8)  # (C,512,512)
        gt = rasterize(ann_p, out_hw=(512, 512))
        # per-image IoU (present classes)
        inter = (pred & gt).reshape(N_CLASSES, -1).sum(1)
        union = ((pred | gt).reshape(N_CLASSES, -1)).sum(1)
        present = union > 0
        ious = {CLASSES[c]: round(float(inter[c] / max(1, union[c])), 3) for c in range(N_CLASSES) if present[c]}
        # ship a downscaled coloured overlay
        base = rz(img[..., :3], (VIEW, VIEW), order=1, preserve_range=True).astype(np.uint8)
        pmask = np.stack([(_resize(pred[c], (VIEW, VIEW), order=0, preserve_range=True) > 0.5).astype(np.uint8) for c in range(N_CLASSES)])
        png = _overlay_png(base, pmask)
        sid = img_p.stem
        (OUT / "overlays" / f"{sid}.png").write_bytes(png)
        samples.append({
            "id": sid,
            "overlay": f"multiclass/overlays/{sid}.png",
            "present_classes": [CLASSES[c] for c in range(N_CLASSES) if present[c]],
            "per_class_iou": ious,
        })
        print(f"  {sid}: {len(ious)} classes present")

    rec = {
        "schema": "fisura.multiclass/v1",
        "dataset": "dacl10k v2 (CC BY-NC 4.0; images local, metrics + low-res overlays only)",
        "arch": args.arch,
        "classes": CLASSES,
        "damage_classes": sorted(DAMAGE_CLASSES),
        "palette": PALETTE,
        "val_mIoU": results["val_mIoU"],
        # the metric DEFINITION travels with the number. An earlier build averaged a per-batch IoU
        # and quoted it against this pooled baseline, which was not a like-for-like comparison.
        "val_mIoU_protocol": results.get(
            "val_mIoU_protocol",
            "pooled (dataset-level) macro IoU over classes present in the val ground truth",
        ),
        "pos_weight": results.get("pos_weight"),
        "baseline_mIoU": results["baseline_mIoU"],
        "baseline_source": results["baseline_source"],
        "per_class_IoU": results.get("per_class_IoU", {}),
        "n_train": results["n_train"],
        "n_val": results["n_val"],
        "epochs": results["epochs"],
        "samples": samples,
    }
    # the honest split the mean hides: how many classes the model actually learned at all
    pcv = [v for v in rec["per_class_IoU"].values()]
    nz = [v for v in pcv if v > 0]
    rec["classes_present"] = len(pcv)
    rec["classes_nonzero"] = len(nz)
    rec["mean_IoU_over_nonzero"] = round(sum(nz) / len(nz), 4) if nz else 0.0
    (OUT / "dacl10k.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"val mIoU {rec['val_mIoU']} vs baseline {rec['baseline_mIoU']}")
    print(f"-> {OUT / 'dacl10k.json'}")


if __name__ == "__main__":
    main()
