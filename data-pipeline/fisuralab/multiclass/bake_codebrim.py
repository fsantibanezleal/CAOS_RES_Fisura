"""Bake the committed CODEBRIM detection artifacts for the App (BL-010 detection rung).

License (dossier section 2.2): CODEBRIM is a bespoke NC license extending to trained models. So the
raw 4608x3456 images and the weights stay local; what ships is the metrics record + a few SMALL
box-overlay crops (transformative low-res derivatives for academic illustration).

Runs the trained Faster R-CNN on a handful of test images, draws the predicted boxes (class-coloured,
score-labelled) on a downscaled crop, and writes:
  data/derived/multiclass/codebrim.json        (mAP metrics + class palette + samples)
  data/derived/multiclass/overlays/cb_<id>.png (boxes on a 512px image)

    python -m fisuralab.multiclass.bake_codebrim --n-samples 4
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from ..learned.shards import data_root
from .codebrim import DEFECTS, parse_annotations, split_records
from .train_codebrim import LONG_SIDE, _build_model

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "data" / "derived" / "multiclass"
VIEW = 512
# one colour per defect (index 1..5; 0 is background), warm damage palette
PALETTE = [[230, 57, 70], [244, 106, 60], [255, 159, 28], [204, 51, 139], [156, 102, 68]]


def _draw_boxes(img_u8: np.ndarray, boxes, labels, scores, scale: float) -> bytes:
    """Draw class-coloured rectangles + a small filled label tab; return PNG bytes (no PIL fonts)."""
    import imageio.v3 as iio  # noqa: PLC0415

    out = img_u8.astype(np.uint8).copy()
    H, W = out.shape[:2]

    def rect(x0, y0, x1, y1, col, thick=3):
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        x0, x1 = max(0, x0), min(W - 1, x1)
        y0, y1 = max(0, y0), min(H - 1, y1)
        for tv in range(thick):
            if y0 + tv < H:
                out[y0 + tv, x0:x1] = col
            if y1 - tv >= 0:
                out[y1 - tv, x0:x1] = col
            if x0 + tv < W:
                out[y0:y1, x0 + tv] = col
            if x1 - tv >= 0:
                out[y0:y1, x1 - tv] = col

    for b, lbl, sc in zip(boxes, labels, scores, strict=False):
        if sc < 0.3:
            continue
        col = np.array(PALETTE[(int(lbl) - 1) % len(PALETTE)], np.uint8)
        x0, y0, x1, y1 = np.array(b) * scale
        rect(x0, y0, x1, y1, col)
        # a small filled tab at the top-left corner (colour = class), so identity reads without text
        ty0 = max(0, int(round(y0)) - 8)
        out[ty0:ty0 + 6, int(round(x0)):int(round(x0)) + 18] = col
    buf = io.BytesIO()
    iio.imwrite(buf, out, extension=".png")
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(prog="fisuralab.multiclass.bake_codebrim")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from skimage.transform import resize as rz  # noqa: PLC0415

    res_path = OUT.parent / "multiclass" / "codebrim_results.json"
    ckpt = data_root() / "derived" / "multiclass" / "codebrim_fasterrcnn.pt"
    res_local = data_root() / "derived" / "multiclass" / "codebrim_results.json"
    src = res_local if res_local.exists() else res_path
    if not src.exists() or not ckpt.exists():
        raise SystemExit(f"train CODEBRIM first: missing {src} or {ckpt}")
    results = json.loads(src.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build_model().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    recs = parse_annotations()
    test = split_records(recs, seed=42)["test"]
    rng = np.random.default_rng(args.seed)
    pick = [test[i] for i in rng.choice(len(test), size=min(args.n_samples, len(test)), replace=False)]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    samples = []
    for r in pick:
        import imageio.v3 as iio  # noqa: PLC0415

        raw = iio.imread(r["image_path"])  # plain file read from the extracted tree
        if raw.ndim == 2:
            raw = np.stack([raw, raw, raw], -1)
        raw = raw[..., :3]
        H, W = raw.shape[:2]
        scale_in = LONG_SIDE / max(H, W)
        inp = rz(raw, (int(H * scale_in), int(W * scale_in)), order=1, preserve_range=True).astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1))).to(device)
        with torch.no_grad():
            out = net([x])[0]
        # draw on a VIEW-size crop (boxes are in the LONG_SIDE space -> scale to VIEW)
        view = rz(raw, (VIEW, int(VIEW * W / H)) if H >= W else (int(VIEW * H / W), VIEW), order=1, preserve_range=True).astype(np.uint8)
        vh, vw = view.shape[:2]
        box_scale = (vh / (H * scale_in) + vw / (W * scale_in)) / 2
        png = _draw_boxes(view, out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy(), box_scale)
        sid = Path(r["image_path"]).stem
        (OUT / "overlays" / f"cb_{sid}.png").write_bytes(png)
        n_keep = int((out["scores"].cpu().numpy() >= 0.3).sum())
        samples.append({
            "id": sid, "overlay": f"multiclass/overlays/cb_{sid}.png",
            "n_pred": n_keep, "n_gt": len(r["boxes"]),
            "gt_classes": sorted({b["label"] for b in r["boxes"]}),
        })
        print(f"  cb_{sid}: {n_keep} preds, {len(r['boxes'])} gt")

    rec = {
        "schema": "fisura.codebrim/v1",
        "dataset": "CODEBRIM (custom NC license; images + weights local, metrics + low-res overlays only)",
        "arch": results["arch"], "classes": DEFECTS, "palette": PALETTE,
        "mAP_50": results["mAP_50"], "mAP_50_95": results["mAP_50_95"],
        "baseline_yolov5x_mAP50": results["baseline_yolov5x_mAP50"], "baseline_source": results["baseline_source"],
        "n_train": results["n_train"], "n_test": results["n_test"], "epochs": results["epochs"],
        "samples": samples,
    }
    (OUT / "codebrim.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"mAP@0.5 {rec['mAP_50']} vs YOLOv5x {rec['baseline_yolov5x_mAP50']}")
    print(f"-> {OUT / 'codebrim.json'}")


if __name__ == "__main__":
    main()
