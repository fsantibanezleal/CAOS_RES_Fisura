"""Bake the enrichment artifacts for the App workbench (research shortlist
`wip/fisura/research/app-enrichment-viz-methods-2026-07-20.md`, items 1-6 + 10).

Per committed example image, from the ground-truth mask (or, when GT is absent, the best learned
mask) and the per-model masks already baked, compute the rich quantitative views the workbench
needs, as compact committed JSON (git-as-data, no new heavy deps, all scikit-image/scipy):

  QUANTIFICATION (from the GT/reference crack):
    - skeleton graph: nodes (endpoints + junctions, by degree) + edges (polylines) for the overlay
    - width profile w(s): arc-length-ordered dual widths (inscribed-circle EDT + orthogonal profile)
    - orientation rose: length-weighted angular histogram (36 bins over 0..180 deg, mirrored)

  METRICS (per learned model vs GT, on THIS image):
    - tolerance sweep: F1 at tolerance 0..8 px (the crack-segmentation protocol axis)
    - precision-recall style summary at 2px and 5px (already in the artifact; collected here per model)
    - confusion field: TP/FP/FN counts at a chosen tolerance (2px) for the overlay legend
    - ensemble disagreement: per-pixel stdev across the learned masks -> a scalar uncertainty summary

Writes data/derived/enrichment/<sample_id>.json + index.json. Determin­istic; scikit-image only.

    python -m fisuralab.cases.bake_enrichment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..core.artifact import rle_decode
from ..model.geometry import _local_tangents, skeleton_and_edt

REPO_ROOT = Path(__file__).resolve().parents[3]
BCL = REPO_ROOT / "data" / "derived" / "bcl_examples" / "artifact.json"
LEARNED = REPO_ROOT / "data" / "derived" / "learned_on_examples" / "artifact.json"
OUT = REPO_ROOT / "data" / "derived" / "enrichment"

LEARNED_MODELS = ["segformer_b2", "deeplabv3p_r18", "unet_r18", "hrsegnet_b16", "dinov2s14_linear"]


def _decode(rle) -> np.ndarray:
    return rle_decode(rle).astype(bool)


def _skeleton_graph(mask: np.ndarray) -> dict:
    """Nodes (endpoints + junctions) + edges (polylines between them) for the overlay."""
    skel, dist = skeleton_and_edt(mask, method="lee")
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    ncount = ndi.convolve(skel.astype(int), k, mode="constant")
    endpoints = skel & (ncount == 1)
    junctions = skel & (ncount >= 3)
    nodes = []
    for r, c in np.argwhere(endpoints):
        nodes.append({"x": int(c), "y": int(r), "degree": 1})
    for r, c in np.argwhere(junctions):
        nodes.append({"x": int(c), "y": int(r), "degree": int(ncount[r, c])})

    # trace edges: remove nodes, connected components of the remainder are branch interiors; approximate
    # each as a downsampled polyline of its pixels ordered by a nearest-neighbour walk from one end.
    node_mask = endpoints | junctions
    body = skel & ~ndi.binary_dilation(node_mask, iterations=1)
    lbl, n = ndi.label(body, structure=np.ones((3, 3), int))
    edges = []
    for i in range(1, n + 1):
        pix = np.argwhere(lbl == i)
        if len(pix) < 2:
            continue
        # order by a simple PCA-projection (monotone along the dominant axis) -> polyline
        arr = pix.astype(float)
        c0 = arr.mean(axis=0)
        vt = np.linalg.svd(arr - c0, full_matrices=False)[2]
        proj = (arr - c0) @ vt[0]
        order = np.argsort(proj)
        ordered = pix[order]
        step = max(1, len(ordered) // 24)  # cap polyline length
        poly = [[int(c), int(r)] for r, c in ordered[::step]]
        halfw = float(np.median([dist[r, c] for r, c in ordered]))
        edges.append({"polyline": poly, "length_px": int(len(ordered)), "mean_halfwidth_px": round(halfw, 2)})
    return {"nodes": nodes, "edges": edges, "n_endpoints": int(endpoints.sum()), "n_junctions": int(junctions.sum())}


def _width_profile(mask: np.ndarray, mm_per_px: float | None) -> dict:
    """Arc-length-ordered inscribed-circle width along the longest skeleton branch."""
    skel, dist = skeleton_and_edt(mask, method="lee")
    pts = np.argwhere(skel)
    if len(pts) < 2:
        return {"s_px": [], "w_dt_px": [], "mm_per_px": mm_per_px}
    # order the whole skeleton by a nearest-neighbour walk from the topmost point (approx arc length)
    remaining = set(map(tuple, pts))
    start = tuple(pts[np.argmin(pts[:, 0])])
    order = [start]
    remaining.discard(start)
    while remaining:
        r, c = order[-1]
        nxt = min(remaining, key=lambda p: (p[0] - r) ** 2 + (p[1] - c) ** 2)
        if (nxt[0] - r) ** 2 + (nxt[1] - c) ** 2 > 9:  # break on a jump (multi-branch)
            break
        order.append(nxt)
        remaining.discard(nxt)
    s, w = [], []
    acc = 0.0
    for i, (r, c) in enumerate(order):
        if i:
            pr, pc = order[i - 1]
            acc += float(np.hypot(r - pr, c - pc))
        s.append(round(acc, 1))
        w.append(round(2.0 * float(dist[r, c]), 2))
    # downsample to <= 80 points for a compact chart
    if len(s) > 80:
        idx = np.linspace(0, len(s) - 1, 80).astype(int)
        s = [s[i] for i in idx]
        w = [w[i] for i in idx]
    return {"s_px": s, "w_dt_px": w, "mm_per_px": mm_per_px}


def _orientation_rose(mask: np.ndarray, bins: int = 36) -> dict:
    """Length-weighted angular histogram over [0,180) deg (each skeleton pixel weighted equally)."""
    skel, _ = skeleton_and_edt(mask, method="lee")
    tangents = _local_tangents(skel)
    if not tangents:
        return {"bins_deg": [], "weight": []}
    angs = np.rad2deg(np.array(list(tangents.values()))) % 180.0
    hist, edges = np.histogram(angs, bins=bins, range=(0.0, 180.0))
    centers = ((edges[:-1] + edges[1:]) / 2).round(1).tolist()
    return {"bins_deg": centers, "weight": hist.astype(int).tolist()}


def _tolerance_sweep(pred: np.ndarray, gt: np.ndarray, tol_max: int = 8) -> dict:
    """Buffered F1 at tolerance 0..tol_max px: a predicted px is TP if GT within tol; recall symmetric."""
    if not gt.any():
        return {"tol_px": [], "f1": []}
    gt_dt = ndi.distance_transform_edt(~gt)
    pred_dt = ndi.distance_transform_edt(~pred)
    tols, f1s = [], []
    n_pred = int(pred.sum())
    n_gt = int(gt.sum())
    for tol in range(tol_max + 1):
        tp_p = int((pred & (gt_dt <= tol)).sum())     # predicted px near some GT
        tp_r = int((gt & (pred_dt <= tol)).sum())      # GT px near some prediction
        prec = tp_p / max(1, n_pred)
        rec = tp_r / max(1, n_gt)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        tols.append(tol)
        f1s.append(round(f1, 4))
    return {"tol_px": tols, "f1": f1s}


def _confusion(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> dict:
    gt_dt = ndi.distance_transform_edt(~gt)
    pred_dt = ndi.distance_transform_edt(~pred)
    tp = int((pred & (gt_dt <= tol)).sum())
    fp = int((pred & (gt_dt > tol)).sum())
    fn = int((gt & (pred_dt > tol)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tol_px": tol}


def main() -> None:
    bcl = json.loads(BCL.read_text(encoding="utf-8"))
    learned = {s["sample_id"]: s for s in json.loads(LEARNED.read_text(encoding="utf-8"))["samples"]}
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"schema": "fisura.enrichment/v1", "samples": []}

    for s in bcl["samples"]:
        sid = s["sample_id"]
        size = s["size"]
        gt = _decode(s["gt_rle"]) if s.get("gt_rle") else None
        ls = learned.get(sid)

        # the reference crack for quantification: GT if present, else the best learned mask (segformer)
        ref = gt
        if ref is None and ls is not None:
            ref = _decode(ls["levels"]["segformer_b2"]["mask_rle"])
        rec: dict = {"sample_id": sid, "size": size, "material": s["material"], "has_gt": gt is not None}

        if ref is not None and ref.any():
            rec["skeleton"] = _skeleton_graph(ref)
            rec["width_profile"] = _width_profile(ref, s.get("mm_per_px"))
            rec["rose"] = _orientation_rose(ref)

        # per-model metrics vs GT
        models = {}
        if gt is not None and ls is not None:
            stack = []
            for m in LEARNED_MODELS:
                if m not in ls["levels"]:
                    continue
                pred = _decode(ls["levels"][m]["mask_rle"])
                stack.append(pred.astype(np.float32))
                models[m] = {
                    "sweep": _tolerance_sweep(pred, gt),
                    "confusion": _confusion(pred, gt),
                    "f1_2px": ls["levels"][m]["segmentation"]["tol2px"]["f1"] if ls["levels"][m].get("segmentation") else None,
                    "f1_5px": ls["levels"][m]["segmentation"]["tol5px"]["f1"] if ls["levels"][m].get("segmentation") else None,
                }
            # ensemble disagreement: mean per-pixel stdev across the learned masks
            if len(stack) >= 2:
                arr = np.stack(stack, axis=0)
                std = arr.std(axis=0)
                rec["uncertainty"] = {
                    "mean_std": round(float(std.mean()), 4),
                    "disagree_px": int((std > 0.1).sum()),
                    "n_models": len(stack),
                }
        rec["models"] = models

        (OUT / f"{sid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        index["samples"].append(sid)
        nnodes = len(rec.get("skeleton", {}).get("nodes", []))
        print(f"  {sid}: skeleton {nnodes} nodes, {len(models)} model-metrics, gt={gt is not None}")

    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
