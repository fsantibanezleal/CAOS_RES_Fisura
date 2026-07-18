"""Tolerance-buffered segmentation metrics, at BOTH field conventions (numpy + scipy, Pyodide-safe).

The research's central measurement lesson: crack papers report tolerance-based F-measures under
incompatible protocols, so every number this lab produces is computed at BOTH 2 px and 5 px
tolerances and carries its protocol string. Definition (the buffered-matching convention):

- precision(t) = |P intersect dilate(G, t)| / |P|
- recall(t)    = |G intersect dilate(P, t)| / |G|
- F1(t)        = harmonic mean

with P the predicted mask, G the ground truth, dilate(?, t) a euclidean dilation of radius t px.
Strict IoU (t = 0) is reported alongside. Degenerate cases follow the honest convention:
empty G and empty P is a correct rejection (P = R = F1 = 1); empty G with detections is F1 = 0.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

TOLERANCES_PX = (2, 5)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    dist = ndi.distance_transform_edt(~mask)
    return dist <= radius


def buffered_prf(pred: np.ndarray, gt: np.ndarray, tolerance_px: int) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    n_pred = int(pred.sum())
    n_gt = int(gt.sum())
    if n_gt == 0 and n_pred == 0:
        return {"tolerance_px": tolerance_px, "precision": 1.0, "recall": 1.0, "f1": 1.0, "n_pred": 0, "n_gt": 0}
    if n_gt == 0 or n_pred == 0:
        return {"tolerance_px": tolerance_px, "precision": 0.0 if n_pred else 1.0, "recall": 0.0 if n_gt else 1.0, "f1": 0.0, "n_pred": n_pred, "n_gt": n_gt}
    p = float((pred & _dilate(gt, tolerance_px)).sum()) / n_pred
    r = float((gt & _dilate(pred, tolerance_px)).sum()) / n_gt
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"tolerance_px": tolerance_px, "precision": p, "recall": r, "f1": f1, "n_pred": n_pred, "n_gt": n_gt}


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = int((pred | gt).sum())
    if union == 0:
        return 1.0
    return float((pred & gt).sum()) / union


def evaluate_mask(pred: np.ndarray, gt: np.ndarray) -> dict:
    """The full per-sample record: both tolerance conventions + strict IoU + the protocol strings."""
    out = {
        "iou_strict": iou(pred, gt),
        "protocol": "buffered P/R/F1; dilation radius = tolerance; pixel-level; no thinning/NMS",
    }
    for t in TOLERANCES_PX:
        m = buffered_prf(pred, gt, t)
        out[f"tol{t}px"] = m
    return out
