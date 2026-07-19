"""PatchCore, reimplemented faithfully in-repo (torchvision backbone; the anomaly track's engine).

Source algorithm: Roth, Pemula, Zepeda, Scholkopf, Brox, Gehler, "Towards Total Recall in Industrial
Anomaly Detection" (PatchCore), CVPR 2022 (arXiv:2106.08265). The reference framework is anomalib
(Apache-2.0), which pins an older torch and pulls a heavy Lightning/OpenVINO tree that conflicts with
the learned track's torch 2.11; rather than fight that, this module is a clean in-repo re-expression
of the real PatchCore algorithm (torchvision WideResNet50 features + locally-aware patch pooling +
greedy-coreset memory bank + kNN anomaly scoring), the same faithful-reimplementation choice made for
HrSegNet. Not a toy substitute: it is the published algorithm.

Fit is training-free (build a memory bank from GOOD images); scoring is nearest-neighbour distance.
Torch imported lazily; GPU lane only.
"""
from __future__ import annotations

import numpy as np


def _backbone():
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    net.eval()
    feats = {}

    def hook(name):
        def _h(_m, _i, o):
            feats[name] = o
        return _h

    net.layer2.register_forward_hook(hook("l2"))
    net.layer3.register_forward_hook(hook("l3"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, feats, device


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 224


def _prep(image: np.ndarray):
    from skimage.transform import resize as _resize  # noqa: PLC0415

    g = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    if g.ndim == 2:
        g = np.stack([g, g, g], axis=-1)
    g = _resize(g, (INPUT, INPUT), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    g = (g - _MEAN) / _STD
    return np.ascontiguousarray(np.transpose(g, (2, 0, 1)))


def embed(net, feats, device, images: list[np.ndarray]):
    """Locally-aware patch embeddings: layer2 + layer3, 3x3 avg-pooled, layer3 upsampled to layer2's
    grid, channel-concatenated. Returns (N, H*W, C) on CPU as float32 (PatchCore's patch features)."""
    import torch  # noqa: PLC0415
    from torch.nn import functional as F  # noqa: PLC0415

    out = []
    bs = 8
    for i in range(0, len(images), bs):
        batch = torch.from_numpy(np.stack([_prep(im) for im in images[i : i + bs]])).to(device)
        with torch.no_grad():
            net(batch)
            l2 = F.avg_pool2d(feats["l2"], 3, 1, 1)
            l3 = F.avg_pool2d(feats["l3"], 3, 1, 1)
            l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
            emb = torch.cat([l2, l3], dim=1)  # (n, C, H, W)
            n, c, h, w = emb.shape
            emb = emb.permute(0, 2, 3, 1).reshape(n, h * w, c).cpu().numpy().astype(np.float32)
        out.append(emb)
    grid = feats["l2"].shape[-2:]
    return np.concatenate(out, axis=0), (int(grid[0]), int(grid[1]))


def greedy_coreset(bank: np.ndarray, n_select: int, seed: int = 42, proj_dim: int = 128) -> np.ndarray:
    """Greedy k-center coreset (PatchCore's memory reduction), on GPU with a sparse random projection.

    Faithful to PatchCore: a Johnson-Lindenstrauss random projection to ``proj_dim`` makes the
    iterative farthest-point selection cheap, and the running min-distance update runs on the GPU.
    Returns indices INTO the input bank."""
    import torch  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n = bank.shape[0]
    n_select = min(n_select, n)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.from_numpy(bank).to(device)
    if bank.shape[1] > proj_dim:
        gen = torch.Generator(device=device).manual_seed(seed)
        proj = torch.randn(bank.shape[1], proj_dim, generator=gen, device=device) / np.sqrt(proj_dim)
        bp = b @ proj
    else:
        bp = b
    idx = np.empty(n_select, dtype=np.int64)
    start = int(rng.integers(0, n))
    idx[0] = start
    d = torch.cdist(bp, bp[start : start + 1]).squeeze(1)  # (n,)
    for k in range(1, n_select):
        j = int(torch.argmax(d).item())
        idx[k] = j
        dj = torch.cdist(bp, bp[j : j + 1]).squeeze(1)
        d = torch.minimum(d, dj)
    return idx


class PatchCore:
    def __init__(self, coreset_fraction: float = 0.1, coreset_cap: int = 8000, n_neighbors: int = 3, seed: int = 42):
        self.coreset_fraction = coreset_fraction
        self.coreset_cap = coreset_cap
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.memory: np.ndarray | None = None
        self._net = self._feats = self._device = None
        self.grid: tuple[int, int] | None = None

    def _ensure_backbone(self):
        if self._net is None:
            self._net, self._feats, self._device = _backbone()

    def fit(self, good_images: list[np.ndarray], bank_cap: int = 120_000) -> dict:
        self._ensure_backbone()
        emb, self.grid = embed(self._net, self._feats, self._device, good_images)
        bank = emb.reshape(-1, emb.shape[-1])  # (N*H*W, C)
        n_total = int(bank.shape[0])
        # random pre-subsample to keep the coreset (and GPU memory) bounded; the coreset then
        # selects the diverse memory from this pool (standard PatchCore practice on large banks)
        if bank.shape[0] > bank_cap:
            rng = np.random.default_rng(self.seed)
            bank = bank[rng.choice(bank.shape[0], bank_cap, replace=False)]
        n_sel = min(self.coreset_cap, max(1, int(bank.shape[0] * self.coreset_fraction)))
        sel = greedy_coreset(bank, n_sel, seed=self.seed)
        self.memory = bank[sel]
        return {"bank_patches_total": n_total, "bank_pool": int(bank.shape[0]), "memory_patches": int(self.memory.shape[0]), "feature_dim": int(bank.shape[1]), "grid": list(self.grid)}

    def score(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (image_scores (N,), anomaly_maps (N, gh, gw))."""
        assert self.memory is not None, "fit() before score()"
        self._ensure_backbone()
        emb, grid = embed(self._net, self._feats, self._device, images)
        gh, gw = grid
        img_scores = np.empty(len(images), dtype=np.float32)
        maps = np.empty((len(images), gh, gw), dtype=np.float32)
        import torch  # noqa: PLC0415

        mem = torch.from_numpy(self.memory).to(self._device)
        for i in range(emb.shape[0]):
            q = torch.from_numpy(emb[i]).to(self._device)  # (H*W, C)
            d = torch.cdist(q, mem)  # (H*W, M)
            patch_min = d.min(dim=1).values  # nearest memory distance per patch
            pm = patch_min.cpu().numpy()
            maps[i] = pm.reshape(gh, gw)
            img_scores[i] = float(pm.max())  # image anomaly score = max patch distance
        return img_scores, maps


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level AUROC (labels: 1 = anomaly/cracked, 0 = normal). Rank-based, no sklearn."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
