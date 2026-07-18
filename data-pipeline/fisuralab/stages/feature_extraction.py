"""Stage 2, feature_extraction: the S3 enhancement responses the ladder levels consume.

Kept as a named stage (frozen archetype names) that precomputes the shared expensive responses per
sample; ``run_level`` in the model recomputes cheap ones internally, so this stage's product is the
per-sample RIDGE response (the L3+ backbone) plus the oriented top-hat response (L2), both of which
also feed the export overlays.
"""
from __future__ import annotations

import numpy as np

from ..io.image_contract import ImageSample
from ..model.classical import (
    LadderParams,
    denoise_nlm,
    flatten_median,
    oriented_tophat,
    ridge_response,
    to_gray_float,
)


def run(samples: list[ImageSample], params: LadderParams) -> list[dict]:
    """Per sample: {'gray', 'flat', 'ridge', 'tophat'} float32 maps (responses in [0, 1])."""
    out: list[dict] = []
    for s in samples:
        gray = to_gray_float(s.image)
        flat = flatten_median(gray, radius=params.flatten_radius)
        den = denoise_nlm(flat)
        ridge = ridge_response(den, method=params.ridge_method, sigmas=params.sigmas)
        tophat, _ = oriented_tophat(flat, length=params.tophat_length)
        out.append({"gray": gray, "flat": flat, "ridge": ridge, "tophat": np.asarray(tophat, dtype=np.float32)})
    return out
