"""Stage 4, infer: run the ladder L0-L5 over every sample (S4-S6 of the stage graph)."""
from __future__ import annotations

from ..io.image_contract import ImageSample
from ..model.classical import LEVELS, LadderParams, LevelResult, run_level
from ..model.metrics import restrict_to_fov


def run(samples: list[ImageSample], params: LadderParams, rf) -> list[dict[str, LevelResult]]:
    """Per sample: {level: LevelResult}. L5 is skipped (absent) when no classifier is available."""
    out: list[dict[str, LevelResult]] = []
    for s in samples:
        per_level: dict[str, LevelResult] = {}
        for level in LEVELS:
            if level == "L5" and rf is None:
                continue
            res = run_level(s.image, level, params=params, rf=rf)
            # exclude anything outside the region of interest (the retina disc) HERE, at the single
            # point every consumer reads from: scoring, mask_rle and the overlays all see the masked
            # prediction, so a response on the fundus rim never reaches a metric or the screen.
            res.mask = restrict_to_fov(res.mask, s.fov)
            per_level[level] = res
        out.append(per_level)
    return out
