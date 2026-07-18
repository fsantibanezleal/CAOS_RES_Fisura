"""Stage 4, infer: run the ladder L0-L5 over every sample (S4-S6 of the stage graph)."""
from __future__ import annotations

from ..io.image_contract import ImageSample
from ..model.classical import LEVELS, LadderParams, LevelResult, run_level


def run(samples: list[ImageSample], params: LadderParams, rf) -> list[dict[str, LevelResult]]:
    """Per sample: {level: LevelResult}. L5 is skipped (absent) when no classifier is available."""
    out: list[dict[str, LevelResult]] = []
    for s in samples:
        per_level: dict[str, LevelResult] = {}
        for level in LEVELS:
            if level == "L5" and rf is None:
                continue
            per_level[level] = run_level(s.image, level, params=params, rf=rf)
        out.append(per_level)
    return out
