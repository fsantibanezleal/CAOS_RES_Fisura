"""Small JSON helpers shared by the pipeline (standard formats; image IO lives in image_formats)."""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: str | Path, payload: dict) -> int:
    """Write deterministic, diff-friendly JSON; returns the byte size on disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=False, ensure_ascii=False)
    p.write_text(text + "\n", encoding="utf-8")
    return p.stat().st_size


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
