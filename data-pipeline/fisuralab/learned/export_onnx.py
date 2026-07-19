"""ONNX export for the trained ladder-A models, with hash + CPU parity verification.

Exports go to the LOCAL model vault (E:\\_Models\\fisura\\fisura-trained by default via
FISURA_MODEL_ROOT), never into git at this stage; the manifest records file, bytes and sha256 so
the artifact chain stays auditable. The live-lane unit (BL-013) later selects and commits the one
compact model the browser ships.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .training import build_model


def model_root() -> Path:
    root = os.environ.get("FISURA_MODEL_ROOT")
    if root:
        return Path(root)
    return Path("E:/_Models/fisura/fisura-trained")


def export(arch: str, ckpt: Path, out_dir: Path | None = None, opset: int = 17) -> dict:
    import torch  # noqa: PLC0415

    out_dir = out_dir or model_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(arch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.randn(1, 3, 512, 512)
    onnx_path = out_dir / f"{arch}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
        opset_version=opset,
        dynamo=False,  # the legacy exporter handles SMP graphs + dynamic axes reliably on torch 2.11
    )
    # parity: torch vs onnxruntime CPU on the same input
    import onnxruntime as ort  # noqa: PLC0415

    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(ref - got)))
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    rec = {
        "arch": arch,
        "onnx": str(onnx_path),
        "bytes": onnx_path.stat().st_size,
        "sha256": sha,
        "opset": opset,
        "parity_max_abs": max_abs,
        "parity_ok": bool(max_abs < 1e-2),
    }
    (out_dir / f"{arch}.export.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec
