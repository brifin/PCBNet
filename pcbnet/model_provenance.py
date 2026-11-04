"""Inference weight provenance guard (lightweight).

The upstream stage2 expects a verifier when ``cfg["models"]`` is present.
This repository only needs to ensure the configured checkpoints exist and are
non-empty before running; full experiment provenance is kept in
``yolo/train/input_audit.json`` and is not required at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def verify_inference_bundle(paths: Iterable[str | Path | None]) -> None:
    missing = []
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if not p.exists() or p.stat().st_size <= 0:
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("推理权重缺失或为空: " + ", ".join(missing))
