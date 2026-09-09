from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
