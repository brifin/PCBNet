from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # 相对路径相对仓库根解析（dataset_root / test_root / 权重），避免依赖运行 cwd。
    root = project_root()
    for key in ("dataset_root", "test_root"):
        v = cfg.get(key)
        if v and not Path(v).is_absolute():
            cfg[key] = str((root / v).resolve())
    models = cfg.get("models") or {}
    for k in ("yolo_weights", "unet_weights"):
        w = models.get(k)
        if w and not Path(w).is_absolute():
            models[k] = str((root / w).resolve())
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
