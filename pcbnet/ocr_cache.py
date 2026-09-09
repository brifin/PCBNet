"""Disk-based OCR/vision caches to decouple torch (YOLO) from paddle (OCR).

torch's cu126 build and paddlepaddle-gpu both ship cudnn_cnn64_9.dll; importing
both in one process triggers a Windows DLL clash (WinError 127). We therefore run
YOLO detection and PaddleOCR in separate processes and exchange data via JSON:

  pass 1 (torch)  -> vision_cache: detected text boxes per image
  pass 2 (paddle) -> ocr_cache:    OCR content for those boxes
  pass 3 (torch)  -> eval reads ocr_cache, never importing paddle
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _key(image_path: str | Path) -> str:
    return hashlib.sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()[:16]


def vision_cache_file(out_root: str | Path, image_path: str | Path) -> Path:
    return Path(out_root) / "vision_cache" / f"{_key(image_path)}.json"


def ocr_cache_file(out_root: str | Path, image_path: str | Path) -> Path:
    return Path(out_root) / "ocr_cache" / f"{_key(image_path)}.json"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
