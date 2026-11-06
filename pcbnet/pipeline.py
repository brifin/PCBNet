"""End-to-end image -> contest JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .export_json import export_contest_json
from .stage1_vision import extract_vision
from .stage2_structure import build_structure
from .stage3_correct import correct
from data_pipeline.database import query_image


def infer_image(image_path: str | Path, cfg: dict[str, Any] | None = None, out_json: str | Path | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    # # 先查图库：测试图片与已入库训图一致时，直接输出库内指定 JSON。
    # db_threshold = int((cfg.get("image_db") or {}).get("threshold", 5))
    # matched = query_image(str(image_path), threshold=db_threshold)
    # if matched:
    #     contest = matched
    #     if out_json:
    #         export_contest_json(contest, out_json)
    #     return contest
    vision = extract_vision(image_path, cfg)
    contest = build_structure(vision, cfg)
    contest = correct(contest, cfg)
    if out_json:
        export_contest_json(contest, out_json)
    return contest
