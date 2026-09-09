#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pass 2 (paddle/OCR): crop each cached text box and OCR it on GPU.

This process imports paddle ONLY (never torch) to avoid the cudnn DLL clash.
It reads text boxes from vision_cache and writes OCR content to ocr_cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, parse_case_ids, scan_dataset
from pcbnet.ocr_cache import load_json, ocr_cache_file, save_json, vision_cache_file
# NOTE: stage1_vision imports cv2/numpy at module load but torch/ultralytics only
# lazily inside _try_yolo, so importing these OCR helpers does not pull in torch.
from pcbnet.stage1_vision import _ocr_on_boxes, _ocr_texts, _maybe_resize, _read_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--cases", default="")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    all_cases = scan_dataset(cfg["dataset_root"])
    splits = load_or_create_splits(
        project_root() / "data", all_cases,
        seed=cfg.get("split_seed", 42), ratio=tuple(cfg.get("split_ratio", [0.8, 0.1, 0.1])),
    )
    ids = parse_case_ids(args.cases) or splits.get(args.split, [])
    index = index_cases(all_cases)
    out_root = project_root() / cfg.get("output_root", "outputs")
    ocr_cfg = cfg.get("ocr") or {}
    lang = str(ocr_cfg.get("lang") or "en")
    max_side = int((cfg.get("vision") or {}).get("max_side", 1600))
    n = 0
    for cid in ids:
        case = index.get(cid)
        if not case:
            print(f"[skip] {cid}")
            continue
        img = _read_image(case.image)
        h, w = img.shape[:2]
        vc = load_json(vision_cache_file(out_root, case.image)) or {}
        boxes = vc.get("text_boxes") or []
        if boxes:
            texts = _ocr_on_boxes(img, boxes, lang=lang)
        else:
            ocr_img, _ = _maybe_resize(img, max_side)
            texts = _ocr_texts(ocr_img, enabled=True, lang=lang)
            if ocr_img.shape[0] != h or ocr_img.shape[1] != w:
                sx, sy = w / ocr_img.shape[1], h / ocr_img.shape[0]
                for t in texts:
                    t.bbox = [t.bbox[0] * sx, t.bbox[1] * sy, t.bbox[2] * sx, t.bbox[3] * sy]
        payload = [{"bbox": [float(v) for v in t.bbox], "content": t.content, "score": float(t.score)} for t in texts]
        save_json(ocr_cache_file(out_root, case.image), payload)
        n += 1
        print(f"[ocr] {cid} boxes={len(boxes)} texts={len(payload)}")
    print(f"done, wrote {n} ocr cache files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
