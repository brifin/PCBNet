#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pass 1 (torch/YOLO): dump detected text boxes per image to vision_cache.

Runs the YOLO detector only (no paddle import) so it is safe to run in the
torch-GPU process. The saved text boxes feed the paddle-only OCR pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, parse_case_ids, scan_dataset
from pcbnet.ocr_cache import save_json, vision_cache_file
from pcbnet.stage1_vision import _read_image, _split_detections, _try_yolo


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
    weights = (cfg.get("models") or {}).get("yolo_weights") or ""
    n = 0
    for cid in ids:
        case = index.get(cid)
        if not case:
            print(f"[skip] {cid}")
            continue
        img = _read_image(case.image)
        dets = _try_yolo(img, weights)
        _comps, text_boxes, _pins, _endpoints = _split_detections(dets)
        save_json(vision_cache_file(out_root, case.image), {"text_boxes": [list(map(float, b)) for b in text_boxes]})
        n += 1
        print(f"[vision] {cid} text_boxes={len(text_boxes)}")
    print(f"done, wrote {n} vision cache files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
