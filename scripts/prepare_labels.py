#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export YOLO boxes and wire masks. Default: train+val only, never test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, parse_case_ids, scan_dataset
from pcbnet.gt_adapter import load_category_map
from pcbnet.label_export import (
    TRAIN_SPLITS,
    export_split,
    load_coarse_map,
    load_yolo_classes,
    verify_labels,
    write_dataset_yaml,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,val", help="comma list; test is blocked unless --include-test")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--all-in-train", action="store_true",
                        help="submission mode: export all 64 cases and train on train+val+test (val kept for monitoring only)")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    cases = scan_dataset(cfg["dataset_root"])
    splits = load_or_create_splits(
        project_root() / "data",
        cases,
        seed=cfg.get("split_seed", 42),
        ratio=tuple(cfg.get("split_ratio", [0.8, 0.1, 0.1])),
    )
    wanted = parse_case_ids(args.splits)
    if args.all_in_train:
        args.include_test = True
        wanted = ["train", "val", "test"]
    if "test" in wanted and not args.include_test:
        print("refusing to export test labels for training; pass --include-test if you really need it")
        wanted = [s for s in wanted if s != "test"]
    if "all" in wanted and not args.include_test:
        print("refusing --splits all; exporting train,val only")
        wanted = list(TRAIN_SPLITS)
    names = load_yolo_classes()
    cmap = load_category_map()
    coarse = load_coarse_map()
    out = project_root() / cfg.get("output_root", "outputs") / "labels"
    text_bbox_root = project_root() / cfg.get("text_bbox_root", "data/Dataset200/text_bbox")
    index = index_cases(cases)
    if args.verify_only:
        errors = verify_labels(out, wanted, names)
        print(f"verify errors={len(errors)}")
        for e in errors[:20]:
            print(" ", e)
        return 0 if not errors else 1
    manifest = {"splits": {}, "classes": names}
    for split in wanted:
        ids = splits.get(split, [])
        print(f"[export] {split} n={len(ids)}")
        manifest["splits"][split] = export_split(
            index, ids, split, out, names, cmap, coarse, text_bbox_root=text_bbox_root
        )
        for row in manifest["splits"][split]["cases"]:
            print(
                f"  {row.get('case_id')} {row.get('status')} boxes={row.get('boxes')} "
                f"text={row.get('text_boxes')} pin={row.get('pin_boxes')} mask_fg={row.get('mask_fg')}"
            )
    train_dirs = ["train", "val", "test"] if args.all_in_train else None
    yaml_path = write_dataset_yaml(
        out, names, has_val="val" in wanted and bool(splits.get("val")), train_dirs=train_dirs
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = verify_labels(out, wanted, names)
    print(f"wrote {yaml_path}")
    print(f"verify errors={len(errors)}")
    for e in errors[:20]:
        print(" ", e)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
