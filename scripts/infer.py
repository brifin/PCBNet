#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, parse_case_ids, scan_dataset
from pcbnet.pipeline import infer_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--cases", default="")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    all_cases = scan_dataset(cfg["dataset_root"])
    splits = load_or_create_splits(project_root() / "data", all_cases, seed=cfg.get("split_seed", 42), ratio=tuple(cfg.get("split_ratio", [0.8, 0.1, 0.1])))
    ids = parse_case_ids(args.cases) or splits.get(args.split, [])
    index = index_cases(all_cases)
    out_root = project_root() / cfg.get("output_root", "outputs")
    for cid in ids:
        case = index.get(cid)
        if not case:
            print(f"[skip] {cid}")
            continue
        dest = out_root / "pred" / f"{cid}.json"
        print(f"[infer] {cid} -> {dest}")
        infer_image(case.image, cfg, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
