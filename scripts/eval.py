#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, scan_dataset
from pcbnet.export_json import export_contest_json
from pcbnet.gt_adapter import cvjson_to_contest
from pcbnet.pipeline import infer_image
from pcbnet.schema import load_json
from pcbnet.scorer import score_batch

from tqdm import tqdm

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--self-check", action="store_true", help="score GT against itself")
    parser.add_argument("--reuse-pred", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    # test 用赛题 10_GTcase（独立目录），train/val 用 200_train_cases 的划分
    use_test_root = args.split == "test" and bool(cfg.get("test_root"))
    all_cases = scan_dataset(cfg.get("test_root") if use_test_root else cfg["dataset_root"])
    if use_test_root:
        ids = [c.case_id for c in all_cases]
    else:
        splits = load_or_create_splits(
            project_root() / "data",
            all_cases,
            seed=cfg.get("split_seed", 42),
            ratio=tuple(cfg.get("split_ratio", [0.8, 0.1, 0.1])),
        )
        ids = splits.get(args.split, [])
    index = index_cases(all_cases)
    out_root = project_root() / cfg.get("output_root", "outputs")
    items = []
    for cid in tqdm(ids):
        case = index.get(cid)
        if not case:
            continue
        gt = cvjson_to_contest(case.cvjson)
        export_contest_json(gt, out_root / "gt" / f"{cid}.json")
        if args.self_check:
            pred = gt
        else:
            pred_path = out_root / "pred" / cid / f"{cid}.json"
            if args.reuse_pred and pred_path.exists():
                pred = load_json(pred_path)
            else:
                pred = infer_image(case.image, cfg, pred_path)
        items.append((cid, pred, gt))
        print(f"[eval] {cid}")
    thr = cfg.get("thresholds") or {}
    kw = dict(
        bbox_thr=float(thr.get("bbox_px", 20)),
        pin_thr=float(thr.get("pin_px", 5)),
        line_thr=float(thr.get("line_px", 5)),
        weights=cfg.get("weights"),
    )
    report = score_batch(items, **kw)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Score report ({args.split}{' / GT self-check' if args.self_check else ''})",
        "",
        f"- n = {report.get('n', 0)}",
        f"- mean Score = {report.get('mean', {}).get('Score', 0):.3f}",
        f"- ComponentF1 = {report.get('mean', {}).get('ComponentF1', 0):.3f}",
        f"- PinF1 = {report.get('mean', {}).get('PinF1', 0):.3f}",
        f"- NetHypergraphF1 = {report.get('mean', {}).get('NetHypergraphF1', 0):.3f}",
        f"- NetLineF1 = {report.get('mean', {}).get('NetLineF1', 0):.3f}",
        "",
        "| case | Score | Comp | Pin | Net | Line |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("cases", []):
        lines.append(
            f"| {row['case_id']} | {row['Score']:.2f} | {row['ComponentF1']:.3f} | {row['PinF1']:.3f} | {row['NetHypergraphF1']:.3f} | {row['NetLineF1']:.3f} |"
        )
    report_name = "score_report_selfcheck.md" if args.self_check else "score_report.md"
    (out_root / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.self_check:
        (out_root / "score_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"wrote {out_root / report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
