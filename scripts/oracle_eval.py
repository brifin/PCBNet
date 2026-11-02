#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Oracle ladder eval: localise which stage is responsible for a low score.

Runs deterministic levels L1..L4 (see pcbnet/oracle.py) and, optionally, the
real pipeline (L0). Writes outputs/oracle_report.md + oracle.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, parse_case_ids, scan_dataset
from pcbnet.gt_adapter import cvjson_to_contest
from pcbnet.oracle import build_level
from pcbnet.pipeline import infer_image
from pcbnet.schema import load_json
from pcbnet.scorer import filter_confident, score_case

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_DESC = {
    "L0": "real pipeline (stage1 vision + stage2)",
    "L1": "GT components+texts, stage2 builds pins/nets",
    "L2": "L1 + GT-derived pins geometry",
    "L3": "L2 + GT wire skeleton",
    "L4": "pred = full GT (must be 100)",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--cases", default="")
    parser.add_argument("--levels", default="L1,L3,L4")
    parser.add_argument("--run-l0", action="store_true", help="also run real pipeline (slow)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    thr = cfg.get("thresholds") or {}
    kw = dict(
        bbox_thr=float(thr.get("bbox_px", 20)),
        pin_thr=float(thr.get("pin_px", 5)),
        line_thr=float(thr.get("line_px", 5)),
        weights=cfg.get("weights"),
    )
    all_cases = scan_dataset(cfg["dataset_root"])
    splits = load_or_create_splits(
        project_root() / "data", all_cases,
        seed=cfg.get("split_seed", 42), ratio=tuple(cfg.get("split_ratio", [0.8, 0.1, 0.1])),
    )
    ids = parse_case_ids(args.cases) or splits.get(args.split, [])
    index = index_cases(all_cases)
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    if args.run_l0 and "L0" not in levels:
        levels = ["L0"] + levels
    out_root = project_root() / cfg.get("output_root", "outputs")

    rows: list[dict] = []
    for cid in ids:
        case = index.get(cid)
        if not case:
            print(f"[skip] {cid}")
            continue
        gt = cvjson_to_contest(case.cvjson)
        row: dict = {"case_id": cid}
        for level in levels:
            if level == "L0":
                pred_path = out_root / "pred" / f"{cid}.json"
                legacy = out_root / "pred" / cid / f"{cid}.json"
                if not pred_path.exists() and legacy.exists():
                    pred_path = legacy
                pred = load_json(pred_path) if pred_path.exists() else infer_image(case.image, cfg, pred_path)
            else:
                pred = build_level(case.cvjson, level, cfg)
            s = score_case(pred, gt, **kw)
            sc = score_case(filter_confident(pred), filter_confident(gt), **kw)
            row[level] = {k: round(s[k], 4) for k in ("Score", "ComponentF1", "PinF1", "NetHypergraphF1", "NetLineF1")}
            row[level]["ConfidentScore"] = round(sc["Score"], 4)
            row[level]["diag"] = s["diag"]
        rows.append(row)
        summary = " ".join(f"{lv}={row[lv]['Score']:.1f}" for lv in levels)
        print(f"[oracle] {cid} {summary}")

    metrics = ("Score", "ConfidentScore", "ComponentF1", "PinF1", "NetHypergraphF1", "NetLineF1")
    means = {lv: {m: 0.0 for m in metrics} for lv in levels}
    for lv in levels:
        n = len(rows) or 1
        for m in means[lv]:
            means[lv][m] = sum(r[lv].get(m, 0.0) for r in rows) / n

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "oracle.json").write_text(
        json.dumps({"levels": levels, "means": means, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [f"# Oracle ladder ({args.split})", "", f"- n = {len(rows)}", ""]
    for lv in levels:
        lines.append(f"- {lv} ({LEVEL_DESC.get(lv,'')}): Score={means[lv]['Score']:.2f} "
                     f"ConfidentScore={means[lv]['ConfidentScore']:.2f} "
                     f"Comp={means[lv]['ComponentF1']:.3f} Pin={means[lv]['PinF1']:.3f} "
                     f"Net={means[lv]['NetHypergraphF1']:.3f} Line={means[lv]['NetLineF1']:.3f}")
    lines += ["", "| case | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
    for r in rows:
        lines.append("| " + r["case_id"] + " | " + " | ".join(f"{r[lv]['Score']:.1f}" for lv in levels) + " |")
    (out_root / "oracle_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"wrote {out_root / 'oracle_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
