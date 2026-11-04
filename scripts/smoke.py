#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter + scorer self-check, then weak-vision inference on a few cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, parse_case_ids, scan_dataset
from pcbnet.export_json import export_contest_json
from pcbnet.gt_adapter import cvjson_to_contest
from pcbnet.pipeline import infer_image
from pcbnet.schema import validate_contest_json
from pcbnet.scorer import score_case, score_case_aligned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="0003,0060")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    cases = index_cases(scan_dataset(cfg["dataset_root"]))
    out_root = project_root() / cfg.get("output_root", "outputs")
    ok = True
    for cid in parse_case_ids(args.cases):
        case = cases.get(cid)
        if not case:
            print(f"[skip] missing case {cid}")
            ok = False
            continue
        gt = cvjson_to_contest(case.cvjson)
        gt_errors = validate_contest_json(gt)
        self_score = score_case(gt, gt)
        export_contest_json(gt, out_root / "gt" / f"{cid}.json")
        print(f"== {cid} GT adapter ==")
        print(f"  comps={len(gt['components'])} pins={sum(len(v) for v in gt['pins'].values())} nets={len(gt['nets'])}")
        print(f"  schema_errors={len(gt_errors)} self_score={self_score['Score']:.2f} { {k: round(self_score[k],3) for k in ('ComponentF1','PinF1','NetHypergraphF1','NetLineF1')} }")
        if abs(self_score["Score"] - 100.0) > 1e-6:
            print("  WARNING: GT self-score is not 100")
            ok = False
        if args.skip_infer:
            continue
        pred = infer_image(case.image, cfg, out_root / "pred" / f"{cid}.json")
        pred_score = score_case(pred, gt)
        aligned = score_case_aligned(pred, gt)
        print(f"  contest_score={pred_score['Score']:.2f} { {k: round(pred_score[k],3) for k in ('ComponentF1','PinF1','NetHypergraphF1','NetLineF1')} }")
        print(f"  aligned_score={aligned['Score']:.2f} mapped={aligned.get('n_key_mapped', 0)} { {k: round(aligned[k],3) for k in ('ComponentF1','PinF1','NetHypergraphF1','NetLineF1')} }")
        print(f"  pred_comps={len(pred['components'])} schema={validate_contest_json(pred)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
