#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10gt 元件检测 + bbox 统计（支持 NMS 去重），按 race type 输出。

用法：
  .venv/bin/python scripts/eval_comp_detect.py --weights <pt> --tag <名> [--nms-iou 0.5] [--conf 0.2]

检测判定：同粗类 + 中心距≤20px（一对一贪心）；另统计四角差/IoU 与赛题规则合格率，
并列出漏检清单（含"四角≤20 可救回"检查）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def nms(boxes: list, iou_thr: float = 0.5) -> list:
    """按类 NMS：同类框 IoU > iou_thr 时保留置信度高者。boxes: [(cls,x1,y1,x2,y2,conf)]"""
    if iou_thr <= 0 or not boxes:
        return boxes
    by_cls: dict[int, list] = defaultdict(list)
    for b in boxes:
        by_cls[int(b[0])].append(b)
    out: list = []
    for lst in by_cls.values():
        lst.sort(key=lambda b: -b[5])
        keep = []
        for b in lst:
            x1, y1, x2, y2 = b[1:5]
            drop = False
            for k in keep:
                kx1, ky1, kx2, ky2 = k[1:5]
                ix1, iy1, ix2, iy2 = max(x1, kx1), max(y1, ky1), min(x2, kx2), min(y2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
                if union > 0 and inter / union > iou_thr:
                    drop = True
                    break
            if not drop:
                keep.append(b)
        out.extend(keep)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt-dirs", default=str(ROOT / "data" / "Dataset200" / "10gt"))
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--nms-iou", type=float, default=0.5, help="0 = 不启用 NMS")
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()

    cls = yaml.safe_load((ROOT / "data" / "yolo_classes.yaml").read_text(encoding="utf-8"))
    names = cls["names"][:14]
    coarse = cls["coarse_map"]
    id_of = {n: i for i, n in enumerate(names)}

    model = YOLO(args.weights)
    agg = defaultdict(lambda: {"gt": 0, "tp": 0, "ok": 0, "cd": [], "ce": [], "iou": []})
    missed_all = []
    overall = {"gt": 0, "tp": 0, "ok": 0, "cd": [], "ce": [], "iou": [], "pred_raw": 0, "pred": 0}

    for d in sorted(Path(args.gt_dirs).iterdir()):
        if not d.is_dir():
            continue
        gt_js = next(iter(sorted(d.glob("*_target*.json"))), None) or next(iter(sorted(d.glob("*.json"))), None)
        img_p = next((p for p in sorted(d.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}), None)
        if gt_js is None or img_p is None:
            continue
        gt = json.loads(gt_js.read_text(encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        res = model.predict(img, imgsz=1280, conf=args.conf, verbose=False)[0]
        preds_raw = []
        if res.boxes is not None:
            for b, cl, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                if int(cl) < 14:
                    preds_raw.append((int(cl), b[0], b[1], b[2], b[3], float(cf)))
        preds = nms(preds_raw, args.nms_iou)
        overall["pred_raw"] += len(preds_raw)
        overall["pred"] += len(preds)

        gts = []
        for key, comp in (gt.get("components") or {}).items():
            bb = comp.get("bbox") or []
            if len(bb) != 4:
                continue
            race = str(comp.get("type"))
            ci = id_of.get(coarse.get(race, "other"), id_of["other"])
            gts.append((race, ci, key, (bb[0], H - bb[3], bb[2], H - bb[1])))

        used = set()
        for race, ci, key, (gx1, gy1, gx2, gy2) in gts:
            agg[race]["gt"] += 1
            overall["gt"] += 1
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            best, bestd = None, 1e9
            for pi, (pci, px1, py1, px2, py2, cf) in enumerate(preds):
                if pi in used or pci != ci:
                    continue
                dd = ((gcx - (px1 + px2) / 2) ** 2 + (gcy - (py1 + py2) / 2) ** 2) ** 0.5
                if dd < bestd:
                    bestd, best = dd, pi
            if best is not None and bestd <= 20.0:
                used.add(best)
                agg[race]["tp"] += 1
                overall["tp"] += 1
                px1, py1, px2, py2 = preds[best][1:5]
                ce = max(abs(gx1 - px1), abs(gy1 - py1), abs(gx2 - px2), abs(gy2 - py2))
                ix1, iy1 = max(gx1, px1), max(gy1, py1)
                ix2, iy2 = min(gx2, px2), min(gy2, py2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                iou = inter / union if union > 0 else 0
                ok = bestd <= 20.0 or ce <= 20.0
                for kk, vv in (("cd", bestd), ("ce", ce), ("iou", iou)):
                    agg[race][kk].append(vv)
                    overall[kk].append(vv)
                if ok:
                    agg[race]["ok"] += 1
                    overall["ok"] += 1
            else:
                bestce = 1e9
                for _pi, (_pci, px1, py1, px2, py2, cf) in enumerate(preds):
                    if _pci != ci:
                        continue
                    bestce = min(bestce, max(abs(gx1 - px1), abs(gy1 - py1), abs(gx2 - px2), abs(gy2 - py2)))
                missed_all.append((d.name, key, race, bestce <= 20.0))

    def q(a):
        return (float(np.mean(a)), float(np.percentile(a, 95))) if a else (0.0, 0.0)

    print(f"===== {args.tag}  10gt（{len([m for m in missed_all]) or '13 例'}） conf={args.conf} NMS_iou={args.nms_iou} =====")
    print(f"预测框：原始 {overall['pred_raw']} → NMS 后 {overall['pred']}（GT {overall['gt']}）")
    print(f"{'type':16s} {'GT':>4s} {'检出':>4s} {'检出率':>7s} {'中心距均/p95':>13s} {'角点差均/p95':>13s} {'IoU均':>6s}")
    for race in sorted(agg):
        a = agg[race]
        cd, cdp = q(a["cd"])
        ce, cep = q(a["ce"])
        print(f"{race:16s} {a['gt']:4d} {a['tp']:4d} {a['tp']/a['gt']:7.1%} {cd:6.1f}/{cdp:6.1f} {ce:6.1f}/{cep:6.1f} "
              f"{np.mean(a['iou']) if a['iou'] else 0:6.2f}")
    cd, cdp = q(overall["cd"])
    ce, cep = q(overall["ce"])
    print(f"{'总计':16s} {overall['gt']:4d} {overall['tp']:4d} {overall['tp']/overall['gt']:7.1%} {cd:6.1f}/{cdp:6.1f} {ce:6.1f}/{cep:6.1f} "
          f"{np.mean(overall['iou']) if overall['iou'] else 0:6.2f}")
    rescued = [m for m in missed_all if m[3]]
    print(f"\n漏检清单（{len(missed_all)} 个；四角≤20 可救回 {len(rescued)} 个）：")
    for cid, key, race, res in missed_all:
        print(f"  case {cid:>2}  {key:10s}  type={race}{'  [角点≤20 可救]' if res else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
