#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""box（IC）引脚检测专项分析 + 可视化（10gt）。

流程：
  1. YOLO 检测 → ic 框（class 0）+ pin_tip 点（class 15）；
  2. pin_tip 点归属到最近的 ic 框（框外扩 15px），并按 8px 去重；
  3. GT：type==box 的元件，其 pins 的 point（左下→左上坐标）；
  4. GT box 与 pred ic 框按中心距≤20px 配对；配对框内 GT pin 与 pred pin 按 20px 贪心匹配；
  5. 输出：总体/逐 case 的 pin recall、precision、tip 距离分布、漏检清单；
  6. --viz 生成对比图（左 GT 框+引脚绿点 / 右 pred 框+引脚橙点+漏检红圈）。
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

GT_DIR = ROOT / "data" / "Dataset200" / "10gt"
OUT_DIR = ROOT / "outputs" / "viz_pin_detect"


def dedup_points(pts: list, eps: float = 8.0) -> list:
    out = []
    for p in sorted(pts, key=lambda x: -x[2]):
        if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < eps * eps for q in out):
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--pin-eps", type=float, default=8.0, help="预测 pin 点去重半径")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    if args.viz:
        (OUT_DIR / args.tag).mkdir(parents=True, exist_ok=True)

    tot = {"gt_pin": 0, "tp": 0, "pred_pin": 0, "pred_orphan": 0, "box_gt": 0, "box_tp": 0}
    dists = []
    missed = []   # (case, comp, pin_key, pinname)
    per_case = []

    for d in sorted(p for p in GT_DIR.iterdir() if p.is_dir()):
        gt_js = next(iter(sorted(d.glob("*_target*.json"))), None) or next(iter(sorted(d.glob("*.json"))), None)
        img_p = next((p for p in sorted(d.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}), None)
        if gt_js is None or img_p is None:
            continue
        gt = json.loads(gt_js.read_text(encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]

        # ---- GT：box 元件 + 其引脚 ----
        gt_boxes = []   # (key, x1,y1,x2,y2)
        gt_pins = []    # (comp_key, pin_key, pinname, x, y)
        for key, comp in (gt.get("components") or {}).items():
            if str(comp.get("type")) != "box":
                continue
            bb = comp.get("bbox") or []
            if len(bb) != 4:
                continue
            gt_boxes.append((key, bb[0], H - bb[3], bb[2], H - bb[1]))
            for pk, pv in ((gt.get("pins") or {}).get(key) or {}).items():
                pt = pv.get("point") or {}
                gt_pins.append((key, pk, str(pv.get("pinname") or ""), float(pt.get("x", 0)), H - float(pt.get("y", 0))))
        if not gt_boxes:
            continue

        # ---- 检测 ----
        res = model.predict(img, imgsz=1280, conf=args.conf, verbose=False)[0]
        ic_boxes, pin_pts = [], []
        if res.boxes is not None:
            for b, cl, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                if int(cl) == 0:
                    ic_boxes.append((b[0], b[1], b[2], b[3], float(cf)))
                elif int(cl) == 15:
                    pin_pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(cf)))
        pin_pts = dedup_points(pin_pts, args.pin_eps)

        # pin 点归属到 ic 框（外扩 15px）
        assigned = defaultdict(list)   # ic 框 idx -> [(x,y,conf)]
        orphan = 0
        for (px, py, cf) in pin_pts:
            best, bestd = None, 1e9
            for i, (x1, y1, x2, y2, _) in enumerate(ic_boxes):
                dx = max(x1 - 15 - px, 0, px - (x2 + 15))
                dy = max(y1 - 15 - py, 0, py - (y2 + 15))
                dd = (dx * dx + dy * dy) ** 0.5
                if dd < bestd:
                    bestd, best = dd, i
            if best is not None and bestd <= 0:
                assigned[best].append((px, py, cf))
            else:
                orphan += 1
        tot["pred_orphan"] += orphan

        # ---- GT box ↔ pred ic 框配对（中心距≤20）----
        used_box = set()
        case = {"gt_pin": len(gt_pins), "tp": 0, "pred_pin": 0, "box_gt": len(gt_boxes), "box_tp": 0}
        for key, gx1, gy1, gx2, gy2 in gt_boxes:
            tot["box_gt"] += 1
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            best, bestd = None, 1e9
            for i, (x1, y1, x2, y2, cf) in enumerate(ic_boxes):
                if i in used_box:
                    continue
                dd = ((gcx - (x1 + x2) / 2) ** 2 + (gcy - (y1 + y2) / 2) ** 2) ** 0.5
                if dd < bestd:
                    bestd, best = dd, i
            if best is None or bestd > 20.0:
                # 框没配对：该 box 的全部引脚都算漏
                for (ck, pk, pn, px, py) in gt_pins:
                    if ck == key:
                        missed.append((d.name, key, pk, pn))
                continue
            used_box.add(best)
            tot["box_tp"] += 1
            case["box_tp"] += 1

            pred_pts = assigned.get(best, [])
            case["pred_pin"] += len(pred_pts)
            tot["pred_pin"] += len(pred_pts)

            # GT pin 与 pred pin 贪心匹配（≤20px）
            used_p = set()
            for (ck, pk, pn, px, py) in gt_pins:
                if ck != key:
                    continue
                tot["gt_pin"] += 1
                case["gt_pin"] += 1
                b, bd = None, 1e9
                for pi, (qx, qy, qf) in enumerate(pred_pts):
                    if pi in used_p:
                        continue
                    dd = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
                    if dd < bd:
                        bd, b = dd, pi
                if b is not None and bd <= 20.0:
                    used_p.add(b)
                    tot["tp"] += 1
                    case["tp"] += 1
                    dists.append(bd)
                else:
                    missed.append((d.name, key, pk, pn))
        per_case.append((d.name, case))

        # ---- 可视化 ----
        if args.viz:
            gt_img = img.copy()
            for key, x1, y1, x2, y2 in gt_boxes:
                cv2.rectangle(gt_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 2)
            for (ck, pk, pn, px, py) in gt_pins:
                cv2.circle(gt_img, (int(px), int(py)), 4, (0, 200, 0), -1)
            pred_img = img.copy()
            for (x1, y1, x2, y2, cf) in ic_boxes:
                cv2.rectangle(pred_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 140, 0), 2)
            for i, lst in assigned.items():
                for (px, py, cf) in lst:
                    cv2.circle(pred_img, (int(px), int(py)), 4, (255, 0, 255), -1)
            for (cid, key, pk, pn) in missed:
                if cid == d.name:
                    for (ck, pk2, pn2, px, py) in gt_pins:
                        if ck == key and pk2 == pk:
                            cv2.circle(pred_img, (int(px), int(py)), 12, (0, 0, 255), 2)
            panel = np.hstack([gt_img, pred_img])
            scale = min(1.0, 2800.0 / panel.shape[1])
            if scale < 1.0:
                panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            cv2.line(panel, (int(gt_img.shape[1] * scale), 0), (int(gt_img.shape[1] * scale), panel.shape[0]), (0, 0, 0), 4)
            cv2.putText(panel, "GT box+pins", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
            cv2.putText(panel, "PRED ic+pin_tip", (int(gt_img.shape[1] * scale) + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 3)
            cv2.imencode(".png", panel)[1].tofile(str(OUT_DIR / args.tag / f"{d.name}.png"))

    R = tot["tp"] / tot["gt_pin"] if tot["gt_pin"] else 1.0
    P = tot["tp"] / tot["pred_pin"] if tot["pred_pin"] else 1.0
    print(f"===== {args.tag}  box 引脚检测（10gt 13 例，conf={args.conf}，pin去重 {args.pin_eps}px）=====")
    print(f"GT box 元件 {tot['box_gt']} 个，pred ic 框配对 {tot['box_tp']} 个")
    print(f"GT 引脚 {tot['gt_pin']} / 检出 {tot['tp']} / 预测 pin 点 {tot['pred_pin']}（另孤儿点 {tot['pred_orphan']}）")
    print(f"recall={R:.3f}（漏检 {tot['gt_pin']-tot['tp']}）  precision={P:.3f}（多检 {tot['pred_pin']-tot['tp']}）")
    if dists:
        a = np.array(dists)
        print(f"tip 距离：mean={a.mean():.1f}  median={np.median(a):.1f}  p95={np.percentile(a,95):.1f}  max={a.max():.1f}")
    print(f"{'case':>5} {'box配对':>6} {'GT引脚':>6} {'TP':>4} {'pred点':>6} {'recall':>7} {'prec':>7}")
    for cid, c in per_case:
        r = c["tp"] / c["gt_pin"] if c["gt_pin"] else 1.0
        p = c["tp"] / c["pred_pin"] if c["pred_pin"] else 1.0
        print(f"{cid:>5} {c['box_tp']}/{c['box_gt']:>4} {c['gt_pin']:>6} {c['tp']:>4} {c['pred_pin']:>6} {r:7.1%} {p:7.1%}")
    print(f"漏检引脚清单（{len(missed)} 个）：")
    for cid, key, pk, pn in missed:
        print(f"  case {cid:>2}  {key}.{pk}  pinname={pn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
