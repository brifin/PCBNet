#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""元件检测结果可视化对比：pred vs GT（10gt）。

左图 = GT 元件框（绿框 + race type + key）；
右图 = 新模型（train_comp_only）预测框（蓝框 + 粗类 + 置信度），
      右上角标出漏检（红圈 = GT 有而模型没检到）与多检数量。
输出：outputs/viz_comp_detect/<case>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

GT_DIR = ROOT / "data" / "Dataset200" / "10gt"
OUT_DIR = ROOT / "outputs" / "viz_comp_detect"
CONF = 0.2
NMS_IOU = 0.5  # 非极大值抑制：同类框 IoU 超过该值只留置信度最高者

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICK = 2


def _race_color(race: str):
    return (0, 180, 0)  # GT 统一绿色


def draw_gt(img, gt: dict, H: int):
    for key, comp in (gt.get("components") or {}).items():
        bb = comp.get("bbox") or []
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)
        # contest(左下) -> image(左上)
        p1, p2 = (int(x1), int(H - y2)), (int(x2), int(H - y1))
        cv2.rectangle(img, p1, p2, (0, 180, 0), THICK)
        race = str(comp.get("type"))
        cv2.putText(img, f"{race}", (p1[0], max(10, p1[1] - 4)), FONT, FONT_SCALE, (0, 120, 0), 1)


def draw_pred(img, preds, missed, n_extra, H: int, names):
    for (ci, x1, y1, x2, y2, conf) in preds:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (255, 140, 0), THICK)
        cv2.putText(img, f"{names[ci]} {conf:.2f}", (p1[0], max(10, p1[1] - 4)), FONT, FONT_SCALE, (180, 90, 0), 1)
    # 漏检：GT 有而模型没有 → 红圈标在原位置
    for (gx, gy, race) in missed:
        cv2.circle(img, (int(gx), int(gy)), 14, (0, 0, 255), 2)
        cv2.putText(img, f"MISS:{race}", (int(gx) + 16, int(gy)), FONT, 0.5, (0, 0, 220), 1)
    cv2.putText(img, f"pred boxes={len(preds)}  missed={len(missed)}  extra~{n_extra}", (10, 24), FONT, 0.7, (0, 0, 220), 2)


def nms(boxes: list, iou_thr: float = 0.5) -> list:
    """按类 NMS：boxes=[(cls,x1,y1,x2,y2,conf)]，同类 IoU>阈值保留高置信度框。"""
    if iou_thr <= 0 or not boxes:
        return boxes
    from collections import defaultdict

    by_cls = defaultdict(list)
    for b in boxes:
        by_cls[int(b[0])].append(b)
    out = []
    for lst in by_cls.values():
        lst.sort(key=lambda b: -b[5])
        keep = []
        for b in lst:
            x1, y1, x2, y2 = b[1:5]
            drop = False
            for k in keep:
                kx1, ky1, kx2, ky2 = k[1:5]
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/models/yolo/train_comp_only/weights/best.pt")
    ap.add_argument("--tag", default="comp_only")
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU)
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    cls = yaml.safe_load((ROOT / "data" / "yolo_classes.yaml").read_text(encoding="utf-8"))
    names = cls["names"][:14]
    coarse = cls["coarse_map"]
    id_of = {n: i for i, n in enumerate(names)}

    model = YOLO(str(ROOT / args.weights))
    out_root = OUT_DIR / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    n_raw_total = n_total = 0

    n_done = 0
    for d in sorted(p for p in GT_DIR.iterdir() if p.is_dir()):
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
        n_raw_total += len(preds_raw)
        n_total += len(preds)
        gts = []
        for key, comp in (gt.get("components") or {}).items():
            bb = comp.get("bbox") or []
            if len(bb) != 4:
                continue
            race = str(comp.get("type"))
            ci = id_of.get(coarse.get(race, "other"), id_of["other"])
            gts.append((ci, race, key, bb))
        used = set()
        missed = []
        for (ci, race, key, bb) in gts:
            gcx = (bb[0] + bb[2]) / 2
            gcy = (bb[1] + bb[3]) / 2
            gy = H - gcy
            best, bestd = None, 1e9
            for pi, (pci, px1, py1, px2, py2, cf) in enumerate(preds):
                if pi in used or pci != ci:
                    continue
                dd = ((gcx - (px1 + px2) / 2) ** 2 + (gy - (py1 + py2) / 2) ** 2) ** 0.5
                if dd < bestd:
                    bestd, best = dd, pi
            if best is not None and bestd <= 20.0:
                used.add(best)
            else:
                missed.append((gcx, gy, race))
        n_extra = len(preds) - len(used)

        gt_img = img.copy()
        draw_gt(gt_img, gt, H)
        pred_img = img.copy()
        draw_pred(pred_img, preds, missed, n_extra, H, names)

        panel = np.hstack([gt_img, pred_img])
        scale = min(1.0, 2800.0 / panel.shape[1])
        if scale < 1.0:
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        # 分隔线与标题
        cv2.line(panel, (int(gt_img.shape[1] * scale), 0), (int(gt_img.shape[1] * scale), panel.shape[0]), (0, 0, 0), 4)
        cv2.putText(panel, "GT", (10, 24), FONT, 1.0, (0, 200, 0), 3)
        cv2.putText(panel, "PRED", (int(gt_img.shape[1] * scale) + 10, 24), FONT, 1.0, (0, 140, 255), 3)
        out = out_root / f"{d.name}.png"
        cv2.imencode(".png", panel)[1].tofile(str(out))
        n_done += 1
        print(f"[viz] {d.name}: GT={len(gts)} pred={len(preds)} missed={len(missed)} extra={n_extra} -> {out}")

    print(f"[viz] {args.tag} 完成 {n_done} 张（NMS: 原始框 {n_raw_total} -> {n_total}），输出目录 {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
