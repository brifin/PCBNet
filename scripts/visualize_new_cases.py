#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize YOLO detection boxes (GT vs Pred) for a few new augmented cases.

Draws the GT component bbox + pin tips (green) and the YOLO-predicted component
boxes + pin tips (red) on a side-by-side image, to eyeball whether the trained
detector produces correct boxes on the single-component crops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "data" / "Dataset200" / "200_train_cases"
WEIGHTS = ROOT / "outputs1" / "models" / "yolo" / "train" / "weights" / "best.pt"
OUT_DIR = ROOT / "outputs1" / "viz_new"

# coarse class names (index order = yolo_classes.yaml names)
CLASS_NAMES = ["ic", "resistor", "capacitor", "inductor", "diode", "transistor",
               "gnd", "vcc", "net_label", "connector", "crystal", "switch",
               "testpoint", "other", "text", "pin_tip"]
PIN_CLASS = 15


def read_img(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def flip_y_bbox(bbox, H):
    x1, y1, x2, y2 = bbox
    return int(x1), int(H - y2), int(x2), int(H - y1)  # (left, top, right, bottom)


def draw_gt(img, data, H):
    comp = list((data.get("components") or {}).values())[0]
    key = list((data.get("components") or {}).keys())[0]
    b = comp.get("bbox") or [0, 0, 0, 0]
    l, t, r, bb = flip_y_bbox(b, H)
    cv2.rectangle(img, (l, t), (r, bb), (0, 180, 0), 2)
    cv2.putText(img, f"GT {comp.get('type')}", (l, max(0, t - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    for pin in ((data.get("pins") or {}).get(key) or {}).values():
        pt = pin.get("point") or {}
        cv2.circle(img, (int(pt["x"]), int(H - pt["y"])), 3, (0, 180, 0), -1)
    return img


def draw_pred(img, res, H):
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls == PIN_CLASS:
            continue
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 230), 2)
        cv2.putText(img, f"{CLASS_NAMES[cls]} {float(b.conf[0]):.2f}",
                    (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    # pin tips
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls == PIN_CLASS:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, (0, 0, 230), -1)
    return img


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    ids = sys.argv[1:] or ["0301", "0302", "0303", "0305", "0308", "0312"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cid in ids:
        img_path = next((TRAIN_DIR / cid).glob("*-crop.png"), None)
        json_path = next((TRAIN_DIR / cid).glob("*-crop_target_named.json"), None)
        if img_path is None or json_path is None:
            print(f"[skip] {cid} not a crop case")
            continue
        img = read_img(img_path)
        H, W = img.shape[:2]
        data = json.loads(json_path.read_text(encoding="utf-8"))

        gt_img = np.full((H, W, 3), 255, np.uint8)
        draw_gt(gt_img, data, H)
        pred_img = np.full((H, W, 3), 255, np.uint8)
        res = model.predict(img, verbose=False)[0]
        draw_pred(pred_img, res, H)

        sep = np.full((H, 12, 3), 200, np.uint8)
        combo = np.hstack([gt_img, sep, pred_img])
        out = OUT_DIR / f"{cid}_gt_pred.png"
        cv2.imencode(".png", combo)[1].tofile(str(out))
        ncomp = sum(1 for b in res.boxes if int(b.cls[0]) != PIN_CLASS)
        npin = sum(1 for b in res.boxes if int(b.cls[0]) == PIN_CLASS)
        print(f"[viz] {cid} -> {out}  pred_comp={ncomp} pred_pin={npin}")

    print(f"done, wrote to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
