#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render recognition results onto the original schematic image.

Draws predicted component boxes (+designator), pin tips and non-degenerate net
edges. Small images are upscaled for legibility, and component labels are placed
with greedy anti-overlap so they don't pile on top of each other.

Reads pred JSON from outputs/pred/<id>.json; runs the pipeline if missing.
Output PNGs go to outputs/viz/<id>.png.

Usage:
  py -3.12 scripts/visualize.py --split test
  py -3.12 scripts/visualize.py --cases 0025,0038 --verbose --pin-names --with-gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import index_cases, load_or_create_splits, parse_case_ids, scan_dataset
from pcbnet.gt_adapter import cvjson_to_contest
from pcbnet.schema import load_json

COMP_COLOR = (0, 170, 0)      # green (BGR)
PIN_COLOR = (0, 0, 230)       # red
NET_COLOR = (230, 130, 0)     # blue
GT_COLOR = (0, 140, 255)      # orange (GT boxes when --with-gt)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _read_bgr(path: str | Path) -> np.ndarray | None:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))


def _overlaps(a, boxes) -> bool:
    ax1, ay1, ax2, ay2 = a
    for bx1, by1, bx2, by2 in boxes:
        if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
            return True
    return False


def _place_label(img, text, bx1, by1, bx2, by2, color, scale, placed, W, H):
    """Draw a text label anchored to its box, with a leader line so the label is
    always visually tied to the component even if nudged to avoid overlaps."""
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    tw += 4
    th += 4
    # anchor point on the box we always connect the label back to (top-left)
    ax, ay = int(bx1), int(by1)
    # candidate label positions, all immediately adjacent to the box
    cands = [
        (bx1, by1 - th - 1),      # just above
        (bx1, by2 + 1),           # just below
        (bx2 + 2, by1),           # right of box
        (bx1 - tw - 2, by1),      # left of box
        (bx1, by1 + 1),           # inside top
    ]
    chosen = None
    for cx, cy in cands:
        cx = int(max(0, min(cx, W - tw)))
        cy = int(max(0, min(cy, H - th)))
        rect = (cx, cy, cx + tw, cy + th)
        if not _overlaps(rect, placed):
            chosen = (cx, cy)
            break
    if chosen is None:
        # bounded search on a small ring around the box, keep it close
        cx0, cy0 = int(max(0, min(bx1, W - tw))), int(max(0, min(by1 - th - 1, H - th)))
        cx, cy = cx0, cy0
        for step in range(1, 12):
            cy = int(max(0, min(cy0 - step * th, H - th)))
            rect = (cx, cy, cx + tw, cy + th)
            if not _overlaps(rect, placed):
                break
        chosen = (cx, cy)
    cx, cy = chosen
    # leader line from label to the box anchor (drawn under the label box)
    lx = cx + tw // 2 if cx + tw < ax else cx + tw
    ly = cy + th // 2
    if abs(lx - ax) > 4 or abs(ly - ay) > 4:
        cv2.line(img, (int(cx + tw // 2), int(cy + th // 2)), (ax, ay), color, 1, cv2.LINE_AA)
    _blit(img, text, cx, cy, tw, th, color, scale)
    placed.append((cx, cy, cx + tw, cy + th))


def _blit(img, text, x, y, tw, th, color, scale):
    cv2.rectangle(img, (x, y), (x + tw, y + th), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + tw, y + th), color, 1)
    cv2.putText(img, text, (x + 2, y + th - 3), FONT, scale, color, 1, cv2.LINE_AA)


def _draw(img, contest, comp_color, s, h0, verbose, pin_names, boxes_only=False):
    """Draw contest-JSON geometry onto the (upscaled) image.

    Contest coords use a bottom-left origin (Y flipped vs the image top-left),
    so every Y is converted with y_img = (h0 - y_contest) before scaling by s.
    """
    H, W = img.shape[:2]
    scale = max(0.4, min(0.7, W / 2200.0))
    thick = max(1, int(round(W / 1400.0)))
    placed: list = []

    def fx(x):
        return int(x * s)

    def fy(y):
        return int((h0 - y) * s)

    if not boxes_only:
        for net in (contest.get("nets") or {}).values():
            for edge in (net.get("edges") or {}).values():
                if isinstance(edge, list) and len(edge) == 2:
                    p1, p2 = edge
                    x1, y1, x2, y2 = fx(p1["x"]), fy(p1["y"]), fx(p2["x"]), fy(p2["y"])
                    if abs(x1 - x2) < 1 and abs(y1 - y2) < 1:  # skip degenerate
                        continue
                    cv2.line(img, (x1, y1), (x2, y2), NET_COLOR, thick, cv2.LINE_AA)
    for key, comp in (contest.get("components") or {}).items():
        bb = comp.get("bbox") or [0, 0, 0, 0]
        x1, x2 = fx(bb[0]), fx(bb[2])
        # bbox is [x1, y1, x2, y2] in contest coords with y1<y2; flip both edges
        y_top, y_bot = fy(bb[3]), fy(bb[1])
        cv2.rectangle(img, (x1, y_top), (x2, y_bot), comp_color, thick)
        if not boxes_only:
            tag = str(key)
            if verbose:
                extra = comp.get("value") or comp.get("name")
                tag = f"{key} {extra}" if extra else str(key)
            _place_label(img, tag, x1, y_top, x2, y_bot, comp_color, scale, placed, W, H)
    if not boxes_only:
        for pins in (contest.get("pins") or {}).values():
            for pin in (pins or {}).values():
                pt = pin.get("point") or {}
                x, y = fx(pt.get("x", 0)), fy(pt.get("y", 0))
                cv2.circle(img, (x, y), max(2, thick + 1), PIN_COLOR, -1)
                if pin_names:
                    _place_label(img, str(pin.get("pinname") or ""), x + 3, y - 6, x + 3, y, PIN_COLOR, scale * 0.8, placed, W, H)
    return img


def _legend(img, with_gt):
    lines = [("component (pred)", COMP_COLOR), ("pin", PIN_COLOR), ("net edge", NET_COLOR)]
    if with_gt:
        lines.append(("component (GT)", GT_COLOR))
    cv2.rectangle(img, (6, 6), (250, 20 + 24 * len(lines)), (255, 255, 255), -1)
    cv2.rectangle(img, (6, 6), (250, 20 + 24 * len(lines)), (60, 60, 60), 1)
    y = 12
    for text, color in lines:
        y += 24
        cv2.rectangle(img, (14, y - 12), (34, y + 2), color, -1)
        cv2.putText(img, text, (42, y), FONT, 0.5, (30, 30, 30), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--cases", default="")
    parser.add_argument("--with-gt", action="store_true", help="also draw GT component boxes (orange)")
    parser.add_argument("--verbose", action="store_true", help="append value/name to component label")
    parser.add_argument("--pin-names", action="store_true", help="draw pin numbers (dense)")
    parser.add_argument("--min-width", type=int, default=1600, help="upscale small images to at least this width")
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
    viz_dir = out_root / "viz"
    n = 0
    for cid in ids:
        case = index.get(cid)
        if not case:
            print(f"[skip] {cid}")
            continue
        img = _read_bgr(case.image)
        if img is None:
            print(f"[skip] {cid} (image unreadable)")
            continue
        h, w = img.shape[:2]
        s = max(1.0, args.min_width / float(w))
        if s > 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        # 本地测试现在是扁平 pred/<case>.json；兼容旧的 pred/<case>/<case>.json
        pred_path = out_root / "pred" / f"{cid}.json"
        legacy = out_root / "pred" / cid / f"{cid}.json"
        if not pred_path.exists() and legacy.exists():
            pred_path = legacy
        if pred_path.exists():
            pred = load_json(pred_path)
        else:
            from pcbnet.pipeline import infer_image
            pred = infer_image(case.image, cfg, pred_path)
        if args.with_gt:
            _draw(img, cvjson_to_contest(case.cvjson), GT_COLOR, s, h, args.verbose, False, boxes_only=True)
        _draw(img, pred, COMP_COLOR, s, h, args.verbose, args.pin_names)
        _legend(img, args.with_gt)
        dest = viz_dir / f"{cid}.png"
        _write_png(dest, img)
        nc = len(pred.get("components") or {})
        np_ = sum(len(v) for v in (pred.get("pins") or {}).values())
        nn = len(pred.get("nets") or {})
        print(f"[viz] {cid} -> {dest}  scale={s:.2f} comp={nc} pins={np_} nets={nn}")
        n += 1
    print(f"done, wrote {n} images to {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
