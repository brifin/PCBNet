"""Export YOLO boxes and wire masks from CVJsonStd. Never writes into Dataset300."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .gt_adapter import load_category_map

TRAIN_SPLITS = ("train", "val")


def _load_classes_yaml(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "yolo_classes.yaml"
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_yolo_classes(path: str | Path | None = None) -> list[str]:
    return [str(x) for x in (_load_classes_yaml(path).get("names") or [])]


def load_coarse_map(path: str | Path | None = None) -> dict[str, str]:
    return {str(k): str(v) for k, v in (_load_classes_yaml(path).get("coarse_map") or {}).items()}


def coarse_component_class(contest_type: str, coarse_map: dict[str, str], class_ids: dict[str, int]) -> str:
    coarse = coarse_map.get(contest_type, contest_type)
    return coarse if coarse in class_ids else "other"


def class_to_id(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _is_active(item: dict) -> bool:
    return str(item.get("state") or "active") != "deleted"


def _box_xyxy(box: dict | list) -> tuple[float, float, float, float] | None:
    if isinstance(box, dict):
        x1, y1, x2, y2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    else:
        return None
    if None in (x1, y1, x2, y2):
        return None
    x1, x2 = float(min(x1, x2)), float(max(x1, x2))
    y1, y2 = float(min(y1, y2)), float(max(y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def yolo_line(box: tuple[float, float, float, float], w: int, h: int, cls: int) -> str | None:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    if min(cx, cy, bw, bh) < 0 or max(cx, cy, bw, bh) > 1:
        return None
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _as_int_pts(coords) -> np.ndarray | None:
    pts: list[list[int]] = []
    for p in coords or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    if len(pts) > 2000:
        step = max(1, len(pts) // 2000)
        pts = pts[::step]
        if pts[-1] != [int(round(float(coords[-1][0]))), int(round(float(coords[-1][1])))]:
            try:
                pts.append([int(round(float(coords[-1][0]))), int(round(float(coords[-1][1])))])
            except Exception:
                pass
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


def _edge_polylines(edge: dict) -> list:
    paths = (edge.get("keyPointPaths") or {}).get("wire") or []
    if paths:
        return paths
    skel = ((edge.get("keyPoints") or {}).get("skeleton") or {}).get("wire")
    if skel:
        return [skel]
    coords = edge.get("coordinates") or []
    return [coords] if coords else []


def _draw_wire_mask(edges: dict, w: int, h: int) -> np.ndarray:
    mask = np.zeros((max(1, h), max(1, w)), np.uint8)
    for edge in edges.values():
        if not _is_active(edge):
            continue
        drawn = False
        for path in _edge_polylines(edge):
            pts = _as_int_pts(path)
            if pts is None:
                continue
            cv2.polylines(mask, [pts], False, 255, 2)
            drawn = True
        if drawn:
            continue
        if edge.get("isFlywire"):
            for item in (edge.get("connect") or {}).values():
                loc = item.get("location") or [0, 0]
                try:
                    cv2.circle(mask, (int(loc[0]), int(loc[1])), 2, 255, -1)
                except Exception:
                    continue
    return mask


def _tip_box(tip, half: float = 6.0) -> tuple[float, float, float, float] | None:
    if not isinstance(tip, (list, tuple)) or len(tip) < 2:
        return None
    try:
        x, y = float(tip[0]), float(tip[1])
    except (TypeError, ValueError):
        return None
    return (x - half, y - half, x + half, y + half)


def export_split(
    cases: dict[str, Any],
    case_ids: list[str],
    split: str,
    out_root: Path,
    names: list[str],
    category_map: dict[str, str],
    coarse_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    mask_dir = out_root / "masks" / split
    for d in (img_dir, lbl_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)
    cls_ids = class_to_id(names)
    other_id = cls_ids.get("other", 0)
    text_id = cls_ids.get("text")
    pin_id = cls_ids.get("pin_tip")
    coarse_map = coarse_map or load_coarse_map()
    hist: Counter[str] = Counter()
    rows = []
    for cid in case_ids:
        case = cases.get(cid)
        if not case:
            rows.append({"case_id": cid, "status": "missing_case"})
            continue
        data = json.loads(case.cvjson.read_text(encoding="utf-8"))
        raw = np.fromfile(str(case.image), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            rows.append({"case_id": cid, "status": "failed_image"})
            continue
        ih, iw = img.shape[:2]
        canvas = data.get("canvas") or {}
        cw, ch = int(canvas.get("width") or iw), int(canvas.get("height") or ih)
        sx, sy = iw / max(cw, 1), ih / max(ch, 1)
        dest_img = img_dir / f"{cid}{case.image.suffix.lower()}"
        shutil.copy2(case.image, dest_img)

        lines = []
        n_skip = 0
        for comp in (data.get("components") or {}).values():
            if not _is_active(comp):
                n_skip += 1
                continue
            box = _box_xyxy(comp.get("realBBox") or comp.get("location"))
            if not box:
                n_skip += 1
                continue
            if sx != 1.0 or sy != 1.0:
                box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            contest_type = category_map.get(comp.get("category"), "other")
            coarse = coarse_component_class(contest_type, coarse_map, cls_ids)
            hist[coarse] += 1
            line = yolo_line(box, iw, ih, cls_ids.get(coarse, other_id))
            if line:
                lines.append(line)
            else:
                n_skip += 1

        n_text = n_pin = 0
        if text_id is not None:
            for t in (data.get("texts") or {}).values():
                if not str(t.get("content") or "").strip():
                    continue
                box = _box_xyxy(t.get("location"))
                if not box:
                    continue
                if sx != 1.0 or sy != 1.0:
                    box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
                line = yolo_line(box, iw, ih, text_id)
                if line:
                    lines.append(line)
                    n_text += 1

        if pin_id is not None:
            for pin in (data.get("pins") or {}).values():
                tb = _tip_box(pin.get("tip"))
                if not tb:
                    continue
                if sx != 1.0 or sy != 1.0:
                    tb = (tb[0] * sx, tb[1] * sy, tb[2] * sx, tb[3] * sy)
                line = yolo_line(tb, iw, ih, pin_id)
                if line:
                    lines.append(line)
                    n_pin += 1
                    
        (lbl_dir / f"{cid}.txt").write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        mask = _draw_wire_mask(data.get("edges") or {}, cw, ch)
        if (cw, ch) != (iw, ih):
            mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
        cv2.imencode(".png", mask)[1].tofile(str(mask_dir / f"{cid}.png"))
        rows.append(
            {
                "case_id": cid,
                "status": "ok",
                "boxes": len(lines),
                "text_boxes": n_text,
                "pin_boxes": n_pin,
                "skipped": n_skip,
                "mask_fg": int((mask > 0).sum()),
                "width": iw,
                "height": ih,
            }
        )
    return {"split": split, "n": len(rows), "class_hist": dict(hist), "cases": rows}


def write_dataset_yaml(out_root: Path, names: list[str], has_val: bool, train_dirs: list[str] | None = None) -> Path:
    path = out_root / "dataset.yaml"
    if train_dirs:
        train_line = "train:\n" + "".join(f"  - images/{d}\n" for d in train_dirs)
    else:
        train_line = "train: images/train\n"
    text = (
        f"path: {out_root.resolve().as_posix()}\n"
        + train_line
        + f"val: images/{'val' if has_val else 'train'}\n"
        + "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
    )
    path.write_text(text, encoding="utf-8")
    (out_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def verify_labels(out_root: Path, splits: list[str], names: list[str]) -> list[str]:
    errors: list[str] = []
    n_cls = len(names)
    for split in splits:
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        mask_dir = out_root / "masks" / split
        if not img_dir.exists():
            errors.append(f"missing images/{split}")
            continue
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        if not images:
            errors.append(f"no images in {split}")
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            mask = mask_dir / f"{img.stem}.png"
            if not lbl.exists():
                errors.append(f"missing label {split}/{img.stem}.txt")
                continue
            for i, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{lbl.name}:{i} expected 5 fields")
                    continue
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
                if cls < 0 or cls >= n_cls:
                    errors.append(f"{lbl.name}:{i} class {cls} out of range")
                if any(v < 0 or v > 1 for v in vals):
                    errors.append(f"{lbl.name}:{i} value not in [0,1]")
            if not mask.exists():
                errors.append(f"missing mask {split}/{img.stem}.png")
    return errors
