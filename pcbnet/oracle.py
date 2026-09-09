"""Oracle ladder: inject GT visual elements to localise where score leaks.

Levels
------
- L0  : real pipeline (stage1 vision -> stage2 structure). Current standing.
- L1  : GT components (bbox/type/key/name/value) + real texts already bound in
        adapter, but nets/pins rebuilt by our stage2 from a GT-derived VisionOut.
        Isolates structure construction quality given perfect component perception.
- L2  : L1 + GT pin tips/numbers injected. Isolates net building given perfect pins.
- L3  : L2 + GT wire skeleton injected. Isolates connectivity rules only.
- L4  : pred = full GT. Must score 100, else adapter/scorer disagree with contest.

The ladder is intentionally deterministic (no ML) so it runs anywhere and pins
down which stage is responsible for a low score before touching models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .gt_adapter import cvjson_to_contest, load_category_map
from .stage1_vision import DetBox, PinBox, TextBox, VisionOut, _skel_to_segments
from .stage2_structure import build_structure


def _load(cvjson: str | Path | dict) -> dict:
    if isinstance(cvjson, dict):
        return cvjson
    return json.loads(Path(cvjson).read_text(encoding="utf-8"))


def _is_active(item: dict) -> bool:
    return str(item.get("state") or "active") != "deleted"


def _canvas(cvjson: dict) -> tuple[int, int]:
    canvas = cvjson.get("canvas") or {}
    return int(canvas.get("width") or 1), int(canvas.get("height") or 1)


def _gt_component_boxes(cvjson: dict) -> list[DetBox]:
    boxes: list[DetBox] = []
    for comp in (cvjson.get("components") or {}).values():
        if not _is_active(comp):
            continue
        b = comp.get("realBBox") or comp.get("location") or {}
        try:
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        boxes.append(
            DetBox(
                bbox=[min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                label=str(comp.get("category") or "other"),
                score=1.0,
            )
        )
    return boxes


def _gt_text_boxes(cvjson: dict) -> list[TextBox]:
    texts: list[TextBox] = []
    for t in (cvjson.get("texts") or {}).values():
        content = str(t.get("content") or "").strip()
        loc = t.get("location") or {}
        try:
            x1, y1, x2, y2 = float(loc["x1"]), float(loc["y1"]), float(loc["x2"]), float(loc["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if content:
            texts.append(TextBox(bbox=[x1, y1, x2, y2], content=content, score=1.0))
    return texts


def _gt_wire_skeleton(cvjson: dict, w: int, h: int) -> np.ndarray:
    import cv2

    mask = np.zeros((max(1, h), max(1, w)), np.uint8)
    for edge in (cvjson.get("edges") or {}).values():
        if not _is_active(edge):
            continue
        paths = (edge.get("keyPointPaths") or {}).get("wire") or []
        drawn = False
        for path in paths:
            if not isinstance(path, (list, tuple)) or len(path) < 2:
                continue
            pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in path], np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, 1)
            drawn = True
        if drawn:
            continue
        coords = edge.get("coordinates") or []
        if len(coords) >= 2:
            pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in coords], np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, 1)
    return mask


def _txt(texts: dict, tid) -> str | None:
    if not tid:
        return None
    c = str((texts.get(tid) or {}).get("content") or "").strip()
    return c or None


def _gt_pins(cvjson: dict) -> list[PinBox]:
    texts = cvjson.get("texts") or {}
    pins: list[PinBox] = []
    for pin in (cvjson.get("pins") or {}).values():
        tip = pin.get("tip")
        if not (isinstance(tip, (list, tuple)) and len(tip) >= 2):
            continue
        number = _txt(texts, pin.get("pinNumberByOCR"))
        name = _txt(texts, pin.get("pinNameByOCR"))
        pins.append(PinBox(xy=(float(tip[0]), float(tip[1])), score=1.0, number=number, name=name))
    return pins


def vision_from_gt(
    cvjson: str | Path | dict,
    with_components: bool = True,
    with_texts: bool = True,
    with_pins: bool = False,
    with_wires: bool = True,
) -> VisionOut:
    data = _load(cvjson)
    w, h = _canvas(data)
    comps = _gt_component_boxes(data) if with_components else []
    texts = _gt_text_boxes(data) if with_texts else []
    pins = _gt_pins(data) if with_pins else []
    skel = _gt_wire_skeleton(data, w, h) if with_wires else None
    segs = _skel_to_segments(skel) if skel is not None else []
    return VisionOut(height=h, width=w, components=comps, texts=texts, pins=pins, skeleton=skel, wire_segments=segs)


def build_level(cvjson: str | Path | dict, level: str, cfg: dict[str, Any] | None = None) -> dict:
    """Return a contest-format prediction for the requested oracle level.

    L1: GT components + texts, geometry pins.       L2: + GT pin tips/numbers.
    L3: + GT wire skeleton.                          L4: full GT identity.
    """
    data = _load(cvjson)
    if level == "L4":
        return cvjson_to_contest(data)
    with_pins = level in {"L2", "L3"}
    vision = vision_from_gt(data, with_components=True, with_texts=True, with_pins=with_pins, with_wires=True)
    return build_structure(vision, cfg or {})
