"""Image (top-left) <-> contest (bottom-left) coordinates."""

from __future__ import annotations


def image_to_contest_xy(x: float, y: float, height: float) -> tuple[float, float]:
    return float(x), float(height) - float(y)


def contest_to_image_xy(x: float, y: float, height: float) -> tuple[float, float]:
    return float(x), float(height) - float(y)


def image_bbox_to_contest(bbox: list[float] | dict, height: float) -> list[float]:
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    else:
        x1, y1, x2, y2 = bbox
    x1, x2 = float(min(x1, x2)), float(max(x1, x2))
    y1, y2 = float(min(y1, y2)), float(max(y1, y2))
    return [x1, height - y2, x2, height - y1]


def contest_point(x: float, y: float, height: float) -> dict:
    cx, cy = image_to_contest_xy(x, y, height)
    return {"x": cx, "y": cy}


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def point_dist(a: dict | tuple, b: dict | tuple) -> float:
    ax, ay = (a["x"], a["y"]) if isinstance(a, dict) else a
    bx, by = (b["x"], b["y"]) if isinstance(b, dict) else b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
