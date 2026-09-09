"""Polyline simplification and pin-wire geometry."""

from __future__ import annotations

import math


def _dist_point_to_seg(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        ax, ay = points[start]
        bx, by = points[end]
        max_d, idx = -1.0, start
        for i in range(start + 1, end):
            d = _dist_point_to_seg(points[i][0], points[i][1], ax, ay, bx, by)
            if d > max_d:
                max_d, idx = d, i
        if max_d > epsilon:
            keep[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))
    return [p for p, flag in zip(points, keep) if flag]


def polyline_to_segments(coords: list, epsilon: float = 2.5) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts: list[tuple[float, float]] = []
    for p in coords:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    if not pts:
        return []
    if len(pts) > 400:
        step = max(1, len(pts) // 400)
        decimated = pts[::step]
        if decimated[-1] != pts[-1]:
            decimated.append(pts[-1])
        pts = decimated
    simp = rdp(pts, epsilon)
    if len(simp) == 1:
        return [(simp[0], simp[0])]
    return [(simp[i], simp[i + 1]) for i in range(len(simp) - 1)]


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
