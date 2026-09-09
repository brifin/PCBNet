"""Stage 2: text binding (Hungarian + 3 constraints), pin building, net graph.

Consumes a VisionOut (components / texts / detected pin tips / wire skeleton) and
emits contest-format JSON. Designed so that with perfect perception (the oracle)
it approaches the designator-recoverable ceiling, and degrades gracefully when a
detector is missing.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from .coords import contest_point, image_bbox_to_contest
from .gt_adapter import load_category_map
from .schema import empty_contest_json
from .stage1_vision import PinBox, VisionOut

POWER_NETS = {"VCC", "GND", "VDD", "VSS", "VPP", "VEE", "AGND", "DGND", "3V3", "5V"}

DESIGNATOR_RE = re.compile(r"^[A-Z]{1,3}\d+$")
DESIGNATOR_LOOSE_RE = re.compile(r"^([A-Za-z]{1,3})[.\s-]*(\d+)$")
VALUE_RE = re.compile(r"\d+(\.\d+)?\s*([kKmMuUnNpPµμ]?[FfHhΩRW]?|[kKmM])\b")
PIN_NUM_RE = re.compile(r"^[A-Za-z]?\d{1,3}$|^[A-Z]\d{1,2}$")

PREFIX_TYPE = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "U": "ic",
    "IC": "ic",
    "Q": "transistor",
    "D": "diode",
    "Y": "crystal",
    "X": "crystal",
    "SW": "switch",
    "S": "switch",
    "J": "connector",
    "P": "connector",
    "CN": "connector",
    "TP": "testpoint",
    "F": "fuse",
    "BT": "battery",
}


def _as_designator(text: str) -> str | None:
    raw = (text or "").strip()
    if DESIGNATOR_RE.match(raw):
        return raw
    m = DESIGNATOR_LOOSE_RE.match(raw.replace(" ", "")) or DESIGNATOR_LOOSE_RE.match(raw)
    if not m:
        return None
    return m.group(1).upper() + m.group(2)


def _prefix_of(designator: str) -> str:
    m = re.match(r"^[A-Z]+", designator or "")
    return m.group(0) if m else ""


def _type_of(label: str, cmap: dict[str, str]) -> str:
    return cmap.get(label, label.lower() if label else "other")


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wv = (p[0] - a[0]) * vx + (p[1] - a[1]) * vy
    vv = vx * vx + vy * vy
    t = 0.0 if vv < 1e-9 else max(0.0, min(1.0, wv / vv))
    return math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy))


def _center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    if not cost or not cost[0]:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(np.array(cost, dtype=float))
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        used_r, used_c, pairs = set(), set(), []
        cells = sorted((cost[i][j], i, j) for i in range(len(cost)) for j in range(len(cost[0])))
        for _, i, j in cells:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
        return pairs


# ----------------------------------------------------------------------------
# Text classification
# ----------------------------------------------------------------------------

def _classify_text(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        return "junk"
    if re.search(r"(Schematic|Sheet|Drawn|SchDoc|Title|\.Sch)", t, re.I):
        return "junk"
    if _as_designator(t):
        return "designator"
    if VALUE_RE.search(t) or re.fullmatch(r"\d+(\.\d+)?[kKmMuUnNpP]", t):
        return "value"
    if re.search(r"[A-Za-z]{2,}\d|\d+[A-Za-z]{2,}", t) and len(t) >= 3:
        return "name"
    if PIN_NUM_RE.match(t):
        return "pinnum"
    return "other"


# ----------------------------------------------------------------------------
# Text-component binding: Hungarian + category / alignment / uniqueness
# ----------------------------------------------------------------------------

def _bind_designators(comps: list[dict], texts: list[dict]) -> None:
    des_idx = [i for i, t in enumerate(texts) if t["cls"] == "designator"]
    if not comps or not des_idx:
        return
    BIG = 1e6
    cost: list[list[float]] = []
    for comp in comps:
        cx, cy = _center(comp["img_bbox"])
        row: list[float] = []
        for i in des_idx:
            t = texts[i]
            tcx, tcy = _center(t["bbox"])
            d = math.hypot(cx - tcx, cy - tcy)
            des = _as_designator(t["content"])
            pref = _prefix_of(des or "")
            expect = PREFIX_TYPE.get(pref, "")
            # Category consistency: forbid clear cross-type prefixes.
            if expect and comp["type"] not in {expect, "other", "ic"} and expect != comp["type"]:
                row.append(BIG)
                continue
            aligned = abs(tcy - cy) < 12 or abs(tcx - cx) < 12
            bonus = (25.0 if aligned else 0.0) + (15.0 if expect == comp["type"] else 0.0)
            row.append(max(0.0, d - bonus))
        cost.append(row)
    for ci, k in _hungarian(cost):
        if ci >= len(comps) or k >= len(des_idx) or cost[ci][k] >= BIG:
            continue
        ti = des_idx[k]
        # Local uniqueness: only accept if reasonably close.
        cx, cy = _center(comps[ci]["img_bbox"])
        tcx, tcy = _center(texts[ti]["bbox"])
        if math.hypot(cx - tcx, cy - tcy) > 220:
            continue
        comps[ci]["designator"] = _as_designator(texts[ti]["content"])
        texts[ti]["used"] = True


def _assign_name_value(comp: dict, texts: list[dict]) -> tuple[str | None, str | None]:
    cx, cy = _center(comp["img_bbox"])
    name = value = None
    cands = sorted(
        (t for t in texts if not t.get("used") and t["cls"] in {"value", "name"}),
        key=lambda t: math.hypot(cx - _center(t["bbox"])[0], cy - _center(t["bbox"])[1]),
    )
    for t in cands:
        d = math.hypot(cx - _center(t["bbox"])[0], cy - _center(t["bbox"])[1])
        if d > 160:
            break
        if comp["type"] in {"resistor", "capacitor", "inductor"} and t["cls"] == "value" and value is None:
            value = t["content"]  # keep original case: 10k != 10K
            t["used"] = True
        elif comp["type"] == "ic" and t["cls"] == "name" and name is None:
            name = t["content"]
            t["used"] = True
    return name, value


# ----------------------------------------------------------------------------
# Pins
# ----------------------------------------------------------------------------

def _geometry_pins(bbox: list[float], ctype: str) -> list[tuple[str, tuple[float, float]]]:
    x1, y1, x2, y2 = bbox
    if ctype in {"resistor", "capacitor", "inductor", "diode", "led", "crystal"}:
        if abs(x2 - x1) >= abs(y2 - y1):
            return [("1", (x1, (y1 + y2) / 2)), ("2", (x2, (y1 + y2) / 2))]
        return [("1", ((x1 + x2) / 2, y1)), ("2", ((x1 + x2) / 2, y2))]
    if ctype in {"gnd", "vcc", "net_label", "testpoint"}:
        return [("1", ((x1 + x2) / 2, y2 if ctype == "gnd" else y1))]
    pins = []
    n = max(1, min(8, int(round((y2 - y1) / 24)) or 4))
    for i in range(n):
        pins.append((str(i + 1), (x1, y1 + (y2 - y1) * (i + 0.5) / n)))
        pins.append((str(n + i + 1), (x2, y1 + (y2 - y1) * (i + 0.5) / n)))
    return pins


def _snap_to_wire(pt: tuple[float, float], vision: VisionOut, eps: float) -> tuple[float, float]:
    best, best_d = pt, eps
    for a, b in vision.wire_segments:
        for cand in (a, b):
            d = _dist(pt, cand)
            if d < best_d:
                best_d, best = d, cand
    return best


def _pins_for_component(
    comp: dict,
    det_pins: list[PinBox],
    texts: list[dict],
    vision: VisionOut,
    snap_px: float,
) -> list[tuple[str, str, tuple[float, float]]]:
    """Return [(pin_number, pinname, xy)] using detected tips if available."""
    x1, y1, x2, y2 = comp["img_bbox"]
    m = 14.0
    owned = [p for p in det_pins if (x1 - m) <= p.xy[0] <= (x2 + m) and (y1 - m) <= p.xy[1] <= (y2 + m)]
    out: list[tuple[str, str, tuple[float, float]]] = []
    if owned:
        seq = 1
        seen: set[str] = set()
        for p in owned:
            xy = _snap_to_wire(p.xy, vision, snap_px)
            number = p.number or _nearest_pinnum(p.xy, texts) or str(seq)
            if number in seen:
                number = str(seq)
            seen.add(number)
            pinname = p.name or _nearest_pinname(p.xy, texts, comp) or number
            out.append((number, pinname, xy))
            seq += 1
        return out
    for number, xy in _geometry_pins(comp["img_bbox"], comp["type"]):
        out.append((number, number, _snap_to_wire(xy, vision, snap_px)))
    return out


def _nearest_pinnum(pt: tuple[float, float], texts: list[dict]) -> str | None:
    best, best_d = None, 22.0
    for t in texts:
        if t.get("used") or t["cls"] not in {"pinnum", "value"}:
            continue
        c = _center(t["bbox"])
        d = _dist(pt, c)
        if d < best_d and PIN_NUM_RE.match(t["content"].strip()):
            best, best_d = t["content"].strip(), d
    return best


def _nearest_pinname(pt: tuple[float, float], texts: list[dict], comp: dict) -> str | None:
    if comp["type"] != "ic":
        return None
    best, best_d = None, 40.0
    for t in texts:
        if t.get("used") or t["cls"] not in {"name", "other"}:
            continue
        c = _center(t["bbox"])
        d = _dist(pt, c)
        if d < best_d and re.search(r"[A-Za-z]", t["content"]):
            best, best_d = t["content"].strip(), d
    return best


# ----------------------------------------------------------------------------
# Net construction
# ----------------------------------------------------------------------------

def _union_find(refs: list[str]):
    parent = {r: r for r in refs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    return parent, find, union


def _connect_pins(pin_nodes: list[dict], vision: VisionOut, eps: float, union) -> None:
    # Prefer skeleton connected components (robust for multi-segment nets).
    if vision.skeleton is not None:
        try:
            import cv2
            import numpy as np

            skel = (vision.skeleton > 0).astype(np.uint8)
            _, labels = cv2.connectedComponents(skel)
            h, w = labels.shape
            radius = max(1, int(eps))
            by_label: dict[int, list[str]] = defaultdict(list)
            for p in pin_nodes:
                x, y = int(round(p["xy"][0])), int(round(p["xy"][1]))
                found = 0
                for r in range(radius + 1):
                    for dy in range(-r, r + 1):
                        yy = y + dy
                        if 0 <= yy < h:
                            for dx in range(-r, r + 1):
                                xx = x + dx
                                if 0 <= xx < w and skel[yy, xx]:
                                    found = int(labels[yy, xx])
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    by_label[found].append(p["ref"])
            for refs in by_label.values():
                for r in refs[1:]:
                    union(refs[0], r)
            return
        except Exception:
            pass
    # Fallback: shared segment adjacency + coincident tips.
    pin_on_seg: dict[int, list[str]] = defaultdict(list)
    for p in pin_nodes:
        for si, (a, b) in enumerate(vision.wire_segments):
            if _seg_dist(p["xy"], a, b) <= eps:
                pin_on_seg[si].append(p["ref"])
    for refs in pin_on_seg.values():
        for r in refs[1:]:
            union(refs[0], r)
    for i, p in enumerate(pin_nodes):
        for q in pin_nodes[i + 1 :]:
            if _dist(p["xy"], q["xy"]) <= eps:
                union(p["ref"], q["ref"])


def _merge_flywires(out: dict, comps: list[dict], union) -> None:
    label_groups: dict[str, list[str]] = defaultdict(list)
    for comp in comps:
        if comp["type"] not in {"net_label", "testpoint"}:
            continue
        label = comp.get("designator") or comp.get("net_label")
        if not label or label.upper() in POWER_NETS:
            continue
        base = re.sub(r"_\d+$", "", label)
        for pk in (out["pins"].get(comp["key"]) or {}):
            label_groups[base].append(f"{comp['key']}.{pk[4:]}")
    for refs in label_groups.values():
        if len(refs) >= 2:
            for r in refs[1:]:
                union(refs[0], r)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_structure(vision: VisionOut, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    cmap = load_category_map()
    thr = cfg.get("thresholds") or {}
    vcfg = cfg.get("vision") or {}
    eps = float(thr.get("pin_wire_eps", 5.0))
    snap_px = float(vcfg.get("pin_snap_px", 5.0))
    height = float(vision.height)
    out = empty_contest_json()

    texts = [{"bbox": t.bbox, "content": t.content, "score": t.score, "cls": _classify_text(t.content)} for t in vision.texts]

    # 构建元件列表 comps
    comps: list[dict] = []
    for det in vision.components:
        comps.append({"type": _type_of(det.label, cmap), "img_bbox": det.bbox, "designator": None, "net_label": None})

    # Reading order (top then left) so duplicate designators number stably.
    comps.sort(key=lambda c: (round(c["img_bbox"][1] / 10), round(c["img_bbox"][0] / 10)))

    _bind_designators(comps, texts)
    # net_label / power text near label-type components

    # 给网络类元件绑定网络标签
    for comp in comps:
        if comp["type"] in {"net_label", "testpoint", "gnd", "vcc"} and not comp.get("designator"):
            cx, cy = _center(comp["img_bbox"])
            for t in texts:
                if t.get("used") or t["cls"] not in {"name", "other", "value"}:
                    continue
                if _dist((cx, cy), _center(t["bbox"])) < 40:
                    comp["net_label"] = t["content"].strip()
                    break

    # 生成元件唯一 key，处理重名
    used_keys: dict[str, int] = {}
    pin_nodes: list[dict] = []
    for i, comp in enumerate(comps):
        des = comp.get("designator")
        raw_key = des or comp.get("net_label") or f"{comp['type']}_{i+1}"
        # Keep type consistent with a recovered designator prefix.
        if des:
            mapped = PREFIX_TYPE.get(_prefix_of(des))
            if mapped:
                comp["type"] = mapped
        used_keys[raw_key] = used_keys.get(raw_key, 0) + 1
        key = raw_key if used_keys[raw_key] == 1 else f"{raw_key}_{used_keys[raw_key]}"
        comp["key"] = key

        # 填充元件信息，提取引脚
        name, value = _assign_name_value(comp, texts)
        out["components"][key] = {
            "type": comp["type"],
            "name": name,
            "value": value,
            "bbox": image_bbox_to_contest(comp["img_bbox"], height),
        }
        out["pins"][key] = {}
        seen_pk: set[str] = set()
        for number, pinname, xy in _pins_for_component(comp, vision.pins, texts, vision, snap_px):
            pk = f"pin_{number}"
            if pk in seen_pk:
                continue
            seen_pk.add(pk)
            out["pins"][key][pk] = {"pinname": pinname, "point": contest_point(xy[0], xy[1], height)}
            pin_nodes.append({"ref": f"{key}.{number}", "xy": xy})

    # 并查集做引脚连通分析
    parent, find, union = _union_find([p["ref"] for p in pin_nodes])
    _connect_pins(pin_nodes, vision, eps, union)
    _merge_flywires(out, comps, union)

    # 按连通根分组，绑定导线 segments，输出 nets
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in pin_nodes:
        groups[find(p["ref"])].append(p)

    for i, (_root, members) in enumerate(sorted(groups.items()), start=1):
        refs = [m["ref"] for m in members]
        pts = {m["ref"]: m["xy"] for m in members}
        segs: list[list[dict]] = []
        for a, b in vision.wire_segments:
            for ref in refs:
                if _seg_dist(pts[ref], a, b) <= eps:
                    segs.append([contest_point(a[0], a[1], height), contest_point(b[0], b[1], height)])
                    break
            if len(segs) >= 200:
                break
        if not segs:  # flywire / direct pin-pin: zero-length point edges
            for ref in refs:
                p = contest_point(pts[ref][0], pts[ref][1], height)
                segs.append([p, p])
        edges = {f"edge_{j}": seg for j, seg in enumerate(segs, start=1)}
        out["nets"][f"NET_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": edges}

    return out
