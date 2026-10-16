"""Stage 2: text binding (Hungarian + 3 constraints), pin building, net graph.

Consumes a VisionOut (components / texts / detected pin tips / wire skeleton) and
emits contest-format JSON. Designed so that with perfect perception (the oracle)
it approaches the designator-recoverable ceiling, and degrades gracefully when a
detector is missing.
"""

from __future__ import annotations

import math
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .coords import contest_point, image_bbox_to_contest
from .gt_adapter import load_category_map
from .schema import empty_contest_json
from .stage1_vision import PinBox, VisionOut
from .visual import run_wire_visualize

POWER_NETS = {"VCC", "GND", "VDD", "VSS", "VPP", "VEE", "AGND", "DGND", "3V3", "5V"}

DESIGNATOR_RE = re.compile(r"^[A-Z]{1,3}\d{1,3}$")
DESIGNATOR_LOOSE_RE = re.compile(r"^([A-Za-z]{1,3})[.\s-]*(\d{1,3})$")
DESIGNATOR_FIND_RE = re.compile(r"\b([A-Za-z]{1,3})[.\s-]*(\d{1,3})\b")
VALUE_RE = re.compile(r"\d+(\.\d+)?\s*([kKmMuUnNpPµμ]?[FfHhΩRW]?|[kKmM])\b")
PIN_NUM_RE = re.compile(r"^[A-Za-z]?\d{1,3}$|^[A-Z]\d{1,2}$")
PACKAGE_RE = re.compile(r"^(0201|0402|0603|0805|1206|1210|2512)$")

PREFIX_TYPE = {
    "R": "r",
    "C": "c",
    "L": "l",
    "U": "box",
    "IC": "box",
    "Q": "mosfet",
    "D": "d",
    "Y": "crystal_2pin",
    "X": "crystal_2pin",
    "SW": "switch",
    "S": "switch",
    "J": "pin",
    "P": "pin",
    "CN": "pin",
    "TP": "testpoint",
    "F": "fuse",
    "BT": "battary",
}
# Valid designator prefixes. K is a button/key ref (GT type is often `other`, so
# it is allowed as a key but never used to overwrite the detected type).
VALID_PREFIX = set(PREFIX_TYPE) | {"K"}
TYPE_OVERRIDE_PREFIX = {"R", "C", "L", "U", "IC", "Q", "D", "Y", "X", "J", "P", "CN", "TP", "F", "BT"}


# ----------------------------------------------------------------------------
# Knowledge base (type <-> key prefix, from 10gt/type_list.md)
# ----------------------------------------------------------------------------

_KB_CACHE = None


def _load_kb() -> dict:
    global _KB_CACHE
    if _KB_CACHE is None:
        try:
            from .knowledge_base import load_key_type_kb

            _KB_CACHE = load_key_type_kb()
        except Exception:
            _KB_CACHE = {}
    return _KB_CACHE


def _canonical_prefixes(ctype: str) -> list[str]:
    """Canonical key prefixes for a component type (KB first, hand-coded fallback)."""
    kb = _load_kb()
    prefs = kb.get("canonical_prefix", {}).get(ctype, [])
    if prefs:
        return prefs
    return [p for p, t in PREFIX_TYPE.items() if t == ctype]


def _type_for_prefix(prefix: str) -> str:
    """Dominant type for a key prefix (KB first, hand-coded fallback)."""
    kb = _load_kb()
    dist = kb.get("prefix_to_type", {}).get(prefix, {})
    if dist:
        return max(dist, key=dist.get)
    return PREFIX_TYPE.get(prefix, "")


_BOX_NAMES_CACHE = None


def _known_box_names() -> set:
    """Whitelist of box-type model names (BoxICLibrary.json)."""
    global _BOX_NAMES_CACHE
    if _BOX_NAMES_CACHE is None:
        try:
            from .knowledge_base import known_box_names

            _BOX_NAMES_CACHE = known_box_names()
        except Exception:
            _BOX_NAMES_CACHE = set()
    return _BOX_NAMES_CACHE


def _strip_token(text: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", (text or "").strip())


_TYPE_PRIOR_CACHE: dict[str, dict] | None = None


def _load_type_priors() -> dict[str, dict]:
    """惰性加载 data/type_priors.json（type -> {pin_count, text_count, ...}）。"""
    global _TYPE_PRIOR_CACHE
    if _TYPE_PRIOR_CACHE is None:
        p = Path(__file__).resolve().parent.parent / "data" / "type_priors.json"
        try:
            _TYPE_PRIOR_CACHE = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _TYPE_PRIOR_CACHE = {}
    return _TYPE_PRIOR_CACHE


def _type_pin_prior(ctype: str) -> int | None:
    """返回该 type 的先验引脚数（众数），box 或无先验返回 None。"""
    if ctype == "box":
        return None
    rec = _load_type_priors().get(ctype)
    if not rec:
        return None
    n = rec.get("pin_count")
    return int(n) if n is not None else None


def _as_designator(text: str) -> str | None:
    """Accept only real schematic refs (R5, U3, C21), not packages / part numbers / net names."""
    raw = _strip_token(text)
    if not raw:
        return None
    compact = raw.replace(" ", "")
    cand = None
    if DESIGNATOR_RE.match(raw.upper()) and raw[0].isalpha():
        cand = raw.upper()
    else:
        m = DESIGNATOR_LOOSE_RE.match(compact) or DESIGNATOR_LOOSE_RE.match(raw)
        if m:
            cand = m.group(1).upper() + m.group(2)
    if not cand:
        return None
    pref = _prefix_of(cand)
    digits = cand[len(pref) :]
    if pref not in VALID_PREFIX:
        return None
    if PACKAGE_RE.match(digits):
        return None
    if len(digits) > 3:
        return None
    return cand


def _extract_designators(text: str) -> list[str]:
    """Whole-string match first, then tokens inside mixed OCR like 'RTC_I2C_SDA R5'."""
    found: list[str] = []
    whole = _as_designator(text)
    if whole:
        found.append(whole)
    for tok in re.split(r"[\s,;:/|]+", text or ""):
        d = _as_designator(tok)
        if d and d not in found:
            found.append(d)
    for m in DESIGNATOR_FIND_RE.finditer(text or ""):
        d = _as_designator(m.group(1) + m.group(2))
        if d and d not in found:
            found.append(d)
    return found


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


def _bbox_gap(a: list[float], b: list[float]) -> float:
    """Min gap between two axis-aligned boxes (0 if they overlap)."""
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


def _end_boxes(bbox: list[float]) -> list[list[float]]:
    """Left and right 30% slices — mixed OCR often puts the ref at one end."""
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    cut = 0.30 * w
    return [[x1, y1, x1 + cut, y2], [x2 - cut, y1, x2, y2]]


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

# 有 value 的 type（统计训练 300 + 测试 10 全部数据，value != null 的 type 全集）：
#   c / r / v / gnd / l / crystal_2pin / crystal_3pin / crystal_4pin / led / circle_header / crystal
# 其中 gnd/v 的 value 是电源网络名（GND、3.3V、+5V…），由 build_structure 的 net_label 逻辑写入 value，
# 不在此处参与 _assign_values（否则会把附近电容/电阻的 value 文本错绑到 gnd/v 符号上）。
_VALUE_TYPES = {"r", "c", "l", "v", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin",
                "led", "circle_header"}


def _looks_value(txt: str) -> bool:
    t = (txt or "").strip().lstrip("=>~")
    if not t or _as_designator(t):
        return False
    if PACKAGE_RE.match(t) or re.fullmatch(r"0[468]0[235]|1206|1210|2512", t):
        return False
    # Net-style tokens (KEY1, SCL) are not values even if they contain a digit.
    if re.fullmatch(r"[A-Z]{2,}\d*", t):
        return False
    if re.fullmatch(r"[+-]?\d+(\.\d+)?\s*[kKmMuUnNpPµμ]?[FfHhΩRWVv]", t):
        return True
    if re.fullmatch(r"[+-]?\d+(\.\d+)?[kKmMuUnNpP]", t):
        return True
    if re.search(r"[+-]?\d+(\.\d+)?\s*[kKmMuUnNpPµμFfHhΩRWVv]/?", t):
        return True
    return False


def _classify_text(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        return "junk"
    if re.search(r"(Schematic|Sheet|Drawn|SchDoc|Title|\.Sch|CSDN|@)", t, re.I):
        return "junk"
    if _as_designator(t):
        return "designator"
    if _extract_designators(t):
        return "mixed"
    if _looks_value(t):
        return "value"
    if re.search(r"[A-Za-z]{2,}\d|\d+[A-Za-z]{2,}", t) and len(t) >= 3:
        return "name"
    if PIN_NUM_RE.match(t):
        return "pinnum"
    return "other"


def _prepare_texts(raw: list[dict]) -> list[dict]:
    """Classify OCR tokens and explode mixed strings into virtual designator texts."""
    out: list[dict] = []
    for t in raw:
        content = (t.get("content") or "").strip()
        cls = _classify_text(content)
        item = {
            "bbox": list(t["bbox"]),
            "content": content,
            "score": float(t.get("score") or 1.0),
            "cls": cls,
            "used": False,
        }
        if cls == "designator":
            item["content"] = _as_designator(content) or content
            out.append(item)
            continue
        if cls == "mixed":
            for des in _extract_designators(content):
                for box in _end_boxes(item["bbox"]):
                    out.append(
                        {
                            "bbox": box,
                            "content": des,
                            "score": item["score"],
                            "cls": "designator",
                            "used": False,
                            "virtual": True,
                        }
                    )
            rest = content
            for des in _extract_designators(content):
                rest = re.sub(re.escape(des), " ", rest, flags=re.I)
            rest = rest.strip(" -_/|")
            if rest and _looks_value(rest):
                out.append({**item, "content": rest, "cls": "value"})
            elif rest:
                out.append({**item, "content": rest, "cls": "name"})
            continue
        out.append(item)
    return out


# ----------------------------------------------------------------------------
# Text-component binding: Hungarian + category / alignment / uniqueness
# ----------------------------------------------------------------------------

def _prefix_ok_for_type(pref: str, ctype: str) -> bool:
    """A component's key prefix must be one of its type's valid formats (from the KB).

    YOLO detects the type first; the OCR'd designator text is only accepted if its
    prefix matches the type's canonical prefixes (e.g. switch only accepts SW/S, so a
    'K1' text is rejected here and later corrected to 'SW1')."""
    prefs = _canonical_prefixes(ctype)
    return bool(prefs) and pref in prefs


def _correct_key_format(text: str, ctype: str) -> str | None:
    """Apply the type -> key-format prior to a matched text.

    Extracts a `prefix+number` designator. If the text's prefix is missing or not
    among the type's valid prefixes, replace it with the type's most common prefix,
    keeping the number from the text (e.g. '84' near a box -> 'U84'; 'P4' near a
    resistor -> 'R4'). Returns None when the text carries no number info, so the
    caller falls back to free numbering (∅)."""
    prefs = _canonical_prefixes(ctype)
    if not prefs:
        return None  # type has no named format -> always ∅ (free numbering)
    raw = _strip_token(text)
    if not raw:
        return None
    m = re.match(r"^([A-Za-z]*)[.\s-]*(\d{1,3})$", raw)
    if not m:
        return None  # no number info -> free numbering
    prefix, number = m.group(1).upper(), m.group(2)
    if PACKAGE_RE.match(number.zfill(4)):
        return None
    if prefix and prefix in prefs:
        return prefix + number  # already valid format
    return prefs[0] + number  # wrong/missing prefix -> most common format


def _bind_designators(comps: list[dict], texts: list[dict]) -> None:
    des_idx = [i for i, t in enumerate(texts) if t["cls"] == "designator" and not t.get("used")]
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
            d = _bbox_gap(comp["img_bbox"], t["bbox"]) + 0.25 * math.hypot(cx - tcx, cy - tcy)
            des = _as_designator(t["content"])
            pref = _prefix_of(des or "")
            if des and not _prefix_ok_for_type(pref, comp["type"]):
                row.append(BIG)
                continue
            aligned = abs(tcy - cy) < 16 or abs(tcx - cx) < 16
            expect = PREFIX_TYPE.get(pref, "")
            bonus = (25.0 if aligned else 0.0) + (20.0 if expect and expect == comp["type"] else 0.0)
            row.append(max(0.0, d - bonus))
        cost.append(row)
    assigned_des: set[str] = set()
    for ci, k in _hungarian(cost):
        if ci >= len(comps) or k >= len(des_idx) or cost[ci][k] >= BIG:
            continue
        if comps[ci].get("designator"):
            continue
        ti = des_idx[k]
        if _bbox_gap(comps[ci]["img_bbox"], texts[ti]["bbox"]) > 80:
            cx, cy = _center(comps[ci]["img_bbox"])
            tcx, tcy = _center(texts[ti]["bbox"])
            if math.hypot(cx - tcx, cy - tcy) > 180:
                continue
        des = _as_designator(texts[ti]["content"])
        if not des or des in assigned_des:
            continue
        comps[ci]["designator"] = des
        texts[ti]["used"] = True
        assigned_des.add(des)
    _bind_leftover_designators(comps, texts, assigned_des)


def _bind_leftover_designators(comps: list[dict], texts: list[dict], assigned: set[str]) -> None:
    """Second pass: leftover components pick the nearest unused matching ref."""
    unused = [t for t in texts if t["cls"] == "designator" and not t.get("used")]
    for comp in comps:
        if comp.get("designator"):
            continue
        best, best_d = None, 70.0
        for t in unused:
            if t.get("used"):
                continue
            des = _as_designator(t["content"])
            if not des or des in assigned or not _prefix_ok_for_type(_prefix_of(des), comp["type"]):
                continue
            d = _bbox_gap(comp["img_bbox"], t["bbox"])
            if d < best_d:
                best, best_d = t, d
        if best is not None:
            des = _as_designator(best["content"])
            if des:
                comp["designator"] = des
                best["used"] = True
                assigned.add(des)

    # Third pass (sub-A): correct a wrong-prefix designator (e.g. 'K4' near a switch
    # -> 'SW4'). These carry an explicit number, so they take priority over bare
    # numbers (which are usually pin labels, not designators).
    for comp in comps:
        if comp.get("designator"):
            continue
        best, best_d = None, 70.0
        for t in texts:
            if t.get("used") or t["cls"] != "designator":
                continue
            des = _correct_key_format(t["content"], comp["type"])
            if not des or des in assigned:
                continue
            d = _bbox_gap(comp["img_bbox"], t["bbox"])
            if d < best_d:
                best, best_d = (des, t), d
        if best is not None:
            des, t = best
            comp["designator"] = des
            t["used"] = True
            assigned.add(des)

    # Third pass (sub-B): bare-number OCR (letter prefix dropped) -> prepend the
    # type's most common prefix (e.g. '84' -> 'U84'). Only for components still
    # without a designator. No number info -> stays ∅ (free numbering from 1).
    for comp in comps:
        if comp.get("designator"):
            continue
        best, best_d = None, 70.0
        for t in texts:
            if t.get("used") or t["cls"] != "pinnum":
                continue
            des = _correct_key_format(t["content"], comp["type"])
            if not des or des in assigned:
                continue
            d = _bbox_gap(comp["img_bbox"], t["bbox"])
            if d < best_d:
                best, best_d = (des, t), d
        if best is not None:
            des, t = best
            comp["designator"] = des
            t["used"] = True
            assigned.add(des)


def _assign_values(comps: list[dict], texts: list[dict]) -> None:
    """One-to-one value binding for types that carry a value (r/c/l/v/crystal*/led/...)。

    IC names are not written onto the contest record: local GT names are empty
    ~75% of the time, and a filled name zeros an otherwise correct U5/R29-style match.
    """
    val_idx = [i for i, t in enumerate(texts) if not t.get("used") and t["cls"] == "value" and _looks_value(t["content"])]
    # 有 value 的 type（即使部分有值也视为有，如 v 的 3.3V、crystal 的 12MHz）
    targets = [c for c in comps if c["type"] in _VALUE_TYPES]
    if not targets or not val_idx:
        return
    BIG = 1e6
    cost = []
    for comp in targets:
        row = []
        for i in val_idx:
            d = _bbox_gap(comp["img_bbox"], texts[i]["bbox"])
            row.append(d if d <= 90 else BIG)
        cost.append(row)
    used_t: set[int] = set()
    for ci, k in _hungarian(cost):
        if ci >= len(targets) or k >= len(val_idx) or cost[ci][k] >= BIG:
            continue
        if k in used_t:
            continue
        ti = val_idx[k]
        targets[ci]["value"] = texts[ti]["content"].strip().lstrip("=>~")
        texts[ti]["used"] = True
        used_t.add(k)


def _as_ic_partname(text: str, known: set[str] | None = None) -> str | None:
    """Chip PNs used as contest keys when the sheet has no U-number (e.g. LME49723).

    Prefers known box names from the KB (whitelist), so unrelated OCR text is not
    bound to a box component; falls back to the part-number regex shape."""
    raw = _strip_token(text)
    if not raw:
        return None
    if known and raw in known:
        return raw
    if len(raw) < 5:
        return None
    if not re.fullmatch(r"[A-Za-z]{2,6}\d{3,6}[A-Za-z0-9]*", raw):
        return None
    digits = re.sub(r"^[A-Za-z]+", "", raw)
    if PACKAGE_RE.match(digits[:4] if len(digits) >= 4 else digits):
        return None
    if re.match(r"(KEY|SCL|SDA|VDD|VSS|GND|VCC)", raw, re.I):
        return None
    return raw


def _nearby_ic_partname(comp: dict, texts: list[dict]) -> str | None:
    """Chip PN (box Name) near an IC, verified against the box-name KB whitelist.

    Only a text whose content (or first token) is a known box name in
    BoxICLibrary.json is accepted, so unrelated OCR text is not bound to the box."""
    if comp["type"] != "box":
        return None
    known = _known_box_names()

    def _score(t: dict, pn: str) -> float:
        d = _bbox_gap(comp["img_bbox"], t["bbox"])
        cx, cy = _center(comp["img_bbox"])
        tcx, tcy = _center(t["bbox"])
        return min(d, math.hypot(cx - tcx, cy - tcy))

    best, best_d = None, 220.0
    for t in texts:
        if t.get("used"):
            continue
        content = (t.get("content") or "").strip()
        for token in (content.split()[0], content):
            pn = _strip_token(token)
            if pn in known:
                d = _score(t, pn)
                if d < best_d:
                    best, best_d = pn, d
                break
    return best


def _nearby_partname(comp: dict, texts: list[dict]) -> str | None:
    """元件附近型号文字：box 用 BoxICLibrary 白名单，二极管(d) 用 1Nxxxx 型号正则。"""
    if comp["type"] == "box":
        return _nearby_ic_partname(comp, texts)
    if comp["type"] == "d":
        best, best_d = None, 120.0
        for t in texts:
            if t.get("used"):
                continue
            content = _strip_token((t.get("content") or "").strip())
            if re.fullmatch(r"1N\d{2,}[A-Za-z]*", content, re.I):
                d = _bbox_gap(comp["img_bbox"], t["bbox"])
                if d < best_d:
                    best, best_d = content, d
        return best
    return None


# ----------------------------------------------------------------------------
# Pins
# ----------------------------------------------------------------------------

def _geometry_pins(bbox: list[float], ctype: str) -> list[tuple[str, tuple[float, float]]]:
    x1, y1, x2, y2 = bbox
    if ctype in {"r", "c", "l", "d", "led", "crystal_2pin"}:
        if abs(x2 - x1) >= abs(y2 - y1):
            return [("1", (x1, (y1 + y2) / 2)), ("2", (x2, (y1 + y2) / 2))]
        return [("1", ((x1 + x2) / 2, y1)), ("2", ((x1 + x2) / 2, y2))]
    if ctype in {"gnd", "v", "net_input", "testpoint"}:
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


def _order_pins_ccw(pins: list[PinBox], cx: float) -> list[PinBox]:
    """Counterclockwise: left edge top->bottom, then right edge bottom->top."""
    left = sorted([p for p in pins if p.xy[0] <= cx], key=lambda p: p.xy[1])
    right = sorted([p for p in pins if p.xy[0] > cx], key=lambda p: -p.xy[1])
    return left + right


def _dedup_pins(pins: list[PinBox], eps: float) -> list[PinBox]:
    """去掉坐标过近（≤ eps px）的重复引脚，保留置信度最高者。

    YOLO 有时在同一位置（很近）检出两个 pin_tip，或匹配错乱导致同一引脚重复；
    这会让非 box 元件多出 pin_1/pin_2 两个几乎重叠的端点，进而拉低 PinF1。
    按置信度降序贪心：每个 pin 若与已保留的任一 pin 距离 ≤ eps 则丢弃。
    """
    if not pins:
        return []
    ordered = sorted(pins, key=lambda p: getattr(p, "score", 0.0), reverse=True)
    keep: list[PinBox] = []
    for p in ordered:
        if any(_dist(p.xy, k.xy) <= eps for k in keep):
            continue
        keep.append(p)
    return keep


def _box_library_pins(ic_name: str | None, n_pins: int) -> list[tuple[str, str]] | None:
    """Return [(number, name), ...] from BoxICLibrary.json for a box IC, or None.

    Pins are returned in ascending numeric EDA order (library `number`), which is the
    counterclockwise order for most IC symbols. Non-numeric numbers sort last."""
    if not ic_name:
        return None
    try:
        from .pin_library import lookup_pins
    except Exception:
        return None
    lib = lookup_pins(ic_name)
    if not lib:
        return None

    def _key(n: str) -> tuple:
        return (0, int(n)) if n.isdigit() else (1, n)

    ordered = sorted(lib.items(), key=lambda kv: _key(kv[0]))
    return ordered[:n_pins]


def _pins_for_component(
    comp: dict,
    det_pins: list[PinBox],
    texts: list[dict],
    vision: VisionOut,
    snap_px: float,
    ic_name: str | None = None,
    pin_dedup_px: float = 5.0,
) -> list[tuple[str, str, tuple[float, float]]]:
    """Return [(pin_number, pinname, xy)] using detected tips if available.

    box 类优先用 BoxICLibrary.json 查 pinname（按名字查号），查不到则 OCR/弱先验
    （左 1 逆时针）；非 box 空编号引脚 pin key 用 pin_1/pin_2…（只比端点坐标）。
    """
    x1, y1, x2, y2 = comp["img_bbox"]
    # box 引脚伸出方形框外更远（实测最多 ~22px），扩大搜索范围避免漏检
    m = 32.0 if comp["type"] == "box" else 14.0
    owned = [p for p in det_pins if (x1 - m) <= p.xy[0] <= (x2 + m) and (y1 - m) <= p.xy[1] <= (y2 + m)]
    # 去重：YOLO 可能在同一位置（很近）检测到两个 pin_tip，导致同一引脚重复标注。
    # 按坐标距离 ≤ pin_dedup_px 的引脚只保留置信度最高的一个。
    owned = _dedup_pins(owned, pin_dedup_px)
    out: list[tuple[str, str, tuple[float, float]]] = []
    if comp["type"] == "box":
        if not owned:
            # 无检出引脚：几何 fallback + 库顺序（保留原行为）
            cx = (x1 + x2) / 2.0
            n = max(1, min(8, int(round((y2 - y1) / 24)) or 4))
            owned = []
            for i in range(n):
                owned.append(PinBox(xy=(x1, y1 + (y2 - y1) * (i + 0.5) / n)))
                owned.append(PinBox(xy=(x2, y1 + (y2 - y1) * (i + 0.5) / n)))
            owned = _order_pins_ccw(owned, cx)
            lib = _box_library_pins(ic_name, len(owned))
            seen = set()
            for i, p in enumerate(owned):
                xy = _snap_to_wire(p.xy, vision, snap_px)
                number, pinname = (lib[i] if lib and i < len(lib) else (str(i + 1), str(i + 1)))
                if number in seen:
                    number = str(i + 1)
                seen.add(number)
                out.append((number, pinname, xy))
            return out

        # 新规则（box 引脚）：key 为 pin_<number>，number 是引脚编号，pinname 是引脚名称。
        # 1. 先匹配 pinname（元件框内部文字），再匹配 number（引脚附近数字）
        # 2. 优先使用图中信息（OCR 文字）；只有图中信息缺失时才对照 BoxICLibrary
        # 3. number 缺失且 pinname 对应多个 number 时，按几何顺序分配 number
        try:
            from .pin_library import lookup_pins
        except Exception:
            lookup_pins = lambda *a, **k: {}  # type: ignore[assignment]
        lib = lookup_pins(ic_name) if ic_name else {}   # {number: name}
        # name -> [numbers]（数值升序），用于同名多编号的几何顺序分配
        name_to_nums: dict[str, list[str]] = defaultdict(list)
        for num in sorted(lib, key=lambda n: (0, int(n)) if n.isdigit() else (1, n)):
            name_to_nums[lib[num]].append(num)

        cx = (x1 + x2) / 2.0
        owned = _order_pins_ccw(owned, cx)  # 逆时针几何顺序

        # 第一遍：先 pinname（图中文字），后 number（图中数字）
        items = []
        for p in owned:
            xy = _snap_to_wire(p.xy, vision, snap_px)
            pinname = p.name or _nearest_pinname(p.xy, texts, comp)  # 图中文字（引脚名）
            number = p.number or _nearest_pinnum(p.xy, texts)         # 图中数字（引脚编号）
            items.append({"pinname": pinname, "number": number, "xy": xy})

        # 第二遍：图中信息优先；缺失时才对照 BoxICLibrary
        name_assigned: dict[str, int] = defaultdict(int)
        seen: set[str] = set()
        for item in items:
            pinname, number, xy = item["pinname"], item["number"], item["xy"]
            if number is None and pinname and pinname in name_to_nums:
                # 图中 number 缺失：用 pinname 反查库（同名多编号按几何顺序分配）
                nums = name_to_nums[pinname]
                idx = name_assigned[pinname]
                number = nums[idx] if idx < len(nums) else nums[-1]
                name_assigned[pinname] += 1
            elif pinname is None and number and number in lib:
                # 图中 pinname 缺失：用 number 查库 pinname
                pinname = lib[number]
            if number is None:
                number = str(len(seen) + 1)
            while number in seen:
                number = str(int(number) + 1) if number.isdigit() else f"{number}_{len(seen)+1}"
            seen.add(number)
            out.append((number, pinname or number, xy))
        return out
    if owned:
        # 非 box 引脚无特殊名称：pin key/name 统一 pin_1/pin_2…（从 1 开始），
        # 不做 OCR 名称匹配（避免把周围文字塞进 pin 名），只比端点坐标。
        # 用 type 先验（众数引脚数）限制引脚数量：多检时按置信度保留 top-N。
        prior = _type_pin_prior(comp["type"])
        if prior is not None and len(owned) > prior:
            owned = sorted(owned, key=lambda p: getattr(p, "score", 0.0), reverse=True)[:prior]
        for seq, p in enumerate(owned, start=1):
            xy = _snap_to_wire(p.xy, vision, snap_px)
            out.append((str(seq), str(seq), xy))
        return out
    for number, xy in _geometry_pins(comp["img_bbox"], comp["type"]):
        out.append((number, number, _snap_to_wire(xy, vision, snap_px)))
    return out


def _nearest_pinnum(pt: tuple[float, float], texts: list[dict]) -> str | None:
    best, best_d = None, 60.0
    for t in texts:
        if t.get("used") or t["cls"] not in {"pinnum", "value"}:
            continue
        c = _center(t["bbox"])
        d = _dist(pt, c)
        if d < best_d and PIN_NUM_RE.match(t["content"].strip()):
            best, best_d = t["content"].strip(), d
    return best


def _nearest_pinname(pt: tuple[float, float], texts: list[dict], comp: dict) -> str | None:
    if comp["type"] != "box":
        return None
    x1, y1, x2, y2 = comp["img_bbox"]
    m = 30.0  # 元件框扩展余量：pinname 文字大部分在框内
    # 先框内搜索（避免误匹配框外文字）
    best, best_d = None, 1e9
    for t in texts:
        if t.get("used") or t["cls"] not in {"name", "other"}:
            continue
        c = _center(t["bbox"])
        if not (x1 - m <= c[0] <= x2 + m and y1 - m <= c[1] <= y2 + m):
            continue
        d = _dist(pt, c)
        if d < best_d and re.search(r"[A-Za-z]", t["content"]):
            best, best_d = t["content"].strip(), d
    if best is not None:
        return best
    # 框内没找到：放宽到引脚点附近 100px 兜底（覆盖少部分框外的 pinname）
    best_d = 100.0
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
        if comp["type"] not in {"net_input", "testpoint"}:
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


def _build_nets_from_wire_nets(
    out: dict,
    vision: VisionOut,
    pin_nodes: list[dict],
    height: float,
    match_tol: float = 12.0,
) -> None:
    """用 topo_graph.extract_wire_nets 的结果生成最终 nets（替代旧 union-find）。

    extract_wire_nets 输出形如
        {net_j: {"all_pins": [{"x","y","type"}], "all_node_idxs": {...},
                 "edges": {"edge_i": [{"x","y"}, {"x","y"}]}}}
    其中 all_pins 是连通在同一网络上的引脚节点（image 坐标）。这里把每个引脚坐标
    最近邻匹配到 pin_nodes 的 ref（component.pin_number）得到 hyperGraph；edges 转 contest 坐标。
    net key 统一 nets_<n>，edge 统一 edge_<n>（顺序不判分）。
    """
    wire_nets = getattr(vision, "wire_nets", None) or {}
    if not wire_nets:
        return

    coords = [p["xy"] for p in pin_nodes]
    refs_all = [p["ref"] for p in pin_nodes]

    net_no = 0
    for net_key in sorted(wire_nets.keys()):
        net = wire_nets[net_key]
        all_pins = net.get("all_pins") or []
        seen: set[str] = set()
        refs: list[str] = []
        for pin in all_pins:
            px, py = float(pin["x"]), float(pin["y"])
            best_i, best_d = -1, match_tol
            for i, (cx, cy) in enumerate(coords):
                d = _dist((px, py), (cx, cy))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                ref = refs_all[best_i]
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
        if not refs:
            continue  # 纯端点网络（未匹配到元件引脚）不输出

        net_no += 1
        edges: dict = {}
        for j, (_ek, seg) in enumerate(sorted(net.get("edges", {}).items()), start=1):
            edges[f"edge_{j}"] = [contest_point(float(pt["x"]), float(pt["y"]), height) for pt in seg]
        out["nets"][f"net_{net_no}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": edges}


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
    pin_dedup_px = float(vcfg.get("pin_dedup_px", 5.0))
    height = float(vision.height)
    out = empty_contest_json()

    texts = _prepare_texts([{"bbox": t.bbox, "content": t.content, "score": t.score} for t in vision.texts])

    comps: list[dict] = []
    for det in vision.components:
        comps.append({"type": _type_of(det.label, cmap), "img_bbox": det.bbox, "designator": None, "net_label": None})

    # Reading order (top then left) so duplicate designators number stably.
    comps.sort(key=lambda c: (round(c["img_bbox"][1] / 10), round(c["img_bbox"][0] / 10)))

    _bind_designators(comps, texts)
    _assign_values(comps, texts)
    # net_label / power text near label-type components
    # gnd/v 的 value 即电源网络名（GND、3.3V、+5V…），在这里顺带写入 value，
    # 否则 "GND" 会被 _classify_text 归为 other，_assign_values 永远匹配不到。
    for comp in comps:
        if comp["type"] in {"net_input", "testpoint", "gnd", "v"} and not comp.get("designator"):
            cx, cy = _center(comp["img_bbox"])
            for t in texts:
                if t.get("used") or t["cls"] not in {"name", "other", "value"}:
                    continue
                if _dist((cx, cy), _center(t["bbox"])) < 40:
                    label = t["content"].strip()
                    comp["net_label"] = label
                    if comp["type"] in {"gnd", "v"} and not comp.get("value"):
                        comp["value"] = label
                    break

    used_keys: dict[str, int] = {}
    empty_seq = 0
    pin_nodes: list[dict] = []
    for i, comp in enumerate(comps):
        des = comp.get("designator")
        # box/二极管类型：Name 用附近型号名（其余类型 Name == key）
        ic_pn = _nearby_partname(comp, texts) if comp["type"] in {"box", "d"} else None
        if ic_pn:
            for t in texts:
                if not t.get("used") and _strip_token((t.get("content") or "").split()[0] if t.get("content") else "") == ic_pn:
                    t["used"] = True
                    break
        # key/name 服从 YOLO 检测出的 type：位号在绑定时已经过 _prefix_ok_for_type
        # 校验（前缀必须是该 type 的合法格式），此处直接用，不再反推覆盖 type。
        if des:
            used_keys[des] = used_keys.get(des, 0) + 1
            key = des if used_keys[des] == 1 else f"{des}_{used_keys[des]}"
        else:
            # 无位号元件：赛题规则 key 为 ∅<编号>（编号自由/顺序）
            empty_seq += 1
            key = f"∅{empty_seq}"
        comp["key"] = key

        out["components"][key] = {
            "Name": ic_pn or key,
            "type": comp["type"],
            "value": comp.get("value"),
            "bbox": image_bbox_to_contest(comp["img_bbox"], height),
        }
        out["pins"][key] = {}
        seen_pk: set[str] = set()
        for number, pinname, xy in _pins_for_component(comp, vision.pins, texts, vision, snap_px, ic_name=ic_pn, pin_dedup_px=pin_dedup_px):
            pk = f"pin_{number}"
            if pk in seen_pk:
                continue
            seen_pk.add(pk)
            out["pins"][key][pk] = {"pinname": pinname, "point": contest_point(xy[0], xy[1], height)}
            pin_nodes.append({"ref": f"{key}.{number}", "xy": xy})

    parent, find, union = _union_find([p["ref"] for p in pin_nodes])
    _connect_pins(pin_nodes, vision, eps, union)
    _merge_flywires(out, comps, union)

    # 优先用 topo_graph.extract_wire_nets 的重建结果生成 nets；为空则退回旧的 union-find 分段
    if getattr(vision, "wire_nets", None):
        _build_nets_from_wire_nets(out, vision, pin_nodes, height)
        return out

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
        out["nets"][f"net_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": edges}

    # run_wire_visualize(out, vision.img, vision.image_path)
    return out
