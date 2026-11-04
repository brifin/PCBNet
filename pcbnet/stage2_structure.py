"""Stage 2: value-first text binding, loose key/Name assignment, pins, net graph.

文字匹配只依赖：
  - data/inference_knowledge/key_type.json     （训练侧 type ↔ key 前缀，软约束）
  - data/inference_knowledge/value_priors.json （训练侧 value 格式/占比）
  - data/inference_knowledge/box_pins.json     （训练侧型号 ↔ pin_number/pin_name）
全部通过 provenance.json 校验训练来源与 SHA256；缺失不回退历史测试侧资产。

流程：先用 priors 判断并绑定 value（没有合法 value 时写 null），再把剩余 OCR 文字
按最近邻分给 key/Name；只有一个文字时 key=Name，没有剩余文字时用 ∅ 自由命名。
除 ∅ 和 BoxICLibrary 的引脚名外，不写任何 OCR 中不存在的文字。
"""

from __future__ import annotations

import math
import re
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from .coords import contest_point, image_bbox_to_contest
from .gt_adapter import load_category_map
from .schema import empty_contest_json
from .stage1_vision import PinBox, VisionOut
from .visual import run_wire_visualize

POWER_NETS = {"VCC", "GND", "VDD", "VSS", "VPP", "VEE", "AGND", "DGND", "3V3", "5V"}

# 位号只做"像位号"的宽松判断（前缀字母 + 数字），不限定前缀白名单：
# box 的 key 可能是 USB1/H1，也可能是 RP2350B/W25Q128JVPIM 这类型号。
DESIGNATOR_RE = re.compile(r"^[A-Za-z]{1,6}\d{1,4}$")
DESIGNATOR_LOOSE_RE = re.compile(r"^([A-Za-z]{1,6})[.\s_-]*(\d{1,4})$")
PIN_NUM_RE = re.compile(r"^[A-Za-z]?\d{1,3}$|^[A-Z]\d{1,2}$")
PACKAGE_RE = re.compile(r"^(0201|0402|0603|0805|1206|1210|2512)$")

# 电源/网络标签类元件：key/Name 由标签文字决定（或 ∅），value 是网络名。
SYMBOL_TYPES = {"gnd", "v", "vcc", "net_input", "net_output", "net_short", "net_bidirection"}
# 这些 type 的 Name 允许与 key 不同（型号/描述文字）。
NAME_DIFF_TYPES = {"box", "d", "led", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin"} | SYMBOL_TYPES


# ----------------------------------------------------------------------------
# Knowledge base (type <-> key prefix, from verified training-only assets)
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


def _prefix_affinity(prefix: str, ctype: str) -> float:
    """软亲和度：type_list 里的 type↔前缀关系只作为加分/减分参考，不做硬性拒绝。

    - 前缀在本 type 的示例位号里出现过 → 强加分；
    - 前缀只出现在别的 type 的示例里 → 减分（例如 box 的 key 不可能是 R+数字）；
    - 前缀不在 type_list 里（USB/H/RP…）→ box 中性，无源器件略减分。
    """
    pref = (prefix or "").upper()
    if not pref:
        return -10.0
    if ctype == "box":
        # box 的 key 常见 U/IC/USB/H…；R/C/L/D/SW/Q/F/TP 等按 type_list 不可能
        if pref in {"U", "IC"}:
            return 45.0
        if pref in {"USB", "H", "J", "CN", "P"}:
            return 20.0
        if pref in {"R", "C", "L", "D", "SW", "Q", "F", "TP", "Y", "X", "BT", "LED"}:
            return -70.0
        return 0.0
    kb = _load_kb()
    dist = kb.get("prefix_to_type", {}).get(pref, {})
    if dist:
        if ctype in dist:
            return 45.0
        return -55.0
    if ctype == "led" and pref == "LED":
        return 50.0
    if ctype in {"pin", "testpoint"} and pref == "TP":
        return 50.0
    if ctype in {"r", "c", "l"}:
        return -12.0
    return 0.0


def _strip_token(text: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", (text or "").strip())


_VALUE_PRIOR_CACHE: dict[str, dict] | None = None


def _load_value_priors() -> dict[str, dict]:
    """Load checksum-verified training-only value forms and examples."""
    global _VALUE_PRIOR_CACHE
    if _VALUE_PRIOR_CACHE is None:
        from .knowledge_base import load_inference_asset

        _VALUE_PRIOR_CACHE = load_inference_asset("value_priors.json")
    return _VALUE_PRIOR_CACHE


def _value_forms(ctype: str) -> dict[str, float]:
    """该 type 在 type_value_priors 中出现过的 value 形式 -> 占比。"""
    rec = _load_value_priors().get(ctype) or {}
    forms = rec.get("forms") or {}
    return {str(k): float((v or {}).get("ratio") or 0.0) for k, v in forms.items()}


def _prior_examples(ctype: str) -> set[str]:
    """该 type 的 value 示例（归一化小写），用于在多个 OCR 候选方向中选更可信的。"""
    out: set[str] = set()
    rec = _load_value_priors().get(ctype) or {}
    for form in (rec.get("forms") or {}).values():
        for ex in (form or {}).get("examples") or []:
            s = "".join(str(ex).lower().split())
            if s:
                out.add(s)
    return out


def _as_designator(text: str) -> str | None:
    """宽松判断"位号形状"：字母前缀 + 数字（R5、U3、C21、USB1、H1、S14…）。

    不限定前缀白名单（type_list 的示例只是参考），只排除封装号；型号文字
    （RP2350B、W25Q128JVPIM、NCD0603Y2 等含字母后缀）不会被当成位号。
    """
    raw = _strip_token(text)
    if not raw:
        return None
    compact = raw.replace(" ", "")
    m = DESIGNATOR_LOOSE_RE.match(compact) or DESIGNATOR_RE.match(compact)
    if not m:
        return None
    prefix, digits = m.group(1).upper(), m.group(2)
    if PACKAGE_RE.match(digits):
        return None
    return prefix + digits


def _extract_designators(text: str) -> list[str]:
    """Whole-string match first, then tokens separated by common OCR delimiters.

    不做子串搜索：'IP5306-4.35V' / 'RP2350B' 这类型号内部虽然含'字母+数字'，
    但它们是完整型号，不是粘连的多个位号。
    """
    found: list[str] = []
    whole = _as_designator(text)
    if whole:
        found.append(whole)
    for tok in re.split(r"[\s,;:/|]+", text or ""):
        d = _as_designator(tok)
        if d and d not in found:
            found.append(d)
    return found


def _prefix_of(designator: str) -> str:
    m = re.match(r"^[A-Z]+", (designator or "").upper())
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


def _hungarian_py(cost: list[list[float]]) -> list[tuple[int, int]]:
    """纯 Python 最优指派（O(n^2 m)），不依赖 scipy。

    之前 scipy 缺失时退化成"按 cost 从小到大贪心"，会先挑局部最便宜的格子，
    再被迫把后续元件/文字错位配对（1-4 被占后 2 只能配 5，3 配 6 等）。
    这里用标准的潜在函数最短路增广算法，保证在整张代价矩阵上全局和最小。
    要求 rows <= cols；rows > cols 时转置后交换回来。
    """
    if not cost or not cost[0]:
        return []
    n, m = len(cost), len(cost[0])
    if n > m:
        pairs = _hungarian_py([list(row) for row in zip(*cost)])
        return [(j, i) for i, j in pairs]

    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = 当前匹配到列 j 的行（1-based）
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    if not cost or not cost[0]:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(np.array(cost, dtype=float))
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        return _hungarian_py(cost)


def _local_hungarian(
    cost: list[list[float]],
    row_boxes: list[list[float]],
    col_boxes: list[list[float]],
    radius: float,
) -> list[tuple[int, int]]:
    """局部多元素联合匹配：只在彼此邻近的元件/文字簇内做最优一对一。

    全局匈牙利在代价函数包含先验加分时，可能为了一个远一点但"更像"的文字，
    把一整片邻近元件的配对整体错位。这里先把 row/col 按空间近邻连成若干簇，
    每簇内部再解匈牙利，局部距离和最小；不同簇之间不会互相抢占。
    """
    if not cost or not cost[0]:
        return []
    n, m = len(cost), len(cost[0])
    if n == 0 or m == 0:
        return []
    # 用 bbox 间距 + 0.15*中心距（与 _text_dist 一致）做近邻；
    # 大 IC 的位号在框上/下方时中心距可能很远，但实际与框只隔几像素。
    def _d(i: int, j: int) -> float:
        return _text_dist(row_boxes[i], col_boxes[j])

    parent = list(range(n + m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 所有半径内的 row-col 都连边；簇 = 有效候选图的连通分量。
    # 这样不会因为只连 k 近邻而漏掉一个仍在半径内的合法候选。
    for i in range(n):
        for j in range(m):
            if _d(i, j) <= radius:
                union(i, n + j)

    groups: dict[int, tuple[list[int], list[int]]] = {}
    for i in range(n):
        groups.setdefault(find(i), ([], []))[0].append(i)
    for j in range(m):
        groups.setdefault(find(n + j), ([], []))[1].append(j)

    pairs: list[tuple[int, int]] = []
    for rows, cols in groups.values():
        if not rows or not cols:
            continue
        sub = [[cost[i][j] for j in cols] for i in rows]
        for ii, jj in _hungarian(sub):
            pairs.append((rows[ii], cols[jj]))
    return pairs


# ----------------------------------------------------------------------------
# Text classification
# ----------------------------------------------------------------------------

_NUM_RE = r"[+-]?\d+(?:[.,]\d+)?"


def _value_form(txt: str) -> str | None:
    """把 OCR 文字归到 type_value_priors.json 中的 value 形式；不是 value 返回 None。

    只做"格式识别"，不限定某个 type 一定要有 value；具体 type 允许哪些形式由
    _value_forms(type) 决定（type_value_priors.json）。
    """
    t = (txt or "").strip().lstrip("=>~").strip()
    if not t:
        return None
    compact = re.sub(r"\s+", "", t)
    if not compact or _as_designator(t) or PACKAGE_RE.match(compact):
        return None
    # 范围写法（10-100uF / 1~10kΩ / 1-10k）
    m = re.fullmatch(
        r"\d+(?:[.,]\d+)?\s*[-~]\s*\d+(?:[.,]\d+)?\s*"
        r"([pnuµmμ]?[FfHhVvAa]|[kKmM]?[ΩΩ]|[kKmM])",
        compact,
    )
    if m:
        unit = m.group(1).lower()
        if unit.endswith("f"):
            return "cap"
        if unit.endswith("h"):
            return "ind"
        if unit.endswith("ω"):
            return "res"
        if unit.endswith("v"):
            return "volt"
        if unit.endswith("a"):
            return "amp"
        return "res"
    # 频率（放在电感前面，避免 MHz 被当作 mH）
    if re.fullmatch(_NUM_RE + r"[kKmMgG]?[Hh][Zz]", compact):
        return "freq"
    # 电容：4.7uF / 100nF / 27p / 100n / 10pF/NC
    if re.fullmatch(_NUM_RE + r"[pnuµmμ]?[Ff]", compact):
        return "cap"
    if re.fullmatch(_NUM_RE + r"[pnuµmμ]", compact):
        return "cap"
    if re.fullmatch(_NUM_RE + r"[pnuµmμ][Ff][/|].*", compact):
        return "cap"
    # 电阻：10K / 5k1 / 47.5k / 25mΩ / 5.1KΩ / 0R（5k1 这类只允许 1 位小数位）
    if re.fullmatch(_NUM_RE + r"(?:[kKmM]?[ΩΩ]|[kKmM]\d{0,1}|[Rr]\d{0,1})", compact):
        return "res"
    # 电感
    if re.fullmatch(_NUM_RE + r"[pnuµmμ]?[Hh]", compact):
        return "ind"
    # 电压（3.3V / 50 V / +5V / 20VP / 3,3V / 3V3）
    if re.fullmatch(_NUM_RE + r"[kKmM]?[Vv][A-Za-z0-9]{0,2}", compact):
        return "volt"
    # 电流
    if re.fullmatch(_NUM_RE + r"[mµμ]?[Aa]", compact):
        return "amp"
    # 裸数字（104 / 100 / 470）
    if re.fullmatch(_NUM_RE, compact):
        return "bare_number"
    return None


def _is_junk(txt: str) -> bool:
    return bool(re.search(r"(Schematic|Sheet|Drawn|SchDoc|Title|\.Sch|CSDN|@)", txt or "", re.I))


def _has_letter(txt: str) -> bool:
    return bool(re.search(r"[A-Za-z]", txt or ""))


def _classify_text(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        return "junk"
    if _is_junk(t):
        return "junk"
    if _as_designator(t):
        return "designator"
    if _extract_designators(t):
        return "mixed"
    if _value_form(t):
        return "value"
    if PIN_NUM_RE.match(t) and not _has_letter(t):
        return "pinnum"
    if _has_letter(t) and len(t) >= 2:
        return "name"
    return "other"


def _classify_group(variants: list[dict]) -> str:
    """按置信度最高的识别方向分类，避免旋转/局部识别的噪声方向改变文字性质。"""
    contents = [
        (str(v.get("content") or "").strip(), float(v.get("score") or 0.0))
        for v in variants
    ]
    contents = [(c, s) for c, s in contents if c]
    if not contents:
        return "junk"
    primary = max(contents, key=lambda cs: cs[1])[0]
    cls = _classify_text(primary)
    if cls != "junk":
        return cls
    for c, _ in sorted(contents, key=lambda cs: -cs[1]):
        cls = _classify_text(c)
        if cls != "junk":
            return cls
    return "junk"


# ----------------------------------------------------------------------------
# Text items: 同一位置可能有多个识别方向（正向 / 旋转 / 局部放大）
# ----------------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _span_box(content: str, bbox: list[float], start: int, end: int) -> list[float]:
    """按字符区间在 OCR 文本框内切出子框（用于 'C11|10pF/NC' 这类粘连文本）。"""
    n = max(1, len(content))
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    return [x1 + w * start / n, y1, x1 + w * end / n, y2]


def _token_box(content: str, token: str, bbox: list[float]) -> list[float]:
    idx = content.upper().find(token.upper())
    if idx < 0:
        return list(bbox)
    return _span_box(content, bbox, idx, idx + len(token))


def _rest_span(content: str, tokens: list[str]) -> tuple[int, int] | None:
    mask = [True] * len(content)
    for tok in tokens:
        for m in re.finditer(re.escape(tok), content, re.I):
            for i in range(m.start(), m.end()):
                mask[i] = False
    keep = [i for i, ok in enumerate(mask) if ok and not content[i].isspace() and content[i] not in "-_/|"]
    if not keep:
        return None
    return min(keep), max(keep) + 1


def _variants_of(item: dict) -> list[dict]:
    variants = item.get("variants") or []
    if not variants:
        variants = [{"content": item.get("content") or "", "score": item.get("score") or 1.0,
                     "angle": item.get("angle") or 0, "src": item.get("source") or "main"}]
    return variants


def _text_dist(comp_bbox: list[float], text_bbox: list[float]) -> float:
    gap = _bbox_gap(comp_bbox, text_bbox)
    cx, cy = _center(comp_bbox)
    tx, ty = _center(text_bbox)
    return gap + 0.15 * math.hypot(cx - tx, cy - ty)


def _prepare_texts(raw: list[dict]) -> list[dict]:
    """把 OCR 结果按位置聚合成文本项；同一位置的多方向候选放进 variants。

    粘连文本（如 'C11|10pF/NC'）会按字符位置拆成虚拟项（位号 + value/name），
    每个虚拟项只包含 OCR 中真实出现的子串，不拼接、不创造文字。
    """
    groups: list[dict] = []
    for t in raw:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        bb = [float(v) for v in t["bbox"]]
        variant = {
            "content": content,
            "score": float(t.get("score") or 1.0),
            "angle": int(t.get("angle") or 0),
            "src": str(t.get("source") or "main"),
        }
        placed = False
        for g in groups:
            ga = (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1])
            ba = (bb[2] - bb[0]) * (bb[3] - bb[1])
            ratio = (ba / ga) if ga > 0 else 0.0
            same = (
                (_iou(bb, g["bbox"]) > 0.6 and 0.55 <= ratio <= 1.8)
                or (_dist(_center(bb), _center(g["bbox"])) < 3.0 and ratio >= 0.5)
            )
            if same:
                g["variants"].append(variant)
                # bbox 保持首个检测框（避免相邻文字通过 union 框连锁合并）
                g["span"] = [min(bb[0], g["span"][0]), min(bb[1], g["span"][1]),
                             max(bb[2], g["span"][2]), max(bb[3], g["span"][3])]
                placed = True
                break
        if not placed:
            groups.append({"bbox": bb, "span": list(bb), "variants": [variant]})

    out: list[dict] = []
    for g in groups:
        # 去重 variant（同内容只保留最高分）
        uniq: dict[str, dict] = {}
        for v in g["variants"]:
            key = v["content"]
            if key not in uniq or v["score"] > uniq[key]["score"]:
                uniq[key] = v
        variants = list(uniq.values())
        primary = max(variants, key=lambda v: v["score"])
        item = {
            "bbox": list(g.get("span") or g["bbox"]),
            "content": primary["content"],
            "score": primary["score"],
            "variants": variants,
            "cls": _classify_group(variants),
            "used": False,
        }
        out.append(item)
        if item["cls"] != "mixed":
            continue
        # 拆分粘连文本：位号子串 + 剩余子串（都只取自原 OCR 文字）
        seen: set[tuple[str, int, int]] = set()
        for v in variants:
            content = v["content"]
            des_list = _extract_designators(content)
            for des in des_list:
                box = _token_box(content, des, item["bbox"])
                sig = (des, int(box[0] / 4), int(box[2] / 4))
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({
                    "bbox": box,
                    "content": des,
                    "score": v["score"],
                    "variants": [{**v, "content": des}],
                    "cls": "designator",
                    "used": False,
                    "virtual": True,
                })
            rest_span = _rest_span(content, des_list)
            if rest_span:
                rest = content[rest_span[0]:rest_span[1]].strip(" -_/|")
                if rest:
                    box = _span_box(content, item["bbox"], rest_span[0], rest_span[1])
                    out.append({
                        "bbox": box,
                        "content": rest,
                        "score": v["score"],
                        "variants": [{**v, "content": rest}],
                        "cls": _classify_text(rest),
                        "used": False,
                        "virtual": True,
                    })
    return out


# ----------------------------------------------------------------------------
# Pin text detection (box 内部的小字不参与元件 key/Name 绑定)
# ----------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _lib_pins_exact(ic_name: str | None) -> dict[str, str]:
    """BoxICLibrary.json 精确查表（number -> name）；不做前缀/后缀模糊匹配。"""
    if not ic_name:
        return {}
    try:
        from .pin_library import load_library
    except Exception:
        return {}
    lib = load_library()
    rec = lib.get(ic_name)
    if rec is None:
        low = str(ic_name).lower()
        for k, v in lib.items():
            if str(k).lower() == low:
                rec = v
                break
    if not rec:
        return {}
    by_num: dict[str, list[str]] = defaultdict(list)
    for pin in rec.get("pins") or []:
        num = str(pin.get("number") or "").strip()
        name = str(pin.get("name") or "").strip()
        if num:
            by_num[num].append(name)
    out: dict[str, str] = {}
    for num, names in by_num.items():
        real = [n for n in names if n and n != num]
        out[num] = max(set(real), key=real.count) if real else num
    return out


def _box_lib_candidates(comp: dict, texts: list[dict]) -> tuple[dict[str, str], set[str], str]:
    """在 box 附近找能精确命中 BoxICLibrary 的型号文字。

    同一框内/附近可能有多段文字恰好也是库里的 key（例如引脚名 'LED1' 也可能是
    某个 IC 的型号）。这里用"库内 pinname 与框内文字的重合数"选最可信的库：
    正确型号的引脚名会大量出现在框内，错误型号不会。
    返回 (number->name, pinname 小写集合, 命中的型号文字)。
    """
    x1, y1, x2, y2 = comp["img_bbox"]
    inside: list[list[str]] = []
    for item in texts:
        cx, cy = _center(item["bbox"])
        if x1 - 4 <= cx <= x2 + 4 and y1 - 4 <= cy <= y2 + 4:
            inside.append(["".join(str(v.get("content") or "").lower().split())
                           for v in _variants_of(item)])

    best: tuple[dict[str, str], set[str], str] | None = None
    best_score = -1e9
    for item in texts:
        d = _text_dist(comp["img_bbox"], item["bbox"])
        if d > 140:
            continue
        for v in _variants_of(item):
            for key in {str(v.get("content") or "").strip(), _strip_token(v.get("content") or "")}:
                if len(key) < 2:
                    continue
                des = _as_designator(key)
                lib = _lib_pins_exact(key)
                if not lib:
                    continue
                names = {"".join(str(n).lower().split()) for n in lib.values() if n}
                inside_contents = {c for contents in inside for c in contents}
                matched_names = names & inside_contents
                matches = len(matched_names)
                n_short = sum(1 for c in inside_contents if len(c) <= 10 and " " not in c)
                # 库里的 pinname 必须能在框内文字中找到足够多的对应；
                # 否则说明这个"型号"只是恰好同名（例如 USB1 也可能是另一个 USB 符号的库）。
                # 长型号名允许 0 匹配兜底，短位号不允许。
                need = 1 if len(lib) <= 2 else max(2, math.ceil(0.3 * min(len(names), n_short or len(names))))
                if matches < need and (des or len(key) < 5):
                    continue
                score = 10.0 * matches - 0.05 * d + (4.0 if len(key) >= 4 else 0.0) + (3.0 if not des else 0.0)
                if score > best_score:
                    best_score = score
                    best = (lib, names, key)
    if best is None:
        return {}, set(), ""
    return best


def _mark_pin_texts(
    comps: list[dict],
    texts: list[dict],
    det_pins: list[PinBox],
) -> set[int]:
    """标记 box 内部属于引脚的文字（pinname / pin_number）。

    规则：
      - 文字中心落在某个 box 内；
      - 命中该 box 型号在 BoxICLibrary 里的 pinname，或
      - 紧邻某个 pin_tip 的纯数字（引脚号）/纯字母短词（引脚名）。
    这些文字不参与元件 key/Name 绑定，但保留给引脚匹配使用。
    """
    box_comps = [c for c in comps if c["type"] == "box"]
    if not box_comps:
        return set()
    lib_names: dict[int, set[str]] = {}
    lib_keys: dict[int, str] = {}
    for c in box_comps:
        _, names, key = _box_lib_candidates(c, texts)
        lib_names[id(c)] = names
        lib_keys[id(c)] = "".join(key.lower().split())
        c["lib_key"] = key

    pin_ids: set[int] = set()
    pin_xy = [p.xy for p in det_pins]
    for item in texts:
        if item.get("used") or item["cls"] == "junk":
            continue
        tc = _center(item["bbox"])
        near_pin = any(_dist(tc, xy) <= 60 for xy in pin_xy)
        near_pin_strict = any(_dist(tc, xy) <= 20 for xy in pin_xy)
        owner = None
        for c in box_comps:
            x1, y1, x2, y2 = c["img_bbox"]
            if x1 - 4 <= tc[0] <= x2 + 4 and y1 - 4 <= tc[1] <= y2 + 4:
                if owner is None or (x2 - x1) * (y2 - y1) < (
                    (owner["img_bbox"][2] - owner["img_bbox"][0]) * (owner["img_bbox"][3] - owner["img_bbox"][1])
                ):
                    owner = c
        if owner is None:
            # box 外的引脚号：紧贴 pin_tip 的纯数字（1/2/13…）不是 value/key
            if near_pin_strict and any(
                str(v.get("content") or "").strip().isdigit()
                and len(str(v.get("content") or "").strip()) <= 3
                for v in _variants_of(item)
            ):
                pin_ids.add(id(item))
                item["cls"] = "pinnum"
            continue
        names = lib_names.get(id(owner)) or set()
        lib_key = lib_keys.get(id(owner)) or ""
        bx1, by1, bx2, by2 = owner["img_bbox"]
        relx = (tc[0] - bx1) / max(1.0, bx2 - bx1)
        at_edge = relx <= 0.22 or relx >= 0.78
        is_pin = False
        for v in _variants_of(item):
            content = str(v.get("content") or "").strip()
            if not content:
                continue
            norm = "".join(content.lower().split())
            # Parentheses are printed pin annotations too: VOUT(TAB) must not
            # compete with an external part number merely because the library
            # calls that port VOUT. Preserve the full OCR string for pin output.
            base_name = re.sub(r"\([^()]*\)$", "", norm)
            annotation = re.search(r"\(([^()]*)\)$", norm)
            if (norm in names or (base_name != norm and base_name in names)
                    or (annotation and annotation.group(1) in names
                        and not re.search(r"\d", norm))):
                is_pin = True
                break
            # 命中的型号文字本身是 Name，不是引脚文字
            if lib_key and norm == lib_key:
                continue
            short = len(content) <= 10 and " " not in content
            if not short:
                continue
            if names:
                # 有引脚库时：框内靠近引脚/位于左右边缘的短文字（CC2/DP1/SBU1…）是 pinname
                if near_pin or at_edge:
                    is_pin = True
                    break
            else:
                if near_pin and (not _has_letter(content) or not re.search(r"\d", content)):
                    is_pin = True
                    break
                # 无引脚库时：左右边缘的短字母数字文字也按 pinname 处理
                if at_edge and re.search(r"\d", content) and _has_letter(content):
                    is_pin = True
                    break
        if is_pin:
            pin_ids.add(id(item))
            item["cls"] = "pinnum" if any(
                not _has_letter(str(v.get("content") or "")) for v in _variants_of(item)
            ) else "pinname"
    return pin_ids


# ----------------------------------------------------------------------------
# Value binding (value-first, 以 type_value_priors.json 为准)
# ----------------------------------------------------------------------------

def _symbol_red_mask(img):
    import cv2
    import numpy as np

    b,g,r=cv2.split(img.astype(np.int16))
    return ((r-g>20)&(r-b>15)&(r>60)).astype(np.uint8)


def _ground_bars(img, bbox):
    """Return visible three/four-bar ground geometry, or None without evidence."""
    import itertools
    import cv2
    import numpy as np

    h,w=img.shape[:2]
    x1,y1,x2,y2=map(float,bbox)
    pad=max(14.,min(30.,.6*min(x2-x1,y2-y1)))
    left,top=max(0,int(x1-pad)),max(0,int(y1-pad))
    right,bottom=min(w,int(math.ceil(x2+pad))),min(h,int(math.ceil(y2+pad)))
    mask=_symbol_red_mask(img[top:bottom,left:right])
    cand=[]
    for trans in [False,True]:
        src=mask.T if trans else mask
        lines=cv2.morphologyEx(src,cv2.MORPH_OPEN,np.ones((1,5),np.uint8))
        n,_,stats,_=cv2.connectedComponentsWithStats(lines,8)
        bars=[]
        for x,y,bw,bh,area in stats[1:]:
            if bw>=5 and bw>=2.2*bh and bh<=7:
                bars.append((float(x),float(y),float(bw),float(bh),float(area)))
        for triple in itertools.chain.from_iterable(itertools.combinations(bars,n) for n in [3,4] if len(bars)>=n):
            triple=sorted(triple,key=lambda z:z[1])
            centers=[b[0]+b[2]/2 for b in triple]
            ys=[b[1]+b[3]/2 for b in triple]
            lengths=[b[2] for b in triple]
            gaps=[ys[i+1]-ys[i] for i in range(len(ys)-1)]
            if min(gaps)<2 or max(gaps)>15 or max(gaps)>1.75*min(gaps):continue
            if max(centers)-min(centers)>max(3.,.1*max(lengths)):continue
            if not (all(lengths[i]>=1.15*lengths[i+1] for i in range(len(lengths)-1)) or
                    all(lengths[i+1]>=1.15*lengths[i] for i in range(len(lengths)-1))):continue
            if min(lengths)<.12*max(lengths):continue
            bb=[min(b[0] for b in triple),min(b[1] for b in triple),
                max(b[0]+b[2] for b in triple),max(b[1]+b[3] for b in triple)]
            core=list(bb)
            # Include the visible central lead on the long-bar side. Its
            # length is bounded by half a bar so an attached wire/junction
            # cannot expand the symbol to a whole network branch.
            big=max(triple,key=lambda b:b[2]);center=int(round(big[0]+big[2]/2-.5))
            step=-1 if big is triple[0] else 1
            start=int(big[1])-1 if step<0 else int(big[1]+big[3])
            end=start-step
            for depth in range(min(14,int(max(lengths)/3))):
                yy=start+step*depth
                if yy<0 or yy>=src.shape[0]:break
                if not np.any(src[yy,max(0,center-1):min(src.shape[1],center+2)]):break
                end=yy
            if step<0:bb[1]=min(bb[1],end)
            else:bb[3]=max(bb[3],end+1)
            if trans:
                bb=[bb[1],bb[0],bb[3],bb[2]]
                core=[core[1],core[0],core[3],core[2]]
            bb=[bb[0]+left,bb[1]+top,bb[2]+left,bb[3]+top]
            core=[core[0]+left,core[1]+top,core[2]+left,core[3]+top]
            # The detector must actually overlap the observed symbol; a nearby
            # different ground is never used solely because it is in the crop.
            ix=max(0.,min(bb[2],x2)-max(bb[0],x1));iy=max(0.,min(bb[3],y2)-max(bb[1],y1))
            area=(bb[2]-bb[0])*(bb[3]-bb[1])
            if ix*iy<.15*area:continue
            cx,cy=(bb[0]+bb[2])/2,(bb[1]+bb[3])/2
            d=math.hypot(cx-(x1+x2)/2,cy-(y1+y2)/2)
            cand.append((d+.3*(max(centers)-min(centers)),bb,trans,core,len(triple)))
    # A three-bar subset must not beat the complete four-bar glyph merely
    # because the detector is shifted toward its label. This dominance check
    # is local to the same body; a larger neighbouring ground gets no bonus.
    maximal=[]
    for c in cand:
        a=c[3]
        dominated=False
        for d in cand:
            b=d[3]
            if c is d or d[2]!=c[2] or d[4]<c[4]:continue
            contains=all([b[0]<=a[0]+1,b[1]<=a[1]+1,b[2]>=a[2]-1,b[3]>=a[3]-1])
            bigger=(b[2]-b[0])*(b[3]-b[1])>1.15*(a[2]-a[0])*(a[3]-a[1])
            if contains and bigger:
                dominated=True;break
        if not dominated:maximal.append(c)
    return min(maximal,key=lambda c:c[0])[1:4] if maximal else None


def _clean_symbol_components(comps, img):
    """Return copied components plus removed src_idx -> retained src_idx aliases."""
    out=[dict(c,img_bbox=list(c['img_bbox'])) for c in comps]
    if img is None:return out,{}
    observations={};original={c['src_idx']:list(c['img_bbox']) for c in out}
    for c in out:
        if c['type']=='gnd':
            obs=_ground_bars(img,c['img_bbox'])
            if obs:
                observations[c['src_idx']]=obs[2]
                c['img_bbox']=list(obs[0])
    aliases={};removed=set()
    red=_symbol_red_mask(img)
    def support(bb):
        h,w=red.shape
        l,t=max(0,int(bb[0])),max(0,int(bb[1]));r,b=min(w,int(math.ceil(bb[2]))),min(h,int(math.ceil(bb[3])))
        return int(red[t:b,l:r].sum()) if r>l and b>t else 0
    for ai,a in enumerate(out):
        ia=a['src_idx']
        if ia in removed or a['type'] not in {'gnd','v','vcc','net_input','net_output','net_short','net_bidirection'}:continue
        for b in out[ai+1:]:
            ib=b['src_idx']
            if ib in removed or a['type']!=b['type']:continue
            ba,bb=a['img_bbox'],b['img_bbox']
            if max(ba[0],bb[0])>=min(ba[2],bb[2]) or max(ba[1],bb[1])>=min(ba[3],bb[3]):continue
            same_ground=ia in observations and ib in observations and max(abs(x-y) for x,y in zip(observations[ia],observations[ib]))<=3
            inter=[max(ba[0],bb[0]),max(ba[1],bb[1]),min(ba[2],bb[2]),min(ba[3],bb[3])]
            common=support(inter);na,nb=support(ba),support(bb)
            # Two detections of a power end-bar can have low rectangle IoU,
            # yet share almost every coloured symbol pixel of the smaller one.
            pixel_same=(a['type'] in {'v','vcc'} and common>=10 and min(na,nb)>=10 and
                        common/min(na,nb)>=.8 and
                        math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2,(ba[1]+ba[3]-bb[1]-bb[3])/2)<12.)
            network_same=(a['type'].startswith('net_') and common>=20 and min(na,nb)>=20 and
                          common/min(na,nb)>=.85 and
                          math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2,(ba[1]+ba[3]-bb[1]-bb[3])/2)
                          <.65*max(ba[2]-ba[0],ba[3]-ba[1],bb[2]-bb[0],bb[3]-bb[1]))
            if not (same_ground or pixel_same or network_same):continue
            # Prefer actual symbol support, then confidence/stable source ID.
            rank_a=(support(original[ia]),float(a.get('score',0)),-ia)
            rank_b=(support(original[ib]),float(b.get('score',0)),-ib)
            if rank_b>rank_a:
                aliases[ia]=ib;removed.add(ia);break
            aliases[ib]=ia;removed.add(ib)
    for old in aliases:
        target=aliases[old]
        while target in aliases:
            target=aliases[target]
        aliases[old]=target
    return [c for c in out if c['src_idx'] not in removed],aliases


def _component_ocr_requests(comps: list[dict], texts: list[dict], pin_ids: set[int],
                            vision: VisionOut) -> list[dict]:
    """Find uncertain component labels and separate text from coloured wires.

    No expected string is supplied to recognition. Without an independent
    coloured text layer, keep the original OCR and matching behaviour.
    """
    import cv2
    import numpy as np

    img = getattr(vision, "img", None)
    if img is None:
        return []
    h, w = img.shape[:2]
    requests = []
    for item in texts:
        if id(item) in pin_ids or item.get("virtual"):
            continue
        content, bb = str(item.get("content") or ""), item["bbox"]
        context = None
        for comp in sorted(comps, key=lambda c: _text_dist(c["img_bbox"], bb)):
            typ = comp["type"]
            if _text_dist(comp["img_bbox"], bb) > 65:
                break
            if typ in {"gnd", "v", "vcc"}:
                score = _symbol_label_score(typ, content)
                if (score is None and _bbox_gap(comp["img_bbox"], bb) < 18
                        and not _as_designator(content) and not _value_form(content)):
                    context = typ
                    break
                if score is not None and (float(item["score"]) < .985
                        or (bb[3]-bb[1] > 1.2*(bb[2]-bb[0])
                            and not any(v.get("angle") for v in _variants_of(item)))):
                    context = typ
                    break
            if typ in {"r", "c", "l"} and _value_form(content):
                context = typ
                break
        if context is None:
            continue
        x1, y1 = max(0, int(bb[0])-3), max(0, int(bb[1])-3)
        x2, y2 = min(w, math.ceil(bb[2])+3), min(h, math.ceil(bb[3])+3)
        crop = img[y1:y2, x1:x2]
        if not crop.size:
            continue
        b, g, r = cv2.split(crop.astype(np.int16))
        blue = (b-r > 20) & (b-g > 20)
        yy, xx = np.where(blue)
        if len(xx) < 12:
            continue
        bx1, by1, bx2, by2 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
        obstruction = ((r-g > 20) & (r-b > 15)) | ((g-r > 15) & (g-b > 15))
        if context in {"r", "c", "l"} and int(obstruction[by1:by2, bx1:bx2].sum()) < 3:
            continue
        clean = np.full(crop.shape[:2], 255, dtype=np.uint8)
        clean[blue] = 0
        tight = clean[max(0, by1-2):min(clean.shape[0], by2+2),
                      max(0, bx1-2):min(clean.shape[1], bx2+2)]
        variants = [cv2.cvtColor(cv2.copyMakeBorder(tight, 6, 6, 6, 6,
                    cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)]
        # A second threshold/padding read supplies independent evidence when
        # a crossing arrow obscures only part of a printed unit.
        if context in {"r", "c", "l"}:
            strict = np.full(crop.shape[:2], 255, dtype=np.uint8)
            strict[(b-r > 35) & (b-g > 35)] = 0
            yy2, xx2 = np.where(strict == 0)
            if len(xx2) >= 12:
                tight2 = strict[max(0,int(yy2.min())-2):min(strict.shape[0],int(yy2.max())+3),
                                max(0,int(xx2.min())-2):min(strict.shape[1],int(xx2.max())+3)]
                variants.append(cv2.cvtColor(cv2.copyMakeBorder(tight2, 2, 2, 2, 2,
                                cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR))
        else:
            # Retain antialiasing as a second read for small rotated supplies;
            # a hard threshold can erase the short strokes of VCC/VDD.
            contrast = (255-np.clip(3*(b-np.maximum(r,g)),0,255)).astype(np.uint8)
            contrast[~blue] = 255
            tight2 = contrast[max(0,by1-2):min(contrast.shape[0],by2+2),
                              max(0,bx1-2):min(contrast.shape[1],bx2+2)]
            variants.append(cv2.cvtColor(cv2.copyMakeBorder(tight2,2,2,2,2,
                            cv2.BORDER_CONSTANT,value=255),cv2.COLOR_GRAY2BGR))
        requests.append({"item": item, "type": context,
                         "crops": variants,
                         "bbox": list(bb), "old_content": content})
    return requests[:24]


def _component_ocr_worker(input_path: str, output_path: str, lang: str = "en") -> None:
    """Paddle-only subprocess: batch the existing 3.7 recognition model."""
    import os
    import numpy as np
    import paddleocr
    from ._ocr_worker import _quiet_paddle_logs

    if not str(paddleocr.__version__).startswith("3.7"):
        raise RuntimeError("Component text refinement requires PaddleOCR 3.7")
    _quiet_paddle_logs()
    # Match the existing OCR worker, including its Paddle 3.3 CPU workaround.
    engine = paddleocr.PaddleOCR(
        lang=lang, device=os.environ.get("PCBOCR_DEVICE", "cpu"),
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, enable_mkldnn=False,
        text_det_limit_side_len=int(os.environ.get("PCBOCR_DET_LIMIT", "2560")),
    )
    recognizer = engine.paddlex_pipeline._pipeline.text_rec_model
    with np.load(input_path, allow_pickle=False) as data:
        crops = [data[f"crop_{i}"] for i in range(len(data.files))]
    results = list(recognizer.predict(crops, batch_size=8))
    if len(results) != len(crops):
        raise RuntimeError("OCR refinement returned an incomplete crop batch")
    Path(output_path).write_text(json.dumps([
        {"content": str(r.get("rec_text") or "").strip(),
         "score": float(r.get("rec_score") or 0.)} for r in results
    ], ensure_ascii=False), encoding="utf-8")


def _run_component_ocr(crops: list, lang: str) -> list[dict]:
    """Reuse Stage1's runtime setup and process isolation, without changing it."""
    import os
    import subprocess
    import sys
    import tempfile
    import numpy as np
    from .stage1_vision import _ensure_ocr37, _ensure_ocr_models, _ocr_subprocess_env

    runtime = _ensure_ocr37()
    _ensure_ocr_models()
    with tempfile.TemporaryDirectory(prefix="pcbnet_component_ocr_") as tmp:
        src, dest = Path(tmp)/"crops.npz", Path(tmp)/"result.json"
        np.savez(src, **{f"crop_{i}": crop for i, crop in enumerate(crops)})
        env = _ocr_subprocess_env(runtime)
        if runtime:
            env["PYTHONPATH"] = runtime + (os.pathsep + os.environ["PYTHONPATH"]
                                          if os.environ.get("PYTHONPATH") else "")
        env.setdefault("PCBOCR_DEVICE", "cpu")
        cmd = [sys.executable] + (["-S"] if runtime else [])
        cmd += ["-c", "import sys; "
                "sys.path.insert(0, sys.argv[1]) if sys.argv[1] else None; "
                "from pcbnet.stage2_structure import _component_ocr_worker; "
                "_component_ocr_worker(*sys.argv[2:])", runtime or "", str(src), str(dest), lang]
        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent,
                                   env=env, capture_output=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[-600:])
        return json.loads(dest.read_text(encoding="utf-8"))


def _refine_component_texts(comps: list[dict], texts: list[dict], pin_ids: set[int],
                             vision: VisionOut, cfg: dict) -> None:
    """Add OCR-observed alternatives only; do not modify the input VisionOut."""
    import cv2
    import numpy as np
    import warnings

    if not (cfg.get("ocr") or {}).get("enabled", True):
        return
    requests = _component_ocr_requests(comps, texts, pin_ids, vision)
    if not requests:
        return
    crops, metadata = [], []
    for i, req in enumerate(requests):
        for crop in req["crops"]:
            for scale in (2., 3.):
                resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for angle in range(4):
                    crops.append(np.ascontiguousarray(np.rot90(resized, angle)))
                    metadata.append((i, angle*90))
    try:
        results = _run_component_ocr(crops, str((cfg.get("ocr") or {}).get("lang", "en")))
        if len(results) != len(metadata):
            raise ValueError("Incomplete component OCR results")
    except Exception as exc:
        warnings.warn(f"Component OCR refinement unavailable; using existing text: {exc}", RuntimeWarning)
        return
    by_request: dict[int, list[dict]] = defaultdict(list)
    for (i, angle), result in zip(metadata, results):
        result = dict(result, angle=angle, src="component_reocr")
        content, score = result["content"], float(result["score"])
        typ, old = requests[i]["type"], requests[i]["old_content"]
        if typ in {"gnd", "v", "vcc"}:
            if score < .95 or _symbol_label_score(typ, content) is None:
                continue
            if (_symbol_label_score(typ, old) is not None
                    and re.sub(r"[.,\s]", "", old).lower() != re.sub(r"[.,\s]", "", content).lower()):
                continue
        else:
            if (score < .94 or _value_form(content) not in _value_forms(typ)
                    or re.findall(r"\d+", old) != re.findall(r"\d+", content)):
                continue
            old_form, new_form = _value_form(old), _value_form(content)
            old_unit = re.sub(r"[\d.,+\-\s]", "", old)
            new_unit = re.sub(r"[\d.,+\-\s]", "", content)
            if not ((old_form == "bare_number" and new_form != "bare_number")
                    or (typ == "c" and old_unit.lower() == "f" and len(new_unit) > 1)):
                continue
            result["unit_recovered"] = True
        by_request[i].append(result)
    for i, candidates in by_request.items():
        item = requests[i]["item"]
        best = max(candidates, key=lambda r: r["score"])
        old_valid = _symbol_label_score(requests[i]["type"], requests[i]["old_content"])
        if (not best.get("unit_recovered") and old_valid is not None
                and best["score"] < float(item["score"])+.01):
            continue
        item["variants"].append(best)
        if best.get("unit_recovered"):
            item["recovered_value"] = best["content"]
        else:
            item["recovered_symbol"] = best["content"]
        item["content"], item["score"] = best["content"], best["score"]
        item["cls"] = _classify_text(best["content"])


def _value_variant(item: dict, ctype: str) -> tuple[str, str, float] | None:
    forms = _value_forms(ctype)
    if not forms:
        return None
    best = None
    best_sc = -1e9
    for v in _variants_of(item):
        content = str(v.get("content") or "").strip()
        form = _value_form(content)
        if not form or form not in forms:
            continue
        # 单个数字几乎都是引脚号，不作为 value（priors 里的裸数字示例也都是 2 位以上）
        if form == "bare_number" and len(re.sub(r"\D", "", content)) < 2:
            continue
        sc = 40.0 * forms.get(form, 0.0) + 3.0 * float(v.get("score") or 0.0)
        if content == item.get("recovered_value"):
            sc += 45.0  # independently re-read unit, never an inferred one
        if sc > best_sc:
            best_sc = sc
            best = (content.lstrip("=>~").strip(), form, sc)
    return best


def _assign_values(comps: list[dict], texts: list[dict], pin_ids: set[int]) -> None:
    """按 type_value_priors 给"有 value"的 type 一对一绑定最近的合法 value 文字。

    没有合法 value 的元件保持 value=None（字段始终存在）。
    """
    priors = _load_value_priors()
    targets = [c for c in comps if c["type"] in priors and c["type"] not in SYMBOL_TYPES]
    if not targets:
        return
    cand = [
        item for item in texts
        if not item.get("used") and id(item) not in pin_ids and item["cls"] != "junk"
    ]

    BIG = 1e6
    cost: list[list[float]] = []
    infos: list[list[tuple[str, str, float] | None]] = []
    for comp in targets:
        row: list[float] = []
        info_row: list[tuple[str, str, float] | None] = []
        for item in cand:
            info = _value_variant(item, comp["type"])
            info_row.append(info)
            if info is None:
                row.append(BIG)
                continue
            d = _text_dist(comp["img_bbox"], item["bbox"])
            if d > 105:
                row.append(BIG)
                continue
            # 距离为主，value 形式先验只做有界加分（避免先验把邻近配对整体拉错）
            row.append(d - min(12.0, 0.3 * info[2]))
        cost.append(row)
        infos.append(info_row)
    if not cost or not cost[0]:
        return
    used_t: set[int] = set()
    for ci, k in _local_hungarian(
        cost, [t["img_bbox"] for t in targets], [item["bbox"] for item in cand], radius=120.0
    ):
        if ci >= len(targets) or k >= len(cand) or cost[ci][k] >= BIG or k in used_t:
            continue
        item = cand[k]
        info = infos[ci][k]
        if info is None:
            continue
        targets[ci]["value"] = info[0]
        item["used"] = True
        used_t.add(k)


def _symbol_label_score(ctype: str, content: str) -> float | None:
    t = (content or "").strip()
    if not t or _is_junk(t) or not _has_letter(t) or len(t) < 2:
        return None
    norm = "".join(t.lower().split())
    # A short capitalised OCR fragment is not evidence of a supply label.
    # In particular, rotated ground text can be read together with its symbol,
    # while capacitor values and IC part names must remain available to their
    # own components. Keep arbitrary names for net_* labels only.
    if ctype == "gnd":
        if not (re.fullmatch(r"[a-z0-9_+.-]*gnd[a-z0-9_+.-]*", norm)
                or norm in {"vss", "vssa", "vssd", "earth", "pe", "chassis", "0v"}):
            return None
    elif ctype in {"v", "vcc"}:
        supply = re.fullmatch(
            r"(?:[adp]?v_?(?:cc|dd|ss|ee|pp|bat|bus|in|out|ref|core|io|usb)|bat)[a-z0-9_+.-]*", norm)
        if not supply and norm not in {"v+", "v-"} and _value_form(t) != "volt":
            return None
    elif _as_designator(t):
        return None
    score = 0.0
    if norm in _prior_examples(ctype):
        score += 50.0
    if ctype == "gnd" and "gnd" in norm:
        score += 60.0
    if ctype in {"v", "vcc"} and re.search(r"(vcc|vdd|vss|vbat|vin|vout|vref|vbus|\d+(?:[.,]\d+)?v)", norm):
        score += 30.0
    if t.upper() == t:
        score += 8.0
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+\-.]{0,17}", t):
        score += 10.0
    return score


def _assign_symbol_values(comps: list[dict], texts: list[dict], pin_ids: set[int]) -> None:
    """gnd/v/net_* 的 value 是电源/网络标签；按最近文字绑定，无则 None。"""
    targets = [c for c in comps if c["type"] in SYMBOL_TYPES]
    if not targets:
        return

    def best_label(item, ctype):
        best, best_sc = None, -1e9
        for v in _variants_of(item):
            content = str(v.get("content") or "").strip()
            if item.get("recovered_symbol") and content != item["recovered_symbol"]:
                continue
            sc = _symbol_label_score(ctype, content)
            if sc is None:
                continue
            sc += 2.0 * float(v.get("score") or 0.0)
            if sc > best_sc:
                best, best_sc = content, sc
        return None if best is None else (best, best_sc)

    BIG = 1e6
    cost: list[list[float]] = []
    infos: list[list[tuple[str, float] | None]] = []
    for comp in targets:
        row: list[float] = []
        info_row: list[tuple[str, float] | None] = []
        for item in texts:
            if id(item) in pin_ids or item["cls"] in {"junk", "mixed"}:
                info_row.append(None)
                row.append(BIG)
                continue
            info = best_label(item, comp["type"])
            info_row.append(info)
            if info is None or item.get("used") or info[1] < 8.0:
                row.append(BIG)
                continue
            d = _text_dist(comp["img_bbox"], item["bbox"])
            if d > 85:
                row.append(BIG)
                continue
            row.append(d - min(18.0, 0.3 * info[1]))
        cost.append(row)
        infos.append(info_row)
    if not cost or not cost[0]:
        return
    used_t: set[int] = set()
    for ci, k in _local_hungarian(
        cost, [c["img_bbox"] for c in targets], [item["bbox"] for item in texts], radius=110.0
    ):
        if ci >= len(targets) or k >= len(texts) or cost[ci][k] >= BIG or k in used_t:
            continue
        item = texts[k]
        info = infos[ci][k]
        if info is None:
            continue
        comp = targets[ci]
        comp["value"] = info[0]
        comp["_value_item"] = item
        if comp["type"].startswith("net_"):
            comp["net_label"] = info[0]
        item["used"] = True
        used_t.add(k)

    # 同一符号被重复检出时（框高度重叠），两个框共享同一网络标签，
    # 避免对齐到没有 value 的那个重复框。
    for comp in targets:
        if comp.get("value"):
            continue
        for other in targets:
            if other is comp or other["type"] != comp["type"] or not other.get("value"):
                continue
            if _iou(comp["img_bbox"], other["img_bbox"]) > 0.3 or \
                    _dist(_center(comp["img_bbox"]), _center(other["img_bbox"])) < 8:
                comp["value"] = other["value"]
                if comp["type"].startswith("net_"):
                    comp["net_label"] = other["value"]
                break

    # 一排相邻的 gnd/v 符号可能共用同一个 GND/+5V 文字：允许同 type 的邻近符号
    # 复用已经识别到的标签文字（仍然只使用 OCR 中真实存在的文字）。
    for comp in targets:
        if comp.get("value") or comp["type"] not in {"gnd", "v", "vcc"}:
            continue
        for other in targets:
            item = other.get("_value_item")
            if other is comp or other["type"] != comp["type"] or not other.get("value") or item is None:
                continue
            if _text_dist(comp["img_bbox"], item["bbox"]) <= 120:
                comp["value"] = other["value"]
                break
    for comp in targets:
        comp.pop("_value_item", None)


# ----------------------------------------------------------------------------
# Key / Name binding (value 完成后再用剩余文字决定 key/Name)
# ----------------------------------------------------------------------------

def _designator_variant(item: dict, ctype: str) -> tuple[str, float] | None:
    """从候选方向里选一个位号形状的文字，并按 type_list 的前缀关系给软亲和度。"""
    best = None
    best_sc = -1e9
    for v in _variants_of(item):
        des = _as_designator(str(v.get("content") or "").strip())
        if not des:
            continue
        sc = _prefix_affinity(_prefix_of(des), ctype) + 4.0 * float(v.get("score") or 0.0)
        if sc > best_sc:
            best, best_sc = des, sc
    return None if best is None else (best, best_sc)


def _is_part_number(content: str, ctype: str) -> bool:
    """型号文字（SS34 / 1N4007 / NCD0603Y2 / 78M12…），不是本 type 的位号。"""
    t = _strip_token(content)
    if len(t) < 3 or not _has_letter(t) or not re.search(r"\d", t):
        return False
    if _value_form(t) or _is_junk(t):
        return False
    des = _as_designator(t)
    if des and _prefix_affinity(_prefix_of(des), ctype) != 0.0:
        return False
    return True


def _name_content(item: dict) -> tuple[str, float] | None:
    best, best_sc = None, -1e9
    for v in _variants_of(item):
        content = str(v.get("content") or "").strip()
        if len(content) < 2 or _is_junk(content) or _value_form(content):
            continue
        sc = len(content) + 2.0 * float(v.get("score") or 0.0)
        if _as_designator(content) and not _lib_pins_exact(content):
            sc -= 18.0
        if sc > best_sc:
            best, best_sc = content, sc
    return None if best is None else (best, best_sc)


def _name_score(comp: dict, item: dict, lib_names: set[str]) -> float | None:
    info = _name_content(item)
    if info is None:
        return None
    content, sc = info
    d = _text_dist(comp["img_bbox"], item["bbox"])
    if d > 120:
        return None
    x1, y1, x2, y2 = comp["img_bbox"]
    cx, cy = _center(item["bbox"])
    score = 80.0 - d + 0.5 * sc
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        score += 25.0
    if re.search(r"\d", content):
        score += 8.0
    if len(content) >= 5:
        score += 8.0
    if _as_designator(content):
        # 真实存在于 BoxICLibrary 的"位号形状"文字其实是型号（TP4333/78M12…）
        score += 15.0 if _lib_pins_exact(content) else -25.0
    if content.lower() in lib_names:
        score -= 200.0
    # 纯网络/电源标签（BAT+、GND、VCC…）不是型号/描述
    if re.fullmatch(r"[A-Za-z0-9_+\-]+", content) and not re.search(r"\d", content) \
            and len(content) <= 8 and content.upper().strip("+-") in (POWER_NETS | {"BAT", "VBUS", "VIN", "VOUT"}):
        return None
    return score


def _pick_name(comp: dict, texts: list[dict], pin_ids: set[int],
               exclude: set[int] | None = None) -> tuple[str, dict] | None:
    exclude = exclude or set()
    lib_names = _box_lib_candidates(comp, texts)[1] if comp["type"] == "box" else set()
    best, best_sc = None, -1e9
    for item in texts:
        if item.get("used") or id(item) in pin_ids or id(item) in exclude:
            continue
        if item["cls"] in {"junk", "value", "mixed"}:
            continue
        if comp["type"] != "box":
            info = _name_content(item)
            if info is None or not _is_part_number(info[0], comp["type"]):
                continue
        else:
            info = _name_content(item)
            if info is None:
                continue
            content = info[0]
            # 纯电源/网络标签不是 box 的型号/描述（BAT+ / GND / VCC…）
            if re.fullmatch(r"[A-Za-z0-9_+\-]+", content) and not re.search(r"\d", content) \
                    and len(content) <= 8:
                continue
        sc = _name_score(comp, item, lib_names)
        if sc is not None and sc > best_sc:
            best, best_sc = (_name_content(item)[0], item), sc
    # 只剩下的都是电源/网络标签等低分文字时，不把 Name 写成它们（Name=key 更合理）
    return best if best_sc >= 0.0 else None


def _assign_box_names(comps: list[dict], texts: list[dict],
                      pin_ids: set[int]) -> dict[int, tuple[str, dict]]:
    """box 的 Name/型号文字做全局一对一分配，避免相邻的误检框抢走型号。

    例如 U3 的 'XB7608A' 不能被旁边一个误检成 box 的 H2 抢走。
    """
    boxes = [c for c in comps if c["type"] == "box"]
    if not boxes:
        return {}
    cands = [
        item for item in texts
        if not item.get("used") and id(item) not in pin_ids
        and item["cls"] not in {"junk", "value", "mixed"}
    ]
    # 仍是位号形状的文字通常是别的元件漏绑的 key；型号形状（SX1308）或
    # BoxICLibrary 中真实存在的型号（TP4333）才允许作为 box 的 Name。
    def _name_ok(c: str) -> bool:
        if not _as_designator(c):
            return True
        return _is_part_number(c, "box") or bool(_lib_pins_exact(c))

    cands = [it for it in cands if _name_ok((_name_content(it) or ("",))[0])]
    BIG = 1e6
    cost: list[list[float]] = []
    for comp in boxes:
        lib_names = _box_lib_candidates(comp, texts)[1]
        row: list[float] = []
        for item in cands:
            sc = _name_score(comp, item, lib_names)
            row.append(BIG if sc is None or sc < 0.0 else -sc)
        cost.append(row)
    out: dict[int, tuple[str, dict]] = {}
    if not cost or not cost[0]:
        return out
    for ci, k in _local_hungarian(
        cost, [c["img_bbox"] for c in boxes], [item["bbox"] for item in cands], radius=160.0
    ):
        if ci >= len(boxes) or k >= len(cands) or cost[ci][k] >= BIG:
            continue
        item = cands[k]
        info = _name_content(item)
        if info is None:
            continue
        out[id(boxes[ci])] = (info[0], item)
        item["used"] = True
    return out


def _bind_keys_and_names(comps: list[dict], texts: list[dict], pin_ids: set[int]) -> None:
    """按"value 之后剩余文字"决定 key/Name。

    - 有多个剩余文字：优先把位号形状文字给 key，box/d/led 再取型号/描述给 Name；
    - 只有一个剩余文字：key = Name = 该文字；
    - 没有任何剩余文字：key 留空（build_structure 会命名为 ∅ 开头）。
    """
    targets = [c for c in comps if c["type"] not in SYMBOL_TYPES]
    key_items: list[dict] = []
    for item in texts:
        if item.get("used") or id(item) in pin_ids:
            continue
        if item["cls"] in {"junk", "value", "pinnum", "mixed"}:
            continue
        if not _designator_variant(item, "box"):
            continue
        key_items.append(item)

    BIG = 1e6
    cost: list[list[float]] = []
    infos: list[list[tuple[str, float] | None]] = []
    affinities: list[list[float]] = []
    for comp in targets:
        row: list[float] = []
        info_row: list[tuple[str, float] | None] = []
        aff_row: list[float] = []
        for item in key_items:
            info = _designator_variant(item, comp["type"])
            info_row.append(info)
            aff = _prefix_affinity(_prefix_of(info[0]), comp["type"]) if info else 0.0
            aff_row.append(aff)
            if info is None:
                row.append(BIG)
                continue
            d = _text_dist(comp["img_bbox"], item["bbox"])
            if d > 130:
                row.append(BIG)
                continue
            row.append(d - min(15.0, 0.35 * info[1]))
        cost.append(row)
        infos.append(info_row)
        affinities.append(aff_row)

    used_des: set[str] = set()

    def _apply_key_pairs(pairs, use_cost):
        for ci, k in pairs:
            if ci >= len(targets) or k >= len(key_items) or use_cost[ci][k] >= BIG:
                continue
            comp = targets[ci]
            item = key_items[k]
            if comp.get("designator") or item.get("used"):
                continue
            info = infos[ci][k]
            if info is None:
                continue
            des = info[0]
            if des in used_des:
                continue
            comp["designator"] = des
            comp["_key_item"] = item
            item["used"] = True
            used_des.add(des)

    # 第一轮只让“前缀符合该 type”的位号参与，防止型号文字（如 SX1308）
    # 因为距离近而抢走真正的 U2 位号；第二轮再给仍无 key 的元件放宽到
    # 未知/非典型前缀（box 的型号 key、d 的 SS34 等）。
    if cost and cost[0]:
        cost1 = [
            [cost[i][j] if affinities[i][j] > 0.0 else BIG for j in range(len(key_items))]
            for i in range(len(targets))
        ]
        _apply_key_pairs(_local_hungarian(
            cost1, [c["img_bbox"] for c in targets], [item["bbox"] for item in key_items], radius=150.0
        ), cost1)
        cost2 = [
            [BIG if (targets[i].get("designator") or key_items[j].get("used")) else cost[i][j]
             for j in range(len(key_items))]
            for i in range(len(targets))
        ]
        _apply_key_pairs(_local_hungarian(
            cost2, [c["img_bbox"] for c in targets], [item["bbox"] for item in key_items], radius=150.0
        ), cost2)

    # Name：box / d / led / crystal 等允许与 key 不同
    box_names = _assign_box_names(comps, texts, pin_ids)
    for comp in comps:
        if comp["type"] in SYMBOL_TYPES:
            continue
        key = comp.get("designator")
        key_item = comp.get("_key_item")
        if comp["type"] == "box":
            picked = box_names.get(id(comp))
        elif comp["type"] in NAME_DIFF_TYPES:
            picked = _pick_name(comp, texts, pin_ids, exclude={id(key_item)} if key_item else None)
        else:
            picked = None
        if comp["type"] in NAME_DIFF_TYPES:
            if picked and comp["type"] != "box":
                picked[1]["used"] = True
            if key:
                comp["name"] = picked[0] if picked else key
            elif picked:
                comp["designator"] = picked[0]
                comp["name"] = picked[0]
            else:
                comp["name"] = None
        elif key:
            comp["name"] = key

    # 兜底：没有 key 的元件，如果只剩唯一一个可用文字，则 key=Name=该文字
    for comp in comps:
        if comp["type"] in SYMBOL_TYPES or comp.get("designator"):
            continue
        nearby = []
        for item in texts:
            if item.get("used") or id(item) in pin_ids:
                continue
            if item["cls"] in {"junk", "value", "pinnum", "mixed"}:
                continue
            info = _name_content(item)
            if info is None:
                continue
            if _text_dist(comp["img_bbox"], item["bbox"]) <= 110:
                nearby.append((info[0], item))
        if len(nearby) == 1:
            content, item = nearby[0]
            comp["designator"] = content
            comp["name"] = content
            item["used"] = True
    for comp in comps:
        comp.pop("_key_item", None)


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
    """Return [(number, name), ...] from BoxICLibrary.json for a box IC, or None."""
    lib = _lib_pins_exact(ic_name)
    if not lib:
        return None

    def _key(n: str) -> tuple:
        return (0, int(n)) if n.isdigit() else (1, n)

    ordered = sorted(lib.items(), key=lambda kv: _key(kv[0]))
    return ordered[:n_pins]


def _box_pin_observations(comp: dict, vision: VisionOut) -> tuple[list[float], list[dict]] | None:
    """Find a rectangular symbol and its visible stubs, without using GT or a part layout.

    The detector box is a search window, not the pin endpoint.  Long coloured
    strokes identify the body; short perpendicular strokes outside it identify
    left/right/top/bottom pins.  Plain black symbols retain the existing path.
    """
    import cv2
    import numpy as np

    image = getattr(vision, "img", None)
    if image is None or image.ndim != 3:
        return None
    bx1, by1, bx2, by2 = map(float, comp["img_bbox"])
    bw, bh = bx2 - bx1, by2 - by1
    if min(bw, bh) < 12:
        return None
    h, w = image.shape[:2]
    pad = 40
    ox, oy = max(0, int(bx1) - pad), max(0, int(by1) - pad)
    ex, ey = min(w, int(bx2) + pad + 1), min(h, int(by2) + pad + 1)
    patch = image[oy:ey, ox:ex].astype(np.int16)
    if patch.size == 0:
        return None
    b, g, r = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    masks = [((r-g > 20) & (r-b > 15)), ((b-g > 20) & (b-r > 15))]
    found = None
    for raw in masks:
        mask = raw.astype(np.uint8)
        vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((max(6, round(bh * .45)), 1), np.uint8))
        horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                 np.ones((1, max(6, round(bw * .45))), np.uint8))
        xs = np.flatnonzero(vert.sum(axis=0) >= bh * .42) + ox
        ys = np.flatnonzero(horiz.sum(axis=1) >= bw * .42) + oy

        def edge(values, target, lower, upper):
            values = [float(v) for v in values if lower <= v <= upper and abs(v-target) <= 30]
            if not values:
                return None
            nearest = min(values, key=lambda v: abs(v-target))
            close = [v for v in values if abs(v-nearest) <= 2]
            return float(np.median(close))

        cx, cy = (bx1+bx2)/2, (by1+by2)/2
        x1 = edge(xs, bx1, bx1-30, cx-3)
        x2 = edge(xs, bx2, cx+3, bx2+30)
        y1 = edge(ys, by1, by1-30, cy-3)
        y2 = edge(ys, by2, cy+3, by2+30)
        if None in (x1, x2, y1, y2):
            continue
        if x2-x1 < bw*.45 or y2-y1 < bh*.45:
            continue
        strength = float(vert.sum()+horiz.sum())
        if found is None or strength > found[0]:
            found = (strength, mask, [x1, y1, x2, y2])
    if found is None:
        return None
    _, mask, frame = found
    x1, y1, x2, y2 = frame
    observations = []
    for side in ("left", "right", "top", "bottom"):
        horizontal = side in {"left", "right"}
        boundary = {"left": x1, "right": x2, "top": y1, "bottom": y2}[side]
        sign = -1 if side in {"left", "top"} else 1
        low, high = (y1+3, y2-3) if horizontal else (x1+3, x2-3)
        runs = []
        for tangent in range(round(low), round(high)+1):
            points = []
            misses = 0
            for d in range(2, 39):
                normal = round(boundary) + sign*d
                x, y = (normal, tangent) if horizontal else (tangent, normal)
                ix, iy = x-ox, y-oy
                if not (0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]):
                    break
                if mask[iy, ix]:
                    points.append((x, y)); misses = 0
                else:
                    misses += 1
                    if misses >= 2:
                        break
            if len(points) >= 3:
                runs.append((tangent, points[-1]))
        groups = []
        for tangent, tip in runs:
            if not groups or tangent > groups[-1][-1][0]+1:
                groups.append([])
            groups[-1].append((tangent, tip))
        for group in groups:
            tangent = float(np.median([v[0] for v in group]))
            normal = float(np.median([v[1][0 if horizontal else 1] for v in group]))
            point = (normal, tangent) if horizontal else (tangent, normal)
            observations.append({"side": side, "xy": point})
    # A red wire can continue beyond the actual stub.  Estimate the common
    # length from parallel stubs on this body; keep top/bottom independent of
    # left/right (an exposed pad can legitimately have a longer pin).
    for sides in ({"left", "right"}, {"top", "bottom"}):
        group = [t for t in observations if t["side"] in sides]
        lengths = []
        for t in group:
            px, py = t["xy"]
            lengths.append({"left": x1-px, "right": px-x2,
                            "top": y1-py, "bottom": py-y2}[t["side"]])
        if len(lengths) < 3:
            continue
        clusters = [[v for v in lengths if abs(v-base) <= 3] for base in lengths]
        cluster = max(clusters, key=lambda vs: (len(vs), -float(np.median(vs))))
        if len(cluster) < max(2, len(lengths)*.4):
            continue
        typical = float(np.median(cluster))
        for t, length in zip(group, lengths):
            if length <= typical+5:
                continue
            px, py = t["xy"]
            t["xy"] = {"left": (x1-typical, py), "right": (x2+typical, py),
                       "top": (px, y1-typical), "bottom": (px, y2+typical)}[t["side"]]
    return (frame, observations) if len(observations) >= 2 else None


def _refine_box_pin_points(comp: dict, owned: list[PinBox], vision: VisionOut) -> list[PinBox]:
    from dataclasses import replace

    observation = _box_pin_observations(comp, vision)
    if not observation:
        return owned
    frame, tips = observation
    x1, y1, x2, y2 = frame
    result = []
    sources = {}
    for pin in owned:
        px, py = pin.xy
        side = min({"left": abs(px-x1), "right": abs(px-x2),
                    "top": abs(py-y1), "bottom": abs(py-y2)},
                   key={"left": abs(px-x1), "right": abs(px-x2),
                        "top": abs(py-y1), "bottom": abs(py-y2)}.get)
        horizontal = side in {"left", "right"}
        tangent = py if horizontal else px
        choices = [t for t in tips if t["side"] == side and
                   abs(t["xy"][1 if horizontal else 0] - tangent) <= 5]
        if not choices:
            result.append(pin)
            continue
        tip = min(choices, key=lambda t: _dist(t["xy"], pin.xy))["xy"]
        outward = {"left": px-tip[0], "right": tip[0]-px,
                   "top": py-tip[1], "bottom": tip[1]-py}[side]
        # An existing NC/X-centre can be outside the surviving red stroke.
        # Keep that endpoint; move inward points along a visible stub only.
        before = {"left": x1-px, "right": px-x2, "top": y1-py, "bottom": py-y2}[side]
        after = before + outward
        move = outward > 5.0 or before <= .76*after
        xy = tip if 1.5 < outward <= 32 and move else pin.xy
        refined = replace(pin, xy=xy)
        result.append(refined)
        sources[tuple(xy)] = pin
    comp["_box_pin_source_by_xy"] = sources
    return result


def _match_box_pin_rows(comp: dict, owned: list[PinBox], texts: list[dict],
                        vision: VisionOut, ic_name: str | None,
                        fallback: list[tuple[str, str, tuple[float, float]]]) -> list[tuple[str, str, tuple[float, float]]] | None:
    """Match visible pin rows on four sides; explicit numbers precede names.

    Missing numbers are inferred only from a consistent observed sequence or
    an unambiguous library name.  No package-wide clockwise numbering is used.
    """
    observed = _box_pin_observations(comp, vision)
    if not observed:
        return None
    frame, slots = observed
    x1, y1, x2, y2 = frame
    lib = _lib_pins_exact(ic_name)
    norm = lambda s: "".join(str(s).lower().split())
    library_names = {norm(n): n for n in lib.values()}
    names_to_numbers: dict[str, list[str]] = defaultdict(list)
    for number, name in lib.items():
        names_to_numbers[norm(name)].append(number)

    def side_for_point(point):
        px, py = point
        if y1 <= py <= y2:
            if px < x1:
                return "left"
            if px > x2:
                return "right"
        if x1 <= px <= x2:
            if py < y1:
                return "top"
            if py > y2:
                return "bottom"
        distances = {"left": abs(px-x1), "right": abs(px-x2),
                     "top": abs(py-y1), "bottom": abs(py-y2)}
        return min(distances, key=distances.get)

    def text_side(bb):
        distances = {"left": min(abs(bb[0]-x1), abs(bb[2]-x1)),
                     "right": min(abs(bb[0]-x2), abs(bb[2]-x2)),
                     "top": min(abs(bb[1]-y1), abs(bb[3]-y1)),
                     "bottom": min(abs(bb[1]-y2), abs(bb[3]-y2))}
        return min(distances, key=distances.get)

    nums, names = [], []
    excluded = {norm(comp.get(k) or "") for k in ("name", "designator", "lib_key")}
    for text in texts:
        if text.get("used"):
            continue
        bb = text["bbox"]
        tx, ty = _center(bb)
        if not (x1-45 <= tx <= x2+45 and y1-45 <= ty <= y2+45):
            continue
        side = text_side(bb)
        variants = sorted(_variants_of(text), key=lambda v: -float(v.get("score") or 0))
        for variant in variants:
            content = str(variant.get("content") or "").strip()
            if not content or norm(content) in excluded:
                continue
            if re.fullmatch(r"\d{1,3}|[A-Z]\d{1,2}", content):
                if not (x1+4 < tx < x2-4 and y1+4 < ty < y2-4):
                    # A digit near a body corner belongs to the edge outside
                    # its centre, not whichever text-box edge touches a corner.
                    number_side = ("left" if tx < x1 else "right" if tx > x2 else
                                   "top" if ty < y1 else "bottom" if ty > y2 else side)
                    nums.append((text, content, number_side))
                    break
                elif _has_letter(content):
                    names.append((text, content, side))
                    break
            elif (x1+1 <= tx <= x2-1 and y1+1 <= ty <= y2-1
                  and _has_letter(content) and 2 <= len(content) <= 24):
                # Parentheses and polarity markers belong to the printed name;
                # ADJ(GND) and VOUT(TAB) must not collapse to GND and VOUT.
                names.append((text, content, side))
                break

    pitch = {}
    sequence_locked: set[int] = set()
    for side in ("left", "right", "top", "bottom"):
        axis = 1 if side in {"left", "right"} else 0
        positions = sorted(s["xy"][axis] for s in slots if s["side"] == side)
        gaps = sorted(b-a for a, b in zip(positions, positions[1:]) if b-a > 4)
        pitch[side] = gaps[len(gaps)//2] if gaps else 24.

    def assign(candidates, numeric):
        if not candidates:
            return {}, {}
        big = 1e6
        number_edges = {}
        if numeric:
            # Choose one printing convention per edge, rather than allowing
            # each digit to independently steal an adjacent row.  OCR boxes
            # can sit above/below horizontal stubs or left/right of verticals.
            for side in ("left", "right", "top", "bottom"):
                axis = 1 if side in {"left", "right"} else 0
                local_slots = [s for s in slots if s["side"] == side]
                local_texts = [t for t, _content, s in candidates if s == side]
                if not local_slots or not local_texts:
                    continue
                limit = min(14., max(7., pitch[side]*.6))
                choices = []
                for edge in (0, 2):
                    matrix = []
                    for slot in local_slots:
                        row = []
                        for text in local_texts:
                            d = abs(slot["xy"][axis]-text["bbox"][axis+edge])
                            row.append(d if d <= limit else big)
                        row += [limit+2.] * len(local_slots)
                        matrix.append(row)
                    pairs = [(i,j) for i,j in _hungarian(matrix)
                             if j < len(local_texts) and matrix[i][j] < big]
                    choices.append((len(pairs), -sum(matrix[i][j] for i,j in pairs), edge))
                number_edges[side] = max(choices)[2]
        costs = []
        for slot in slots:
            side, xy = slot["side"], slot["xy"]
            axis = 1 if side in {"left", "right"} else 0
            limit = min(14., max(7., pitch[side]*.6)) if numeric else min(24., max(10., pitch[side]*.72))
            row = []
            for text, content, candidate_side in candidates:
                if side != candidate_side:
                    row.append(big)
                    continue
                bb = text["bbox"]
                centre = _center(bb)
                if numeric:
                    d = abs(xy[axis]-bb[axis+number_edges.get(side, 2)]) + .08*abs(xy[axis]-centre[axis])
                else:
                    d = abs(xy[axis]-centre[axis])
                row.append(d + .005*_dist(xy, centre) if d <= limit else big)
            row.extend([limit+2.] * len(slots))
            costs.append(row)
        matched, quality = {}, {}
        for i, j in _hungarian(costs):
            if j < len(candidates) and costs[i][j] < big:
                matched[i] = candidates[j][1]
                quality[i] = costs[i][j]
        return matched, quality

    numbers, num_cost = assign(nums, True)
    labels, _ = assign(names, False)
    if len(numbers) < 2 and not (lib and len(labels) >= 2):
        return None
    # Repeated numeric labels in OCR are resolved by geometric evidence before
    # the sequence fit.  Do not let an earlier weak assignment claim the ID.
    claims: dict[str, list[int]] = defaultdict(list)
    for i, number in numbers.items():
        claims[number].append(i)
    for number, indices in claims.items():
        for i in sorted(indices, key=lambda k: num_cost[k])[1:]:
            numbers.pop(i)
    for side in ("left", "right", "top", "bottom"):
        axis = 1 if side in {"left", "right"} else 0
        ordered = sorted([i for i, s in enumerate(slots) if s["side"] == side],
                         key=lambda i: slots[i]["xy"][axis])
        numeric = [(j, int(numbers[i])) for j, i in enumerate(ordered)
                   if i in numbers and numbers[i].isdigit()]
        if len(numeric) < 3:
            continue
        best = None
        for step in (-1, 1):
            offsets = [number-step*j for j, number in numeric]
            offset = max(set(offsets), key=offsets.count)
            agreement = offsets.count(offset)
            if best is None or agreement > best[0]:
                best = (agreement, step, offset)
        agreement, step, offset = best
        if agreement < max(3, len(numeric)*.8):
            continue
        sequence_locked.update(ordered)
        for j, i in enumerate(ordered):
            candidate = str(offset+step*j)
            if i not in numbers and int(candidate) > 0 and candidate not in numbers.values():
                numbers[i] = candidate

    for i, label in labels.items():
        if i not in numbers:
            available = [n for n in names_to_numbers.get(norm(label), []) if n not in numbers.values()]
            if len(available) == 1:
                numbers[i] = available[0]
    # Reconcile an ambiguous number with an exact, unique spatial pin name.
    # A complete sequence read from the drawing remains authoritative over
    # the catalogue, since package variants can differ in pin IDs.
    proposals = {}
    locked_ids = {numbers[i] for i in sequence_locked if i in numbers}
    for i, label in labels.items():
        candidates = names_to_numbers.get(norm(label), [])
        if i not in sequence_locked and len(candidates) == 1 and candidates[0] not in locked_ids:
            proposals[i] = candidates[0]
    proposed_ids = set(proposals.values())
    for i in list(numbers):
        if i not in proposals and i not in sequence_locked and numbers[i] in proposed_ids:
            numbers.pop(i)
    numbers.update(proposals)
    remaining_slots = [i for i in range(len(slots)) if i not in numbers]
    remaining_ids = [n for n in lib if n not in numbers.values()]
    if len(slots) == len(lib) and len(numbers) >= 2 and len(remaining_slots) == len(remaining_ids) == 1:
        numbers[remaining_slots[0]] = remaining_ids[0]
    for i, slot in enumerate(slots):
        if i in numbers:
            continue
        candidates = [row for row in fallback if row[0] not in numbers.values()
                      and _dist(row[2], slot["xy"]) <= 6]
        if len(candidates) == 1:
            numbers[i] = candidates[0][0]
            labels.setdefault(i, candidates[0][1])
    if len(numbers) < max(2, len(slots)*.9) or len(set(numbers.values())) != len(numbers):
        return None

    result = []
    sources = {}
    used_sources = set()
    for i, slot in enumerate(slots):
        number = numbers.get(i)
        if number is None:
            continue
        xy, side = slot["xy"], slot["side"]
        axis = 1 if side in {"left", "right"} else 0
        source_options = [q for q in owned if id(q) not in used_sources
                          and side_for_point(q.xy) == side and abs(q.xy[axis]-xy[axis]) <= 5]
        source = min(source_options, key=lambda q: _dist(q.xy, xy), default=None)
        if source is not None:
            used_sources.add(id(source))
            # Preserve the outside centre of an NC cross/terminal circle.
            outward = {"left": source.xy[0]-xy[0], "right": xy[0]-source.xy[0],
                       "top": source.xy[1]-xy[1], "bottom": xy[1]-source.xy[1]}[side]
            px, py = source.xy
            before = {"left": x1-px, "right": px-x2, "top": y1-py, "bottom": py-y2}[side]
            after = before + outward
            # Preserve a nearby NC/terminal centre, but not a skeleton point
            # farther out on a connected wire than the visible pin stub.
            if -10.0 <= outward <= 1.5 or (0 < outward <= 5.0 and before > .76*after):
                xy = source.xy
            sources[tuple(xy)] = source
        label = labels.get(i)
        expected = lib.get(number)
        if label and expected:
            if norm(label) == norm(expected) or norm(label).lstrip("o0|") == norm(expected):
                label = expected
            elif len(label) <= 3 and norm(label)[::-1] == norm(expected) and any(
                norm(v.get("content")) == norm(label) and int(v.get("angle") or 0) in {90, 270}
                for text, _content, _side in names for v in _variants_of(text)
            ):
                label = expected
        if not label:
            label = expected or (getattr(source, "name", None) if source else None) or number
        near_legacy = [row for row in fallback if row[0] == number and
                       _dist(row[2], xy) <= 6 and norm(row[1]) == norm(label)]
        if near_legacy:
            label = near_legacy[0][1]
        result.append((number, label, xy))
    comp["_box_pin_source_by_xy"] = sources
    return result


def _pins_for_component_legacy(
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
    if any(getattr(p, "owner", None) is not None for p in det_pins) and comp.get("src_idx") is not None:
        # skele_wire 输出的引脚自带归属元件下标：直接按归属取，避免相邻器件抢引脚
        owned = [p for p in det_pins if getattr(p, "owner", None) == comp["src_idx"]]
    else:
        owned = [p for p in det_pins if (x1 - m) <= p.xy[0] <= (x2 + m) and (y1 - m) <= p.xy[1] <= (y2 + m)]
    if not owned and comp.get("_symbol_inherited_pin") is not None:
        owned = [comp["_symbol_inherited_pin"]]
    # 去重：YOLO 可能在同一位置（很近）检测到两个 pin_tip，导致同一引脚重复标注。
    # 按坐标距离 ≤ pin_dedup_px 的引脚只保留置信度最高的一个。
    owned = _dedup_pins(owned, pin_dedup_px)
    out: list[tuple[str, str, tuple[float, float]]] = []
    if comp["type"] == "box":
        owned = _refine_box_pin_points(comp, owned, vision)
        lib = _lib_pins_exact(ic_name)   # {number: name}（精确型号查表）
        # YOLO 有时在框边缘多检 pin_tip；库已知引脚数时按置信度保留最可信的 N 个，
        # 否则多出来的 tip 会抢占相邻 pin 的 name/number，造成整体错位。
        if owned and lib and len(owned) > len(lib):
            owned = sorted(owned, key=lambda p: getattr(p, "score", 0.0), reverse=True)[:len(lib)]
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
        # name -> [numbers]（数值升序），用于同名多编号的几何顺序分配
        name_to_nums: dict[str, list[str]] = defaultdict(list)
        for num in sorted(lib, key=lambda n: (0, int(n)) if n.isdigit() else (1, n)):
            name_to_nums[lib[num]].append(num)

        cx = (x1 + x2) / 2.0
        owned = _order_pins_ccw(owned, cx)  # 逆时针几何顺序

        # 第一遍：先 pinname（图中文字），后 number（图中数字）。
        # 库校验：OCR 噪声——数字不在库编号里、文字不在库引脚名里的，一律当缺失处理，
        # 交给第二遍的库反查/几何分配，避免编号整体移位、'n' 误绑。
        valid_nums = set(lib.keys()) if lib else set()
        valid_nums_lower = {str(n).lower(): str(n) for n in valid_nums}
        valid_names = {"".join(str(n).lower().split()): str(n) for n in lib.values() if n} if lib else {}

        # 引脚文字候选：name 和 number 分别建候选集合，之后做“引脚 tip × 文字”的
        # 联合一对一匹配（距离和最小），而不是按逆时针顺序逐个抢最近文字。
        name_cands: list[tuple[dict, str]] = []
        num_cands: list[tuple[dict, str]] = []
        for t in texts:
            if t.get("used"):
                continue
            tc = _center(t["bbox"])
            inside_box = (x1 - 6 <= tc[0] <= x2 + 6 and y1 - 6 <= tc[1] <= y2 + 6)
            near_box = _text_dist(comp["img_bbox"], t["bbox"]) <= 35.0
            name = None
            num = None
            for v in _variants_of(t):
                c = str(v.get("content") or "").strip()
                if not c:
                    continue
                # 库内合法引脚号优先（Type-C 的 A12 这类编号也含字母）
                if lib:
                    if c in valid_nums:
                        num = c
                        break
                    if c.lower() in valid_nums_lower:
                        num = valid_nums_lower[c.lower()]
                        break
                elif t["cls"] == "pinnum" and c.isdigit() and len(c) <= 3:
                    num = c
                    break
                if len(c) < 2 or len(c) > 14 or not _has_letter(c):
                    continue
                canon = valid_names.get("".join(c.lower().split()))
                if canon:
                    name = canon
                    break
                # 库外名称（如 VOUT(TAB)/TAB）只接受已被判定为框内 pinname 的文字
                if t["cls"] == "pinname" and not _value_form(c):
                    name = c
                    break
            # pinname 只取元件框内文字；pin number 可在框边缘外侧一点
            if name and inside_box:
                name_cands.append((t, name))
            elif num and near_box:
                num_cands.append((t, num))

        # 同一库引脚名可能被 OCR/局部重识别输出多份（如 U1 的 EP 出现两次）。
        # 库内名映射到几个编号，最多保留几份候选，其余按"离最近 pin tip 最近"丢弃，
        # 避免多余副本去抢占别的引脚。
        if lib and name_cands:
            grouped: dict[str, list[tuple[dict, str]]] = defaultdict(list)
            for tn in name_cands:
                grouped[tn[1]].append(tn)
            pruned: list[tuple[dict, str]] = []
            for nm, lst in grouped.items():
                nums = name_to_nums.get(nm)
                if nums is None:
                    pruned.extend(lst)
                    continue
                limit = max(1, len(nums))
                if len(lst) > limit:
                    lst = sorted(
                        lst,
                        key=lambda tn: min(
                            (_dist(p.xy, _center(tn[0]["bbox"])), -float(tn[0].get("score") or 0.0))
                            for p in owned
                        ),
                    )[:limit]
                pruned.extend(lst)
            name_cands = pruned

        def _joint_text_map(cands: list[tuple[dict, str]]) -> dict[int, str]:
            """pin tip × 候选文字做全局一对一，返回 {tip_index: 文字内容}。"""
            if not owned or not cands:
                return {}
            BIG = 1e6
            mcost: list[list[float]] = []
            for p in owned:
                row = []
                for t, _c in cands:
                    d = _dist(p.xy, _center(t["bbox"]))
                    row.append(d if d <= 90 else BIG)
                mcost.append(row)
            out_map: dict[int, str] = {}
            for ti, k in _hungarian(mcost):
                if ti < len(owned) and k < len(cands) and mcost[ti][k] < BIG:
                    out_map[ti] = cands[k][1]
            return out_map

        name_map = _joint_text_map(name_cands)
        num_map = _joint_text_map(num_cands)

        items = []
        for pi, p in enumerate(owned):
            xy = _snap_to_wire(p.xy, vision, snap_px)
            pinname = name_map.get(pi)
            if lib:
                if p.name:
                    canon = valid_names.get("".join(str(p.name).lower().split()))
                    if canon:
                        pinname = canon
                number = None
                ocr_num = num_map.get(pi)
                if pinname and pinname in name_to_nums:
                    nums = name_to_nums[pinname]
                    if ocr_num in nums:
                        number = ocr_num
                elif pinname and ocr_num:
                    number = ocr_num
                elif not pinname and ocr_num in lib:
                    number = ocr_num
                    pinname = lib[ocr_num]
                elif ocr_num:
                    number = ocr_num
            else:
                if p.name:
                    pinname = pinname or p.name
                number = num_map.get(pi) or p.number
            items.append({"pinname": pinname, "number": number, "xy": xy})

        # 第二遍：把 OCR 编号与库内编号对齐；同名多号按 tip 顺序分配未占用编号，
        # 冲突/缺失时用几何顺序编号兜底。
        name_lower_to_nums: dict[str, list[str]] = {
            "".join(n.lower().split()): v for n, v in name_to_nums.items()
        }
        seen: set[str] = set()
        if lib and items:
            cx_ = (x1 + x2) / 2.0
            left_idx = sorted([i for i in range(len(items)) if items[i]["xy"][0] <= cx_],
                              key=lambda i: items[i]["xy"][1])          # 左列上→下
            right_idx = sorted([i for i in range(len(items)) if items[i]["xy"][0] > cx_],
                               key=lambda i: items[i]["xy"][1])         # 右列上→下
            geom: dict[int, str] = {}
            n = 1
            for i in left_idx:
                geom[i] = str(n)
                n += 1
            for i in right_idx:
                geom[i] = str(n)
                n += 1
            for i, item in enumerate(items):
                pinname, number, xy = item["pinname"], item["number"], item["xy"]
                norm = "".join(pinname.lower().split()) if pinname else ""
                nums = name_lower_to_nums.get(norm)
                if nums:
                    if number in nums and number not in seen:
                        pass
                    else:
                        avail = [x for x in nums if x not in seen]
                        number = avail[0] if avail else None
                if number is None or number in seen:
                    cand = geom.get(i)
                    if not cand or cand in seen:
                        k = 1
                        while str(k) in seen:
                            k += 1
                        cand = str(k)
                    number = cand
                if pinname is None and number in lib:
                    pinname = lib[number]
                seen.add(number)
                out.append((number, pinname or number, xy))
            return out

        for item in items:
            pinname, number, xy = item["pinname"], item["number"], item["xy"]
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
        for seq, p in enumerate(owned, start=1):
            xy = _snap_to_wire(p.xy, vision, snap_px)
            out.append((str(seq), str(seq), xy))
        return out
    for number, xy in _geometry_pins(comp["img_bbox"], comp["type"]):
        out.append((number, number, _snap_to_wire(xy, vision, snap_px)))
    return out


def _pins_for_component(
    comp: dict, det_pins: list[PinBox], texts: list[dict], vision: VisionOut,
    snap_px: float, ic_name: str | None = None, pin_dedup_px: float = 5.0,
) -> list[tuple[str, str, tuple[float, float]]]:
    legacy = _pins_for_component_legacy(comp, det_pins, texts, vision, snap_px,
                                        ic_name=ic_name, pin_dedup_px=pin_dedup_px)
    if comp["type"] != "box":
        return legacy
    x1, y1, x2, y2 = comp["img_bbox"]
    has_owners = any(getattr(p, "owner", None) is not None for p in det_pins)
    owned = [p for p in det_pins if
             (getattr(p, "owner", None) == comp.get("src_idx") if has_owners and comp.get("src_idx") is not None
              else x1-32 <= p.xy[0] <= x2+32 and y1-32 <= p.xy[1] <= y2+32)]
    owned = _dedup_pins(owned, pin_dedup_px)
    rows = _match_box_pin_rows(comp, owned, texts, vision, ic_name, legacy)
    observation = _box_pin_observations(comp, vision)
    if rows is None:
        rows = legacy
    elif observation:
        frame, _ = observation
        fx1, fy1, fx2, fy2 = frame
        lib = _lib_pins_exact(ic_name)
        for number, label, xy in legacy:
            if any(number == r[0] or _dist(xy, r[2]) <= 6 for r in rows):
                continue
            px, py = xy
            # Retain an existing, library-supported pin whose stroke was partly
            # obscured by a wire/NC cross, but never keep a rectangle corner.
            at_vertical = min(abs(px-fx1), abs(px-fx2)) < 45 and fy1+3 < py < fy2-3
            at_horizontal = min(abs(py-fy1), abs(py-fy2)) < 45 and fx1+3 < px < fx2-3
            if (at_vertical or at_horizontal) and lib.get(number) == label:
                rows.append((number, label, xy))
    # A complete printed name takes precedence over a shortened catalogue alias.
    # This only restores an OCR substring at that pin row; no label is invented.
    result = []
    for number, label, xy in rows:
        candidates = []
        canonical_number = str(int(number)) if number.isdigit() else number
        for text in texts:
            bb = text["bbox"]
            tx, ty = _center(bb)
            if not (x1 <= tx <= x2 and y1 <= ty <= y2):
                continue
            pin_distances = {"left": abs(xy[0]-x1), "right": abs(xy[0]-x2),
                             "top": abs(xy[1]-y1), "bottom": abs(xy[1]-y2)}
            pin_side = ("left" if xy[0] < x1 else "right" if xy[0] > x2 else
                        "top" if xy[1] < y1 else "bottom" if xy[1] > y2 else
                        min(pin_distances, key=pin_distances.get))
            text_distances = {"left": min(abs(bb[0]-x1), abs(bb[2]-x1)),
                              "right": min(abs(bb[0]-x2), abs(bb[2]-x2)),
                              "top": min(abs(bb[1]-y1), abs(bb[3]-y1)),
                              "bottom": min(abs(bb[1]-y2), abs(bb[3]-y2))}
            if pin_side != min(text_distances, key=text_distances.get):
                continue
            axis = 1 if pin_side in {"left", "right"} else 0
            # A merged OCR crop may join a connector circle with its digit
            # ("01").  Remove the prefix only when an independent, shorter
            # OCR crop explicitly reads the canonical number on this row.
            if (number != canonical_number and number.startswith("0")
                    and canonical_number not in {row[0] for row in rows}
                    and abs(_center(bb)[axis]-xy[axis]) <= 10
                    and str(text.get("content") or "").strip() == canonical_number):
                if label == number:
                    label = canonical_number
                number = canonical_number
            for variant in _variants_of(text):
                content = str(variant.get("content") or "").strip()
                if "(" not in content or ")" not in content:
                    continue
                if "".join(label.lower().split()) not in "".join(content.lower().split()):
                    continue
                distance = abs(ty-xy[1]) if min(abs(xy[0]-x1), abs(xy[0]-x2)) < min(abs(xy[1]-y1), abs(xy[1]-y2)) else abs(tx-xy[0])
                if distance <= 8:
                    candidates.append((distance, content))
        if candidates:
            label = min(candidates)[1]
        result.append((number, label, xy))
    return result


def _nearest_pinnum(
    pt: tuple[float, float], texts: list[dict], used_ids: set[int] | None = None
) -> tuple[str, int] | None:
    """引脚附近的数字/编号文字（多方向候选中取最像编号的一个）。"""
    used_ids = used_ids or set()
    best, best_d, best_id = None, 60.0, None
    for t in texts:
        if t.get("used") or id(t) in used_ids or t["cls"] not in {"pinnum", "pinname", "value"}:
            continue
        c = _center(t["bbox"])
        d = _dist(pt, c)
        if d >= best_d:
            continue
        for v in _variants_of(t):
            content = str(v.get("content") or "").strip()
            if PIN_NUM_RE.match(content):
                best, best_d, best_id = content, d, id(t)
                break
    return (best, best_id) if best is not None and best_id is not None else None


def _nearest_pinname(
    pt: tuple[float, float],
    texts: list[dict],
    comp: dict,
    used_ids: set[int] | None = None,
    marked_only: bool = False,
) -> tuple[str, int] | None:
    """引脚附近的短字母文字（库不存在时使用；不把 value/普通位号当 pinname）。

    marked_only=True 时只用已被判定为框内引脚文字（cls=pinname）的项，
    用于"库存在但库里没有这个具体 pinname（如 VOUT(TAB)）"时的 OCR 兜底。
    """
    if comp["type"] != "box":
        return None
    used_ids = used_ids or set()
    x1, y1, x2, y2 = comp["img_bbox"]
    m = 30.0  # 元件框扩展余量：pinname 文字大部分在框内

    def _pick(require_inside: bool) -> tuple[str, int] | None:
        best, best_d, best_id = None, (1e9 if require_inside else 100.0), None
        for t in texts:
            if t.get("used") or id(t) in used_ids or t["cls"] not in {"pinname", "name", "other"}:
                continue
            if marked_only and t["cls"] != "pinname":
                continue
            c = _center(t["bbox"])
            if require_inside and not (x1 - m <= c[0] <= x2 + m and y1 - m <= c[1] <= y2 + m):
                continue
            d = _dist(pt, c)
            if d >= best_d:
                continue
            for v in _variants_of(t):
                content = str(v.get("content") or "").strip()
                if not content or len(content) < 2 or len(content) > 14 or not _has_letter(content):
                    continue
                if t["cls"] != "pinname" and (_as_designator(content) or _value_form(content)):
                    continue
                best, best_d, best_id = content, d, id(t)
                break
        return (best, best_id) if best is not None and best_id is not None else None

    # pinname 必须位于元件框内部；框外 100px 内的文字可能是相邻元件/网络的标签
    return _pick(True)


def _nearest_pinname_from_set(
    pt: tuple[float, float],
    texts: list[dict],
    comp: dict,
    name_map: dict[str, str],
    used_ids: set[int] | None = None,
) -> tuple[str, int] | None:
    """在引脚附近文字里找库内合法引脚名（大小写不敏感），返回(规范拼写, 文本 id)。

    同一段 OCR 文字只能被一个引脚使用，避免 'VIN' 同时贴到相邻两个 pin 上。
    """
    if not name_map:
        return None
    used_ids = used_ids or set()
    x1, y1, x2, y2 = comp["img_bbox"]
    m = 30.0
    best, best_d, best_id = None, 1e9, None
    for t in texts:
        if t.get("used") or id(t) in used_ids or t["cls"] not in {"pinname", "name", "other", "pinnum"}:
            continue
        c = _center(t["bbox"])
        if not (x1 - m <= c[0] <= x2 + m and y1 - m <= c[1] <= y2 + m):
            continue
        d = _dist(pt, c)
        if d >= best_d:
            continue
        for v in _variants_of(t):
            content = str(v.get("content") or "").strip()
            canon = name_map.get("".join(content.lower().split()))
            if canon:
                best, best_d, best_id = canon, d, id(t)
                break
    return (best, best_id) if best is not None and best_id is not None else None


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


def _build_nets_from_skele_nets(
    out: dict,
    vision: VisionOut,
    pin_nodes: list[dict],
    height: float,
    pin_ref_map: dict[tuple[int, int], str],
    match_tol: float = 12.0,
) -> bool:
    """用 skele_wire.extract_nets 的原始输出生成 nets。

    skele_wire 的 hyperGraph 以 `comp_<cid>.pin_<idx>` 引用元件引脚，cid 是
    vision.components 的下标。stage2 生成 pins 时记录了 (cid, idx) -> 最终 pin ref
    的映射，这里直接查表，避免坐标最近邻在密集 IC 上匹配错引脚。
    返回是否生成了至少一个 net。
    """
    skele_nets = getattr(vision, "skele_nets", None) or {}
    if not skele_nets:
        return False

    coords = [p["xy"] for p in pin_nodes]
    refs_all = [p["ref"] for p in pin_nodes]
    skele_pins = getattr(vision, "skele_pins", None) or {}

    def _pin_pos(cid: int, idx: int) -> tuple[float, float] | None:
        info = (skele_pins.get(f"comp_{cid}") or {}).get(f"pin_{idx}") or {}
        pt = info.get("point") or {}
        if "x" in pt and "y" in pt:
            return float(pt["x"]), float(pt["y"])
        return None

    def _nearest_ref(xy: tuple[float, float]) -> str | None:
        best_i, best_d = -1, match_tol
        for i, (cx, cy) in enumerate(coords):
            d = _dist(xy, (cx, cy))
            if d < best_d:
                best_d, best_i = d, i
        return refs_all[best_i] if best_i >= 0 else None

    def _net_no(k: str) -> int:
        tail = str(k).rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    net_no = 0
    for net_key in sorted(skele_nets, key=_net_no):
        net = skele_nets[net_key] or {}
        hg = str(net.get("hyperGraph") or "").strip("()")
        seen: set[str] = set()
        refs: list[str] = []
        for token in hg.split(","):
            token = token.strip()
            if not token or "." not in token:
                continue
            cpart, ppart = token.rsplit(".", 1)
            try:
                cid = int(cpart.rsplit("_", 1)[1])
                idx = int(ppart.rsplit("_", 1)[1])
            except Exception:  # noqa: BLE001
                continue
            ref = pin_ref_map.get((cid, idx))
            if ref is None:
                xy = _pin_pos(cid, idx)
                ref = _nearest_ref(xy) if xy is not None else None
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
        if not refs:
            continue

        net_no += 1
        edges: dict = {}
        for j, seg in enumerate((net.get("edges") or {}).values(), start=1):
            pts = [contest_point(float(pt["x"]), float(pt["y"]), height)
                   for pt in seg if "x" in pt and "y" in pt]
            if pts:
                edges[f"edge_{j}"] = pts
        out["nets"][f"net_{net_no}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": edges}
    return net_no > 0


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


def _propagate_symbol_values(out: dict) -> None:
    """把同一全局网络/同一导线网络上的电源标签传播给缺 value 的符号。

    只传播图中**已经识别到**的文字：例如只要有一个接地符号识别到 'GND'，
    其余没有标签的接地符号都属于同一个全局 GND 网；同一根电源线上的多个
    v 符号共享同一个标签。不凭空创造文字。
    """
    comps = out.get("components") or {}
    gnd_vals = [c.get("value") for c in comps.values()
                if c.get("type") == "gnd" and c.get("value")]
    if gnd_vals:
        common = max(set(gnd_vals), key=gnd_vals.count)
        if str(common).upper() == "GND":
            for c in comps.values():
                if c.get("type") == "gnd" and not c.get("value"):
                    c["value"] = common
    for net in (out.get("nets") or {}).values():
        text = str(net.get("hyperGraph") or "").strip("()")
        keys = [r.split(".", 1)[0].strip() for r in text.split(",") if r.strip()]
        for ctype in ({"v", "vcc"},):
            vals = [comps[k].get("value") for k in keys
                    if k in comps and comps[k].get("type") in ctype and comps[k].get("value")]
            if not vals:
                continue
            common = max(set(vals), key=vals.count)
            for k in keys:
                c = comps.get(k)
                if c and c.get("type") in ctype and not c.get("value"):
                    c["value"] = common


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_structure(vision: VisionOut, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    if cfg.get("models"):
        from .model_provenance import verify_inference_bundle

        verify_inference_bundle([cfg["models"].get("yolo_weights")])
    cmap = load_category_map()
    thr = cfg.get("thresholds") or {}
    vcfg = cfg.get("vision") or {}
    eps = float(thr.get("pin_wire_eps", 5.0))
    snap_px = float(vcfg.get("pin_snap_px", 5.0))
    pin_dedup_px = float(vcfg.get("pin_dedup_px", 5.0))
    height = float(vision.height)
    out = empty_contest_json()

    texts = _prepare_texts([
        {"bbox": t.bbox, "content": t.content, "score": t.score,
         "angle": getattr(t, "angle", 0), "source": getattr(t, "source", "main")}
        for t in vision.texts
    ])

    comps: list[dict] = []
    for source_idx, det in enumerate(vision.components):
        comps.append({"type": _type_of(det.label, cmap), "img_bbox": list(det.bbox), "designator": None,
                      "score": det.score, "net_label": None, "src_idx": source_idx})
    comps, symbol_aliases = _clean_symbol_components(comps, getattr(vision, "img", None))

    # Reading order (top then left) so duplicate designators number stably.
    comps.sort(key=lambda c: (round(c["img_bbox"][1] / 10), round(c["img_bbox"][0] / 10)))

    # 文字匹配顺序：先标记 box 内部引脚文字 → value → key/Name。
    # value 先做，剩下的文字再决定 key/Name；没有剩余文字时 key 用 ∅ 自由命名。
    pin_ids = _mark_pin_texts(comps, texts, vision.pins)
    _refine_component_texts(comps, texts, pin_ids, vision, cfg)
    _assign_symbol_values(comps, texts, pin_ids)
    _assign_values(comps, texts, pin_ids)
    _bind_keys_and_names(comps, texts, pin_ids)

    # 电源/网络符号：net_* 的 key/Name 直接用标签文字（如 BAT/VIN），gnd/v 保持 ∅。
    for comp in comps:
        if comp["type"] in SYMBOL_TYPES:
            val = comp.get("value")
            if comp["type"].startswith("net_") and val:
                comp["designator"] = val
            comp["name"] = comp.get("designator") or None

    used_keys: dict[str, int] = {}
    empty_seq = 0
    pin_nodes: list[dict] = []
    # skele_wire 引脚按归属元件分组，用于把最终 pin ref 反查回 (comp_idx, pin_idx)
    skele_by_owner: dict[int, list[PinBox]] = defaultdict(list)
    for p in vision.pins:
        owner = getattr(p, "owner", None)
        if owner is not None and getattr(p, "idx", None) is not None:
            skele_by_owner[owner].append(p)
    # If the retained duplicate has no skeleton tip, preserve a unique tip
    # from the same observed glyph. Do not import distant text-box artefacts.
    for comp in comps:
        owner = comp["src_idx"]
        if skele_by_owner.get(owner):
            continue
        bb = comp["img_bbox"]
        inherited = [p for old, survivor in symbol_aliases.items() if survivor == owner
                     for p in skele_by_owner.get(old, [])
                     if bb[0]-5 <= p.xy[0] <= bb[2]+5 and bb[1]-5 <= p.xy[1] <= bb[3]+5]
        if inherited and len(_dedup_pins(inherited, pin_dedup_px)) == 1:
            comp["_symbol_inherited_pin"] = inherited[0]
    pin_ref_map: dict[tuple[int, int], str] = {}
    mapped_skele: set[tuple[int, int]] = set()
    for i, comp in enumerate(comps):
        des = comp.get("designator")
        if des:
            used_keys[des] = used_keys.get(des, 0) + 1
            key = des if used_keys[des] == 1 else f"{des}_{used_keys[des]}"
        else:
            # 无 key 文字：赛题规则 key 为 ∅<编号>（编号自由/顺序）
            empty_seq += 1
            key = f"∅{empty_seq}"
        comp["key"] = key
        if comp["type"] in NAME_DIFF_TYPES and comp["type"] not in SYMBOL_TYPES:
            name = comp.get("name") or key
        else:
            name = key

        # box 的引脚库只使用"型号文字 ↔ 框内 pinname 重合度"校验过的库；
        # Name/key 恰好也是库 key 但引脚名对不上时（例如 USB1）不套用。
        ic_name = (comp.get("lib_key") or None) if comp["type"] == "box" else None

        out["components"][key] = {
            "Name": name,
            "type": comp["type"],
            "value": comp.get("value"),
            "bbox": image_bbox_to_contest(comp["img_bbox"], height),
        }
        out["pins"][key] = {}
        seen_pk: set[str] = set()
        for number, pinname, xy in _pins_for_component(comp, vision.pins, texts, vision, snap_px, ic_name=ic_name, pin_dedup_px=pin_dedup_px):
            pk = f"pin_{number}"
            if pk in seen_pk:
                continue
            seen_pk.add(pk)
            out["pins"][key][pk] = {"pinname": pinname, "point": contest_point(xy[0], xy[1], height)}
            pin_nodes.append({"ref": f"{key}.{number}", "xy": xy})
            # 记录该 pin 对应的 skele_wire (元件下标, 引脚序号)，供 nets 精确映射
            src_idx = comp.get("src_idx")
            if src_idx is not None:
                hint = (comp.get("_box_pin_source_by_xy") or {}).get(tuple(xy))
                if hint is not None and getattr(hint, "idx", None) is not None and (src_idx, hint.idx) not in mapped_skele:
                    mapped_skele.add((src_idx, hint.idx))
                    pin_ref_map[(src_idx, hint.idx)] = f"{key}.{number}"
                    continue
                best, best_d = None, 3.0
                for q in skele_by_owner.get(src_idx, []):
                    if (src_idx, q.idx) in mapped_skele:
                        continue
                    d = _dist(xy, q.xy)
                    if d <= best_d:
                        best_d, best = d, q
                if best is not None:
                    mapped_skele.add((src_idx, best.idx))
                    pin_ref_map[(src_idx, best.idx)] = f"{key}.{number}"

    # Retained components keep their original owner indices. Transfer deleted
    # duplicate owners' references to an existing nearby pin of the survivor,
    # so component cleanup cannot silently disconnect a skeleton branch.
    for old_owner, new_owner in symbol_aliases.items():
        visited = {old_owner}
        while new_owner in symbol_aliases and new_owner not in visited:
            visited.add(new_owner)
            new_owner = symbol_aliases[new_owner]
        for pin in skele_by_owner.get(old_owner, []):
            retained = next((c for c in comps if c["src_idx"] == new_owner), None)
            inherited = retained.get("_symbol_inherited_pin") if retained else None
            if inherited is not None and _dist(pin.xy, inherited.xy) <= pin_dedup_px:
                ports = list(out["pins"].get(retained["key"], {}))
                if len(ports) == 1:
                    pin_ref_map[(old_owner, pin.idx)] = retained["key"]+"."+ports[0][4:]
                    continue
            choices = [(q, pin_ref_map.get((new_owner, q.idx)))
                       for q in skele_by_owner.get(new_owner, [])]
            choices = [(q, ref) for q, ref in choices if ref and _dist(pin.xy, q.xy) <= 5.]
            if choices:
                _, ref = min(choices, key=lambda pair: _dist(pin.xy, pair[0].xy))
                pin_ref_map[(old_owner, pin.idx)] = ref

    parent, find, union = _union_find([p["ref"] for p in pin_nodes])
    _connect_pins(pin_nodes, vision, eps, union)
    _merge_flywires(out, comps, union)

    # 优先用 skele_wire.extract_nets 的重建结果（fix net extraction）；
    # 其次退回 topo_graph 的 wire_nets，最后才是旧的 union-find 分段。
    if getattr(vision, "skele_nets", None) and _build_nets_from_skele_nets(
        out, vision, pin_nodes, height, pin_ref_map
    ):
        _propagate_symbol_values(out)
        return out
    if getattr(vision, "wire_nets", None):
        _build_nets_from_wire_nets(out, vision, pin_nodes, height)
        _propagate_symbol_values(out)
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
    _propagate_symbol_values(out)
    return out
