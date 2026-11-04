"""Contest-style scoring for 赛题六 PCB 原理图解析（2026-10-20《评测要求》版）。

与 2026-10-14 版的主要变化：
① 位置阈值：元件 bbox 四坐标最大误差 ≤ 20 px；IC 引脚点 ≤ 5 px；导线端点 ≤ 5 px；
② 元件比较：IC（box）只比 Name + type；其余元件比 type + Name + value；
   VCC/GND/net_input/net_output/net_short/net_bidirection 作为独立符号，按
   类别 + 位置一对一匹配，自动编号/自动名称不参与；
③ 有标识符的元件必须 key 精确匹配（R1 写成 R7 不算匹配），不再做命名元件宽松匹配；
④ 网络：先按全局网络身份（GND / 电源 value / net label value）合并原始分支，
   IC 顶点 = key.port，普通元件 = key；空集或零重合网不参与匹配，匈牙利最大化
   重合度之和，NetHypergraphF1 使用软 F1（部分重合给部分分）；
⑤ 导线：只在已匹配逻辑网对内比较，汇总原始分支线段后一对一匹配，
   max 端点误差 ≤ 5 px，方向可交换。
"""

from __future__ import annotations

import math
import re
from typing import Any

from .coords import bbox_center, point_dist

PIN_REF_RE = re.compile(r"^\((.*)\)$")
DESIGNATOR_RE = re.compile(r"^[A-Z]{1,3}\d+$|^[A-Za-z]{1,4}\d+_\d+$")

IC_TYPES = frozenset({"box", "ic"})
# VCC/GND + 网络标签符号：自动编号/自动名称不参与，按类别+位置匹配
SYMBOL_TYPES = frozenset({
    "gnd", "v", "vcc",
    "net_input", "net_output", "net_short", "net_bidirection",
})
# 兼容旧代码里的名字
POWER_TYPES = SYMBOL_TYPES


def _is_ic_type(ctype: str | None) -> bool:
    return str(ctype or "").strip().lower() in IC_TYPES


def _is_symbol_type(ctype: str | None) -> bool:
    return str(ctype or "").strip().lower() in SYMBOL_TYPES


def _is_power_type(ctype: str | None) -> bool:
    return _is_symbol_type(ctype)


def filter_confident(contest: dict[str, Any]) -> dict[str, Any]:
    """Keep only components whose key is a real designator (diagnostic helper)."""
    import copy

    comps = contest.get("components") or {}
    keep = {k for k in comps if DESIGNATOR_RE.match(k)}
    out: dict[str, Any] = {
        "components": {k: copy.deepcopy(v) for k, v in comps.items() if k in keep},
        "pins": {k: copy.deepcopy(v) for k, v in (contest.get("pins") or {}).items() if k in keep},
        "nets": {},
    }
    for nk, net in (contest.get("nets") or {}).items():
        refs = [r for r in _parse_hypergraph(net.get("hyperGraph", "")) if r.split(".", 1)[0] in keep]
        if not refs:
            continue
        out["nets"][nk] = {"hyperGraph": "(" + ",".join(sorted(refs)) + ")",
                           "edges": copy.deepcopy(net.get("edges") or {})}
    return out


def _f1(tp: float, fp: float, fn: float) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _prf_soft(tp: float, pred_n: int, gt_n: int) -> float:
    """软 F1：tp 为所有匹配网对的集合重合度之和。"""
    if gt_n == 0 and pred_n == 0:
        return 1.0
    if gt_n == 0 or pred_n == 0:
        return 0.0
    p = tp / pred_n
    r = tp / gt_n
    return 2 * p * r / (p + r) if p + r else 0.0


def _parse_hypergraph(text: str) -> frozenset[str]:
    text = (text or "").strip()
    m = PIN_REF_RE.match(text)
    inner = m.group(1) if m else text
    if not inner:
        return frozenset()
    return frozenset(x.strip() for x in inner.split(",") if x.strip())


def _pin_number_from_key(pin_key: str) -> str:
    return pin_key[4:] if pin_key.startswith("pin_") else pin_key


def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    if not cost:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(np.array(cost, dtype=float))
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        used_r, used_c, pairs = set(), set(), []
        cells = [(cost[i][j], i, j) for i in range(len(cost)) for j in range(len(cost[0]))]
        cells.sort()
        for _, i, j in cells:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
        return pairs


def _bbox_ok(pred: list[float], gt: list[float], thr: float) -> bool:
    """四对应坐标绝对误差均不超过阈值（不能用中心距/重叠替代）。"""
    if len(pred) != 4 or len(gt) != 4:
        return False
    return all(abs(float(pred[i]) - float(gt[i])) <= thr for i in range(4))


def _bbox_max_err(pred: list[float], gt: list[float]) -> float:
    return max(abs(float(pred[i]) - float(gt[i])) for i in range(4))


def _need_text(ctype: str) -> str | None:
    if _is_ic_type(ctype):
        return "Name"
    return "value"


def _component_ok(key: str, pred: dict, gt: dict, bbox_thr: float, relax_name: bool = False) -> bool:
    """身份/计分属性/位置同时满足。relax_name 仅为兼容旧调用，已不生效。"""
    ctype = str(gt.get("type") or "")
    if str(pred.get("type") or "") != ctype:
        return False
    pb, gb = pred.get("bbox") or [0, 0, 0, 0], gt.get("bbox") or [0, 0, 0, 0]
    if not _bbox_ok(pb, gb, bbox_thr):
        return False

    if _is_symbol_type(ctype):
        # VCC/GND/net_*：自动编号/自动名称不参与，value 仍按原文比较
        return str(pred.get("value") or "") == str(gt.get("value") or "")
    if str(key).startswith("∅"):
        # ∅ 普通元件：只比较 type + bbox + value 是否存在
        return (gt.get("value") not in (None, "")) == (pred.get("value") not in (None, ""))
    if _is_ic_type(ctype):
        # IC：只比较 Name + type，value 不参与
        return str(pred.get("Name") or "") == str(gt.get("Name") or "")
    # 其余元件：type + Name + value
    return (str(pred.get("Name") or "") == str(gt.get("Name") or "")
            and str(pred.get("value") or "") == str(gt.get("value") or ""))


def apply_empty_gt_text_policy(pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    """GT 文本为空时不要求 pred 也空（仅本地评测辅助，不改变提交 JSON）。"""
    import copy

    out = copy.deepcopy(pred)
    for key, gc in (gt.get("components") or {}).items():
        pc = (out.get("components") or {}).get(key)
        if not pc:
            continue
        field = _need_text(gc.get("type") or "")
        if field and not str(gc.get(field) or "").strip():
            pc[field] = None
    return out


def _pin_ok(pred: dict, gt: dict, pin_thr: float) -> bool:
    if str(pred.get("pinname") or "") != str(gt.get("pinname") or ""):
        return False
    return point_dist(pred.get("point") or {"x": 0, "y": 0},
                      gt.get("point") or {"x": 0, "y": 0}) <= pin_thr


def _seg_error(a, b) -> float:
    try:
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == 2 and len(b) == 2):
            return float("inf")
        d1 = max(point_dist(a[0], b[0]), point_dist(a[1], b[1]))
        d2 = max(point_dist(a[0], b[1]), point_dist(a[1], b[0]))
        return min(d1, d2)
    except Exception:
        return float("inf")


def _seg_match(a, b, thr: float) -> bool:
    return _seg_error(a, b) <= thr


def align_prediction(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0) -> dict[str, Any]:
    """Legacy diagnostic: remap pred keys by type + bbox. 官方 score_case 不使用。"""
    import copy

    p_comp = pred.get("components") or {}
    g_comp = gt.get("components") or {}
    p_keys, g_keys = list(p_comp), list(g_comp)
    if not p_keys or not g_keys:
        return copy.deepcopy(pred)
    cost = []
    for gk in g_keys:
        gc = g_comp[gk]
        row = []
        for pk in p_keys:
            pc = p_comp[pk]
            if pc.get("type") != gc.get("type") or not _bbox_ok(
                pc.get("bbox") or [0, 0, 0, 0], gc.get("bbox") or [0, 0, 0, 0], bbox_thr
            ):
                row.append(1e6)
                continue
            pc_, gc_ = bbox_center(pc.get("bbox") or [0, 0, 0, 0]), bbox_center(gc.get("bbox") or [0, 0, 0, 0])
            row.append(math.hypot(pc_[0] - gc_[0], pc_[1] - gc_[1]))
        cost.append(row)
    mapping: dict[str, str] = {}
    for i, j in _hungarian(cost):
        if i < len(g_keys) and j < len(p_keys) and cost[i][j] < 1e5:
            mapping[p_keys[j]] = g_keys[i]

    def remap_ref(ref: str) -> str:
        if "." not in ref:
            return mapping.get(ref, ref)
        ck, rest = ref.split(".", 1)
        return f"{mapping.get(ck, ck)}.{rest}"

    out = copy.deepcopy(pred)
    new_comps, new_pins = {}, {}
    for pk, comp in p_comp.items():
        nk = mapping.get(pk, pk)
        if nk in new_comps and pk not in mapping:
            continue
        new_comps[nk] = copy.deepcopy(comp)
        new_pins[nk] = copy.deepcopy((pred.get("pins") or {}).get(pk) or {})
    new_nets = {}
    for i, (_nk, net) in enumerate((pred.get("nets") or {}).items(), start=1):
        refs = [remap_ref(x) for x in _parse_hypergraph(net.get("hyperGraph", ""))]
        new_nets[f"NET_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")",
                                "edges": copy.deepcopy(net.get("edges") or {})}
    out["components"], out["pins"], out["nets"] = new_comps, new_pins, new_nets
    out["_key_map"] = mapping
    return out


def _apply_key_mapping(pred: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """把 pred 的元件 key 按 mapping 重映射，并同步 pins / nets.hyperGraph。"""
    import copy

    if not mapping:
        return pred

    def remap_ref(ref: str) -> str:
        if "." not in ref:
            return mapping.get(ref, ref)
        ck, rest = ref.split(".", 1)
        return f"{mapping.get(ck, ck)}.{rest}"

    out = copy.deepcopy(pred)
    new_comps, new_pins = {}, {}
    for pk, comp in (pred.get("components") or {}).items():
        nk = mapping.get(pk, pk)
        new_comps[nk] = copy.deepcopy(comp)
        new_pins[nk] = copy.deepcopy((pred.get("pins") or {}).get(pk) or {})
    new_nets = {}
    for i, (_nk, net) in enumerate((pred.get("nets") or {}).items(), start=1):
        refs = [remap_ref(x) for x in _parse_hypergraph(net.get("hyperGraph", ""))]
        new_nets[f"NET_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")",
                                "edges": copy.deepcopy(net.get("edges") or {})}
    out["components"], out["pins"], out["nets"] = new_comps, new_pins, new_nets
    return out


def _align_bbox_typematch(p_sub: dict, g_sub: dict, bbox_thr: float,
                          extra_gate=None) -> dict[str, str]:
    p_keys, g_keys = list(p_sub), list(g_sub)
    if not p_keys or not g_keys:
        return {}
    cost = []
    for gk in g_keys:
        gc = g_sub[gk]
        row = []
        for pk in p_keys:
            pc = p_sub[pk]
            if pc.get("type") != gc.get("type") or not _bbox_ok(
                pc.get("bbox") or [0, 0, 0, 0], gc.get("bbox") or [0, 0, 0, 0], bbox_thr
            ):
                row.append(1e6)
                continue
            if extra_gate is not None and not extra_gate(pc, gc):
                row.append(1e6)
                continue
            pc_, gc_ = bbox_center(pc.get("bbox") or [0, 0, 0, 0]), bbox_center(gc.get("bbox") or [0, 0, 0, 0])
            row.append(math.hypot(pc_[0] - gc_[0], pc_[1] - gc_[1]))
        cost.append(row)
    mapping: dict[str, str] = {}
    for i, j in _hungarian(cost):
        if i < len(g_keys) and j < len(p_keys) and cost[i][j] < 1e5:
            mapping[p_keys[j]] = g_keys[i]
    return mapping


def _align_symbols(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0) -> dict[str, Any]:
    """VCC/GND/net_*：类别 + 位置一对一；自动 key/Name 不做比较。"""
    p_comp = pred.get("components") or {}
    g_comp = gt.get("components") or {}
    p_sym = {k: v for k, v in p_comp.items() if _is_symbol_type(v.get("type"))}
    g_sym = {k: v for k, v in g_comp.items() if _is_symbol_type(v.get("type"))}
    if not p_sym or not g_sym:
        return pred
    mapping = _align_bbox_typematch(p_sym, g_sym, bbox_thr)
    out = _apply_key_mapping(pred, mapping)
    out["_symbol_key_map"] = mapping
    return out


# 兼容旧名字
_align_power_symbols = _align_symbols


def _align_empty_keys(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0) -> dict[str, Any]:
    """`∅<编号>`（非符号类）：key 不比较，只按 type+bbox 匈牙利匹配。"""
    p_comp = pred.get("components") or {}
    g_comp = gt.get("components") or {}
    p_empty = {k: v for k, v in p_comp.items()
               if str(k).startswith("∅") and not _is_symbol_type(v.get("type"))}
    g_empty = {k: v for k, v in g_comp.items()
               if str(k).startswith("∅") and not _is_symbol_type(v.get("type"))}
    if not p_empty or not g_empty:
        return pred
    mapping = _align_bbox_typematch(p_empty, g_empty, bbox_thr)
    out = _apply_key_mapping(pred, mapping)
    out["_empty_key_map"] = mapping
    return out


def align_for_scoring(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0):
    """打分前对齐：符号类 → 其余 ∅ 元件；命名元件 key 必须精确（不做宽松匹配）。

    返回 (aligned_pred, relaxed_named_keys)；第二个值恒为空集，保留是为了兼容旧调用。
    """
    pred = _align_symbols(pred, gt, bbox_thr)
    pred = _align_empty_keys(pred, gt, bbox_thr)
    return pred, set()


def _global_identity(comp: dict, key: str) -> str | None:
    """全局网络身份：GND / 电源 value / 网络标签 value。"""
    ctype = str(comp.get("type") or "")
    if ctype == "gnd":
        return "GLOBAL:GND"
    if ctype in {"v", "vcc"}:
        return "GLOBAL:SUPPLY:" + str(comp.get("value") or "VCC")
    if ctype in {"net_input", "net_output", "net_short", "net_bidirection"}:
        val = comp.get("value")
        if val not in (None, ""):
            return "GLOBAL:LABEL:" + str(val)
        return "SYMBOL:" + str(key)  # 没有 value 时只保留自身，不做合并
    return None


def _net_nodes(hypergraph_text: str, comps: dict[str, Any]) -> frozenset[str]:
    """网络顶点：IC=key.port；普通元件=key；符号=全局网络身份。"""
    nodes: set[str] = set()
    for ref in _parse_hypergraph(hypergraph_text):
        ck, _, pin = ref.partition(".")
        comp = comps.get(ck) or {}
        if _is_ic_type(comp.get("type")) and pin:
            nodes.add(f"{ck}.{pin}")
        else:
            gid = _global_identity(comp, ck)
            nodes.add(gid or ck)
    return frozenset(nodes)


def _logical_nets(comps: dict[str, Any], nets: dict[str, Any]) -> list[dict]:
    """先合并共享全局网络身份的原始分支，返回逻辑网列表。"""
    ids = list(nets)
    if not ids:
        return []
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    raw = {nid: _net_nodes((nets.get(nid) or {}).get("hyperGraph", ""), comps) for nid in ids}
    by_gid: dict[str, list[str]] = {}
    for nid in ids:
        for v in raw[nid]:
            if v.startswith("GLOBAL:"):
                by_gid.setdefault(v, []).append(nid)
    for lst in by_gid.values():
        for x in lst[1:]:
            union(lst[0], x)

    groups: dict[str, list[str]] = {}
    for nid in ids:
        groups.setdefault(find(nid), []).append(nid)

    out = []
    for gids in groups.values():
        verts: set[str] = set()
        edges = []
        for nid in gids:
            verts |= set(raw[nid])
            for eid, seg in ((nets.get(nid) or {}).get("edges") or {}).items():
                edges.append({"net": nid, "edge": eid, "seg": seg})
        out.append({"ids": sorted(gids), "vertices": frozenset(verts), "edges": edges})
    out.sort(key=lambda g: g["ids"][0] if g["ids"] else "")
    return out


def _match_logical_nets(gt_groups: list[dict], pred_groups: list[dict]) -> list[tuple[int, int, float]]:
    if not gt_groups or not pred_groups:
        return []
    cost = []
    for g in gt_groups:
        row = []
        for p in pred_groups:
            inter = len(g["vertices"] & p["vertices"])
            denom = len(g["vertices"]) + len(p["vertices"])
            if inter == 0 or denom == 0:
                row.append(1e6)
            else:
                row.append(1.0 - 2.0 * inter / denom)
        cost.append(row)
    pairs = []
    for i, j in _hungarian(cost):
        if i < len(gt_groups) and j < len(pred_groups) and cost[i][j] < 1e5:
            inter = len(gt_groups[i]["vertices"] & pred_groups[j]["vertices"])
            denom = len(gt_groups[i]["vertices"]) + len(pred_groups[j]["vertices"])
            pairs.append((i, j, 2.0 * inter / denom if denom else 0.0))
    return pairs


def _match_edges(gt_edges: list[dict], pred_edges: list[dict], thr: float):
    cands = []
    for i, ge in enumerate(gt_edges):
        for j, pe in enumerate(pred_edges):
            err = _seg_error(ge.get("seg"), pe.get("seg"))
            if err <= thr:
                cands.append((err, i, j))
    cands.sort(key=lambda x: (x[0], x[1], x[2]))
    ug, up = set(), set()
    matched = []
    for err, i, j in cands:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        matched.append({"gt": gt_edges[i], "pred": pred_edges[j], "err": round(err, 4)})
    gt_un = [e for i, e in enumerate(gt_edges) if i not in ug]
    pred_un = [e for i, e in enumerate(pred_edges) if i not in up]
    return matched, gt_un, pred_un


def score_case(
    pred: dict[str, Any],
    gt: dict[str, Any],
    bbox_thr: float = 20.0,
    pin_thr: float = 5.0,
    line_thr: float = 5.0,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = weights or {"component": 0.30, "pin": 0.25, "net_hypergraph": 0.35, "net_line": 0.10}
    pred, _ = align_for_scoring(pred, gt, bbox_thr)
    p_comp, g_comp = pred.get("components") or {}, gt.get("components") or {}
    p_pins, g_pins = pred.get("pins") or {}, gt.get("pins") or {}
    p_nets, g_nets = pred.get("nets") or {}, gt.get("nets") or {}

    # ---------------- Component ----------------
    tp = fp = fn = 0
    key_fail = 0
    for key, gc in g_comp.items():
        pc = p_comp.get(key)
        if pc is None:
            fn += 1
            key_fail += 1
            continue
        if _component_ok(key, pc, gc, bbox_thr):
            tp += 1
        else:
            fn += 1
            fp += 1
    for key in p_comp:
        if key not in g_comp:
            fp += 1
    component_f1 = _f1(tp, fp, fn)

    # ---------------- Pin（仅 IC） ----------------
    ptp = pfp = pfn = 0
    gt_scored = pred_scored = 0
    for ck, gpins in g_pins.items():
        gc = g_comp.get(ck) or {}
        if not _is_ic_type(gc.get("type")):
            continue
        gt_scored += len(gpins or {})
        owner_ok = ck in p_comp
        ppins = (p_pins.get(ck) or {}) if owner_ok else {}
        pred_scored += len(ppins)
        for pk, gpin in (gpins or {}).items():
            pp = ppins.get(pk)
            if pp is None:
                pfn += 1
                continue
            if _pin_ok(pp, gpin, pin_thr):
                ptp += 1
            else:
                pfn += 1
                pfp += 1
        if owner_ok:
            for pk in ppins:
                if pk not in (gpins or {}):
                    pfp += 1
        else:
            # GT 元件在 pred components 中不存在：其下若还有无主 pins，全部计 FP
            stray = p_pins.get(ck) or {}
            pred_scored += len(stray)
            pfp += len(stray)
    # 预测多出的 IC 元件（未与 GT 对齐）的引脚也计 FP
    for pk, pc in p_comp.items():
        if not _is_ic_type(pc.get("type")) or pk in g_comp:
            continue
        pins = p_pins.get(pk) or {}
        pred_scored += len(pins)
        pfp += len(pins)
    pin_f1 = 0.0 if (gt_scored == 0 and pred_scored == 0) else _f1(ptp, pfp, pfn)

    # ---------------- Net（并全局网 + 软 F1） ----------------
    gt_groups = _logical_nets(g_comp, g_nets)
    pred_groups = _logical_nets(p_comp, p_nets)
    net_pairs = _match_logical_nets(gt_groups, pred_groups)
    soft_tp = sum(sim for _, _, sim in net_pairs)
    net_f1 = _prf_soft(soft_tp, len(pred_groups), len(gt_groups))

    # ---------------- Wire（只在匹配网对里比较） ----------------
    ltp = lfp = lfn = 0
    matched_g = {i for i, _, _ in net_pairs}
    matched_p = {j for _, j, _ in net_pairs}
    for i, j, _ in net_pairs:
        m, gun, pun = _match_edges(gt_groups[i]["edges"], pred_groups[j]["edges"], line_thr)
        ltp += len(m)
        lfn += len(gun)
        lfp += len(pun)
    for i, g in enumerate(gt_groups):
        if i not in matched_g:
            lfn += len(g["edges"])
    for j, p in enumerate(pred_groups):
        if j not in matched_p:
            lfp += len(p["edges"])
    line_f1 = _f1(ltp, lfp, lfn)

    score = 100.0 * (
        weights["component"] * component_f1
        + weights["pin"] * pin_f1
        + weights["net_hypergraph"] * net_f1
        + weights["net_line"] * line_f1
    )
    return {
        "ComponentF1": component_f1,
        "PinF1": pin_f1,
        "NetHypergraphF1": net_f1,
        "NetLineF1": line_f1,
        "Score": score,
        "diag": {
            "component_tp_fp_fn": [tp, fp, fn],
            "pin_tp_fp_fn": [ptp, pfp, pfn],
            "net_tp_fp_fn": [round(soft_tp, 6), round(len(pred_groups) - soft_tp, 6),
                            round(len(gt_groups) - soft_tp, 6)],
            "line_tp_fp_fn": [ltp, lfp, lfn],
            "key_fail": key_fail,
            "n_pred_comp": len(p_comp),
            "n_gt_comp": len(g_comp),
            "n_gt_nets": len(gt_groups),
            "n_pred_nets": len(pred_groups),
        },
    }


def score_case_aligned(pred: dict[str, Any], gt: dict[str, Any], **kwargs) -> dict[str, Any]:
    aligned = align_prediction(pred, gt, bbox_thr=float(kwargs.get("bbox_thr", 20.0)))
    out = score_case(aligned, gt, **kwargs)
    out["AlignedScore"] = out["Score"]
    out["n_key_mapped"] = len(aligned.get("_key_map") or {})
    return out


def score_batch(items: list[tuple[str, dict, dict]], **kwargs) -> dict[str, Any]:
    rows = []
    for case_id, pred, gt in items:
        one = score_case(pred, gt, **kwargs)
        one["case_id"] = case_id
        rows.append(one)
    if not rows:
        return {"cases": [], "mean": {}}
    keys = ["ComponentF1", "PinF1", "NetHypergraphF1", "NetLineF1", "Score"]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    return {"cases": rows, "mean": mean, "n": len(rows)}
