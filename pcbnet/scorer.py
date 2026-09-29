"""Contest-style scoring: Component / Pin / NetHypergraph / NetLine F1."""

from __future__ import annotations

import math
import re
from typing import Any

from .coords import bbox_center, point_dist

PIN_REF_RE = re.compile(r"^\((.*)\)$")
DESIGNATOR_RE = re.compile(r"^[A-Z]{1,3}\d+$|^[A-Za-z]{1,4}\d+_\d+$")


def filter_confident(contest: dict[str, Any]) -> dict[str, Any]:
    """Keep only components whose key is a real designator (e.g. U3, R1, RUN_1).

    Components keyed by a cvjson fallback id (e.g. C_15) are not recoverable from
    the image, so scoring against them is not meaningful. This yields an honest
    upper-bound metric on the designator-recoverable universe. Pins/nets are
    restricted to the surviving components.
    """
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
        out["nets"][nk] = {"hyperGraph": "(" + ",".join(sorted(refs)) + ")", "edges": copy.deepcopy(net.get("edges") or {})}
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


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
    pc, gc = bbox_center(pred), bbox_center(gt)
    if math.hypot(pc[0] - gc[0], pc[1] - gc[1]) <= thr:
        return True
    return all(abs(pred[i] - gt[i]) <= thr for i in range(4))


def _need_text(ctype: str) -> str | None:
    if ctype == "box":
        return "Name"
    if ctype in {"r", "c"}:
        return "value"
    return None


def _component_ok(key: str, pred: dict, gt: dict, bbox_thr: float, relax_name: bool = False) -> bool:
    if pred.get("type") != gt.get("type"):
        return False
    if not _bbox_ok(pred.get("bbox") or [0, 0, 0, 0], gt.get("bbox") or [0, 0, 0, 0], bbox_thr):
        return False
    if str(key).startswith("∅"):
        # ∅ 元件：只比较 type + bbox + value（是否存在），Name/key 不比较
        gt_has = gt.get("value") not in (None, "")
        pred_has = pred.get("value") not in (None, "")
        return gt_has == pred_has
    if relax_name:
        # 命名元件 bbox 宽松匹配（key 前缀合法但编号不同）：不比较 Name，只比 value
        return str(pred.get("value") or "") == str(gt.get("value") or "")
    # 非 ∅ 元件：Name 与 value 均严格匹配
    if str(pred.get("Name") or "") != str(gt.get("Name") or ""):
        return False
    return str(pred.get("value") or "") == str(gt.get("value") or "")


def apply_empty_gt_text_policy(pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    """GT-empty text policy: if the annotation has no name/value, do not compare
    against a filled prediction. Used only at local eval (Dataset300 often leaves
    those fields blank). Submission JSON is unchanged."""
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
    return point_dist(pred.get("point") or {"x": 0, "y": 0}, gt.get("point") or {"x": 0, "y": 0}) <= pin_thr


def _seg_match(a: list[dict], b: list[dict], thr: float) -> bool:
    d1 = point_dist(a[0], b[0]) + point_dist(a[1], b[1])
    d2 = point_dist(a[0], b[1]) + point_dist(a[1], b[0])
    return min(d1, d2) <= 2 * thr


def align_prediction(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0) -> dict[str, Any]:
    """Remap pred component keys onto GT keys by type + bbox. For Dataset300 local eval only."""
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
            if pc.get("type") != gc.get("type") or not _bbox_ok(pc.get("bbox") or [0, 0, 0, 0], gc.get("bbox") or [0, 0, 0, 0], bbox_thr):
                row.append(1e6)
                continue
            pc_, gc_ = bbox_center(pc.get("bbox") or [0, 0, 0, 0]), bbox_center(gc.get("bbox") or [0, 0, 0, 0])
            row.append(math.hypot(pc_[0] - gc_[0], pc_[1] - gc_[1]))
        cost.append(row)
    mapping: dict[str, str] = {}
    used_gt: set[str] = set()
    for i, j in _hungarian(cost):
        if i < len(g_keys) and j < len(p_keys) and cost[i][j] < 1e5:
            mapping[p_keys[j]] = g_keys[i]
            used_gt.add(g_keys[i])

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
        pins = copy.deepcopy((pred.get("pins") or {}).get(pk) or {})
        new_pins[nk] = pins
    new_nets = {}
    for i, (_nk, net) in enumerate((pred.get("nets") or {}).items(), start=1):
        refs = [remap_ref(x) for x in _parse_hypergraph(net.get("hyperGraph", ""))]
        name = f"NET_{i}"
        new_nets[name] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": copy.deepcopy(net.get("edges") or {})}
    out["components"], out["pins"], out["nets"] = new_comps, new_pins, new_nets
    out["_key_map"] = mapping
    return out


def _align_empty_keys(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0) -> dict[str, Any]:
    """∅<编号> 元件的 key 不做对比，只按 type+bbox 匈牙利匹配（评测标准 §2.3）。

    把 pred 的 ∅ key 重映射到 GT 的 ∅ key，并同步 remap pins / nets.hyperGraph 里的
    component key 引用；命名元件（R5/U1/…）key 严格匹配，不受此函数影响。
    """
    import copy

    p_comp = pred.get("components") or {}
    g_comp = gt.get("components") or {}
    p_empty = {k: v for k, v in p_comp.items() if str(k).startswith("∅")}
    g_empty = {k: v for k, v in g_comp.items() if str(k).startswith("∅")}
    if not p_empty or not g_empty:
        return pred

    p_keys = list(p_empty)
    g_keys = list(g_empty)
    cost: list[list[float]] = []
    for gk in g_keys:
        gc = g_empty[gk]
        row: list[float] = []
        for pk in p_keys:
            pc = p_empty[pk]
            if pc.get("type") != gc.get("type") or not _bbox_ok(pc.get("bbox") or [0, 0, 0, 0], gc.get("bbox") or [0, 0, 0, 0], bbox_thr):
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
        new_comps[nk] = copy.deepcopy(comp)
        new_pins[nk] = copy.deepcopy((pred.get("pins") or {}).get(pk) or {})
    new_nets = {}
    for i, (_nk, net) in enumerate((pred.get("nets") or {}).items(), start=1):
        refs = [remap_ref(x) for x in _parse_hypergraph(net.get("hyperGraph", ""))]
        new_nets[f"NET_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": copy.deepcopy(net.get("edges") or {})}
    out["components"], out["pins"], out["nets"] = new_comps, new_pins, new_nets
    out["_empty_key_map"] = mapping
    return out


_PREFIX_CACHE: dict[str, set[str]] | None = None


def _canonical_prefixes(ctype: str) -> set[str]:
    """type 的合法 key 前缀集合（来自 type_list.md 的 canonical_prefix）。"""
    global _PREFIX_CACHE
    if _PREFIX_CACHE is None:
        try:
            from .knowledge_base import load_key_type_kb

            kb = load_key_type_kb()
            _PREFIX_CACHE = {t: set(p) for t, p in (kb.get("canonical_prefix") or {}).items() if p}
        except Exception:
            _PREFIX_CACHE = {}
    return _PREFIX_CACHE.get(ctype, set())


def _key_prefix(key: str) -> str:
    m = re.match(r"^([A-Za-z]{1,3})", key or "")
    return m.group(1).upper() if m else ""


def _align_named_keys(pred: dict[str, Any], gt: dict[str, Any], bbox_thr: float = 20.0) -> tuple[dict[str, Any], set[str]]:
    """命名元件 key 匹配不到时，按 bbox 宽松匹配（评测标准扩展）。

    GT 命名元件（非 ∅）在 pred 中无同名 key 时，在 pred 的命名元件里找 type 相同、
    bbox 匹配、value 一致、且 key 前缀满足 type_list 规范者，建立 key 映射并 remap。
    返回 (remap 后的 pred, 走宽松匹配的 GT key 集合)。
    """
    import copy

    p_comp = pred.get("components") or {}
    g_comp = gt.get("components") or {}

    g_named_missing = {k: v for k, v in g_comp.items() if not str(k).startswith("∅") and k not in p_comp}
    p_named_extra = {k: v for k, v in p_comp.items() if not str(k).startswith("∅") and k not in g_comp}
    if not g_named_missing or not p_named_extra:
        return pred, set()

    g_keys = list(g_named_missing)
    p_keys = list(p_named_extra)
    cost: list[list[float]] = []
    for gk in g_keys:
        gc = g_named_missing[gk]
        row: list[float] = []
        for pk in p_keys:
            pc = p_named_extra[pk]
            if pc.get("type") != gc.get("type"):
                row.append(1e6)
                continue
            if not _bbox_ok(pc.get("bbox") or [0, 0, 0, 0], gc.get("bbox") or [0, 0, 0, 0], bbox_thr):
                row.append(1e6)
                continue
            if str(pc.get("value") or "") != str(gc.get("value") or ""):
                row.append(1e6)
                continue
            # pred key 前缀必须满足该 type 的 type_list 规范
            if _key_prefix(pk) not in _canonical_prefixes(gc.get("type")):
                row.append(1e6)
                continue
            pc_, gc_ = bbox_center(pc.get("bbox") or [0, 0, 0, 0]), bbox_center(gc.get("bbox") or [0, 0, 0, 0])
            row.append(math.hypot(pc_[0] - gc_[0], pc_[1] - gc_[1]))
        cost.append(row)

    mapping: dict[str, str] = {}
    for i, j in _hungarian(cost):
        if i < len(g_keys) and j < len(p_keys) and cost[i][j] < 1e5:
            mapping[p_keys[j]] = g_keys[i]

    if not mapping:
        return pred, set()

    def remap_ref(ref: str) -> str:
        if "." not in ref:
            return mapping.get(ref, ref)
        ck, rest = ref.split(".", 1)
        return f"{mapping.get(ck, ck)}.{rest}"

    out = copy.deepcopy(pred)
    new_comps, new_pins = {}, {}
    for pk, comp in p_comp.items():
        nk = mapping.get(pk, pk)
        new_comps[nk] = copy.deepcopy(comp)
        new_pins[nk] = copy.deepcopy((pred.get("pins") or {}).get(pk) or {})
    new_nets = {}
    for i, (_nk, net) in enumerate((pred.get("nets") or {}).items(), start=1):
        refs = [remap_ref(x) for x in _parse_hypergraph(net.get("hyperGraph", ""))]
        new_nets[f"NET_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": copy.deepcopy(net.get("edges") or {})}
    out["components"], out["pins"], out["nets"] = new_comps, new_pins, new_nets
    relaxed = set(mapping.values())
    return out, relaxed


def score_case(
    pred: dict[str, Any],
    gt: dict[str, Any],
    bbox_thr: float = 20.0,
    pin_thr: float = 5.0,
    line_thr: float = 5.0,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = weights or {"component": 0.30, "pin": 0.25, "net_hypergraph": 0.35, "net_line": 0.10}
    # 评测标准 §2.3：∅ 元件 key 不对比，按 type+bbox 匈牙利对齐后再算
    pred = _align_empty_keys(pred, gt, bbox_thr)
    # 命名元件 key 匹配不到时，按 bbox+type+value+key 前缀 宽松匹配
    pred, relaxed_named = _align_named_keys(pred, gt, bbox_thr)
    p_comp, g_comp = pred.get("components") or {}, gt.get("components") or {}
    p_pins, g_pins = pred.get("pins") or {}, gt.get("pins") or {}
    p_nets, g_nets = pred.get("nets") or {}, gt.get("nets") or {}

    tp = fp = fn = 0
    key_fail = 0
    for key, gc in g_comp.items():
        if key not in p_comp:
            fn += 1
            key_fail += 1
            continue
        if _component_ok(key, p_comp[key], gc, bbox_thr, relax_name=key in relaxed_named):
            tp += 1
        else:
            fn += 1
            fp += 1
    for key in p_comp:
        if key not in g_comp:
            fp += 1
    component_f1 = _f1(tp, fp, fn)

    ptp = pfp = pfn = 0
    for ck, gpins in g_pins.items():
        ppins = p_pins.get(ck) or {}
        is_box = str((g_comp.get(ck) or {}).get("type") or "") == "box"
        if is_box:
            # box 类：pin key（pin_<pin_id>）+ pinname + point 严格匹配
            for pk, gpin in gpins.items():
                pred_pin = ppins.get(pk)
                if pred_pin is None:
                    pfn += 1
                    continue
                if _pin_ok(pred_pin, gpin, pin_thr):
                    ptp += 1
                else:
                    pfn += 1
                    pfp += 1
            for pk in ppins:
                if pk not in gpins:
                    pfp += 1
        else:
            # 非 box 类：pin key/name 顺序随机，只看端点坐标，匈牙利匹配
            g_keys_list = list(gpins.keys())
            p_keys_list = list(ppins.keys())
            gpts = [gpins[k].get("point") or {"x": 0, "y": 0} for k in g_keys_list]
            ppts = [ppins[k].get("point") or {"x": 0, "y": 0} for k in p_keys_list]
            matched = 0
            if gpts and ppts:
                cost = [[point_dist(g, p) for p in ppts] for g in gpts]
                for gi, pi in _hungarian(cost):
                    if gi < len(gpts) and pi < len(ppts) and cost[gi][pi] <= pin_thr:
                        matched += 1
            ptp += matched
            pfn += len(gpts) - matched
            pfp += max(0, len(ppts) - matched)
    pin_f1 = _f1(ptp, pfp, pfn)

    g_sets = {nk: _parse_hypergraph(nv.get("hyperGraph", "")) for nk, nv in g_nets.items()}
    p_sets = {nk: _parse_hypergraph(nv.get("hyperGraph", "")) for nk, nv in p_nets.items()}
    g_keys, p_keys = list(g_sets), list(p_sets)
    matched_pairs: list[tuple[str, str]] = []
    if g_keys and p_keys:
        cost = []
        for gk in g_keys:
            row = []
            for pk in p_keys:
                inter = len(g_sets[gk] & p_sets[pk])
                union = len(g_sets[gk] | p_sets[pk]) or 1
                row.append(1.0 - inter / union)
            cost.append(row)
        for i, j in _hungarian(cost):
            if i < len(g_keys) and j < len(p_keys):
                matched_pairs.append((g_keys[i], p_keys[j]))
    used_p = {p for _, p in matched_pairs}
    ntp = nfp = nfn = 0
    for gk, pk in matched_pairs:
        gs, ps = g_sets[gk], p_sets[pk]
        ntp += len(gs & ps)
        nfp += len(ps - gs)
        nfn += len(gs - ps)
    for gk in g_keys:
        if gk not in {a for a, _ in matched_pairs}:
            nfn += len(g_sets[gk])
    for pk in p_keys:
        if pk not in used_p:
            nfp += len(p_sets[pk])
    net_f1 = _f1(ntp, nfp, nfn)

    ltp = lfp = lfn = 0
    for gk, pk in matched_pairs:
        g_edges = list((g_nets[gk].get("edges") or {}).values())
        p_edges = list((p_nets[pk].get("edges") or {}).values())
        used = set()
        for ge in g_edges:
            hit = None
            for j, pe in enumerate(p_edges):
                if j in used:
                    continue
                if isinstance(ge, list) and isinstance(pe, list) and len(ge) == 2 and len(pe) == 2 and _seg_match(ge, pe, line_thr):
                    hit = j
                    break
            if hit is None:
                lfn += 1
            else:
                ltp += 1
                used.add(hit)
        lfp += max(0, len(p_edges) - len(used))
    for gk in g_keys:
        if gk not in {a for a, _ in matched_pairs}:
            lfn += len(g_nets[gk].get("edges") or {})
    for pk in p_keys:
        if pk not in used_p:
            lfp += len(p_nets[pk].get("edges") or {})
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
            "net_tp_fp_fn": [ntp, nfp, nfn],
            "line_tp_fp_fn": [ltp, lfp, lfn],
            "key_fail": key_fail,
            "n_pred_comp": len(p_comp),
            "n_gt_comp": len(g_comp),
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
