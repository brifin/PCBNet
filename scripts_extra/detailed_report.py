#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""细粒度错误归因报告（2026-10-14 版评测标准）。

对每个测例，把 Component / Pin / Net 三类判错原因拆到最小单位：
  - Component: key(缺失/错位) / type / bbox / Name / value（GND、电源符号按 类别+值+几何 匹配）
  - Pin: 仅 box IC 引脚计分（pin key / pinname / point）；普通元件引脚不计位置分
  - Net: hyperGraph 按两级节点比对（IC 记 `U1.7`，普通元件折叠成 `C6`）；edges 线段（20 px）

空间阈值：元件 bbox / IC 引脚 / 导线端点 统一 20 px。
输出 outputs/detailed_report.md 与 outputs/detailed_report.json。
只依赖已生成的 outputs/pred/<id>/<id>.json 与 outputs/gt/<id>.json。
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.coords import point_dist
from pcbnet.scorer import (  # type: ignore
    _bbox_ok,
    _hungarian,
    _is_power_type,
    _net_nodes,
    align_for_scoring,
)

OUT_MD = ROOT / "outputs" / "detailed_report.md"
OUT_JSON = ROOT / "outputs" / "detailed_report.json"

# 与 configs/default.yaml thresholds 一致；GT 自检/评测口径统一
BBOX_THR = 20.0
PIN_THR = 20.0
LINE_THR = 20.0


# ---------------------------------------------------------------------------
# Component 归因
# ---------------------------------------------------------------------------
def _component_reason(key, pred, gt, bbox_thr, relax_name=False):
    """返回 None（命中）或错误原因字符串。"""
    if pred.get("type") != gt.get("type"):
        return "type"
    if not _bbox_ok(pred.get("bbox") or [0, 0, 0, 0], gt.get("bbox") or [0, 0, 0, 0], bbox_thr):
        return "bbox"
    if _is_power_type(gt.get("type")):
        # ④ GND/电源符号：按 类别 + value + 几何位置 匹配；value 不等则配不上、判错
        return None if str(pred.get("value") or "") == str(gt.get("value") or "") else "value"
    if str(key).startswith("∅"):
        # 其余 ∅ 元件：只比 type + bbox + value（是否存在）
        gt_has = gt.get("value") not in (None, "")
        pred_has = pred.get("value") not in (None, "")
        return None if gt_has == pred_has else "value"
    if relax_name:
        # 命名元件 bbox 宽松匹配：不比较 Name
        return None if str(pred.get("value") or "") == str(gt.get("value") or "") else "value"
    if str(pred.get("Name") or "") != str(gt.get("Name") or ""):
        return "Name"
    if str(pred.get("value") or "") != str(gt.get("value") or ""):
        return "value"
    return None


# ---------------------------------------------------------------------------
# Pin 归因（② 只统计 box IC 引脚）
# ---------------------------------------------------------------------------
def _pin_reason(pred_pin, gt_pin, pin_thr):
    """返回 None（命中）或错误原因字符串。仅 box IC：比 pinname + point。"""
    if str(pred_pin.get("pinname") or "") != str(gt_pin.get("pinname") or ""):
        return "pinname"
    if point_dist(pred_pin.get("point") or {"x": 0, "y": 0}, gt_pin.get("point") or {"x": 0, "y": 0}) > pin_thr:
        return "point"
    return None


def analyze_case(pred, gt, bbox_thr=BBOX_THR, pin_thr=PIN_THR, line_thr=LINE_THR):
    pred, relaxed_named = align_for_scoring(pred, gt, bbox_thr)
    p_comp, g_comp = pred.get("components") or {}, gt.get("components") or {}
    p_pins, g_pins = pred.get("pins") or {}, gt.get("pins") or {}
    p_nets, g_nets = pred.get("nets") or {}, gt.get("nets") or {}

    # ---- Component ----
    comp_errors = []  # (key, gt_type, 原因)
    for key, gc in g_comp.items():
        pc = p_comp.get(key)
        if pc is None:
            comp_errors.append((key, gc.get("type"), "key缺失"))
            continue
        reason = _component_reason(key, pc, gc, bbox_thr, relax_name=key in relaxed_named)
        if reason:
            comp_errors.append((key, gc.get("type"), reason))
    comp_fp = [k for k in p_comp if k not in g_comp]

    # ---- Pin（只计 box IC）----
    pin_errors = []  # (comp_key, pin_key, gt_type, 原因)
    n_gt_pins_scored = 0
    n_gt_pins_unscored = 0
    for ck, gpins in g_pins.items():
        ctype = (g_comp.get(ck) or {}).get("type") or ""
        if ctype != "box":
            # ② 普通元件的 pin 不再计位置分
            n_gt_pins_unscored += len(gpins)
            continue
        n_gt_pins_scored += len(gpins)
        ppins = p_pins.get(ck) or {}
        for pk, gpin in gpins.items():
            pp = ppins.get(pk)
            if pp is None:
                pin_errors.append((ck, pk, ctype, "pin_key缺失"))
                continue
            reason = _pin_reason(pp, gpin, pin_thr)
            if reason:
                pin_errors.append((ck, pk, ctype, reason))

    # ---- Net（③ 两级节点）----
    g_sets = {nk: _net_nodes(nv.get("hyperGraph", ""), g_comp) for nk, nv in g_nets.items()}
    p_sets = {nk: _net_nodes(nv.get("hyperGraph", ""), p_comp) for nk, nv in p_nets.items()}
    g_keys, p_keys = list(g_sets), list(p_sets)
    matched_pairs = []
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

    net_errors = []  # (gt_net_key, 原因, 详情)
    used_p = set()
    for gk, pk in matched_pairs:
        used_p.add(pk)
        gs, ps = g_sets[gk], p_sets[pk]
        missing_pins = gs - ps
        extra_pins = ps - gs
        # edges 比较（线段端点 20 px，顺配/反配取小）
        g_edges = list((g_nets[gk].get("edges") or {}).values())
        p_edges = list((p_nets[pk].get("edges") or {}).values())
        used = set()
        edge_miss = 0
        for ge in g_edges:
            hit = False
            for j, pe in enumerate(p_edges):
                if j in used:
                    continue
                if isinstance(ge, list) and isinstance(pe, list) and len(ge) == 2 and len(pe) == 2:
                    d1 = point_dist(ge[0], pe[0]) + point_dist(ge[1], pe[1])
                    d2 = point_dist(ge[0], pe[1]) + point_dist(ge[1], pe[0])
                    if min(d1, d2) <= 2 * line_thr:
                        hit = True
                        used.add(j)
                        break
            if not hit:
                edge_miss += 1
        if missing_pins or extra_pins:
            detail = f"缺{len(missing_pins)}节点/多{len(extra_pins)}节点"
            if missing_pins:
                detail += "；缺: " + ", ".join(sorted(missing_pins)[:6])
            net_errors.append((gk, "hyperGraph", detail))
        if edge_miss:
            net_errors.append((gk, "edges", f"缺{edge_miss}段线"))

    unmatched_g = [gk for gk in g_keys if gk not in {a for a, _ in matched_pairs}]
    for gk in unmatched_g:
        net_errors.append((gk, "hyperGraph整体缺失", f"整网{len(g_sets[gk])}节点未匹配"))
    unmatched_p = [pk for pk in p_keys if pk not in used_p]
    for pk in unmatched_p:
        net_errors.append((pk, "hyperGraph整体多余", "pred多出的网"))

    return {
        "comp_errors": comp_errors,
        "comp_fp": comp_fp,
        "pin_errors": pin_errors,
        "net_errors": net_errors,
        "n_gt_comp": len(g_comp),
        "n_gt_pins": n_gt_pins_scored,
        "n_gt_pins_unscored": n_gt_pins_unscored,
        "n_gt_nets": len(g_nets),
    }


def main() -> int:
    pred_dir = ROOT / "outputs" / "pred"
    gt_dir = ROOT / "outputs" / "gt"
    ids = sorted(
        int(p.name) for p in pred_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )

    # 汇总统计
    comp_reason_cnt = defaultdict(int)
    pin_reason_cnt = defaultdict(int)
    net_reason_cnt = defaultdict(int)
    per_case = {}
    total = {"comp": 0, "pin": 0, "pin_unscored": 0, "net": 0}

    for cid in ids:
        pred_path = pred_dir / str(cid) / f"{cid}.json"
        gt_path = gt_dir / f"{cid}.json"
        if not pred_path.exists() or not gt_path.exists():
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        r = analyze_case(pred, gt)
        per_case[str(cid)] = r

        for _k, _t, reason in r["comp_errors"]:
            comp_reason_cnt[reason] += 1
        for _ck, _pk, _t, reason in r["pin_errors"]:
            pin_reason_cnt[reason] += 1
        for _gk, reason, _d in r["net_errors"]:
            net_reason_cnt[reason] += 1

        total["comp"] += r["n_gt_comp"]
        total["pin"] += r["n_gt_pins"]
        total["pin_unscored"] += r["n_gt_pins_unscored"]
        total["net"] += r["n_gt_nets"]

    # ---- 写 JSON ----
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"per_case": per_case, "summary": {
            "comp": dict(comp_reason_cnt), "pin": dict(pin_reason_cnt),
            "net": dict(net_reason_cnt), "total": total,
        }}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 写 MD ----
    L = ["# PCB 原理图解析 —— 细粒度错误归因报告", "", "## 评测口径（2026-10-14 版）", ""]
    L.append(f"- 空间阈值：元件 bbox / IC 引脚 / 导线端点 **统一 {BBOX_THR:.0f} px**")
    L.append("- PinF1 只统计 **box IC 引脚**（pin key + pinname + 坐标）；普通元件引脚不计位置分")
    L.append("- 连接关系两级：IC 按 **pin 节点**（`U1.7`）、普通元件按 **元件节点**（`C6`，多引脚折叠）")
    L.append("- GND/电源符号（type `gnd`/`v`）按 **类别 + value + 几何位置** 匹配，不比自动编号 key")
    L.append("")
    L.append(f"- 测例数：{len(per_case)}；GT 元件 {total['comp']} / 计分引脚(IC) {total['pin']} / "
             f"不计分引脚(普通元件) {total['pin_unscored']} / 网络 {total['net']}")
    L.append("")

    L.append("## 1. 总体错误归因（判错到最小单位）")
    L.append("")

    L.append("### 1.1 元件（Component）判错原因")
    L.append("")
    L.append("| 原因 | 数量 | 占比(所有GT元件) |")
    L.append("|---|---:|---:|")
    for reason in sorted(comp_reason_cnt, key=lambda x: -comp_reason_cnt[x]):
        n = comp_reason_cnt[reason]
        L.append(f"| {reason} | {n} | {100.0*n/max(1,total['comp']):.1f}% |")
    L.append("")

    L.append("### 1.2 引脚（Pin）判错原因（仅 box IC 引脚计分）")
    L.append("")
    L.append("| 原因 | 数量 | 占比(计分 GT 引脚) |")
    L.append("|---|---:|---:|")
    for reason in sorted(pin_reason_cnt, key=lambda x: -pin_reason_cnt[x]):
        n = pin_reason_cnt[reason]
        L.append(f"| {reason} | {n} | {100.0*n/max(1,total['pin']):.1f}% |")
    L.append("")

    L.append("### 1.3 网络（Net）判错原因（两级节点：IC pin / 普通元件）")
    L.append("")
    L.append("| 原因 | 数量 | 占比(所有GT网络) |")
    L.append("|---|---:|---:|")
    for reason in sorted(net_reason_cnt, key=lambda x: -net_reason_cnt[x]):
        n = net_reason_cnt[reason]
        L.append(f"| {reason} | {n} | {100.0*n/max(1,total['net']):.1f}% |")
    L.append("")

    # ---- 每个测例明细 ----
    L.append("## 2. 每个测例明细")
    L.append("")
    for cid in ids:
        cid_s = str(cid)
        if cid_s not in per_case:
            continue
        r = per_case[cid_s]
        L.append(f"### 测例 {cid}")
        L.append("")
        if r["comp_errors"]:
            L.append("**元件判错**：")
            L.append("")
            for key, t, reason in r["comp_errors"]:
                L.append(f"- `{key}` (type=`{t}`) → `{reason}`")
            L.append("")
        if r["comp_fp"]:
            L.append(f"**元件多余(FP)**：{', '.join(f'`{k}`' for k in r['comp_fp'])}")
            L.append("")
        if r["pin_errors"]:
            L.append("**引脚判错（仅 box IC）**：")
            L.append("")
            for ck, pk, t, reason in r["pin_errors"]:
                L.append(f"- `{ck}.{pk}` (type=`{t}`) → `{reason}`")
            L.append("")
        if r.get("n_gt_pins_unscored"):
            L.append(f"> 另有普通元件引脚 {r['n_gt_pins_unscored']} 个不计位置分（新标准②）。")
            L.append("")
        if r["net_errors"]:
            L.append("**网络判错**：")
            L.append("")
            for gk, reason, detail in r["net_errors"]:
                L.append(f"- `{gk}` → `{reason}`（{detail}）")
            L.append("")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"元件判错原因：{dict(comp_reason_cnt)}")
    print(f"引脚判错原因(仅IC)：{dict(pin_reason_cnt)}  计分IC引脚={total['pin']} 不计分普通引脚={total['pin_unscored']}")
    print(f"网络判错原因：{dict(net_reason_cnt)}")
    print(f"已写 {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
