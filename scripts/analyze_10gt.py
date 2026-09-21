#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-type components & pins analysis on the 10gt test set.

For each case, match pred components to GT (∅-key policy), evaluate pins (box: key+name
+point; others: point only), and aggregate per-type ComponentF1 / PinF1.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re as _re
_EMPTY_RE = _re.compile(r"^∅")


def _is_empty_key(k: str) -> bool:
    return bool(_EMPTY_RE.match(str(k or "")))

TEST_ROOT = ROOT / "data" / "Dataset200" / "10gt"
PRED_ROOT = ROOT / "outputs1" / "pred"


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _bbox_ok(pb, gb, thr=20.0):
    pc, gc = _center(pb), _center(gb)
    return ((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2) ** 0.5 <= thr


def _f1(tp, fp, fn):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def match_components(pred, gt):
    p_comp = pred.get("components") or {}
    g_comp = gt.get("components") or {}
    key_map, matched = {}, set()
    tp, fn, fp = Counter(), Counter(), Counter()

    def text_ok(pc, gc):
        f = "Name" if gc.get("type") == "box" else ("value" if gc.get("type") in ("r", "c") else None)
        return (not f) or str(pc.get(f) or "") == str(gc.get(f) or "")

    for k, gc in g_comp.items():
        if _is_empty_key(k):
            continue
        pc = p_comp.get(k)
        if pc is None:
            fn[gc.get("type")] += 1
            continue
        if pc.get("type") == gc.get("type") and _bbox_ok(pc.get("bbox") or [0,0,0,0], gc.get("bbox") or [0,0,0,0]) and text_ok(pc, gc):
            tp[gc.get("type")] += 1
            key_map[k] = k
            matched.add(k)
        else:
            fn[gc.get("type")] += 1
            fp[pc.get("type")] += 1

    rem = {k: v for k, v in p_comp.items() if k not in matched}
    for k, gc in g_comp.items():
        if not _is_empty_key(k):
            continue
        best_k, best_d = None, 1e18
        for pk, pc in rem.items():
            if pc.get("type") != gc.get("type") or not _bbox_ok(pc.get("bbox") or [0,0,0,0], gc.get("bbox") or [0,0,0,0]):
                continue
            d = ((_center(pc.get("bbox") or [0,0,0,0])[0]-_center(gc.get("bbox") or [0,0,0,0])[0])**2 + (_center(pc.get("bbox") or [0,0,0,0])[1]-_center(gc.get("bbox") or [0,0,0,0])[1])**2) ** 0.5
            if d < best_d:
                best_d, best_k = d, pk
        if best_k:
            rem.pop(best_k)
            tp[gc.get("type")] += 1
            key_map[k] = best_k
        else:
            fn[gc.get("type")] += 1
    for pk, pc in rem.items():
        fp[pc.get("type")] += 1
    return tp, fn, fp, key_map, rem


def pin_eval(pred, gt, key_map, rem, thr=5.0):
    p_pins = pred.get("pins") or {}
    g_pins = gt.get("pins") or {}
    g_comp = gt.get("components") or {}
    tp, fn, fp = Counter(), Counter(), Counter()

    def pd(a, b):
        return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5

    for gk, gpins in g_pins.items():
        ctype = (g_comp.get(gk) or {}).get("type")
        pk = key_map.get(gk)
        ppins = p_pins.get(pk) if pk else {}
        for gpin_key, gpin in gpins.items():
            gpt = gpin.get("point") or {}
            if ctype == "box":
                pred_pin = ppins.get(gpin_key)
                if pred_pin is None:
                    fn[ctype] += 1
                    continue
                ok = (str(pred_pin.get("pinname") or "") == str(gpin.get("pinname") or "")) and pd(pred_pin.get("point") or {"x":0,"y":0}, gpt) <= thr
            else:
                best_d = 1e18
                for p2 in ppins.values():
                    best_d = min(best_d, pd(p2.get("point") or {"x":0,"y":0}, gpt))
                ok = best_d <= thr
            if ok:
                tp[ctype] += 1
            else:
                fn[ctype] += 1
        # FP (pred pins of this component not matched)
        for p2 in ppins.values():
            matched = False
            if ctype == "box":
                matched = any(p2 is ppins.get(gk2) for gk2 in gpins)
            else:
                matched = any(pd(p2.get("point") or {"x":0,"y":0}, gpin.get("point") or {}) <= thr for gpin in gpins.values())
            if not matched:
                fp[ctype] += 1
    for pk in rem:
        ctype = (pred.get("components") or {}).get(pk, {}).get("type")
        fp[ctype] += len((p_pins.get(pk) or {}))
    return tp, fn, fp


def main() -> int:
    tp_c, fn_c, fp_c = Counter(), Counter(), Counter()
    tp_p, fn_p, fp_p = Counter(), Counter(), Counter()
    for d in sorted(TEST_ROOT.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        gt = json.loads((d / f"{d.name}.json").read_text(encoding="utf-8"))
        pred_path = PRED_ROOT / d.name / f"{d.name}.json"
        if not pred_path.exists():
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        a, b, c, key_map, rem = match_components(pred, gt)
        for t, v in a.items(): tp_c[t] += v
        for t, v in b.items(): fn_c[t] += v
        for t, v in c.items(): fp_c[t] += v
        a, b, c = pin_eval(pred, gt, key_map, rem)
        for t, v in a.items(): tp_p[t] += v
        for t, v in b.items(): fn_p[t] += v
        for t, v in c.items(): fp_p[t] += v

    def report(name, tp, fn, fp):
        types = sorted(set(tp) | set(fn) | set(fp))
        print(f"\n=== {name} (逐类型) ===")
        ttp = tfn = tfp = 0
        for t in types:
            a, b, c = tp.get(t, 0), fn.get(t, 0), fp.get(t, 0)
            ttp += a; tfn += b; tfp += c
            print(f"  {t:16s} TP={a:3d} FN={b:3d} FP={c:3d} F1={_f1(a,c,b):.3f}")
        print(f"  {'TOTAL':16s} TP={ttp:3d} FN={tfn:3d} FP={tfp:3d} F1={_f1(ttp,tfp,tfn):.3f}")

    report("ComponentF1", tp_c, fn_c, fp_c)
    report("PinF1", tp_p, fn_p, fp_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
