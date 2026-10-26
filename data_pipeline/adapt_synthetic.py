#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合成数据 → Dataset200 适配。

把 ``data/Synthetic_Data/<n>/{board_render.png,annotation.json}`` 适配成
``data/Dataset200/200_train_cases/<id>/<id>-Synthetic.png`` +
``<id>-Synthetic_target_named.json``，编号从 0301 起（前 300 个是真实 case）。

与赛题/真实数据保持一致的约定
-----------------------------
* JSON 顶层字段顺序 ``components`` → ``pins`` → ``nets``；
* component 字段顺序 ``Name`` → ``type`` → ``value`` → ``bbox``；
  net 字段顺序 ``hyperGraph`` → ``edges``；
* components / pins / nets 内部顺序与 ``annotation.json`` 完全一致（dict 插入序）；
* component key 去掉合成器加的真实来源前缀（``0138_U10.2`` → ``U10.2``），
  ``∅`` 元件保留 ``∅<编号>``；同一 case 内重名按出现顺序编 ``_1/_2``（赛题重名规则）；
* ``Name`` == key（合成元件无 box 类型号名），``value`` 为 ``null``（合成器未标注 value）；
* pin 沿用 ``pin_<n>``，只保留 ``pinname`` + ``point``（丢弃 route_start / outward_direction）；
* net key 规范化为 ``net_<i>``、edge key 规范化为 ``edge_<i>``，顺序不变，
  丢弃 ``junctions``（赛题 schema 无此字段）；
* ``hyperGraph`` 形如 ``(U1.7,R5.1)``，即 ``<key>.<pin_key 去掉 pin_ 前缀>``。

合成数据自带的文本框 bbox 写入 ``data/Dataset200/text_bbox/<id>.json``
（图像左上角坐标，供 YOLO 的 text 类标签使用）。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "data" / "Synthetic_Data"
CASES_ROOT = ROOT / "data" / "Dataset200" / "200_train_cases"
TEXT_ROOT = ROOT / "data" / "Dataset200" / "text_bbox"

SOURCE_SUFFIX = "Synthetic"
START_ID = 301  # 真实 case 0001~0300 之后

_SOURCE_PREFIX_RE = re.compile(r"^\d{4}_")


def strip_source_prefix(name: str) -> str:
    """``0138_U10.2`` → ``U10.2``；``0185_∅399`` → ``∅399``。"""
    return _SOURCE_PREFIX_RE.sub("", str(name))


def build_case(annotation: dict, text_boxes: list[tuple[str, list[float]]]) -> dict:
    """annotation.json（合成器格式）→ 赛题 contest JSON。"""
    raw_comps: dict = annotation.get("components") or {}
    raw_pins: dict = annotation.get("pins") or {}
    raw_nets: dict = annotation.get("nets") or {}

    # 1) component key 重命名（去来源前缀 + 同 case 内去重）
    rename: dict[str, str] = {}
    used: dict[str, int] = {}
    components: dict = {}
    for raw_key, comp in raw_comps.items():
        key = strip_source_prefix(raw_key)
        n = used.get(key, 0) + 1
        used[key] = n
        key = key if n == 1 else f"{key}_{n}"
        rename[raw_key] = key
        components[key] = {
            "Name": key,
            "type": comp.get("type"),
            "value": None,
            "bbox": [float(v) for v in comp.get("bbox") or []],
        }

    # 2) pins：只保留 pinname + point，顺序不变
    pins: dict = {}
    for raw_key, pin_dict in raw_pins.items():
        key = rename.get(raw_key)
        if key is None:
            continue
        out_pins: dict = {}
        for pin_key, pin in (pin_dict or {}).items():
            point = pin.get("point") or {}
            out_pins[pin_key] = {
                "pinname": pin.get("pinname"),
                "point": {"x": float(point.get("x", 0.0)), "y": float(point.get("y", 0.0))},
            }
        pins[key] = out_pins

    # 3) nets：hyperGraph + edges，顺序不变
    nets: dict = {}
    for i, (_, net) in enumerate(raw_nets.items(), start=1):
        refs: list[str] = []
        for item in net.get("pins") or []:
            key = rename.get(item.get("comp_name"))
            if key is None:
                continue
            refs.append(f"{key}.{str(item.get('pin_key', '')).replace('pin_', '', 1)}")
        edges = {}
        for j, seg in enumerate((net.get("edges") or {}).values(), start=1):
            edges[f"edge_{j}"] = [
                {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))} for p in seg
            ]
        nets[f"net_{i}"] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": edges}

    return {"components": components, "pins": pins, "nets": nets}


def convert_text_boxes(annotation: dict, case_id: str) -> dict:
    """合成 ``text`` 字段（左下角原点）→ 图像左上角坐标的 text_bbox 记录。"""
    width, height = (float(v) for v in annotation.get("board_size") or [0.0, 0.0])
    texts = []
    for text, bbox in (annotation.get("text") or {}).items():
        x1, y1, x2, y2 = (float(v) for v in bbox)
        texts.append(
            {
                "text": text,
                "bbox": [x1, height - max(y1, y2), x2, height - min(y1, y2)],
            }
        )
    return {
        "case_id": case_id,
        "width": width,
        "height": height,
        "source": "synthetic",
        "coordinate_system": "image_top_left",
        "texts": texts,
    }


def iter_source_cases() -> list[Path]:
    dirs = [p for p in SRC_ROOT.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0))


def clean_stale(keep_ids: set[str], start_id: int) -> list[str]:
    """删除上一次适配留下、本次不再生成的合成 case（只动 <id>-Synthetic.* 与 ≥start_id 的 text_bbox）。"""
    removed: list[str] = []
    for d in sorted(CASES_ROOT.iterdir()):
        if not d.is_dir() or not d.name.isdigit() or d.name in keep_ids or int(d.name) < start_id:
            continue
        mine = [p for p in d.iterdir() if f"-{SOURCE_SUFFIX}" in p.name]
        if not mine:
            continue
        shutil.rmtree(d)
        for p in TEXT_ROOT.glob(f"{d.name}.json"):
            p.unlink()
        removed.append(d.name)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-id", type=int, default=START_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-clean", action="store_true", help="不删除上次遗留、本次不再生成的合成 case")
    args = parser.parse_args()

    sources = iter_source_cases()
    if not sources:
        print(f"[adapt] 没有找到合成 case：{SRC_ROOT}")
        return 1

    CASES_ROOT.mkdir(parents=True, exist_ok=True)
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[adapt] {len(sources)} 个合成 case → 编号 {args.start_id:04d} 起")
    rows = []
    for offset, src in enumerate(sources):
        case_id = f"{args.start_id + offset:04d}"
        annotation = json.loads((src / "annotation.json").read_text(encoding="utf-8"))
        image_src = src / annotation.get("image", "circuit.png")
        if not image_src.exists():
            print(f"  [skip] {src.name} 缺少图片 {image_src.name}")
            continue

        contest = build_case(annotation, [])
        text_bbox = convert_text_boxes(annotation, case_id)

        case_dir = CASES_ROOT / case_id
        image_dst = case_dir / f"{case_id}-{SOURCE_SUFFIX}{image_src.suffix.lower()}"
        json_dst = case_dir / f"{case_id}-{SOURCE_SUFFIX}_target_named.json"
        text_dst = TEXT_ROOT / f"{case_id}.json"

        n_pins = sum(len(v) for v in contest["pins"].values())
        n_edges = sum(len(v["edges"]) for v in contest["nets"].values())
        rows.append((case_id, src.name, len(contest["components"]), n_pins, len(contest["nets"]),
                     n_edges, len(text_bbox["texts"])))

        if args.dry_run:
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, image_dst)
        json_dst.write_text(json.dumps(contest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        text_dst.write_text(json.dumps(text_bbox, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.dry_run and not args.no_clean:
        stale = clean_stale({r[0] for r in rows}, args.start_id)
        if stale:
            print(f"[adapt] 清理 {len(stale)} 个上次遗留的合成 case：{', '.join(stale[:8])}"
                  f"{' ...' if len(stale) > 8 else ''}")

    print(f"{'case':>6} {'src':>4} {'comps':>5} {'pins':>5} {'nets':>5} {'edges':>5} {'text':>5}")
    for row in rows:
        print(f"{row[0]:>6} {row[1]:>4} {row[2]:>5} {row[3]:>5} {row[4]:>5} {row[5]:>5} {row[6]:>5}")
    if args.dry_run:
        print("[adapt] dry-run，未写入文件")
        return 0
    print(f"[adapt] 已写入 {CASES_ROOT} 与 {TEXT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
