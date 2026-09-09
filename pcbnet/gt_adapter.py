"""Convert Dataset300 CVJsonStd into contest JSON."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .coords import contest_point, image_bbox_to_contest
from .geometry import polyline_to_segments, unique_keep_order
from .schema import empty_contest_json

DESIGNATOR_RE = re.compile(r"^[A-Z]{1,3}\d+$")
PLACEHOLDER_RE = re.compile(r"^[A-Z]+\?$")
VALUE_HINT_RE = re.compile(r"\d+(k|K|uF|nF|pF|mF|ohm|Ω)", re.I)
POWER_NETS = {"VCC", "GND", "VDD", "VSS", "VPP", "VEE"}
TYPE_PREFIX = {
    "ic": ("U", "IC"),
    "resistor": ("R",),
    "capacitor": ("C",),
    "inductor": ("L",),
    "diode": ("D",),
    "led": ("D", "LED"),
    "bjt": ("Q",),
    "mosfet": ("Q", "M"),
    "crystal": ("Y", "X"),
    "switch": ("SW", "S"),
    "connector": ("J", "P", "CN"),
    "testpoint": ("TP",),
    "fuse": ("F",),
    "battery": ("BT",),
}


def load_category_map(path: str | Path | None = None) -> dict[str, str]:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "category_map.yaml"
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items()}


def _text_content(texts: dict, tid: str | None) -> str | None:
    if not tid:
        return None
    item = texts.get(tid)
    if not item:
        return None
    content = str(item.get("content") or "").strip()
    return content or None


def _associated_contents(associated: dict[str, Any] | None) -> list[str]:
    if not associated:
        return []
    return [str(v.get("content") or "").strip() for v in associated.values() if str(v.get("content") or "").strip()]


def _contents_by_detail(associated: dict[str, Any] | None, detail: str) -> list[str]:
    """Contents whose CVJsonStd `details` tag equals `detail` (designator/value/searchName)."""
    if not associated:
        return []
    out: list[str] = []
    for v in associated.values():
        if str(v.get("details") or "") != detail:
            continue
        content = str(v.get("content") or "").strip()
        if content:
            out.append(content)
    return out


# Sheet titles / metadata that must never be treated as an IC name.
_SHEET_JUNK_RE = re.compile(r"(Drawn\s*By|Sheet|SchDoc|\.Sch\b|Schematic\d*|Title|Rev\b|Date)", re.I)


def _pick_designator(associated: dict[str, Any] | None, fallback: str, dst_type: str) -> str:
    # Authoritative signal first: texts explicitly tagged as a designator.
    for text in _contents_by_detail(associated, "designator"):
        if DESIGNATOR_RE.match(text):
            return text
    contents = _associated_contents(associated)
    prefixes = TYPE_PREFIX.get(dst_type, ())
    if prefixes:
        for text in contents:
            if DESIGNATOR_RE.match(text) and text.startswith(prefixes):
                return text
        for text in contents:
            if PLACEHOLDER_RE.match(text) and text[:-1] in prefixes:
                return fallback
    elif dst_type not in {"ic", "gnd", "vcc", "net_label"}:
        for text in contents:
            if DESIGNATOR_RE.match(text):
                return text
    return fallback


def _is_plausible_ic_name(content: str) -> bool:
    if not content or len(content) < 3:
        return False
    if DESIGNATOR_RE.match(content) or PLACEHOLDER_RE.match(content):
        return False
    if _SHEET_JUNK_RE.search(content):
        return False
    return bool(re.search(r"[A-Za-z]{2,}\d|\d+[A-Za-z]{2,}", content))


def _pick_ic_name(comp: dict, texts: dict) -> str | None:
    # Prefer library / VLM matched names, then OCR name; each must pass the junk filter.
    for field in ("searchNameByLib", "searchNameByVLM", "searchNameByOCR"):
        cand = _text_content(texts, comp.get(field))
        if cand and _is_plausible_ic_name(cand):
            return cand
    for content in _contents_by_detail(comp.get("associatedTexts"), "searchName"):
        if _is_plausible_ic_name(content):
            return content
    for content in _associated_contents(comp.get("associatedTexts")):
        if _is_plausible_ic_name(content):
            return content
    return None


def _pick_value(associated: dict[str, Any] | None) -> str | None:
    # Authoritative signal first: texts explicitly tagged as a value.
    for content in _contents_by_detail(associated, "value"):
        return content
    for content in _associated_contents(associated):
        compact = content.replace("Ω", "").replace("ω", "").replace(" ", "")
        if VALUE_HINT_RE.search(content) or re.fullmatch(r"\d+(\.\d+)?[kKmMuUnNpP]?", compact):
            return content
    return None


def _sanitize_pin_number(raw: str | None, fallback: str) -> str:
    if not raw:
        return fallback
    text = raw.strip()
    if not text or text in {"∅", "?", "o"}:
        return fallback
    if "," in text:
        return text.split(",")[0].strip() or fallback
    return text


def _pin_key(number: str) -> str:
    return f"pin_{number}"


def _hypergraph(pins: list[str]) -> str:
    return "(" + ",".join(unique_keep_order(pins)) + ")"


def _looks_like_net(label: str) -> bool:
    text = (label or "").strip()
    if not text or len(text) > 40:
        return False
    if any(tok in text for tok in ("\\", ":/", "SchDoc", "Drawn", "Sheet")):
        return False
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        return False
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9_ ./+-]{0,39}$", text))


def _is_active(item: dict) -> bool:
    return str(item.get("state") or "active") != "deleted"


def _edge_image_segments(edge: dict, rdp_eps: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Prefer clean 2-point keyPointPaths; fall back to skeleton, then raw coordinates."""
    paths = (edge.get("keyPointPaths") or {}).get("wire") or []
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if paths:
        for path in paths:
            if isinstance(path, (list, tuple)) and len(path) >= 2:
                a, b = path[0], path[-1]
                segs.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        if segs:
            return segs
    skel = ((edge.get("keyPoints") or {}).get("skeleton") or {}).get("wire")
    if skel:
        return polyline_to_segments(skel, rdp_eps)
    return polyline_to_segments(edge.get("coordinates") or [], rdp_eps)


def _resolve_netlabel(edge: dict, texts: dict) -> str:
    raw = str(edge.get("netlabel") or "").strip()
    if not raw:
        return ""
    if raw in texts:
        return (_text_content(texts, raw) or "").strip()
    return raw


def cvjson_to_contest(
    cvjson: dict[str, Any] | str | Path,
    category_map: dict[str, str] | None = None,
    rdp_eps: float = 2.5,
) -> dict[str, Any]:
    if not isinstance(cvjson, dict):
        cvjson = json.loads(Path(cvjson).read_text(encoding="utf-8"))
    category_map = category_map or load_category_map()
    height = float((cvjson.get("canvas") or {}).get("height") or 0)
    texts = cvjson.get("texts") or {}
    raw_components = cvjson.get("components") or {}
    raw_pins = cvjson.get("pins") or {}
    raw_edges = cvjson.get("edges") or {}

    out = empty_contest_json()
    pin_id_to_ref: dict[str, str] = {}
    used_keys: dict[str, int] = {}

    def uniquify(key: str) -> str:
        n = used_keys.get(key, 0) + 1
        used_keys[key] = n
        return key if n == 1 else f"{key}_{n}"

    for cpnt_id, comp in sorted(raw_components.items(), key=lambda kv: kv[0]):
        if not _is_active(comp):
            continue
        src_type = str(comp.get("category") or "other")
        dst_type = category_map.get(src_type, src_type.lower() or "other")
        key = uniquify(_pick_designator(comp.get("associatedTexts"), cpnt_id, dst_type))
        bbox_src = comp.get("realBBox") or comp.get("location") or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        name = _pick_ic_name(comp, texts) if dst_type == "ic" else None
        value = _pick_value(comp.get("associatedTexts")) if dst_type in {"resistor", "capacitor"} else None
        if dst_type not in {"ic", "resistor", "capacitor"}:
            extras = _associated_contents(comp.get("associatedTexts"))
            name = name or (extras[0] if extras else None)
        out["components"][key] = {
            "type": dst_type,
            "name": name,
            "value": value,
            "bbox": image_bbox_to_contest(bbox_src, height),
        }
        out["pins"][key] = {}
        nested = comp.get("pins") or {}
        pin_items = list(nested.items())
        if not pin_items:
            pin_items = [
                (pid, pin)
                for pid, pin in raw_pins.items()
                if (pin or {}).get("cpntId") == cpnt_id
            ]
        seq = 1
        for pid, pin in sorted(pin_items, key=lambda kv: kv[0]):
            pin_number = _sanitize_pin_number(_text_content(texts, pin.get("pinNumberByOCR")), str(seq))
            pin_name = _text_content(texts, pin.get("pinNameByOCR")) or pin_number
            tip = pin.get("tip") or [0, 0]
            pkey = _pin_key(pin_number)
            if pkey in out["pins"][key]:
                pin_number = str(seq)
                pkey = _pin_key(pin_number)
            out["pins"][key][pkey] = {
                "pinname": pin_name,
                "point": contest_point(float(tip[0]), float(tip[1]), height),
            }
            pin_id_to_ref[pid] = f"{key}.{pin_number}"
            seq += 1

    for pid, pin in raw_pins.items():
        if pid in pin_id_to_ref:
            continue
        tip = pin.get("tip") or [0, 0]
        cx, cy = contest_point(float(tip[0]), float(tip[1]), height)["x"], contest_point(float(tip[0]), float(tip[1]), height)["y"]
        best_key, best_d = None, 1e18
        for key, comp in out["components"].items():
            x1, y1, x2, y2 = comp["bbox"]
            if x1 - 8 <= cx <= x2 + 8 and y1 - 8 <= cy <= y2 + 8:
                d = abs((x1 + x2) / 2 - cx) + abs((y1 + y2) / 2 - cy)
                if d < best_d:
                    best_key, best_d = key, d
        if not best_key:
            continue
        seq = len(out["pins"][best_key]) + 1
        pin_number = _sanitize_pin_number(_text_content(texts, pin.get("pinNumberByOCR")), str(seq))
        pkey = _pin_key(pin_number)
        if pkey in out["pins"][best_key]:
            pin_number = str(seq)
            pkey = _pin_key(pin_number)
        out["pins"][best_key][pkey] = {
            "pinname": _text_content(texts, pin.get("pinNameByOCR")) or pin_number,
            "point": contest_point(float(tip[0]), float(tip[1]), height),
        }
        pin_id_to_ref[pid] = f"{best_key}.{pin_number}"

    groups: dict[str, dict[str, Any]] = {}
    for edge_id, edge in raw_edges.items():
        if not _is_active(edge):
            continue
        refs = []
        for item in (edge.get("connect") or {}).values():
            ref = pin_id_to_ref.get(item.get("pinId"))
            if ref:
                refs.append(ref)
        label = _resolve_netlabel(edge, texts)
        merge_key = label if _looks_like_net(label) and label.upper() not in POWER_NETS else edge_id
        if merge_key not in groups:
            groups[merge_key] = {"refs": [], "segs": [], "fly": bool(edge.get("isFlywire"))}
        groups[merge_key]["refs"].extend(refs)
        groups[merge_key]["fly"] = groups[merge_key]["fly"] or bool(edge.get("isFlywire"))
        image_segs = _edge_image_segments(edge, rdp_eps)
        if not image_segs and groups[merge_key]["fly"]:
            for item in (edge.get("connect") or {}).values():
                loc = item.get("location") or [0, 0]
                image_segs.append(((float(loc[0]), float(loc[1])), (float(loc[0]), float(loc[1]))))
        for a, b in image_segs:
            groups[merge_key]["segs"].append(
                [contest_point(a[0], a[1], height), contest_point(b[0], b[1], height)]
            )

    parent = {k: k for k in groups}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pin_to_nets: dict[str, list[str]] = defaultdict(list)
    for gid, g in groups.items():
        for ref in unique_keep_order(g["refs"]):
            pin_to_nets[ref].append(gid)
    for nets in pin_to_nets.values():
        root = find(nets[0])
        for n in nets[1:]:
            parent[find(n)] = root

    merged: dict[str, dict[str, Any]] = {}
    for gid, g in groups.items():
        root = find(gid)
        if root not in merged:
            merged[root] = {"refs": [], "segs": []}
        merged[root]["refs"].extend(g["refs"])
        merged[root]["segs"].extend(g["segs"])

    for i, (gid, g) in enumerate(sorted(merged.items()), start=1):
        refs = unique_keep_order(g["refs"])
        if not refs and not g["segs"]:
            continue
        net_name = gid if _looks_like_net(str(gid)) else f"NET_{i}"
        segs = list(g["segs"])
        if not segs and refs:
            for ref in refs:
                ck, pnum = ref.split(".", 1)
                pt = ((out["pins"].get(ck) or {}).get(_pin_key(pnum)) or {}).get("point")
                if pt:
                    segs.append([pt, pt])
        edges = {f"edge_{j}": seg for j, seg in enumerate(segs, start=1)}
        out["nets"][net_name] = {"hyperGraph": _hypergraph(refs), "edges": edges}

    return out
