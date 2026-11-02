"""Load contest knowledge bases.

- key-type KB: parsed from type_list.md (type -> key prefix, constrains naming given type)
- box-name KB: BoxICLibrary.json (box model names; pin naming handled by pin_library)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Dataset200" / "10gt"
TYPE_LIST = _DATA_DIR / "type_list.md"
BOX_LIB = _DATA_DIR / "BoxICLibrary.json"

_key_type_cache: dict[str, Any] | None = None
_box_name_cache: set[str] | None = None


def _parse_type_list(path: Path) -> dict[str, Any]:
    """Parse type_list.md (| `type` | 含义 | 示例位号 |) into canonical_prefix + prefix_to_type."""
    canonical: dict[str, list[str]] = {}
    prefix_to_type: dict[str, set[str]] = {}
    designator_re = re.compile(r"^([A-Za-z]{1,3})\d")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        typ = cells[0].strip("`").strip()
        if not typ or typ in ("type", "---") or typ == "含义":
            continue
        if typ == "box":
            # box 的"示例位号"是型号名（如 RP2350B/RM2），key 前缀(U)走 PREFIX_TYPE 兜底
            continue
        prefixes: list[str] = []
        for ex in cells[2].split("、"):
            ex = ex.strip()
            if not ex or ex.startswith("∅"):
                continue
            m = designator_re.match(ex)
            if m:
                p = m.group(1).upper()
                if p not in prefixes:
                    prefixes.append(p)
                prefix_to_type.setdefault(p, set()).add(typ)
        canonical[typ] = prefixes
    return {
        "canonical_prefix": canonical,
        "prefix_to_type": {p: {t: 1 for t in ts} for p, ts in prefix_to_type.items()},
    }


def load_key_type_kb() -> dict[str, Any]:
    global _key_type_cache
    if _key_type_cache is None:
        _key_type_cache = _parse_type_list(TYPE_LIST)
    return _key_type_cache


def load_box_name_kb() -> dict[str, Any]:
    return json.loads(BOX_LIB.read_text(encoding="utf-8"))


def known_box_names() -> set[str]:
    """Whitelist of box model names (keys of BoxICLibrary.json)."""
    global _box_name_cache
    if _box_name_cache is None:
        _box_name_cache = set(load_box_name_kb().keys())
    return _box_name_cache
