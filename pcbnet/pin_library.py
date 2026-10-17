"""Look up IC pin number -> pinname from BoxICLibrary.json."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_LIB = Path(__file__).resolve().parent.parent / "data" / "Dataset200" / "10gt" / "BoxICLibrary.json"

_cache: dict[str, Any] | None = None


def load_library(path: str | Path | None = None) -> dict[str, Any]:
    global _cache
    if _cache is None:
        p = Path(path) if path else DEFAULT_LIB
        _cache = json.loads(p.read_text(encoding="utf-8"))
    return _cache


def _pick_name(names: list[str], num: str) -> str:
    counter = Counter(n for n in names if n)
    if not counter:
        return num
    for name, _ in counter.most_common():
        if name != num:
            return name
    return counter.most_common(1)[0][0]


def lookup_pins(ic_name: str, path: str | Path | None = None) -> dict[str, str]:
    """Return {pin_number: pinname} for an IC name, or {} if not found."""
    lib = load_library(path)
    rec = lib.get(ic_name)
    if not rec:
        # 软匹配：前缀/后缀变体
        for key, val in lib.items():
            if key.startswith(ic_name) or ic_name.startswith(key):
                rec = val
                break
        if not rec:
            return {}
    by_number: dict[str, list[str]] = defaultdict(list)
    for pin in rec.get("pins") or []:
        num = str(pin.get("number") or "").strip()
        name = str(pin.get("name") or "").strip()
        if not num:
            continue
        by_number[num].append(name)
    return {num: _pick_name(names, num) for num, names in by_number.items()}
