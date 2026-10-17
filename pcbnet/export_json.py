from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .schema import dump_json, validate_contest_json


def _round_coords(data: dict[str, Any]) -> dict[str, Any]:
    """最终输出格式：所有定位数字取整为整数（如 540）。

    bbox、pin 的 point、net edge 的端点统一 `round(v)` 成整数，
    再以 float 写出（json 里 540.0 → "540.0"）。
    """
    out = copy.deepcopy(data)

    def i1(v: Any) -> int:
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return int(float(v))

    for comp in (out.get("components") or {}).values():
        bb = comp.get("bbox")
        if isinstance(bb, list) and len(bb) == 4:
            comp["bbox"] = [i1(v) for v in bb]
    for pins in (out.get("pins") or {}).values():
        for pin in (pins or {}).values():
            pt = pin.get("point") if isinstance(pin, dict) else None
            if isinstance(pt, dict):
                if "x" in pt:
                    pt["x"] = i1(pt["x"])
                if "y" in pt:
                    pt["y"] = i1(pt["y"])
    for net in (out.get("nets") or {}).values():
        for seg in (net.get("edges") or {}).values():
            if not isinstance(seg, list):
                continue
            for pt in seg:
                if isinstance(pt, dict):
                    if "x" in pt:
                        pt["x"] = i1(pt["x"])
                    if "y" in pt:
                        pt["y"] = i1(pt["y"])
    return out


def export_contest_json(data: dict[str, Any], path: str | Path, strict: bool = False) -> list[str]:
    # Only the three contest sections belong in the submission file.
    clean = {k: data[k] for k in ("components", "pins", "nets") if k in data}
    clean = _round_coords(clean)
    errors = validate_contest_json(clean)
    dump_json(clean, path)
    if strict and errors:
        raise ValueError("invalid contest json: " + "; ".join(errors[:8]))
    return errors
