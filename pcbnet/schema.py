"""Contest JSON schema helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def empty_contest_json() -> dict[str, Any]:
    return {"components": {}, "pins": {}, "nets": {}}


def validate_contest_json(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["root must be an object"]
    for key in ("components", "pins", "nets"):
        if key not in data or not isinstance(data[key], dict):
            errors.append(f"missing object field: {key}")
    if errors:
        return errors
    for ck, comp in data["components"].items():
        if not isinstance(comp, dict):
            errors.append(f"component {ck} is not an object")
            continue
        if "type" not in comp:
            errors.append(f"component {ck} missing type")
        bbox = comp.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            errors.append(f"component {ck} bbox must be [x1,y1,x2,y2]")
        elif bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            errors.append(f"component {ck} bbox order invalid")
    for ck, pins in data["pins"].items():
        if ck not in data["components"]:
            errors.append(f"pins.{ck} has no matching component")
        if not isinstance(pins, dict):
            errors.append(f"pins.{ck} is not an object")
            continue
        for pk, pin in pins.items():
            if not isinstance(pin, dict):
                errors.append(f"{ck}.{pk} is not an object")
                continue
            if "pinname" not in pin:
                errors.append(f"{ck}.{pk} missing pinname")
            pt = pin.get("point")
            if not (isinstance(pt, dict) and "x" in pt and "y" in pt):
                errors.append(f"{ck}.{pk} point must be {{x,y}}")
    for nk, net in data["nets"].items():
        if not isinstance(net, dict):
            errors.append(f"net {nk} is not an object")
            continue
        hg = net.get("hyperGraph")
        if not isinstance(hg, str) or not hg.startswith("("):
            errors.append(f"net {nk} hyperGraph format invalid")
        edges = net.get("edges")
        if not isinstance(edges, dict):
            errors.append(f"net {nk} edges must be an object")
            continue
        for ek, edge in edges.items():
            if not (isinstance(edge, list) and len(edge) == 2):
                errors.append(f"{nk}.{ek} must be a 2-point list")
                continue
            for p in edge:
                if not (isinstance(p, dict) and "x" in p and "y" in p):
                    errors.append(f"{nk}.{ek} points must be {{x,y}}")
    return errors


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clone(data: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(data)
