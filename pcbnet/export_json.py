from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import dump_json, validate_contest_json


def export_contest_json(data: dict[str, Any], path: str | Path, strict: bool = False) -> list[str]:
    # Only the three contest sections belong in the submission file.
    clean = {k: data[k] for k in ("components", "pins", "nets") if k in data}
    errors = validate_contest_json(clean)
    dump_json(clean, path)
    if strict and errors:
        raise ValueError("invalid contest json: " + "; ".join(errors[:8]))
    return errors
