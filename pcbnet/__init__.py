"""PCBnet baseline: schematic image to contest JSON + scoring."""

from .schema import empty_contest_json, validate_contest_json
from .gt_adapter import cvjson_to_contest
from .scorer import score_batch, score_case

__all__ = [
    "empty_contest_json",
    "validate_contest_json",
    "cvjson_to_contest",
    "score_case",
    "score_batch",
]
