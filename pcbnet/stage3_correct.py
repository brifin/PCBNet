"""Stage 3: confidence-gated multi-agent correction.

Order (cheap -> expensive), matching PCBnet:
  1. Gatekeeper: only tokens that look wrong (junk chars, missing underscore,
     failing naming) are candidates. High-confidence tokens pass through.
  2. Rule fixes: 1/l/I and O/0 confusions on short alnum tokens. Never rewrites
     R/C `value` case (10k != 10K is contest-significant).
  3. LLM correction (qwen3.7-plus): constrained token repair, only when a client
     is configured and within a per-case call budget. Off by default.

Runs on the contest dict (component names / pin names). Value strings are left
untouched to avoid corrupting case-sensitive comparisons.
"""

from __future__ import annotations

import re
from typing import Any

from .correct.qwen_client import QwenClient

_CONFUSE_TO_DIGIT = str.maketrans({"O": "0", "l": "1", "I": "1"})
_DESIGNATOR_RE = re.compile(r"^[A-Z]{1,3}\d+$")


def _looks_wrong(token: str) -> bool:
    t = (token or "").strip()
    if not t:
        return False
    if re.fullmatch(r"[O0Il1]{1,3}", t):
        return True
    if _DESIGNATOR_RE.match(t):
        return False
    # designator-ish but with a confusable char instead of a digit
    return bool(re.match(r"^[A-Z]{1,3}[OlI]$", t))


def _rule_fix(token: str) -> str:
    t = (token or "").strip()
    if re.fullmatch(r"[O0Il1]{1,3}", t):
        return t.translate(_CONFUSE_TO_DIGIT)
    m = re.match(r"^([A-Z]{1,3})([OlI])$", t)
    if m:
        return m.group(1) + m.group(2).translate(_CONFUSE_TO_DIGIT)
    return t


def correct(contest: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    scfg = cfg.get("stage3") or {}
    if not (scfg.get("enabled") or cfg.get("stage3_enabled")):
        return contest
    client = QwenClient(scfg)
    budget = int(scfg.get("max_calls_per_case", 40))
    used = 0

    def maybe_llm(token: str, category: str | None) -> str:
        nonlocal used
        if not client.available or used >= budget:
            return token
        used += 1
        allowed = "A-Z0-9_" if category != "ic" else "A-Za-z0-9_-"
        return client.correct_token(token, allowed, category)

    for comp in (contest.get("components") or {}).values():
        name = comp.get("name")
        if isinstance(name, str) and _looks_wrong(name):
            fixed = _rule_fix(name)
            if fixed == name:
                fixed = maybe_llm(name, comp.get("type"))
            comp["name"] = fixed
        # value is intentionally not case-normalised or LLM-touched.

    for ck, pins in (contest.get("pins") or {}).items():
        ctype = ((contest.get("components") or {}).get(ck) or {}).get("type")
        for pin in (pins or {}).values():
            pn = pin.get("pinname")
            if isinstance(pn, str) and _looks_wrong(pn):
                fixed = _rule_fix(pn)
                if fixed == pn:
                    fixed = maybe_llm(pn, ctype)
                pin["pinname"] = fixed
    contest.setdefault("_stage3", {})["llm_calls"] = used
    return contest
