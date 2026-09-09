"""Thin client for the contest-provided qwen3.7-plus API.

Safety / environment notes
--------------------------
- The competition test platform provides a qwen3.7-plus endpoint. This client
  only talks to an endpoint explicitly configured via env vars; it never calls a
  public service and is a no-op when unconfigured. This keeps local runs offline
  (internal-network policy) while matching the contest runtime when deployed.
- Env vars: QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL (defaults to config model).
- The client is only ever used for short, low-confidence text tokens (never for
  end-to-end schematic parsing), with a per-case call budget enforced upstream.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any


class QwenClient:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.model = os.environ.get("QWEN_MODEL") or cfg.get("model") or "qwen3.7-plus"
        self.base_url = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("QWEN_API_KEY", "")
        self.use_vlm_crop = bool(cfg.get("use_vlm_crop", False))
        self.calls = 0

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def correct_token(self, token: str, allowed: str, category: str | None, crop_png: bytes | None = None) -> str:
        """Return a corrected token constrained to `allowed` charset / naming.

        Offline (no endpoint): returns the input unchanged. Online: sends a tiny
        constrained-generation prompt (and optional crop) to qwen3.7-plus.
        """
        if not self.available:
            return token
        prompt = (
            "You correct a single OCR token from a PCB schematic. "
            f"Component category: {category or 'unknown'}. "
            f"Allowed characters: {allowed}. "
            "Return ONLY the corrected token, no explanation.\n"
            f"OCR token: {token}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if crop_png and self.use_vlm_crop:
            b64 = base64.b64encode(crop_png).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        try:
            import urllib.request

            body = json.dumps(
                {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0}
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self.calls += 1
            out = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            out = re.split(r"\s", out)[0] if out else token
            return out or token
        except Exception:
            return token
