
# -*- coding: utf-8 -*-
"""
LLM parsing utilities (generic).

Responsibilities:
- Extract assistant message content and usage info from an Azure OpenAI response object.
- Robustly parse JSON from model outputs (including cases where the model includes extra text).
- Provide small helpers (confidence clamp) used by pipeline stages.

Design choices:
- Keep this module independent of taxonomy/pipeline logic.
- Avoid importing Azure OpenAI SDK types directly; treat response as a generic object.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


def extract_content_and_usage(resp: Any) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """
    Extract:
      - content (assistant message content)
      - usage (token usage dict if available)
      - finish_reason (if available)

    Works with Azure OpenAI Python SDK style responses.

    Returns:
      (content, usage, finish_reason)
    """
    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None
    content: Optional[str] = None

    # usage: for newer SDKs, resp.model_dump() is available
    try:
        dumped = resp.model_dump()  # type: ignore[attr-defined]
        usage = dumped.get("usage", {}) or {}
    except Exception:
        usage = {}

    # choices -> message -> content
    try:
        choices = getattr(resp, "choices", None)
        if choices:
            choice0 = choices[0]
            finish_reason = getattr(choice0, "finish_reason", None)
            msg = getattr(choice0, "message", None)
            content = getattr(msg, "content", None) if msg else None
    except Exception:
        # Keep defaults
        pass

    return content, usage, finish_reason


def safe_parse_json(s: Optional[str]) -> Dict[str, Any]:
    """
    Safely parse JSON from a string.

    Strategy:
    1) Try json.loads on the whole string
    2) If it fails, try extracting the first {...} block and parse that
    3) If still fails, return {}

    This is resilient to accidental prefix/suffix text around JSON.
    """
    text = (s or "").strip()
    if not text:
        return {}

    # Attempt 1: strict parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt 2: extract first JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    return {}


def clamp_conf(x: Any) -> float:
    """
    Clamp a confidence-like value to [0, 1].
    """
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def get_json_field(data: Dict[str, Any], key: str, default: Any = "") -> Any:
    """
    Convenience getter for parsed JSON dicts.
    """
    try:
        v = data.get(key, default)
        return default if v is None else v
    except Exception:
        return default
