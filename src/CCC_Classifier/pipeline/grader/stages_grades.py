# -*- coding: utf-8 -*-
"""
Grading stages (LLM-backed).

Each stage returns a grade payload:
  {
    "verdict": "Correct" | "Partial" | "Incorrect",
    "score": 1 | 0.5 | 0,
    "suggested_label": "<string>"  # empty if verdict is Correct
  }

This implementation uses prompt builders in:
  CCC_Classifier.pipeline.grader.prompts

Note:
- We pass full taxonomy for subdomain/root_cause (flattened), per design decision.
- This module contains execution + parsing/normalization; prompt text lives in prompts.py.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
import logging
from CCC_Classifier.llm.client import send_chat_request
from CCC_Classifier.llm.parsing import extract_content_and_usage, get_json_field, safe_parse_json
from CCC_Classifier.pipeline.grader.prompts_grades import (
    system_prompt_grade_contact_driver,
    system_prompt_grade_contact_type,
    system_prompt_grade_domain,
    system_prompt_grade_root_cause,
    system_prompt_grade_subdomain,
    user_prompt_grade_contact_driver,
    user_prompt_grade_contact_type,
    user_prompt_grade_domain,
    user_prompt_grade_root_cause,
    user_prompt_grade_subdomain,
)

logger = logging.getLogger(__name__)

_ALLOWED_VERDICTS = {"Correct", "Partial", "Incorrect"}


def _as_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x).strip() if x is not None else ""


def _normalize_grade_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    verdict = _as_str(get_json_field(data, "verdict", "Incorrect"))
    verdict = verdict[:1].upper() + verdict[1:].lower() if verdict else "Incorrect"
    if verdict not in _ALLOWED_VERDICTS:
        verdict = "Incorrect"

    score_raw = get_json_field(data, "score", 0)
    try:
        score_f = float(score_raw)
    except Exception:
        score_f = 0.0

    # snap to {0,0.5,1}
    if score_f >= 0.75:
        score = 1.0
    elif score_f >= 0.25:
        score = 0.5
    else:
        score = 0.0

    suggested = _as_str(get_json_field(data, "suggested_label", ""))
    if verdict == "Correct":
        suggested = ""

    return {"verdict": verdict, "score": score, "suggested_label": suggested}


async def _grade_label(
    *,
    client: Any,
    deployment: str,
    system_text: str,
    user_text: str,
    max_completion_tokens: int,
    use_json_mode: bool,
) -> Dict[str, Any]:
    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=system_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )
    content, usage, finish = extract_content_and_usage(resp)
    # print("\n" + "=" * 100)
    # print("[GRADER DEBUG] raw_model_content:\n" + (content or ""))
    # print("-" * 100)
    # print("[GRADER DEBUG] usage:", usage)
    # print("[GRADER DEBUG] finish_reason:", finish)
    # print("=" * 100 + "\n")
    data = safe_parse_json(content)


    out = _normalize_grade_payload(data if isinstance(data, dict) else {})
    out["_usage"] = usage
    out["_finish"] = finish
    return out


async def stage_grade_contact_type(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    predicted_contact_type: str,
    max_completion_tokens: int = 200,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    system_text = system_prompt_grade_contact_type()
    user_text = user_prompt_grade_contact_type(
        transcript=transcript,
        predicted_contact_type=predicted_contact_type,
    )
    return await _grade_label(
        client=client,
        deployment=deployment,
        system_text=system_text,
        user_text=user_text,
        max_completion_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )


async def stage_grade_domain(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    predicted_domain: str,
    predicted_contact_type: str,
    max_completion_tokens: int = 220,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    system_text = system_prompt_grade_domain()
    user_text = user_prompt_grade_domain(
        transcript=transcript,
        predicted_domain=predicted_domain,
        predicted_contact_type=predicted_contact_type,
    )
    return await _grade_label(
        client=client,
        deployment=deployment,
        system_text=system_text,
        user_text=user_text,
        max_completion_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )


async def stage_grade_subdomain(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    predicted_subdomain: str,
    predicted_domain: str,
    max_completion_tokens: int = 240,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    system_text = system_prompt_grade_subdomain()
    user_text = user_prompt_grade_subdomain(
        transcript=transcript,
        predicted_subdomain=predicted_subdomain,
        predicted_domain=predicted_domain,
    )
    return await _grade_label(
        client=client,
        deployment=deployment,
        system_text=system_text,
        user_text=user_text,
        max_completion_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )


async def stage_grade_root_cause(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    predicted_root_cause: str,
    predicted_subdomain: str,
    max_completion_tokens: int = 260,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    system_text = system_prompt_grade_root_cause()
    user_text = user_prompt_grade_root_cause(
        transcript=transcript,
        predicted_root_cause=predicted_root_cause,
        predicted_subdomain=predicted_subdomain,
    )
    return await _grade_label(
        client=client,
        deployment=deployment,
        system_text=system_text,
        user_text=user_text,
        max_completion_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )


async def stage_grade_contact_driver(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    predicted_contact_driver: str,
    max_completion_tokens: int = 220,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    system_text = system_prompt_grade_contact_driver()
    user_text = user_prompt_grade_contact_driver(
        transcript=transcript,
        predicted_contact_driver=predicted_contact_driver,
    )
    return await _grade_label(
        client=client,
        deployment=deployment,
        system_text=system_text,
        user_text=user_text,
        max_completion_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )   