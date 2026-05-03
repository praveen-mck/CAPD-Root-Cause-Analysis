# -*- coding: utf-8 -*-
"""
Grading orchestrator: grade a single prediction row against the transcript.

Responsibilities:
- Accept transcript_text + predicted labels (5 fields)
- Run grading stages in sequence (or independently):
    contact_type -> domain -> subdomain
- Return a single normalized dict consistent with batchgrades.py expectations

Output contract:
{
  "CONTACT_TYPE": {"verdict": "Correct|Partial|Incorrect", "score": 0|0.5|1, "suggested_label": str},
  "DOMAIN": {...},
  "SUBDOMAIN": {...},
  "overall_score": float
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from CCC_Classifier.pipeline.grader.stages_grades import (
    stage_grade_contact_type,
    stage_grade_domain,
    stage_grade_subdomain
)

logger = logging.getLogger(__name__)

GRADE_FIELDS = ("CONTACT_TYPE", "DOMAIN", "SUBDOMAIN")
_ALLOWED_VERDICTS = {"Correct", "Partial", "Incorrect"}


def _safe_grade_payload() -> Dict[str, Dict[str, object]]:
    return {f: {"verdict": "Incorrect", "score": 0.0, "suggested_label": ""} for f in GRADE_FIELDS}


def _normalize_verdict(v: object) -> str:
    s = str(v or "").strip()
    if not s:
        return "Incorrect"
    s_norm = s[:1].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    return s_norm if s_norm in _ALLOWED_VERDICTS else "Incorrect"


def _normalize_score(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x >= 0.75:
        return 1.0
    if x >= 0.25:
        return 0.5
    return 0.0


def _overall_score(grades: Dict[str, Dict[str, object]]) -> float:
    vals = []
    for f in GRADE_FIELDS:
        vals.append(_normalize_score(grades.get(f, {}).get("score")))
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _normalize_field_grade(x: object) -> Dict[str, object]:
    if not isinstance(x, dict):
        return {"verdict": "Incorrect", "score": 0.0, "suggested_label": ""}

    verdict = _normalize_verdict(x.get("verdict"))
    score = _normalize_score(x.get("score"))

    suggested = "" if x.get("suggested_label") is None else str(x.get("suggested_label"))
    # Enforce contract: Correct => empty suggestion
    if verdict == "Correct":
        suggested = ""

    return {"verdict": verdict, "score": score, "suggested_label": suggested}


async def analyze_predict_row(
    *,
    client: Any,
    deployment: str,
    transcript_text: str,
    predicted: Dict[str, str],
    max_completion_tokens: int = 1024,
    use_json_mode: bool = True,
) -> Dict[str, object]:
    """
    Grade one prediction row (5 labels) vs transcript.
    """
    transcript = (transcript_text or "").strip()
    if not transcript:
        out = _safe_grade_payload()
        return {**out, "overall_score": _overall_score(out)}

    try:
        # Stages return field payloads: {"verdict": ..., "score": ..., "suggested_label": ...}
        ct = await stage_grade_contact_type(
            client=client,
            deployment=deployment,
            transcript=transcript,
            predicted_contact_type=str(predicted.get("CONTACT_TYPE") or ""),
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        dom = await stage_grade_domain(
            client=client,
            deployment=deployment,
            transcript=transcript,
            predicted_domain=str(predicted.get("DOMAIN") or ""),
            predicted_contact_type=str(predicted.get("CONTACT_TYPE") or ""),
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        sub = await stage_grade_subdomain(
            client=client,
            deployment=deployment,
            transcript=transcript,
            predicted_subdomain=str(predicted.get("SUBDOMAIN") or ""),
            predicted_domain=str(predicted.get("DOMAIN") or ""),
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )

        out = {
            "CONTACT_TYPE": _normalize_field_grade(ct),
            "DOMAIN": _normalize_field_grade(dom),
            "SUBDOMAIN": _normalize_field_grade(sub),
        }
        return {**out, "overall_score": _overall_score(out)}

    except Exception:
        preview = transcript[:300].replace("\n", " ")
        logger.exception(
            "analyze_predict_row failed max_completion_tokens=%s use_json_mode=%s transcript_len=%s preview=%r",
            max_completion_tokens,
            use_json_mode,
            len(transcript),
            preview,
        )
        out = _safe_grade_payload()
        return {**out, "overall_score": _overall_score(out)}