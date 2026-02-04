# -*- coding: utf-8 -*-
"""
Prompt builders for Grader stages.

Design goals:
- Grade predicted labels against transcript evidence.
- Return STRICT JSON only (no markdown, no extra keys).
- Include a full allowed taxonomy list (per current design decision).
- Verdicts/scores are discrete:
    - verdict: "Correct" | "Partial" | "Incorrect"
    - score: 1 | 0.5 | 0
- suggested_label:
    - MUST be "" when verdict == "Correct"
    - otherwise prefer a label from the allowed list; if none fit, provide best-effort free text

Each stage returns a JSON object:
{
  "verdict": "Correct" | "Partial" | "Incorrect",
  "score": 1 | 0.5 | 0,
  "suggested_label": "<string>"
}
"""

from __future__ import annotations

from typing import List, Optional

from CCC_Classifier.taxonomy.dictionaries import (
    CONTACT_TYPES_CANON,
    DOMAINS_CANON,
    SUBDOMAINS_BY_DOMAIN_CANON,
    ROOT_CAUSES_BY_SUBDOMAIN_CANON,
    CONTACT_DRIVERS_CANON,
)


def _json_rule_block() -> str:
    return (
        "Output MUST be a single JSON object with EXACTLY these keys: "
        "'verdict', 'score', 'suggested_label'.\n"
        "- 'verdict' MUST be one of: Correct, Partial, Incorrect.\n"
        "- 'score' MUST be one of: 1, 0.5, 0.\n"
        "- 'suggested_label' MUST be a string.\n"
        "Do NOT include markdown, code fences, comments, explanations, or any extra keys.\n"
        "Do NOT wrap JSON in additional text.\n"
    )


def _grading_rules_block() -> str:
    return (
        "TASK:\n"
        "You will be given 3 things:\n"
        "1) TRANSCRIPT: the evidence for what the customer is contacting about.\n"
        "2) PREDICTED LABEL: the classifier's label for the current field.\n"
        "3) ALLOWED LABELS: the taxonomy labels you should prefer.\n\n"
        "YOU MUST RETURN:\n"
        "- verdict: Correct | Partial | Incorrect\n"
        "- score: 1 | 0.5 | 0\n"
        "- suggested_label: string (see rules below)\n\n"
        "HOW TO CHOOSE verdict + score:\n"
        "- Correct (score=1): The predicted label is clearly supported by the transcript.\n"
        "- Partial (score=0.5): The predicted label is related/close, but not the best match.\n"
        "- Incorrect (score=0): The predicted label is not supported by the transcript.\n\n"
        "HOW TO CHOOSE suggested_label:\n"
        "- If score=1 (Correct): suggested_label MUST be \"\".\n"
        "- If score=0.5 (Partial) or score=0 (Incorrect): suggested_label MUST be NON-EMPTY.\n"
        "  Prefer selecting suggested_label from the ALLOWED LABELS list.\n"
        "  Only use free-text if NONE of the allowed labels reasonably fit the transcript.\n\n"
        "EXAMPLES (illustrative):\n"
        "Example 1 (Incorrect -> choose allowed label):\n"
        "- Transcript: user cannot reset password / cannot log in.\n"
        "- Predicted label: \"Billing\"\n"
        "- Allowed labels include: \"Account Access\", \"Billing\", ...\n"
        "=> verdict=Incorrect, score=0, suggested_label=\"Account Access\".\n\n"
        "Example 2 (Partial -> choose better allowed label):\n"
        "- Transcript: user asks to cancel plan and stop future charges.\n"
        "- Predicted label: \"Account\"\n"
        "- Allowed labels include: \"Billing\", \"Account Changes\", \"Account Access\", ...\n"
        "=> verdict=Partial, score=0.5, suggested_label=\"Billing\" (or \"Account Changes\" if that is the best match).\n\n"
        "Example 3 (When to choose FREE-TEXT suggested_label):\n"
        "- Transcript: user reports a brand-new issue not represented in taxonomy (e.g., a new feature name or outage type).\n"
        "- Predicted label: any existing label\n"
        "- None of the allowed labels fit without being misleading.\n"
        "=> verdict=Incorrect, score=0, suggested_label=\"<short free-text label in taxonomy style>\".\n"
        "  (Use free-text ONLY in this situation.)\n\n"
        "Be conservative: do not mark Correct unless clearly supported by the transcript.\n"
    )


def _allowed_block(title: str, allowed: List[str], max_items: int = 2000) -> str:
    allowed = [str(x) for x in allowed if str(x).strip()]
    allowed = allowed[: int(max_items)]
    return f"{title}:\n" + "\n".join(f"- {x}" for x in allowed)


def _transcript_block(transcript: str) -> str:
    t = (transcript or "").strip()
    return f'Transcript:\n"""\n{t}\n"""'


def _flatten_values(d: dict) -> List[str]:
    out: List[str] = []
    for _, v in d.items():
        if not v:
            continue
        out.extend(list(v))
    return out


# -------------------------
# System prompts (per stage)
# -------------------------


def system_prompt_grade_contact_type() -> str:
    return (
        "You are an expert evaluator grading a classifier's predicted CONTACT_TYPE.\n"
        + _grading_rules_block()
        + _json_rule_block()
    )


def system_prompt_grade_domain() -> str:
    return (
        "You are an expert evaluator grading a classifier's predicted DOMAIN.\n"
        + _grading_rules_block()
        + _json_rule_block()
    )


def system_prompt_grade_subdomain() -> str:
    return (
        "You are an expert evaluator grading a classifier's predicted SUBDOMAIN.\n"
        + _grading_rules_block()
        + _json_rule_block()
    )


def system_prompt_grade_root_cause() -> str:
    return (
        "You are an expert evaluator grading a classifier's predicted ROOT_CAUSE.\n"
        + _grading_rules_block()
        + _json_rule_block()
    )


def system_prompt_grade_contact_driver() -> str:
    return (
        "You are an expert evaluator grading a classifier's predicted CONTACT_DRIVER.\n"
        + _grading_rules_block()
        + _json_rule_block()
    )


# ----------------------
# User prompts (per stage)
# ----------------------


def user_prompt_grade_contact_type(
    *,
    transcript: str,
    predicted_contact_type: str,
) -> str:
    return "\n\n".join(
        [
            f"Predicted CONTACT_TYPE: {predicted_contact_type}",
            _allowed_block("Allowed CONTACT_TYPE labels", CONTACT_TYPES_CANON),
            _transcript_block(transcript),
        ]
    )


def user_prompt_grade_domain(
    *,
    transcript: str,
    predicted_domain: str,
    predicted_contact_type: Optional[str] = None,
) -> str:
    parts: List[str] = []
    if predicted_contact_type:
        parts.append(f"Context CONTACT_TYPE (classifier output): {predicted_contact_type}")
    parts.extend(
        [
            f"Predicted DOMAIN: {predicted_domain}",
            _allowed_block("Allowed DOMAIN labels", DOMAINS_CANON),
            _transcript_block(transcript),
        ]
    )
    return "\n\n".join(parts)


def user_prompt_grade_subdomain(
    *,
    transcript: str,
    predicted_subdomain: str,
    predicted_domain: Optional[str] = None,
) -> str:
    all_subdomains = _flatten_values(SUBDOMAINS_BY_DOMAIN_CANON)
    parts: List[str] = []
    if predicted_domain:
        parts.append(f"Context DOMAIN (classifier output): {predicted_domain}")
    parts.extend(
        [
            f"Predicted SUBDOMAIN: {predicted_subdomain}",
            _allowed_block("Allowed SUBDOMAIN labels (full taxonomy)", all_subdomains),
            _transcript_block(transcript),
        ]
    )
    return "\n\n".join(parts)


def user_prompt_grade_root_cause(
    *,
    transcript: str,
    predicted_root_cause: str,
    predicted_subdomain: Optional[str] = None,
) -> str:
    all_root_causes = _flatten_values(ROOT_CAUSES_BY_SUBDOMAIN_CANON)
    parts: List[str] = []
    if predicted_subdomain:
        parts.append(f"Context SUBDOMAIN (classifier output): {predicted_subdomain}")
    parts.extend(
        [
            f"Predicted ROOT_CAUSE: {predicted_root_cause}",
            _allowed_block("Allowed ROOT_CAUSE labels (full taxonomy)", all_root_causes),
            _transcript_block(transcript),
        ]
    )
    return "\n\n".join(parts)


def user_prompt_grade_contact_driver(
    *,
    transcript: str,
    predicted_contact_driver: str,
) -> str:
    return "\n\n".join(
        [
            f"Predicted CONTACT_DRIVER: {predicted_contact_driver}",
            _allowed_block("Allowed CONTACT_DRIVER labels", CONTACT_DRIVERS_CANON),
            _transcript_block(transcript),
        ]
    )