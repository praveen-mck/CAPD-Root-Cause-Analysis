
# -*- coding: utf-8 -*-
"""
Pipeline orchestrator: analyze a single transcript.

Responsibilities:
- If no customer input (rule-based), short-circuit with "No Customer Input" outputs
- Otherwise run stages in sequence:
    contact_type -> domain -> subdomain -> root_cause -> contact_driver -> SHORT_SUMMARY
- Compute overall confidence (min of stage confidences)
- Return a single result dict

Design choices (per your direction):
- Keep no_input.py as-is for now.
- No remapping/synonyms.
- No "Unclassified ..." labels.
- Anything not matching canonical lists becomes "Other: ..." (handled inside stages via canon.py).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from CCC_Classifier.pipeline.stages import (
    stage_SHORT_SUMMARY,
    stage_DETAILED_SUMMARY,
    stage_contact_driver,
    stage_contact_type,
    stage_domain,
    stage_root_cause,
    stage_subdomain,
)
from CCC_Classifier.utils.no_input import is_no_customer_input

logger = logging.getLogger(__name__)

def _min_conf(*vals: Optional[float]) -> float:
    """
    Compute a conservative overall confidence = min of available confidences.
    Missing/None confidences treated as 0.0.
    """
    cleaned = []
    for v in vals:
        try:
            cleaned.append(float(v))
        except Exception:
            cleaned.append(0.0)
    return round(min(cleaned) if cleaned else 0.0, 3)


def _no_input_result() -> Dict[str, Any]:
    """
    Standard output when there is no meaningful customer input.
    Keeps the behavior consistent and deterministic.
    """
    return {
        "contact_type": "Unclear Contact",  # keep existing behavior
        "domain": "No Customer Input",
        "subdomain": "No Customer Input",
        "root_cause": "No Customer Input",
        "contact_driver": "No Customer Input",
        "SHORT_SUMMARY": "Customer did not provide sufficient input to agent.",
        "DETAILED_SUMMARY": "Customer did not provide sufficient input to agent.",
        "confidence": 1.0,
        "IS_NO_INPUT": 1,
    }


async def analyze_transcript(
    *,
    client: Any,
    deployment: str,
    transcript_text: str,
    max_completion_tokens: int = 512,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a single transcript and return:
      {
        contact_type, domain, subdomain, root_cause,
        contact_driver, SHORT_SUMMARY, DETAILED_SUMMARY, confidence, IS_NO_INPUT
      }

    Notes:
- This function does not do any Snowflake IO.
- It is called by batch.py for each row.
    """
    transcript = (transcript_text or "").strip()

    # 1) Short-circuit for no customer input (rule-based, kept as-is)
    if not transcript or is_no_customer_input(transcript):
        return _no_input_result()

    # Initialize stage variable to be used in exception message
    stage = ''

    # 2) Run stages sequentially
    #    We keep it simple: run domain even if contact type is "Unclear Contact",
    #    because you removed "Unclassified" and want Others captured instead of blank placeholders.
    try:
        stage = 'contact_type'
        ct = await stage_contact_type(
            client=client,
            deployment=deployment,
            transcript=transcript,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        contact_type = ct.get("contact_type", "Unclear Contact")
        ct_conf = ct.get("confidence", 0.0)

        stage = 'domain'
        dom = await stage_domain(
            client=client,
            deployment=deployment,
            transcript=transcript,
            contact_type=contact_type,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        domain = dom.get("domain", "")
        dom_conf = dom.get("confidence", 0.0)

        # If the model says No Customer Input here, treat as no-input overall.
        # (This should be rare since we already used is_no_customer_input(), but safe to support.)
        if domain == "No Customer Input":
            return _no_input_result()

        stage = 'subdomain'
        sub = await stage_subdomain(
            client=client,
            deployment=deployment,
            transcript=transcript,
            domain=domain,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        subdomain = sub.get("subdomain", "")
        sub_conf = sub.get("confidence", 0.0)

        stage = 'root_cause'
        rc = await stage_root_cause(
            client=client,
            deployment=deployment,
            transcript=transcript,
            subdomain=subdomain,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        root_cause = rc.get("root_cause", "")
        rc_conf = rc.get("confidence", 0.0)

        stage = 'contact_driver'
        drv = await stage_contact_driver(
            client=client,
            deployment=deployment,
            transcript=transcript,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        contact_driver = drv.get("contact_driver", "")
        drv_conf = drv.get("confidence", 0.0)

        stage = 'short_summary'
        ssy = await stage_SHORT_SUMMARY(
            client=client,
            deployment=deployment,
            transcript=transcript,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        SHORT_SUMMARY = ssy.get("SHORT_SUMMARY", "Context Unspecified")

        stage = 'detailed_summary'
        dsy = await stage_DETAILED_SUMMARY(
            client=client,
            deployment=deployment,
            transcript=transcript,
            max_completion_tokens=max_completion_tokens,
            use_json_mode=use_json_mode,
        )
        DETAILED_SUMMARY = dsy.get("DETAILED_SUMMARY", "Context Unspecified")

        overall_conf = _min_conf(ct_conf, dom_conf, sub_conf, rc_conf, drv_conf)

        return {
            "contact_type": contact_type,
            "domain": domain,
            "subdomain": subdomain,
            "root_cause": root_cause,
            "contact_driver": contact_driver,
            "SHORT_SUMMARY": SHORT_SUMMARY,
            "DETAILED_SUMMARY": DETAILED_SUMMARY,
            "confidence": overall_conf,
            "IS_NO_INPUT": 0,
        }

    except Exception:
        preview = transcript[:300].replace("\n", " ")
        logger.exception(
            "exception analyze_transcript failed at stage=%s max_completion_tokens=%s use_json_mode=%s transcript_len=%s preview=%r",
            stage,
            max_completion_tokens,
            use_json_mode,
            len(transcript),
            preview,
        )
        # Keep the fallback simple and aligned with your "Other" philosophy:
        # we don't return any "Unclassified" labels.
        return {
            "contact_type": "Unclear Contact",
            "domain": "Other: Unspecified",
            "subdomain": "Other: Unspecified",
            "root_cause": "Other: Unspecified",
            "contact_driver": "Other: Unspecified",
            "SHORT_SUMMARY": "Context Unspecified",
            "DETAILED_SUMMARY": "Context Unspecified",
            "confidence": 0.0,
            "IS_NO_INPUT": 0,
        }
