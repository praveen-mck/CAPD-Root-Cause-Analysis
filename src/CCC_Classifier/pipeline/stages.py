
# -*- coding: utf-8 -*-
"""
Stage wrappers for CCC classifier.

Each stage does:
  prompt -> LLM call -> parse JSON -> canonicalize (or Other) -> return structured dict

Design choices:
- No remapping/synonyms.
- No "Unclassified ..." labels.
- If output doesn't match canonical allowed set, return "Other: <free text>".
- "No Customer Input" is allowed, but the decision of short-circuiting lives in orchestrator.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from CCC_Classifier.llm.client import send_chat_request
from CCC_Classifier.llm.parsing import clamp_conf, extract_content_and_usage, get_json_field, safe_parse_json
from CCC_Classifier.pipeline.prompts import (
    system_prompt_case_context,
    system_prompt_contact_driver,
    system_prompt_contact_type,
    system_prompt_domain,
    system_prompt_root_cause,
    system_prompt_subdomain,
)
from CCC_Classifier.taxonomy.canon import (
    canonical_domain_or_other,
    canonical_driver_or_other,
    canonical_or_other,
    canonicalize,
    other_free_text,
    CONTACT_TYPES_MAP,
)
from CCC_Classifier.taxonomy.dictionaries import (
    CONTACT_TYPES_CANON,
    SUBDOMAINS_BY_DOMAIN_CANON,
    ROOT_CAUSES_BY_SUBDOMAIN_CANON,
)


def _user_transcript_block(transcript: str, prefix: Optional[str] = None) -> str:
    """
    Common user message wrapper sent to the LLM.
    """
    t = (transcript or "").strip()
    if prefix:
        return f"{prefix}\nTranscript:\n{t}"
    return f"Transcript:\n{t}"


def _as_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x).strip() if x is not None else ""


def _strict_contact_type(raw_value: str) -> str:
    """
    Contact type is intentionally strict:
    - Must be one of CONTACT_TYPES_CANON
    - Otherwise fallback to "Unclear Contact"
    """
    canon = canonicalize(raw_value, CONTACT_TYPES_MAP)
    return canon if canon in CONTACT_TYPES_CANON else "Unclear Contact"


async def stage_contact_type(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    max_completion_tokens: int = 256,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "contact_type": <canonical contact type>,
        "confidence": <0..1>,
        "_usage": {...},
        "_finish": "...",
      }
    """
    sys_text = system_prompt_contact_type()
    user_text = _user_transcript_block(transcript)

    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=sys_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )

    content, usage, finish = extract_content_and_usage(resp)
    data = safe_parse_json(content)

    raw = _as_str(get_json_field(data, "contact_type", "Unclear Contact"))
    conf = clamp_conf(get_json_field(data, "confidence", 0.0))

    out = _strict_contact_type(raw)

    return {"contact_type": out, "confidence": conf, "_usage": usage, "_finish": finish}


async def stage_domain(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    contact_type: str,
    max_completion_tokens: int = 256,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "domain": <canonical domain or Other: ...>,
        "confidence": <0..1>,
        "_usage": {...},
        "_finish": "...",
      }
    """
    sys_text = system_prompt_domain()
    user_text = _user_transcript_block(transcript, prefix=f"Contact type (context): {contact_type}")

    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=sys_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )

    content, usage, finish = extract_content_and_usage(resp)
    data = safe_parse_json(content)

    raw = _as_str(get_json_field(data, "domain", ""))
    conf = clamp_conf(get_json_field(data, "confidence", 0.0))

    # Domain: canonical or "Other: ..."
    domain = canonical_domain_or_other(raw)

    return {"domain": domain, "confidence": conf, "_usage": usage, "_finish": finish}


async def stage_subdomain(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    domain: str,
    max_completion_tokens: int = 256,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "subdomain": <canonical subdomain for the given domain OR Other: ...>,
        "confidence": <0..1>,
        "_usage": {...},
        "_finish": "...",
      }
    """
    sys_text = system_prompt_subdomain(domain)
    user_text = _user_transcript_block(transcript, prefix=f"Domain: {domain}")

    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=sys_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )

    content, usage, finish = extract_content_and_usage(resp)
    data = safe_parse_json(content)

    raw = _as_str(get_json_field(data, "subdomain", ""))
    conf = clamp_conf(get_json_field(data, "confidence", 0.0))

    allowed: List[str] = SUBDOMAINS_BY_DOMAIN_CANON.get(domain, [])
    # If domain is "Other: ..." there is no subdomain list; default to Other for subdomain.
    if allowed:
    #     subdomain = other_free_text(raw, max_words=5) 
    # else:
        subdomain = canonical_or_other(raw, allowed_values=allowed, max_words=5)

    return {"subdomain": subdomain, "confidence": conf, "_usage": usage, "_finish": finish}


async def stage_root_cause(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    subdomain: str,
    max_completion_tokens: int = 256,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "root_cause": <canonical root cause for the given subdomain OR Other: ...>,
        "confidence": <0..1>,
        "_usage": {...},
        "_finish": "...",
      }
    """
    sys_text = system_prompt_root_cause(subdomain)
    user_text = _user_transcript_block(transcript, prefix=f"Subdomain: {subdomain}")

    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=sys_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )

    content, usage, finish = extract_content_and_usage(resp)
    data = safe_parse_json(content)

    raw = _as_str(get_json_field(data, "root_cause", ""))
    conf = clamp_conf(get_json_field(data, "confidence", 0.0))

    allowed: List[str] = ROOT_CAUSES_BY_SUBDOMAIN_CANON.get(subdomain, [])
    # If no defined root causes for that subdomain, root cause must be Other.
    if allowed:
    #     root_cause = other_free_text(raw, max_words=8)
    # else:
        root_cause = canonical_or_other(raw, allowed_values=allowed, max_words=8)

    return {"root_cause": root_cause, "confidence": conf, "_usage": usage, "_finish": finish}


async def stage_contact_driver(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    max_completion_tokens: int = 256,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "contact_driver": <canonical driver or Other: ...>,
        "confidence": <0..1>,
        "_usage": {...},
        "_finish": "...",
      }
    """
    sys_text = system_prompt_contact_driver()
    user_text = _user_transcript_block(transcript)

    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=sys_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )

    content, usage, finish = extract_content_and_usage(resp)
    data = safe_parse_json(content)

    raw = _as_str(get_json_field(data, "contact_driver", ""))
    conf = clamp_conf(get_json_field(data, "confidence", 0.0))

    driver = canonical_driver_or_other(raw)

    return {"contact_driver": driver, "confidence": conf, "_usage": usage, "_finish": finish}




async def stage_case_context(
    *,
    client: Any,
    deployment: str,
    transcript: str,
    max_completion_tokens: int = 512,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    sys_text = system_prompt_case_context()
    user_text = _user_transcript_block(transcript)

    resp = await send_chat_request(
        client=client,
        deployment=deployment,
        system_text=sys_text,
        user_text=user_text,
        max_out_tokens=max_completion_tokens,
        use_json_mode=use_json_mode,
    )

    # Grab message object (if present)
    msg = None
    try:
        msg = resp.choices[0].message
    except Exception:
        msg = None

    content, usage, finish = extract_content_and_usage(resp)
    data = safe_parse_json(content)

    ctx = _as_str(get_json_field(data, "case_context", "")) or "Context Unspecified"

    # ---- Only print diagnostics when ctx is unspecified ----
    if ctx == "Context Unspecified":
        reasons = []

        # 1) Finish reason
        if finish:
            reasons.append(f"finish_reason={finish}")

        # 2) Content presence
        if content is None:
            reasons.append("content=None (no assistant content returned)")
        elif isinstance(content, str) and content.strip() == "":
            reasons.append("content=empty_string")

        # 3) JSON parsing / key presence
        if not isinstance(data, dict) or not data:
            reasons.append("parsed_json=empty_or_invalid")
        else:
            if "case_context" not in data:
                reasons.append(f"missing_key=case_context (keys={list(data.keys())})")
            else:
                v = data.get("case_context")
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    reasons.append("case_context_value=empty_or_null")

        # 4) Token usage insights (very useful for your earlier issue)
        try:
            comp_details = (usage or {}).get("completion_tokens_details") or {}
            rp = comp_details.get("reasoning_tokens", None)
            ap = comp_details.get("accepted_prediction_tokens", None)
            if rp is not None or ap is not None:
                reasons.append(f"completion_tokens_details(reasoning={rp}, accepted_prediction={ap})")
        except Exception:
            pass

        # 5) Message structure hints (tool calls, function calls)
        try:
            if msg is not None:
                tc = getattr(msg, "tool_calls", None)
                fc = getattr(msg, "function_call", None)
                # Only note if present (or explicitly None)
                reasons.append(f"tool_calls={'present' if tc else 'none'}")
                reasons.append(f"function_call={'present' if fc else 'none'}")
        except Exception:
            reasons.append("message_inspect_error")

        print("\n[CASE_CONTEXT UNSPECIFIED] " + " | ".join(reasons))
        # Optional: print raw content + parsed json for deep debugging
        print("[CASE_CONTEXT UNSPECIFIED] raw_content:", repr(content))
        print("[CASE_CONTEXT UNSPECIFIED] parsed_json:", data)
        print("[CASE_CONTEXT UNSPECIFIED] usage:", usage)

    return {"case_context": ctx, "_usage": usage, "_finish": finish}
