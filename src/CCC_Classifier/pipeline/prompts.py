
# -*- coding: utf-8 -*-
"""
Prompt builders for CCC classifier stages.

Design choices:
- No remapping / synonym logic in prompts.
- No "Unclassified ..." labels.
- If no label fits, model must return: "Other: <short free text>"
- Keep "No Customer Input" for cases where customer never provided intent.
- Output must be valid JSON only (single object).
"""

from __future__ import annotations

from typing import List

from CCC_Classifier.taxonomy.dictionaries import (
    CONTACT_TYPES_CANON,
    DOMAINS_CANON,
    SUBDOMAINS_BY_DOMAIN_CANON
)



def _json_rule_block(required_key: str) -> str:
    return (
        f"Output MUST be a single JSON object with keys: '{required_key}' and 'confidence'.\n"
        f"- '{required_key}' must be a single string.\n"
        "- 'confidence' must be a number between 0 and 1.\n"
        "Do NOT include markdown, code fences, comments, or any extra keys.\n"
        "Do NOT wrap JSON in text.\n"
    )



def _other_rule(max_words: int) -> str:
    return (
        f"If none of the labels fit, output a short free-text label formatted as:\n"
        f"  'Other: <max {max_words} words>'\n"
        "Use title/sentence case and keep it concise.\n"
    )


def system_prompt_contact_type() -> str:
    allowed = ", ".join(CONTACT_TYPES_CANON)
    return (
        "You are an expert classifier for customer support transcripts.\n"
        f"Task: Choose exactly ONE contact_type from: [{allowed}].\n"
        "Definitions:\n"
        "- Issue: Something went wrong or failed; expects investigation/correction.\n"
        "- Inquiry: Asking for information/status/policy/how-to; not claiming a failure.\n"
        "- Request: Asking for an action (cancel/modify/resend/update) without a primary failure.\n"
        "- Unclear Contact: Not enough signal to decide.\n"
        "Rules:\n"
        "- Do not guess. If unclear, choose 'Unclear Contact'.\n"
        "- If customer never spoke or only trivial acknowledgements, choose 'Unclear Contact'.\n"
        + _json_rule_block("contact_type")
        + "Also include numeric key 'confidence' between 0 and 1.\n"
    )


def system_prompt_domain() -> str:
    allowed = ", ".join(DOMAINS_CANON)
    return (
        "You are an expert classifier for customer support transcripts.\n"
        f"Task: Choose exactly ONE domain from: [{allowed}].\n"
        "Definitions:\n"
        "- Product: Item attributes/quality/availability.\n"
        "- Billing: Charges, invoices, payments, credits/refunds, pricing, reports.\n"
        "- Order & Fulfillment: Placing/modifying/canceling orders; shipping/tracking; delivery timing; shortages/wrong items.\n"
        "- Returns: Return workflow, eligibility, RMA, labels, return status.\n"
        "- Technical Support: Portal/app errors, authentication/access issues, outages.\n"
        "- Portal Guidance: How-to guidance, navigation help, portal information (not errors).\n"
        "- Case Management: Callback, case status, case closure, reconnect, contact info.\n"
        "- Programs & Rewards: Programs, campaigns, rebates, rewards cards.\n"
        "- Policy & Compliance: Regulatory/policy documentation, audits, controlled items, closures/notices.\n"
        "- Customer Feedback: Praise/complaints/feedback about service experience.\n"
        "- No Customer Input: Use only when customer never spoke or only trivial acknowledgements.\n"
        "Rules:\n"
        "- Choose the best single fit.\n"
        + _other_rule(max_words=5)
        + _json_rule_block("domain")
        + "Also include numeric key 'confidence' between 0 and 1.\n"
    )



def system_prompt_subdomain(domain: str) -> str:
    subdomains: List[str] = SUBDOMAINS_BY_DOMAIN_CANON.get(domain, [])
    allowed_line = ", ".join(subdomains) if subdomains else ""
    pref = f"Allowed subdomains (choose ONE): [{allowed_line}].\n" if allowed_line else ""

    return (
        "You are an expert classifier for customer support transcripts.\n"
        f"Task: Given domain='{domain}', choose exactly ONE subdomain.\n"
        + pref +
        "Rules:\n"
        "- You MUST return a 'subdomain' that is EXACTLY one of the allowed subdomains listed above.\n"
        "- Do NOT paraphrase. Do NOT invent new labels.\n"
        "- If the transcript uses different wording, map it to the closest allowed subdomain.\n"
        "- Use 'Other: <free text>' ONLY if NONE of the allowed subdomains closely match the intent.\n"
        "- If you output 'Other:', keep the free text concise (<= 5 words) and describe the missing subdomain intent.\n"
        "- Before responding, double-check: if your chosen subdomain is not an exact string match to an allowed label,\n"
        "  then pick the closest allowed label instead (unless truly no close match exists).\n"
        + _json_rule_block("subdomain")
        + "Also include numeric key 'confidence' between 0 and 1.\n"
    )


def system_prompt_SHORT_SUMMARY() -> str:
    return (
        "You are an expert summarizer for customer support transcripts.\n"
        "Task: Write a concise summary describing the key context of this case.\n"
        "Rules:\n"
        "- The summary should be brief (maximum 20 words).\n"
        "- Focus on the main object and the issue or request.\n"
        "- Use clear and simple language.\n"
        "- Avoid including any personal identifiable information (PII).\n"
        "- Do not include any speculation or assumptions.\n"
        + _json_rule_block("SHORT_SUMMARY")
    )


def system_prompt_DETAILED_SUMMARY() -> str:
    return (
        "You are an expert summarizer for customer support transcripts.\n"
        "Task: Write a high-level, structured summary capturing only the essential context of this case.\n"
        "Rules:\n"
        "- Output ONE paragraph containing EXACTLY these four labeled sentences in this order:\n"
        " Key topics discussed: <one sentence>.\n"
        " Important decisions made: <one sentence>.\n"
        " Action items or next steps: <one sentence>.\n"
        " Any unresolved questions or issues: <one sentence>.\n"
        "- Each labeled sentence must be <= 15 words.\n"
        "- Each sentence may describe only ONE primary idea.\n"
        "- Base the summary ONLY on explicit statements in the transcript; do not infer or speculate.\n"
        "- Mention the primary object (order, shipment, invoice, product, return, or account) only once.\n"
        "- Focus on outcomes, not procedures or step-by-step actions.\n"
        "- Avoid instructional or procedural language (e.g., fill out, navigate, submit via, count, click).\n"
        "- Reduce content by selecting only the most important details; prefer omission over inclusion.\n"
        "- If the interaction is a neutral inquiry, do not imply a failure or issue.\n"
        "- Use clear, simple language.\n"
        "- Avoid personal identifiable information (PII).\n"
        "- If no unresolved issues remain, write exactly: \"None.\"\n"
        "- If the transcript lacks sufficient information, set DETAILED_SUMMARY exactly to:\n"
        "\"Insufficient information to generate a detailed summary.\"\n"
        "\n"
        "Example (valid JSON):\n"
        "{\n"
        " \"DETAILED_SUMMARY\": \"Key topics discussed: Return of recalled product without original packaging. "
        "Important decisions made: Agent approved return under recall policy. "
        "Action items or next steps: Customer to submit required return documentation. "
        "Any unresolved questions or issues: None.\",\n"
        " \"confidence\": 0.95\n"
        "}\n"
        + _json_rule_block("DETAILED_SUMMARY")
    )



