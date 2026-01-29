
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
    SUBDOMAINS_BY_DOMAIN_CANON,
    ROOT_CAUSES_BY_SUBDOMAIN_CANON,
    CONTACT_DRIVERS_CANON,
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





def system_prompt_root_cause(subdomain: str) -> str:
    causes: List[str] = ROOT_CAUSES_BY_SUBDOMAIN_CANON.get(subdomain, [])
    allowed_line = ", ".join(causes) if causes else ""
    pref = f"Allowed root causes (choose ONE): [{allowed_line}].\n" if causes else ""

    return (
        "You are an expert classifier for customer support transcripts.\n"
        f"Task: Given subdomain='{subdomain}', choose exactly ONE root_cause.\n"
        + pref +
        "Rules:\n"
        "- Your output MUST be valid JSON with keys: root_cause, confidence.\n"
        # "- If an allowed list is provided above, you MUST choose a root_cause that is EXACTLY one of those strings.\n"
        # "- Do NOT paraphrase. Do NOT invent new labels.\n"
        # "- If the transcript wording differs, MAP it to the closest allowed root cause.\n"
        # "- Choose the most specific single cause supported by the transcript.\n"
        # "- Use 'Other: <free text>' ONLY if NONE of the allowed root causes closely match the intent.\n"
        # "- Keep free text concise (<= 8 words).\n"
        " - You MUST return EITHER:\n"
        "  (a) one EXACT allowed root cause string (preferred), OR\n"
        "  (b) 'Other: <free text>' ONLY when none of the allowed causes reasonably match.\n"
        "- If the allowed list is empty, you MUST return 'Other: <free text>' (<= 8 words).\n"
        "- If the customer never spoke, choose 'No Customer Input'.\n"
        "- SELF-CHECK RULE: Before responding, verify your root_cause is EXACTLY one of the allowed strings.\n"
        "  If not, replace it with the closest allowed cause.\n"
        "\n"
        "Important mapping examples:\n"
        "- 'Wrong tote labelling', 'Incorrect tote label', 'Tote mislabeled' -> 'Incorrect tote labeling'\n"
        "- 'Stop receiving price stickers' -> 'Stop OTC price stickers'\n"
        "- 'Cannot locate credit', 'Unable to find credit invoice', 'Missing credit notes' -> 'Unable to locate credit'\n"
        "- 'Remove items from order', 'Delete item from order', 'Cannot edit open order' -> 'Order modification not possible'\n"
        "- 'Order locked', 'Cannot unlock order', 'Unlock order request' -> 'Order unlock needed'\n"
        "- 'Entered full unit instead of partial unit', 'Return unit code U/Z/P/C confusion', "
        "'Wrong return unit code' -> 'Clarification on return codes / units'\n"
        "\n"
        "- These examples illustrate how to map similar meaning phrases to EXACT canonical labels.\n"
        + _json_rule_block("root_cause")
        + "Also include numeric key 'confidence' between 0 and 1.\n"
    )



def system_prompt_contact_driver() -> str:
    allowed = ", ".join(CONTACT_DRIVERS_CANON)
    return (
        "You are an expert classifier for customer support transcripts.\n"
        f"Task: Choose exactly ONE situational contact_driver from: [{allowed}].\n"
        "Definition:\n"
        "- A contact_driver is the earliest explicit customer-observed stimulus or need that triggered contacting support.\n"
        "- It is NOT the detailed root cause or the requested action.\n"
        "Rules:\n"
        "- Output MUST be valid JSON with keys: contact_driver, confidence.\n"
        "- You MUST choose a contact_driver that is EXACTLY one of the allowed labels.\n"
        "- Do NOT paraphrase. Do NOT invent new labels.\n"
        "- If transcript wording differs, MAP it to the closest allowed contact_driver.\n"
        "- Pick the earliest trigger mentioned by the customer (first stimulus/need), not later troubleshooting.\n"
        "- If multiple triggers are mentioned, choose the earliest one in the transcript.\n"
        "- If customer never spoke or only trivial acknowledgements, choose 'No Customer Input'.\n"
        "- Use 'Other: <free text>' ONLY if NONE of the allowed labels reasonably match.\n"
        "- If you are uncertain between a specific allowed label and using Other, YOU MUST choose the closest allowed label.\n"
        "- SELF-CHECK: ensure your chosen label appears verbatim in the allowed list.\n"
        "\n"
        "Mapping examples (map request-style phrases to stimulus drivers):\n"
        "- 'remove items from order', 'cancel order', 'cancel transaction', 'accidental order placement' -> Urgent / TimeSensitive Need\n"
        "- 'checking case status', 'status update', 'return status inquiry' -> Saw No Progress\n"
        "- 'need invoice number', 'missing invoice', 'need return label/bag' -> Saw Missing Document\n"
        "- 'confirm receipt of POD/email', 'no confirmation received' -> Saw Missing Confirmation\n"
        "- 'wrong tote label', 'received unordered shipment', 'missing/extra item' -> Saw Item / Quantity Mismatch\n"
        "- 'store closed', 'holiday closure', 'closed on specific date' -> Saw Unexpected Status Change\n"
        "- 'promo/banner message' -> Saw Portal Message / Banner\n"
        "- 'portal error', 'site malfunction' -> Saw Portal Error / Malfunction\n"
        "- 'how do I...', 'asked about cutoff time', 'need instructions' -> Was Unfamiliar with Next Step\n"
        "- 'schedule return pickup', 'add delivery day', 'reconnect with prior agent' -> Was Unfamiliar with Next Step\n"
        "- 'close case', 'close claim' -> Saw No Progress\n"
        "- 'return overstock', 'overstock return request' -> Saw Eligibility / Window Denial\n"
        "- 'account change', 'add pharmacist user', 'remove location', 'access/role change' -> Needed Account / User Change\n"
        "- 'expressing appreciation', 'thank you' -> Wanted to Provide Feedback\n"
        "- 'is this in stock', 'available at any warehouse', 'ETA', 'when will it arrive', 'backorder', 'allocation', 'filter', 'can you order this item' -> Needed Availability / Stock Clarity\n"
        "- 'expiry date', 'expiration', 'lot number', 'dating', 'better expiry', 'short-dated', 'need longer dating' -> Needed Expiry / Lot / Dating Clarity\n"
        "- 'damaged', 'leaking', 'broken seal', 'defective', 'contaminated', 'tabs discolored', 'package opened', 'product malfunctioning' -> Observed Product Defect / Damage\n"
        "- 'claim window expired', 'credit denied due to policy', 'return window passed', 'cannot credit because too late' -> Saw Eligibility / Window Denial\n"
        "- 'forward email', 'please send this email', 'upload paperwork', 'submit documents', 'send photos/pictures' -> Needed Document Submission Help\n"
        "\n"
        + _json_rule_block("contact_driver")
        + "Also include numeric key 'confidence' between 0 and 1.\n"
    )




def system_prompt_short_summary() -> str:
    return (
        "You are an expert summarizer for customer support transcripts.\n"
        "Task: Write a concise summary describing the key context of this case.\n"
        "Rules:\n"
        "- The summary should be brief (maximum 20 words).\n"
        "- Focus on the main object and the issue or request.\n"
        "- Use clear and simple language.\n"
        "- Avoid including any personal identifiable information (PII).\n"
        "- Do not include any speculation or assumptions.\n"
        + _json_rule_block("case_context")
    )

