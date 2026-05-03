
# -*- coding: utf-8 -*-
"""
Normalization + canonicalization helpers for CCC classifier.

Design choices:
- Keep "No Customer Input"
- Do NOT use any "Unclassified ..." labels
- If a model output does not match a canonical label, we fall back to:
  "Other: <cleaned free text>".
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Optional

from .dictionaries import (
    CONTACT_TYPES_CANON,
    DOMAINS_CANON,
    SUBDOMAINS_BY_DOMAIN_CANON
)

# -----------------------------
# Normalization helpers
# -----------------------------
_NORM_SPACE_RE = re.compile(r"\s+")
_NORM_PUNCT_RE = re.compile(r"[^a-z0-9&/\- ]+")  # keep & / - and spaces

def _normalize_label(s: str) -> str:
    """
    Normalize a label so that minor differences (case, punctuation, extra spaces)
    map back to the canonical label.

    Example:
      " Order  &  Fulfillment " -> "order & fulfillment"
      "Return Authorization (RMA)" -> "return authorization rma"
    """
    if not s:
        return ""
    s = html.unescape(s)
    # normalize dashes
    s = s.replace("—", "-").replace("–", "-")
    s = s.strip().lower()

    # normalize "and" -> "&" (helps canonical matching)
    s = re.sub(r"\band\b", "&", s)

    # collapse whitespace
    s = _NORM_SPACE_RE.sub(" ", s).strip()

    # strip punctuation except our allowed set
    s = _NORM_PUNCT_RE.sub(" ", s)
    s = _NORM_SPACE_RE.sub(" ", s).strip()
    return s


def _build_norm_map(values: List[str]) -> Dict[str, str]:
    """
    Build a dictionary mapping normalized_label -> canonical_label.
    """
    out: Dict[str, str] = {}
    for v in values:
        out[_normalize_label(v)] = v
    return out


def _build_nested_norm_map(d: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Build a dictionary mapping:
      key -> (normalized_label -> canonical_label)
    for nested structures (e.g., domain -> subdomains list).
    """
    return {k: _build_norm_map(vs) for k, vs in d.items()}


# -----------------------------
# Canonical maps (precomputed)
# -----------------------------
CONTACT_TYPES_MAP: Dict[str, str] = _build_norm_map(CONTACT_TYPES_CANON)
DOMAINS_MAP: Dict[str, str] = _build_norm_map(DOMAINS_CANON)
SUBDOMAINS_MAP_BY_DOMAIN: Dict[str, Dict[str, str]] = _build_nested_norm_map(SUBDOMAINS_BY_DOMAIN_CANON)


def canonicalize(value: str, allowed_map: Dict[str, str]) -> Optional[str]:
    """
    Return canonical label if value matches one of the allowed values (after normalization),
    else return None.
    """
    if not value:
        return None
    norm = _normalize_label(value)
    return allowed_map.get(norm)


def canonicalize_in_context(value: str, allowed_values: List[str]) -> Optional[str]:
    """
    Canonicalize using an explicit list of allowed values.
    Helpful for domain->subdomain and subdomain.
    """
    if not value:
        return None
    local_map = _build_norm_map(allowed_values)
    return canonicalize(value, local_map)


# -----------------------------
# "Other: <free text>" helpers
# -----------------------------
OTHER_PREFIX = "Other:"
_OTHER_PREFIX_RE = re.compile(r"^\s*other\s*[:\-–—]\s*", re.IGNORECASE)

# Keep common acronyms uppercase in "Other:" free text
_ACRONYMS = {"AR", "RMA", "POD", "ETA", "MFA", "SSO", "SKU", "EDI", "DEA", "NDC", "OTC"}

def other_free_text(raw: str, max_words: int = 5) -> str:
    """
    Return canonical free-text in format: 'Other: <Sentence case, <=max_words>'.

    Rules:
    - Always returns a string beginning with "Other:"
    - If raw is empty/unusable, returns "Other: Unspecified"
    - Removes any existing "Other:" prefix variants from raw
    - Keeps acronyms uppercase (ETA, RMA, etc.)
    """
    raw = (raw or "").strip()
    if not raw:
        return f"{OTHER_PREFIX} Unspecified"

    # Remove any "Other:" prefix the model might already include
    content = _OTHER_PREFIX_RE.sub("", raw).strip()
    content = html.unescape(content)
    content = content.replace("—", "-").replace("–", "-")
    content = _NORM_SPACE_RE.sub(" ", content).strip()

    # Remove weird characters but allow letters/digits/&/-/spaces
    content = re.sub(r"[^A-Za-z0-9&/\- ]+", " ", content)
    content = _NORM_SPACE_RE.sub(" ", content).strip()

    if not content:
        return f"{OTHER_PREFIX} Unspecified"

    words = content.split()
    words = words[:max_words]

    # sentence case with acronym preservation
    lowered = [w.lower() for w in words]
    fixed = [w.upper() if w.upper() in _ACRONYMS else w for w in lowered]

    # Capitalize first token if it's not an acronym
    if fixed and fixed[0] and fixed[0].upper() not in _ACRONYMS:
        fixed[0] = fixed[0][0].upper() + fixed[0][1:]

    content_out = " ".join(fixed).strip()
    if not content_out:
        return f"{OTHER_PREFIX} Unspecified"

    return f"{OTHER_PREFIX} {content_out}"


def canonical_or_other(value: str, allowed_values: List[str], max_words: int = 5) -> str:
    """
    Convenience helper:
    - If value canonicalizes to one of allowed_values -> return canonical label
    - Else -> return "Other: <cleaned free text>"

    Use this in stages when you want strict canonical matching.
    """
    canon = canonicalize_in_context((value or "").strip(), allowed_values)
    return canon if canon else other_free_text(value, max_words=max_words)


def canonical_domain_or_other(value: str) -> str:
    """
    Canonicalize domain against DOMAINS_CANON; otherwise return Other: ...
    """
    canon = canonicalize(value, DOMAINS_MAP)
    return canon if canon else other_free_text(value, max_words=5)


def canonical_contact_type_or_other(value: str) -> str:
    """
    Canonicalize contact type against CONTACT_TYPES_CANON; otherwise return Other: ...
    NOTE: You may decide later that contact_type must be strict; for now this follows your rule.
    """
    canon = canonicalize(value, CONTACT_TYPES_MAP)
    return canon if canon else other_free_text(value, max_words=3)
