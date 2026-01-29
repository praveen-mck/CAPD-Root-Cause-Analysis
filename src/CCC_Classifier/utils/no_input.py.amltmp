
# -*- coding: utf-8 -*-
"""
No-customer-input detection utilities.

Purpose:
- Detect when the customer never spoke OR only provided trivial acknowledgements.
- Used to short-circuit the pipeline and assign "No Customer Input" outcomes
  without calling the LLM.

Assumptions (based on your original transcript format):
- Header may contain an agent name like: "Chat Origin: Live Chat Button Agent Wayne"
- Speaker lines may look like: "( 1m 20s ) Name: message"
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Matches very short / trivial customer utterances
_NO_INTENT_SHORT_RE = re.compile(
    r"^(hi|hello|hey|ok|okay|thanks|thank\s+you|thx|yes|no|yep|nope|\?|\."
    r"|…|bye|goodbye)$",
    re.IGNORECASE,
)

# Extract agent name from header like: "... Agent Wayne" or "... Agent ANTHONY"
_AGENT_NAME_RE = re.compile(r"\bAgent\s+([A-Za-z][A-Za-z0-9_\-]*)\b")

# Parse speaker lines like: "( 1m 20s ) Name: message"
_SPEAKER_LINE_RE = re.compile(
    r"\(\s*\d+(?:m\s*)?(?:\d+)?s\s*\)\s*([^:]+):\s*(.+)"
)


def extract_agent_name(transcript: str) -> Optional[str]:
    """
    Extract agent name from header text like:
      "Chat Origin: Live Chat Button Agent Wayne"
    Returns None if not found.
    """
    if not transcript:
        return None
    m = _AGENT_NAME_RE.search(transcript)
    return m.group(1).strip() if m else None


def extract_speaker_lines(transcript: str) -> List[Tuple[str, str]]:
    """
    Extract speaker lines like:
      "( 1m 20s ) Name: message"

    Returns a list of (speaker, message).
    """
    out: List[Tuple[str, str]] = []
    if not transcript:
        return out

    for m in _SPEAKER_LINE_RE.finditer(transcript):
        speaker = (m.group(1) or "").strip()
        msg = (m.group(2) or "").strip()
        if speaker and msg:
            out.append((speaker, msg))
    return out


def is_no_customer_input(transcript: str, max_words: int = 3) -> bool:
    """
    Returns True if:
    - There are no customer messages at all, OR
    - Customer messages exist but are trivial-only and total words <= max_words.

    Logic:
    - Determine agent speaker name from header; if not available, infer agent as first speaker.
    - Consider anyone who is not agent and not SYSTEM/BOT as "customer".
    """
    txt = (transcript or "").strip()
    if not txt:
        return True

    agent = extract_agent_name(txt)
    speaker_lines = extract_speaker_lines(txt)

    # If no parsed speaker lines, be conservative: treat as "has content"
    # (You can change this to True if you want to treat unparseable transcripts as no-input.)
    if not speaker_lines:
        return False

    inferred_agent = speaker_lines[0][0].strip() if speaker_lines else None
    agent_norm = (agent or inferred_agent or "").strip().upper()

    customer_msgs: List[str] = []
    for speaker, msg in speaker_lines:
        sp = speaker.strip().upper()

        # Skip agent lines (header-based or inferred)
        if agent_norm and sp == agent_norm:
            continue

        # Skip system-like speakers
        if sp in {"SYSTEM", "BOT"}:
            continue

        customer_msgs.append(msg)

    # If no customer turns, it's no-input
    if not customer_msgs:
        return True

    # If customer only said a few trivial words/messages, treat as no-input
    joined = " ".join(customer_msgs)
    words = re.findall(r"\b\w+\b", joined)

    if len(words) <= max_words:
        msgs_norm = [re.sub(r"\s+", " ", m.strip().lower()) for m in customer_msgs]
        if all(_NO_INTENT_SHORT_RE.match(x) for x in msgs_norm):
            return True

    return False
