
# -*- coding: utf-8 -*-
"""
Batch runner: process a list of transcript rows concurrently.

Responsibilities:
- For each row:
    - call orchestrator.analyze_transcript()
    - shape output into Snowflake-friendly columns (uppercase)
- Return a pandas DataFrame with consistent schema.

Design choices:
- Keep "as-is" behavior: output ID column is always named CHAT_TRANSCRIPT_NAME
- Concurrency controlled via env var MAX_CONCURRENT (default 8)
"""

from __future__ import annotations

import asyncio
import os
import time
import datetime as dt
from typing import Any, Dict, List

import pandas as pd

from CCC_Classifier.pipeline.orchestrator import analyze_transcript


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


async def process_batch(
    *,
    client: Any,
    deployment: str,
    rows: List[Dict[str, Any]],
    id_col: str,
    text_col: str,
    max_completion_tokens: int = 512,
    use_json_mode: bool = True,
) -> pd.DataFrame:
    """
    Process many transcripts concurrently.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure OpenAI deployment name
        rows: list of dicts (each dict is a row from input data)
        id_col: column name for transcript identifier in `rows`
        text_col: column name for transcript text in `rows`
        max_completion_tokens: max tokens per stage completion (passed to orchestrator/stages)
        use_json_mode: enforce JSON outputs at the API layer

    Returns:
        DataFrame with columns:
          CHAT_TRANSCRIPT_NAME, CONTACT_TYPE, DOMAIN, SUBDOMAIN, ROOT_CAUSE,
          CONTACT_DRIVER, CASE_CONTEXT, CONFIDENCE, ANALYZED_AT, IS_NO_INPUT,
          optional: _DURATION_MS
    """
    max_concurrent = _int_env("MAX_CONCURRENT", 8)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_one(row: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            start = time.perf_counter()

            rid = row.get(id_col)
            transcript = (row.get(text_col) or "")
            transcript = transcript if isinstance(transcript, str) else str(transcript)

            try:
                result = await analyze_transcript(
                    client=client,
                    deployment=deployment,
                    transcript_text=transcript,
                    max_completion_tokens=max_completion_tokens,
                    use_json_mode=use_json_mode,
                )
            except Exception:
                # Very defensive fallback (should be rare because orchestrator already has a fallback)
                result = {
                    "contact_type": "Unclear Contact",
                    "domain": "Other: Unspecified",
                    "subdomain": "Other: Unspecified",
                    "root_cause": "Other: Unspecified",
                    "contact_driver": "Other: Unspecified",
                    "case_context": "Context Unspecified",
                    "confidence": 0.0,
                    "IS_NO_INPUT": 0,
                }

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Shape into Snowflake-friendly schema (uppercase column names)
            out = {
                "CHAT_TRANSCRIPT_NAME": rid,
                "CONTACT_TYPE": result.get("contact_type", "Unclear Contact"),
                "DOMAIN": result.get("domain", "Other: Unspecified"),
                "SUBDOMAIN": result.get("subdomain", "Other: Unspecified"),
                "ROOT_CAUSE": result.get("root_cause", "Other: Unspecified"),
                "CONTACT_DRIVER": result.get("contact_driver", "Other: Unspecified"),
                "CASE_CONTEXT": result.get("case_context", "Context Unspecified"),
                "CONFIDENCE": float(result.get("confidence", 0.0) or 0.0),
                "ANALYZED_AT": dt.datetime.utcnow(),
                "IS_NO_INPUT": int(result.get("IS_NO_INPUT", 0) or 0),
                "_DURATION_MS": round(elapsed_ms, 1),
            }
            return out

    tasks = [_process_one(r) for r in rows]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame(results)

    # Keep "as-is" behavior: output ID is always CHAT_TRANSCRIPT_NAME
    # If something upstream accidentally used a different name, enforce it here.
    if "CHAT_TRANSCRIPT_NAME" not in df.columns and id_col in df.columns:
        df = df.rename(columns={id_col: "CHAT_TRANSCRIPT_NAME"})

    return df
