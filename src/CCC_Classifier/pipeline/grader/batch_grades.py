# -*- coding: utf-8 -*-
"""
batchgrades.py

Grade-mode batch runner (similar to pipeline/batch.py) that:
- Takes prediction rows (joined with transcript text BODY)
- Calls grading orchestrator analyze_predict_row(...) for each row
- Shapes the grading outputs into a Snowflake-ready DataFrame

This module intentionally contains NO Snowflake I/O.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from CCC_Classifier.pipeline.grader.orchestrator_grades import analyze_predict_row

logger = logging.getLogger(__name__)

GRADE_FIELDS: Sequence[str] = (
    "CONTACT_TYPE",
    "DOMAIN",
    "SUBDOMAIN",
    "ROOT_CAUSE",
    "CONTACT_DRIVER",
)

_ALLOWED_VERDICTS = {"Correct", "Partial", "Incorrect"}
_ALLOWED_SCORES = {0.0, 0.5, 1.0}


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name, str(default))
    try:
        return int(v)
    except Exception:
        return default


def _normalize_verdict(v: Any) -> str:
    s = str(v or "").strip()
    if not s:
        return "Incorrect"
    # normalize common casing
    s_norm = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    return s_norm if s_norm in _ALLOWED_VERDICTS else "Incorrect"


def _normalize_score(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    # snap to allowed scores
    if x >= 0.75:
        return 1.0
    if x >= 0.25:
        return 0.5
    return 0.0


def _compute_overall_score(out_row: Dict[str, Any], fields: Sequence[str]) -> float:
    vals: List[float] = []
    for f in fields:
        vals.append(float(out_row.get(f"{f}_SCORE", 0.0) or 0.0))
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _to_rows(pred_df_or_rows: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(pred_df_or_rows, pd.DataFrame):
        return pred_df_or_rows.to_dict(orient="records")
    return pred_df_or_rows


async def process_grade_batch(
    pred_df_or_rows: Union[pd.DataFrame, List[Dict[str, Any]]],
    *,
    client: Any,
    deployment: str,
    grader_run_id: str,
    graded_at: str,
    id_col: str = "CHAT_TRANSCRIPT_NAME",
    text_col: str = "BODY",
    fields: Sequence[str] = GRADE_FIELDS,
    max_concurrent: Optional[int] = None,
    max_completion_tokens: int = 1024,
) -> pd.DataFrame:
    """
    Process grade batch concurrently.

    Input rows require:
      - id_col (default CHAT_TRANSCRIPT_NAME)
      - text_col (default BODY)
      - 5 predicted columns: CONTACT_TYPE, DOMAIN, SUBDOMAIN, ROOT_CAUSE, CONTACT_DRIVER

    Returns a DataFrame with columns matching the grade table schema:
      CHAT_TRANSCRIPT_NAME, GRADER_RUN_ID, GRADED_AT,
      <FIELD>_VERDICT, <FIELD>_SCORE, <FIELD>_SUGGESTED_LABEL (for each field),
      OVERALL_SCORE
    """
    rows = _to_rows(pred_df_or_rows)

    if max_concurrent is None:
        max_concurrent = _int_env("MAX_CONCURRENT_GRADE", 8)
    max_concurrent = max(1, int(max_concurrent))

    sem = asyncio.Semaphore(max_concurrent)

    async def _grade_one(row: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()

            rid = row.get(id_col)
            body = row.get(text_col)
            transcript_text = body if isinstance(body, str) else ("" if body is None else str(body))

            predicted: Dict[str, str] = {f: ("" if row.get(f) is None else str(row.get(f))) for f in fields}

            try:
                grade_result = await analyze_predict_row(
                    client=client,
                    deployment=deployment,
                    transcript_text=transcript_text,
                    predicted=predicted,
                    max_completion_tokens=max_completion_tokens,
                )
            except Exception:
                logger.exception("Grade orchestrator failed for %s=%r", id_col, rid)
                grade_result = {}

            out: Dict[str, Any] = {
                "CHAT_TRANSCRIPT_NAME": rid,
                "GRADER_RUN_ID": grader_run_id,
                "GRADED_AT": graded_at,
            }

            # Expect orchestrator output as:
            # { "DOMAIN": {"verdict": "...", "score": 0|0.5|1, "suggested_label": "..."}, ... }
            for f in fields:
                g = grade_result.get(f, {}) if isinstance(grade_result, dict) else {}
                out[f"{f}_VERDICT"] = _normalize_verdict(g.get("verdict"))
                out[f"{f}_SCORE"] = _normalize_score(g.get("score"))
                out[f"{f}_SUGGESTED_LABEL"] = str(g.get("suggested_label") or "")

            out["OVERALL_SCORE"] = (
                _normalize_score(grade_result.get("overall_score"))
                if isinstance(grade_result, dict) and "overall_score" in grade_result
                else _compute_overall_score(out, fields)
            )

            out["_DURATION_MS"] = round((time.perf_counter() - t0) * 1000.0, 1)
            return out

    results = await asyncio.gather(*[_grade_one(r) for r in rows])
    df = pd.DataFrame(results)

    # Keep Snowflake schema clean: drop diagnostics unless explicitly needed
    if not _int_env("GRADE_INCLUDE_DIAGNOSTICS", 0):
        if "_DURATION_MS" in df.columns:
            df = df.drop(columns=["_DURATION_MS"])
    return df