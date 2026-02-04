"""
scripts/main.py

Thin terminal entrypoint for CCC classifier.

Responsibilities:
- Parse CLI args and environment configuration
- Create Azure OpenAI client
- Orchestrate pipeline execution (predict / grade)
- Delegate Snowflake read/write to CCC_Classifier.io.snowflake helpers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from openai import AsyncAzureOpenAI

# ----------------------------
# Make src/ importable
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/main.py -> project root
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# ----------------------------
# Imports from package
# ----------------------------
from CCC_Classifier.pipeline.batch import process_batch
from CCC_Classifier.pipeline.grader.batch_grades import process_grade_batch
from CCC_Classifier.io.snowflake import (
    load_transcripts,
    write_stage_and_merge,
    load_predictions_for_grading_join_source,
    write_stage_and_merge_grades,
    new_grader_run_id,
)

# ----------------------------
# Local helpers
# ----------------------------
def _setup_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run.log", mode="w", encoding="utf-8"),
        ],
    )


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v or ""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in ("1", "true", "t", "yes", "y")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCC Classifier")
    p.add_argument(
        "--mode",
        choices=("predict", "grade"),
        default="predict",
        help="Run mode. 'predict' runs classification; 'grade' runs pass-2 grading (to be implemented).",
    )
    return p.parse_args()


def _build_snowflake_cfg(*, source_db: str, source_schema: str) -> Dict[str, Any]:
    """
    Build Snowflake cfg expected by CCC_Classifier.io.snowflake.*.

    Note: cfg["database"] / cfg["schema"] are used as defaults by helpers but may be overridden
    internally for output operations.
    """
    return {
        "sf_user": _env("SF_USER", "DEV_MT_BIGBETS_AU"),
        "sf_url": _env("SF_URL", required=True),
        "database": source_db,
        "schema": source_schema,
        "warehouse": _env("SF_WAREHOUSE", "DEV_MT_BIG_BETS_WH"),
        "role": _env("SF_ROLE", "DEV_MT_BIG_BETS_ENGINEER_FR"),
        "rsa_key_secret_name": _env("SF_RSA_KEY_SECRET", "bigbets-dev-snowflake-rsakey"),
        "pem_passphrase_secret_name": _env("SF_PEM_PASSPHRASE_SECRET", "bigbets-dev-snowflake-pem-passphrase"),
        "keyvault_name": _env("SF_KEYVAULT_NAME", "kv-mlops-aibigbets-dev"),
    }


def _build_aoai_client() -> tuple[AsyncAzureOpenAI, str]:
    _env("AZURE_OPENAI_API_KEY", required=True)
    _env("AZURE_OPENAI_API_VERSION", required=True)
    _env("AZURE_OPENAI_ENDPOINT", required=True)
    deployment = _env("AZURE_OPENAI_DEPLOYMENT_NAME", required=True)

    client = AsyncAzureOpenAI(
        api_key=_env("AZURE_OPENAI_API_KEY"),
        api_version=_env("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
    )
    return client, deployment


# ----------------------------
# Modes
# ----------------------------
async def run_predict() -> None:
    _setup_logging(PROJECT_ROOT)

    # Inputs
    SOURCE_DB = _env("SOURCE_DB", "DEV_MT_BIG_BETS_DB")
    SOURCE_SCHEMA = _env("SOURCE_SCHEMA", "POC")
    SOURCE_TABLE = _env("SOURCE_TABLE", "CHAT_TRANSCRIPTS_JULY_TO_SEPT_2025_V2")
    ID_COL = _env("ID_COL", "CHAT_TRANSCRIPT_NAME")
    TEXT_COL = _env("TEXT_COL", "BODY")

    # Outputs
    RESULT_DB = _env("RESULT_DB", SOURCE_DB)
    RESULT_SCHEMA = _env("RESULT_SCHEMA", SOURCE_SCHEMA)
    RESULT_TABLE = _env("RESULT_TABLE", "CHAT_ANALYSIS_TAXONOMY_V9_CD")

    WHERE_CLAUSE = _env("WHERE_CLAUSE", "WHERE BODY IS NOT NULL")
    MAX_ROWS = _int_env("MAX_ROWS", 20)

    MAX_COMPLETION_TOKENS = _int_env("MAX_COMPLETION_TOKENS", 256)
    USE_JSON_MODE = _bool_env("USE_JSON_MODE", True)

    sf_cfg = _build_snowflake_cfg(source_db=SOURCE_DB, source_schema=SOURCE_SCHEMA)
    client, deployment = _build_aoai_client()

    # Read inputs (Snowflake helper)
    print("\n[main] Reading input rows from Snowflake...")
    t0 = time.perf_counter()
    src_df = load_transcripts(
        sf_cfg,
        source_db=SOURCE_DB,
        source_schema=SOURCE_SCHEMA,
        source_table=SOURCE_TABLE,
        id_col=ID_COL,
        text_col=TEXT_COL,
        where_clause=WHERE_CLAUSE,
        limit=MAX_ROWS,
    )
    print(f"[main] Loaded {len(src_df)} rows in {round(time.perf_counter() - t0, 2)}s")

    if src_df.empty:
        print("[main] No rows to process. Exiting.")
        return

    # Run inference
    rows = src_df.to_dict(orient="records")
    print(
        f"\n[main] Classifying {len(rows)} transcripts "
        f"(deployment={deployment}, max_tokens={MAX_COMPLETION_TOKENS}, json_mode={USE_JSON_MODE})"
    )

    t1 = time.perf_counter()
    results_df = await process_batch(
        client=client,
        deployment=deployment,
        rows=rows,
        id_col=ID_COL,
        text_col=TEXT_COL,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        use_json_mode=USE_JSON_MODE,
    )
    infer_s = time.perf_counter() - t1
    print(f"[main] Inference done in {round(infer_s, 2)}s. Results rows: {len(results_df)}")

    # Optional: normalize ANALYZED_AT to string (matches existing behavior)
    if "ANALYZED_AT" in results_df.columns:
        results_df["ANALYZED_AT"] = pd.to_datetime(results_df["ANALYZED_AT"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # Persist results (Snowflake helper does: stage write -> ensure target -> merge -> drop stage)
    print(f"\n[main] Writing stage + merging into target: {RESULT_DB}.{RESULT_SCHEMA}.{RESULT_TABLE}")
    t2 = time.perf_counter()
    success, nchunks, nrows, stage_table = write_stage_and_merge(
        sf_cfg,
        results_df=results_df,
        result_db=RESULT_DB,
        result_schema=RESULT_SCHEMA,
        result_table=RESULT_TABLE,
        id_col="CHAT_TRANSCRIPT_NAME",
        drop_stage=True,
    )
    print(
        f"[main] Snowflake write/merge done in {round(time.perf_counter() - t2, 2)}s "
        f"(stage={stage_table}, success={success}, chunks={nchunks}, rows={nrows})"
    )

    metrics = {
        "records_processed": int(len(results_df)),
        "inference_seconds": round(infer_s, 2),
        "stage_rows": int(nrows),
        "deployment": deployment,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "use_json_mode": USE_JSON_MODE,
        "mode": "predict",
    }
    print("\n=== Batch Metrics ===")
    print(json.dumps(metrics, indent=2))


async def run_grade() -> None:
    _setup_logging(PROJECT_ROOT)

    # Prediction table location
    PRED_DB = _env("PRED_DB", _env("RESULT_DB", "DEV_MT_BIG_BETS_DB"))
    PRED_SCHEMA = _env("PRED_SCHEMA", _env("RESULT_SCHEMA", "POC"))
    PRED_TABLE = _env("PRED_TABLE", _env("RESULT_TABLE", "CHAT_ANALYSIS_TAXONOMY_V9_CD"))

    # Source transcript table location (for BODY)
    SOURCE_DB = _env("SOURCE_DB", PRED_DB)
    SOURCE_SCHEMA = _env("SOURCE_SCHEMA", PRED_SCHEMA)
    SOURCE_TABLE = _env("SOURCE_TABLE", "CHAT_TRANSCRIPTS_JULY_TO_SEPT_2025_V2")
    MAX_COMPLETION_TOKENS = _int_env("MAX_COMPLETION_TOKENS", 1024)
    ID_COL = _env("ID_COL", "CHAT_TRANSCRIPT_NAME")
    TEXT_COL = _env("TEXT_COL", "BODY")

    # Grade output table
    GRADE_DB = _env("GRADE_DB", PRED_DB)
    GRADE_SCHEMA = _env("GRADE_SCHEMA", PRED_SCHEMA)
    GRADE_TABLE = _env("GRADE_TABLE", "CHAT_ANALYSIS_TAXONOMY_V9_CD_GRADE")

    LIMIT = _int_env("GRADE_LIMIT", 0)

    sf_cfg = _build_snowflake_cfg(source_db=PRED_DB, source_schema=PRED_SCHEMA)

    print(
        f"\n[main] Loading prediction rows for grading (INNER JOIN source): "
        f"{PRED_DB}.{PRED_SCHEMA}.{PRED_TABLE} ↔ {SOURCE_DB}.{SOURCE_SCHEMA}.{SOURCE_TABLE}"
    )
    t0 = time.perf_counter()
    pred_df = load_predictions_for_grading_join_source(
        sf_cfg,
        pred_db=PRED_DB,
        pred_schema=PRED_SCHEMA,
        pred_table=PRED_TABLE,
        source_db=SOURCE_DB,
        source_schema=SOURCE_SCHEMA,
        source_table=SOURCE_TABLE,
        id_col=ID_COL,
        text_col=TEXT_COL,
        limit=LIMIT,
    )
    print(f"[main] Loaded {len(pred_df)} rows in {round(time.perf_counter() - t0, 2)}s")
    if pred_df.empty:
        print("[main] No prediction rows found to grade. Exiting.")
        return

    grader_run_id = new_grader_run_id()
    # Build stub grading output (plumbing validation)
    graded_at_str = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
    sf_cfg = _build_snowflake_cfg(source_db=PRED_DB, source_schema=PRED_SCHEMA)
    client, deployment = _build_aoai_client()
    grades_df = await process_grade_batch(
        pred_df,
        client=client,
        deployment=deployment,
        grader_run_id=grader_run_id,
        graded_at=graded_at_str,
        id_col=ID_COL,
        text_col=TEXT_COL,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    # grades_df = pd.DataFrame(
    #     {
    #         "CHAT_TRANSCRIPT_NAME": pred_df["CHAT_TRANSCRIPT_NAME"].astype(str),
    #         "GRADER_RUN_ID": grader_run_id,
    #         "GRADED_AT": graded_at_str,

    #         "CONTACT_TYPE_VERDICT": "Correct",
    #         "CONTACT_TYPE_SCORE": 1.0,
    #         "CONTACT_TYPE_SUGGESTED_LABEL": pred_df["CONTACT_TYPE"].astype(str),

    #         "DOMAIN_VERDICT": "Correct",
    #         "DOMAIN_SCORE": 1.0,
    #         "DOMAIN_SUGGESTED_LABEL": pred_df["DOMAIN"].astype(str),

    #         "SUBDOMAIN_VERDICT": "Correct",
    #         "SUBDOMAIN_SCORE": 1.0,
    #         "SUBDOMAIN_SUGGESTED_LABEL": pred_df["SUBDOMAIN"].astype(str),

    #         "ROOT_CAUSE_VERDICT": "Correct",
    #         "ROOT_CAUSE_SCORE": 1.0,
    #         "ROOT_CAUSE_SUGGESTED_LABEL": pred_df["ROOT_CAUSE"].astype(str),

    #         "CONTACT_DRIVER_VERDICT": "Correct",
    #         "CONTACT_DRIVER_SCORE": 1.0,
    #         "CONTACT_DRIVER_SUGGESTED_LABEL": pred_df["CONTACT_DRIVER"].astype(str),

    #         "OVERALL_SCORE": 1.0,
    #     }
    # )
    print("[main] grades_df dtypes:\n", grades_df.dtypes)
    print(f"\n[main] Writing grades to: {GRADE_DB}.{GRADE_SCHEMA}.{GRADE_TABLE}")
    t1 = time.perf_counter()
    success, nchunks, nrows, stage_table = write_stage_and_merge_grades(
        sf_cfg,
        grades_df=grades_df,
        grade_db=GRADE_DB,
        grade_schema=GRADE_SCHEMA,
        grade_table=GRADE_TABLE,
        drop_stage=True,
    )
    print(
        f"[main] Grade write/merge done in {round(time.perf_counter() - t1, 2)}s "
        f"(stage={stage_table}, success={success}, chunks={nchunks}, rows={nrows}, grader_run_id={grader_run_id})"
    )

# ----------------------------
# Entry
# ----------------------------
async def main() -> None:
    args = _parse_args()
    if args.mode == "predict":
        await run_predict()
    elif args.mode == "grade":
        await run_grade()
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    asyncio.run(main())