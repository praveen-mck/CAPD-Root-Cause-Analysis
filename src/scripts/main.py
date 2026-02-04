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
from CCC_Classifier.io.snowflake import (
    load_transcripts,
    write_stage_and_merge,
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
    raise NotImplementedError(
        "Mode=grade is not implemented yet. Next step: define grade inputs (source table/columns) and output table."
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