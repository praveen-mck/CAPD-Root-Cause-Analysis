
# -*- coding: utf-8 -*-
"""
Snowflake IO utilities + Azure Key Vault secret retrieval.

Implements:
- get_secret_from_keyvault
- load_private_key_der (for Snowflake key-pair auth)
- create_snowflake_connection
- extract_data_from_snowflake (read into pandas)
- execute_snowflake_multi_query (multi-statement exec)
- write_pandas_create_or_replace_stage (Create/Replace stage table and write pandas df)
- merge_chats_results_into_table (MERGE stage into target)
- merge_calls_results_into_table (MERGE stage into target)

Expected cfg keys (as used in your scripts/main.py):
  cfg = {
    "sf_user": "...",
    "sf_url": "...",                # snowflake account identifier/url
    "database": "...",
    "schema": "...",
    "warehouse": "...",
    "role": "...",
    "rsa_key_secret_name": "...",   # secret name in key vault containing private key PEM
    "pem_passphrase_secret_name": "...", # secret name in key vault containing passphrase
    "keyvault_name": "...",         # key vault name (without https://...vault.azure.net)
  }

Optional environment variable:
  AZURE_MI_CLIENT_ID  # if using a user-assigned managed identity
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Tuple

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


# -----------------------------
# Key Vault + Key-pair helpers
# -----------------------------
def get_secret_from_keyvault(cfg: Dict[str, Any], secret_name: str) -> str:
    """
    Read a secret value from Azure Key Vault using Managed Identity.

    If AZURE_MI_CLIENT_ID is set, it uses that client id (user-assigned MI).
    Otherwise uses system-assigned MI.
    """
    kv_name = cfg.get("keyvault_name")
    if not kv_name:
        raise ValueError("cfg['keyvault_name'] is required to read secrets from Key Vault.")

    vault_url = f"https://{kv_name}.vault.azure.net/"
    client_id = os.getenv("AZURE_MI_CLIENT_ID")

    credential = ManagedIdentityCredential(client_id=client_id) if client_id else ManagedIdentityCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)

    return secret_client.get_secret(secret_name).value


def load_private_key_der(pem_passphrase: str, rsa_key_pem: str) -> bytes:
    """
    Convert encrypted PEM private key (string) + passphrase into DER bytes expected by Snowflake connector.
    """
    if rsa_key_pem is None or not str(rsa_key_pem).strip():
        raise ValueError("RSA private key PEM is empty.")
    if pem_passphrase is None:
        pem_passphrase = ""

    p_key = serialization.load_pem_private_key(
        rsa_key_pem.encode("utf-8"),
        password=pem_passphrase.encode("utf-8") if pem_passphrase else None,
        backend=default_backend(),
    )

    der_bytes = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return der_bytes


def create_snowflake_connection(cfg: Dict[str, Any]) -> snowflake.connector.SnowflakeConnection:
    """
    Create Snowflake connection using key-pair auth, where key and passphrase are retrieved from Key Vault.
    """
    required = ["sf_user", "sf_url", "database", "schema", "warehouse", "role",
                "rsa_key_secret_name", "pem_passphrase_secret_name", "keyvault_name"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required Snowflake cfg keys: {missing}")

    rsa_key_pem = get_secret_from_keyvault(cfg, cfg["rsa_key_secret_name"])
    pem_passphrase = get_secret_from_keyvault(cfg, cfg["pem_passphrase_secret_name"])
    private_key_der = load_private_key_der(pem_passphrase, rsa_key_pem)

    return snowflake.connector.connect(
        user=cfg["sf_user"],
        account=cfg["sf_url"],
        role=cfg["role"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"],
        private_key=private_key_der,
    )


# -----------------------------
# Read / Execute helpers
# -----------------------------
def extract_data_from_snowflake(cfg: Dict[str, Any], query: str) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a pandas DataFrame.
    """
    conn = create_snowflake_connection(cfg)
    try:
        df = pd.read_sql(sql=query, con=conn)
        return df
    finally:
        conn.close()


def execute_snowflake_multi_query(cfg: Dict[str, Any], query: str) -> None:
    """
    Execute potentially multi-statement SQL separated by ';'.
    Skips empty statements and comment-only statements.

    Note:
    - This is a simple splitter; if you have semicolons in strings, use a proper SQL parser.
    """
    conn = create_snowflake_connection(cfg)
    cur = conn.cursor()
    try:
        statements = [s.strip() for s in (query or "").split(";")]
        for idx, stmt in enumerate(statements, start=1):
            if not stmt:
                continue
            if stmt.startswith("--"):
                continue
            print(f"\n[Snowflake] Executing statement {idx}:\n{stmt};")
            cur.execute(stmt)
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


# -----------------------------
# Write stage table + merge
# -----------------------------
def _snowflake_type_for_pd_dtype(dtype: Any) -> str:
    """
    Map pandas dtype -> Snowflake type for stage table DDL.
    Keep it conservative.
    """
    dt = str(dtype)
    if dt in ("int64", "Int64"):
        return "BIGINT"
    if dt in ("float64", "Float64"):
        return "FLOAT"
    if dt == "bool":
        return "BOOLEAN"
    # We'll store timestamps as TIMESTAMP_NTZ if column already looks like datetime in pandas.
    if "datetime64" in dt:
        return "TIMESTAMP_NTZ"
    return "STRING"


def write_pandas_create_or_replace_stage(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    table_name: str,
) -> Tuple[bool, int, int]:
    """
    Create OR REPLACE stage table matching DataFrame schema, then write df using write_pandas.

    Returns: (success, nchunks, nrows)
    """
    if df is None:
        raise ValueError("df is None")
    if not table_name:
        raise ValueError("table_name is required")

    # Build DDL from dataframe columns
    columns_ddl = []
    for col, dtype in zip(df.columns, df.dtypes):
        col_up = str(col).upper().replace(" ", "_")
        sf_type = _snowflake_type_for_pd_dtype(dtype)
        columns_ddl.append(f"\"{col_up}\" {sf_type}")

    create_sql = f"""
    CREATE OR REPLACE TABLE "{cfg['database']}"."{cfg['schema']}"."{table_name}" (
      {", ".join(columns_ddl)}
    );
    """
    print("[Snowflake] Creating/Replacing stage table:\n", create_sql.strip())
    execute_snowflake_multi_query(cfg, create_sql)

    # Write df to the stage table
    conn = create_snowflake_connection(cfg)
    try:
        success, nchunks, nrows, _ = write_pandas(
            conn,
            df,
            table_name=table_name,
            schema=cfg["schema"],
            database=cfg["database"],
            overwrite=True,
        )
        return bool(success), int(nchunks), int(nrows)
    finally:
        conn.close()

def load_transcripts(
    cfg: Dict[str, Any],
    *,
    source_db: str,
    source_schema: str,
    source_table: str,
    id_col: str,
    text_col: str,
    where_clause: str = "",
    limit: int = 0,
) -> pd.DataFrame:
    """
    Load (id_col, text_col) rows from Snowflake.
    where_clause may be "" or start with "WHERE ...".
    """
    where = f" {where_clause}" if where_clause.strip() else ""
    lim = f"\nLIMIT {int(limit)}" if int(limit) > 0 else ""
    sql = f"""
    SELECT {id_col}, {text_col}
    FROM {source_db}.{source_schema}.{source_table}
    {where}{lim};
    """
    return extract_data_from_snowflake(cfg, sql)


# CHATS - Create Results Table
def ensure_chats_results_table_exists(cfg: Dict[str, Any], *, result_db: str, result_schema: str, result_table: str) -> None:
    """
    Create base results table (Pass-1 columns). Pass-2 columns are added later by merge_chats_results_into_table.
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS "{result_db}"."{result_schema}"."{result_table}" (
      "CHAT_TRANSCRIPT_NAME" STRING,
      "CONTACT_TYPE" STRING,
      "DOMAIN" STRING,
      "SUBDOMAIN" STRING,
      "ROOT_CAUSE" STRING,
      "CONTACT_DRIVER" STRING,
      "SHORT_SUMMARY" STRING,
      "DETAILED_SUMMARY" STRING,
      "CONFIDENCE" FLOAT,
      "ANALYZED_AT" TIMESTAMP_NTZ,
      "IS_NO_INPUT" NUMBER(1,0)
    );
    """
    execute_snowflake_multi_query(cfg, sql)


# CALLS - Create Results Table
def ensure_call_results_table_exists(cfg: Dict[str, Any], *, result_db: str, result_schema: str, result_table: str) -> None:
    """
    Create base CALL results table (Pass-1 columns).
    Same shape as chat results, but keyed by CALL_ID.
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS "{result_db}"."{result_schema}"."{result_table}" (
      "CALL_ID" STRING,
      "CONTACT_TYPE" STRING,
      "DOMAIN" STRING,
      "SUBDOMAIN" STRING,
      "ROOT_CAUSE" STRING,
      "CONTACT_DRIVER" STRING,
      "SHORT_SUMMARY" STRING,
      "DETAILED_SUMMARY" STRING,
      "CONFIDENCE" FLOAT,
      "ANALYZED_AT" TIMESTAMP_NTZ,
      "IS_NO_INPUT" NUMBER(1,0)
    );
    """
    execute_snowflake_multi_query(cfg, sql)


def drop_table_if_exists(cfg: Dict[str, Any], *, db: str, schema: str, table: str) -> None:
    sql = f'DROP TABLE IF EXISTS "{db}"."{schema}"."{table}";'
    execute_snowflake_multi_query(cfg, sql)


# CHATS - Write results to stage table and merge
def write_stage_and_merge_chats(
    cfg: Dict[str, Any],
    *,
    results_df: pd.DataFrame,
    result_db: str,
    result_schema: str,
    result_table: str,
    id_col: str,
    stage_suffix: str = "_STAGE",
    drop_stage: bool = True,
) -> Tuple[bool, int, int, str]:
    """
    Write results_df to a stage table, MERGE into result_table, optionally drop stage.
    Returns (success, nchunks, nrows, stage_table_name).
    """
    stage_table = f"{result_table}{stage_suffix}"

    out_cfg = dict(cfg)
    out_cfg["database"] = result_db
    out_cfg["schema"] = result_schema

    success, nchunks, nrows = write_pandas_create_or_replace_stage(out_cfg, results_df, stage_table)

    ensure_chats_results_table_exists(out_cfg, result_db=result_db, result_schema=result_schema, result_table=result_table)

    merge_chats_results_into_table(
        cfg=out_cfg,
        target_table=result_table,
        stage_table=stage_table,
        id_col=id_col,
    )

    if drop_stage:
        drop_table_if_exists(out_cfg, db=result_db, schema=result_schema, table=stage_table)

    return success, nchunks, nrows, stage_table


# CALLS - Write results to stage table and merge
def write_stage_and_merge_calls(
    cfg: Dict[str, Any],
    *,
    results_df: pd.DataFrame,
    result_db: str,
    result_schema: str,
    result_table: str,
    id_col: str = "CALL_ID",
    stage_suffix: str = "_STAGE",
    drop_stage: bool = True,
) -> Tuple[bool, int, int, str]:
    """
    Write call results_df to stage table, MERGE into call results table keyed by CALL_ID.
    Returns (success, nchunks, nrows, stage_table_name).
    """
    stage_table = f"{result_table}{stage_suffix}"

    out_cfg = dict(cfg)
    out_cfg["database"] = result_db
    out_cfg["schema"] = result_schema

    success, nchunks, nrows = write_pandas_create_or_replace_stage(out_cfg, results_df, stage_table)

    ensure_call_results_table_exists(out_cfg, result_db=result_db, result_schema=result_schema, result_table=result_table)

    merge_call_results_into_table(
        cfg=out_cfg,
        target_table=result_table,
        stage_table=stage_table,
        id_col=id_col,
    )

    if drop_stage:
        drop_table_if_exists(out_cfg, db=result_db, schema=result_schema, table=stage_table)

    return success, nchunks, nrows, stage_table


# CHATS - Merge results into existing table
def merge_chats_results_into_table(
    cfg: Dict[str, Any],
    target_table: str,
    stage_table: str,
    id_col: str,
) -> None:
    """
    Merge from stage_table into target_table using id_col.

    Assumes your stage dataframe uses uppercase column names as written by batch.py:
      CHAT_TRANSCRIPT_NAME, CONTACT_TYPE, DOMAIN, SUBDOMAIN, ROOT_CAUSE, CONTACT_DRIVER,
      SHORT_SUMMARY, DETAILED_SUMMARY, CONFIDENCE, ANALYZED_AT, IS_NO_INPUT

    Your scripts/main.py passes id_col="CHAT_TRANSCRIPT_NAME" (keep-as-is behavior).
    """
    if not target_table or not stage_table or not id_col:
        raise ValueError("target_table, stage_table, id_col are required")

    sql = f"""
    MERGE INTO "{cfg['database']}"."{cfg['schema']}"."{target_table}" TGT
    USING "{cfg['database']}"."{cfg['schema']}"."{stage_table}" SRC
      ON TGT."{id_col}" = SRC."{id_col}"
    WHEN MATCHED THEN UPDATE SET
      TGT."CONTACT_TYPE"    = SRC."CONTACT_TYPE",
      TGT."DOMAIN"          = SRC."DOMAIN",
      TGT."SUBDOMAIN"       = SRC."SUBDOMAIN",
      TGT."ROOT_CAUSE"      = SRC."ROOT_CAUSE",
      TGT."CONTACT_DRIVER"  = SRC."CONTACT_DRIVER",
      TGT."SHORT_SUMMARY"    = SRC."SHORT_SUMMARY",
      TGT."DETAILED_SUMMARY" = SRC."DETAILED_SUMMARY",
      TGT."CONFIDENCE"      = SRC."CONFIDENCE",
      TGT."ANALYZED_AT"     = SRC."ANALYZED_AT",
      TGT."IS_NO_INPUT"     = SRC."IS_NO_INPUT"
    WHEN NOT MATCHED THEN INSERT (
      "{id_col}",
      "CONTACT_TYPE", "DOMAIN", "SUBDOMAIN", "ROOT_CAUSE",
      "CONTACT_DRIVER", "SHORT_SUMMARY", "DETAILED_SUMMARY", "CONFIDENCE", "ANALYZED_AT", "IS_NO_INPUT"
    ) VALUES (
      SRC."{id_col}",
      SRC."CONTACT_TYPE", SRC."DOMAIN", SRC."SUBDOMAIN", SRC."ROOT_CAUSE",
      SRC."CONTACT_DRIVER", SRC."SHORT_SUMMARY", SRC."DETAILED_SUMMARY", SRC."CONFIDENCE", SRC."ANALYZED_AT", SRC."IS_NO_INPUT"
    );
    """
    print("[Snowflake] Merging stage -> target...\n", sql.strip())
    execute_snowflake_multi_query(cfg, sql)

# CALLS - Merge results into existing table
def merge_call_results_into_table(
    cfg: Dict[str, Any],
    target_table: str,
    stage_table: str,
    id_col: str = "CALL_ID",
) -> None:
    """
    Merge call results from stage_table into target_table using CALL_ID.
    """
    sql = f"""
    MERGE INTO "{cfg['database']}"."{cfg['schema']}"."{target_table}" TGT
    USING "{cfg['database']}"."{cfg['schema']}"."{stage_table}" SRC
      ON TGT."CALL_ID" = SRC."CALL_ID"
    WHEN MATCHED THEN UPDATE SET
      TGT."CONTACT_TYPE"     = SRC."CONTACT_TYPE",
      TGT."DOMAIN"           = SRC."DOMAIN",
      TGT."SUBDOMAIN"        = SRC."SUBDOMAIN",
      TGT."ROOT_CAUSE"       = SRC."ROOT_CAUSE",
      TGT."CONTACT_DRIVER"   = SRC."CONTACT_DRIVER",
      TGT."SHORT_SUMMARY"    = SRC."SHORT_SUMMARY",
      TGT."DETAILED_SUMMARY" = SRC."DETAILED_SUMMARY",
      TGT."CONFIDENCE"       = SRC."CONFIDENCE",
      TGT."ANALYZED_AT"      = SRC."ANALYZED_AT",
      TGT."IS_NO_INPUT"      = SRC."IS_NO_INPUT"
    WHEN NOT MATCHED THEN INSERT (
      "CALL_ID",
      "CONTACT_TYPE", "DOMAIN", "SUBDOMAIN", "ROOT_CAUSE",
      "CONTACT_DRIVER", "SHORT_SUMMARY", "DETAILED_SUMMARY",
      "CONFIDENCE", "ANALYZED_AT", "IS_NO_INPUT"
    ) VALUES (
      SRC."CALL_ID",
      SRC."CONTACT_TYPE", SRC."DOMAIN", SRC."SUBDOMAIN", SRC."ROOT_CAUSE",
      SRC."CONTACT_DRIVER", SRC."SHORT_SUMMARY", SRC."DETAILED_SUMMARY",
      SRC."CONFIDENCE", SRC."ANALYZED_AT", SRC."IS_NO_INPUT"
    );
    """
    execute_snowflake_multi_query(cfg, sql)

def load_predictions_for_grading_join_source_chats(
    cfg: Dict[str, Any],
    *,
    pred_db: str,
    pred_schema: str,
    pred_table: str,
    source_db: str,
    source_schema: str,
    source_table: str,
    id_col: str = "CHAT_TRANSCRIPT_NAME",
    text_col: str = "BODY",
    limit: int = 0,
) -> pd.DataFrame:
    """
    Load rows for grading by joining prediction table to source transcripts table (INNER JOIN).

    Returns columns:
      - CHAT_TRANSCRIPT_NAME
      - BODY (from source table)
      - CONTACT_TYPE, DOMAIN, SUBDOMAIN, ROOT_CAUSE, CONTACT_DRIVER (from prediction table)
    """
    lim = f"\nLIMIT {int(limit)}" if int(limit) > 0 else ""

    sql = f"""
    SELECT
      p."{id_col}" AS "CHAT_TRANSCRIPT_NAME",
      s."{text_col}" AS "BODY",
      p."CONTACT_TYPE",
      p."DOMAIN",
      p."SUBDOMAIN",
      p."ROOT_CAUSE",
      p."CONTACT_DRIVER"
    FROM "{pred_db}"."{pred_schema}"."{pred_table}" p
    INNER JOIN "{source_db}"."{source_schema}"."{source_table}" s
      ON p."{id_col}" = s."{id_col}"
    {lim};
    """
    return extract_data_from_snowflake(cfg, sql)

def load_predictions_for_grading_join_source_calls(
    cfg: Dict[str, Any],
    *,
    pred_db: str,
    pred_schema: str,
    pred_table: str,
    source_db: str,
    source_schema: str,
    source_table: str,
    id_col: str = "COL_ID",
    text_col: str = "DIARIZED_TRANSCRIPT_NAME",
    limit: int = 0,
) -> pd.DataFrame:
    """
    Load rows for grading by joining prediction table to source transcripts table (INNER JOIN).

    Returns columns:
      - COL_ID
      - DIARIZED_TRANSCRIPT_NAME (from source table)
      - CONTACT_TYPE, DOMAIN, SUBDOMAIN, ROOT_CAUSE, CONTACT_DRIVER (from prediction table)
    """
    lim = f"\nLIMIT {int(limit)}" if int(limit) > 0 else ""

    sql = f"""
    SELECT
      p."{id_col}" AS "COL_ID",
      s."{text_col}" AS "DIARIZED_TRANSCRIPT_NAME",
      p."CONTACT_TYPE",
      p."DOMAIN",
      p."SUBDOMAIN",
      p."ROOT_CAUSE",
      p."CONTACT_DRIVER"
    FROM "{pred_db}"."{pred_schema}"."{pred_table}" p
    INNER JOIN "{source_db}"."{source_schema}"."{source_table}" s
      ON p."{id_col}" = s."{id_col}"
    {lim};
    """
    return extract_data_from_snowflake(cfg, sql)

def ensure_grades_table_exists_chats(
    cfg: Dict[str, Any],
    *,
    grade_db: str,
    grade_schema: str,
    grade_table: str,
) -> None:
    """
    Create grading results table if missing.

    Notes:
    - Snowflake doesn't enforce PK in the same way as OLTP; we treat CHAT_TRANSCRIPT_NAME as merge key.
    - Verdict is stored as STRING: 'Correct' | 'Partial' | 'Incorrect'
    - Score is FLOAT: 0 | 0.5 | 1
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS "{grade_db}"."{grade_schema}"."{grade_table}" (
      "CHAT_TRANSCRIPT_NAME" STRING,

      -- grade metadata
      "GRADER_RUN_ID" STRING,
      "GRADED_AT" TIMESTAMP_NTZ,

      -- per-field grade outputs
      "CONTACT_TYPE_VERDICT" STRING,
      "CONTACT_TYPE_SCORE" FLOAT,
      "CONTACT_TYPE_SUGGESTED_LABEL" STRING,

      "DOMAIN_VERDICT" STRING,
      "DOMAIN_SCORE" FLOAT,
      "DOMAIN_SUGGESTED_LABEL" STRING,

      "SUBDOMAIN_VERDICT" STRING,
      "SUBDOMAIN_SCORE" FLOAT,
      "SUBDOMAIN_SUGGESTED_LABEL" STRING,

      "ROOT_CAUSE_VERDICT" STRING,
      "ROOT_CAUSE_SCORE" FLOAT,
      "ROOT_CAUSE_SUGGESTED_LABEL" STRING,

      "CONTACT_DRIVER_VERDICT" STRING,
      "CONTACT_DRIVER_SCORE" FLOAT,
      "CONTACT_DRIVER_SUGGESTED_LABEL" STRING,

      -- overall
      "OVERALL_SCORE" FLOAT
    );
    """
    execute_snowflake_multi_query(cfg, sql)


def ensure_grades_table_exists_calls(
    cfg: Dict[str, Any],
    *,
    grade_db: str,
    grade_schema: str,
    grade_table: str,
) -> None:
    """
    Create grading results table if missing.

    Notes:
    - Snowflake doesn't enforce PK in the same way as OLTP; we treat CHAT_TRANSCRIPT_NAME as merge key.
    - Verdict is stored as STRING: 'Correct' | 'Partial' | 'Incorrect'
    - Score is FLOAT: 0 | 0.5 | 1
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS "{grade_db}"."{grade_schema}"."{grade_table}" (
      "CALL_ID" STRING,

      -- grade metadata
      "GRADER_RUN_ID" STRING,
      "GRADED_AT" TIMESTAMP_NTZ,

      -- per-field grade outputs
      "CONTACT_TYPE_VERDICT" STRING,
      "CONTACT_TYPE_SCORE" FLOAT,
      "CONTACT_TYPE_SUGGESTED_LABEL" STRING,

      "DOMAIN_VERDICT" STRING,
      "DOMAIN_SCORE" FLOAT,
      "DOMAIN_SUGGESTED_LABEL" STRING,

      "SUBDOMAIN_VERDICT" STRING,
      "SUBDOMAIN_SCORE" FLOAT,
      "SUBDOMAIN_SUGGESTED_LABEL" STRING,

      "ROOT_CAUSE_VERDICT" STRING,
      "ROOT_CAUSE_SCORE" FLOAT,
      "ROOT_CAUSE_SUGGESTED_LABEL" STRING,

      "CONTACT_DRIVER_VERDICT" STRING,
      "CONTACT_DRIVER_SCORE" FLOAT,
      "CONTACT_DRIVER_SUGGESTED_LABEL" STRING,

      -- overall
      "OVERALL_SCORE" FLOAT
    );
    """
    execute_snowflake_multi_query(cfg, sql)


def write_stage_and_merge_grades_chats(
    cfg: Dict[str, Any],
    *,
    grades_df: pd.DataFrame,
    grade_db: str,
    grade_schema: str,
    grade_table: str,
    stage_suffix: str = "_STAGE",
    drop_stage: bool = True,
) -> Tuple[bool, int, int, str]:
    """
    Write grades_df to stage, ensure grade table exists, merge, optionally drop stage.
    Returns (success, nchunks, nrows, stage_table_name).
    """
    stage_table = f"{grade_table}{stage_suffix}"

    out_cfg = dict(cfg)
    out_cfg["database"] = grade_db
    out_cfg["schema"] = grade_schema

    success, nchunks, nrows = write_pandas_create_or_replace_stage(out_cfg, grades_df, stage_table)

    ensure_grades_table_exists_chats(out_cfg, grade_db=grade_db, grade_schema=grade_schema, grade_table=grade_table)

    merge_grades_into_table_chats(
        out_cfg,
        grade_db=grade_db,
        grade_schema=grade_schema,
        grade_table=grade_table,
        stage_table=stage_table,
    )

    if drop_stage:
        sql = f'DROP TABLE IF EXISTS "{grade_db}"."{grade_schema}"."{stage_table}";'
        execute_snowflake_multi_query(out_cfg, sql)

    return success, nchunks, nrows, stage_table


# Calls
def write_stage_and_merge_grades_calls(
    cfg: Dict[str, Any],
    *,
    grades_df: pd.DataFrame,
    grade_db: str,
    grade_schema: str,
    grade_table: str,
    stage_suffix: str = "_STAGE",
    drop_stage: bool = True,
) -> Tuple[bool, int, int, str]:
    """
    Write grades_df to stage, ensure grade table exists, merge, optionally drop stage.
    Returns (success, nchunks, nrows, stage_table_name).
    """
    stage_table = f"{grade_table}{stage_suffix}"

    out_cfg = dict(cfg)
    out_cfg["database"] = grade_db
    out_cfg["schema"] = grade_schema

    success, nchunks, nrows = write_pandas_create_or_replace_stage(out_cfg, grades_df, stage_table)

    ensure_grades_table_exists_calls(out_cfg, grade_db=grade_db, grade_schema=grade_schema, grade_table=grade_table)

    merge_grades_into_table_calls(
        out_cfg,
        grade_db=grade_db,
        grade_schema=grade_schema,
        grade_table=grade_table,
        stage_table=stage_table,
    )

    if drop_stage:
        sql = f'DROP TABLE IF EXISTS "{grade_db}"."{grade_schema}"."{stage_table}";'
        execute_snowflake_multi_query(out_cfg, sql)

    return success, nchunks, nrows, stage_table


def new_grader_run_id(prefix: str = "grade") -> str:
    """
    Utility to generate a unique run id for grading runs.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ...existing code...

def merge_grades_into_table_chats(
    cfg: Dict[str, Any],
    *,
    grade_db: str,
    grade_schema: str,
    grade_table: str,
    stage_table: str,
) -> None:
    """
    Merge stage -> grade table. Cast SRC.GRADED_AT to TIMESTAMP_NTZ to avoid stage type inference issues.
    """
    sql = f"""
    MERGE INTO "{grade_db}"."{grade_schema}"."{grade_table}" TGT
    USING "{grade_db}"."{grade_schema}"."{stage_table}" SRC
      ON TGT."CHAT_TRANSCRIPT_NAME" = SRC."CHAT_TRANSCRIPT_NAME"
    WHEN MATCHED THEN UPDATE SET
      TGT."GRADER_RUN_ID" = SRC."GRADER_RUN_ID",
      TGT."GRADED_AT" = TRY_TO_TIMESTAMP_NTZ(SRC."GRADED_AT"),

      TGT."CONTACT_TYPE_VERDICT" = SRC."CONTACT_TYPE_VERDICT",
      TGT."CONTACT_TYPE_SCORE" = SRC."CONTACT_TYPE_SCORE",
      TGT."CONTACT_TYPE_SUGGESTED_LABEL" = SRC."CONTACT_TYPE_SUGGESTED_LABEL",

      TGT."DOMAIN_VERDICT" = SRC."DOMAIN_VERDICT",
      TGT."DOMAIN_SCORE" = SRC."DOMAIN_SCORE",
      TGT."DOMAIN_SUGGESTED_LABEL" = SRC."DOMAIN_SUGGESTED_LABEL",

      TGT."SUBDOMAIN_VERDICT" = SRC."SUBDOMAIN_VERDICT",
      TGT."SUBDOMAIN_SCORE" = SRC."SUBDOMAIN_SCORE",
      TGT."SUBDOMAIN_SUGGESTED_LABEL" = SRC."SUBDOMAIN_SUGGESTED_LABEL",

      TGT."ROOT_CAUSE_VERDICT" = SRC."ROOT_CAUSE_VERDICT",
      TGT."ROOT_CAUSE_SCORE" = SRC."ROOT_CAUSE_SCORE",
      TGT."ROOT_CAUSE_SUGGESTED_LABEL" = SRC."ROOT_CAUSE_SUGGESTED_LABEL",

      TGT."CONTACT_DRIVER_VERDICT" = SRC."CONTACT_DRIVER_VERDICT",
      TGT."CONTACT_DRIVER_SCORE" = SRC."CONTACT_DRIVER_SCORE",
      TGT."CONTACT_DRIVER_SUGGESTED_LABEL" = SRC."CONTACT_DRIVER_SUGGESTED_LABEL",

      TGT."OVERALL_SCORE" = SRC."OVERALL_SCORE"
    WHEN NOT MATCHED THEN INSERT (
      "CHAT_TRANSCRIPT_NAME",
      "GRADER_RUN_ID",
      "GRADED_AT",

      "CONTACT_TYPE_VERDICT",
      "CONTACT_TYPE_SCORE",
      "CONTACT_TYPE_SUGGESTED_LABEL",

      "DOMAIN_VERDICT",
      "DOMAIN_SCORE",
      "DOMAIN_SUGGESTED_LABEL",

      "SUBDOMAIN_VERDICT",
      "SUBDOMAIN_SCORE",
      "SUBDOMAIN_SUGGESTED_LABEL",

      "ROOT_CAUSE_VERDICT",
      "ROOT_CAUSE_SCORE",
      "ROOT_CAUSE_SUGGESTED_LABEL",

      "CONTACT_DRIVER_VERDICT",
      "CONTACT_DRIVER_SCORE",
      "CONTACT_DRIVER_SUGGESTED_LABEL",

      "OVERALL_SCORE"
    ) VALUES (
      SRC."CHAT_TRANSCRIPT_NAME",
      SRC."GRADER_RUN_ID",
      TRY_TO_TIMESTAMP_NTZ(SRC."GRADED_AT"),

      SRC."CONTACT_TYPE_VERDICT",
      SRC."CONTACT_TYPE_SCORE",
      SRC."CONTACT_TYPE_SUGGESTED_LABEL",

      SRC."DOMAIN_VERDICT",
      SRC."DOMAIN_SCORE",
      SRC."DOMAIN_SUGGESTED_LABEL",

      SRC."SUBDOMAIN_VERDICT",
      SRC."SUBDOMAIN_SCORE",
      SRC."SUBDOMAIN_SUGGESTED_LABEL",

      SRC."ROOT_CAUSE_VERDICT",
      SRC."ROOT_CAUSE_SCORE",
      SRC."ROOT_CAUSE_SUGGESTED_LABEL",

      SRC."CONTACT_DRIVER_VERDICT",
      SRC."CONTACT_DRIVER_SCORE",
      SRC."CONTACT_DRIVER_SUGGESTED_LABEL",

      SRC."OVERALL_SCORE"
    );
    """
    execute_snowflake_multi_query(cfg, sql)



def merge_grades_into_table_calls(
    cfg: Dict[str, Any],
    *,
    grade_db: str,
    grade_schema: str,
    grade_table: str,
    stage_table: str,
) -> None:
    """
    Merge stage -> grade table. Cast SRC.GRADED_AT to TIMESTAMP_NTZ to avoid stage type inference issues.
    """
    sql = f"""
    MERGE INTO "{grade_db}"."{grade_schema}"."{grade_table}" TGT
    USING "{grade_db}"."{grade_schema}"."{stage_table}" SRC
      ON TGT."CALL_ID" = SRC."CALL_ID"
    WHEN MATCHED THEN UPDATE SET
      TGT."GRADER_RUN_ID" = SRC."GRADER_RUN_ID",
      TGT."GRADED_AT" = TRY_TO_TIMESTAMP_NTZ(SRC."GRADED_AT"),

      TGT."CONTACT_TYPE_VERDICT" = SRC."CONTACT_TYPE_VERDICT",
      TGT."CONTACT_TYPE_SCORE" = SRC."CONTACT_TYPE_SCORE",
      TGT."CONTACT_TYPE_SUGGESTED_LABEL" = SRC."CONTACT_TYPE_SUGGESTED_LABEL",

      TGT."DOMAIN_VERDICT" = SRC."DOMAIN_VERDICT",
      TGT."DOMAIN_SCORE" = SRC."DOMAIN_SCORE",
      TGT."DOMAIN_SUGGESTED_LABEL" = SRC."DOMAIN_SUGGESTED_LABEL",

      TGT."SUBDOMAIN_VERDICT" = SRC."SUBDOMAIN_VERDICT",
      TGT."SUBDOMAIN_SCORE" = SRC."SUBDOMAIN_SCORE",
      TGT."SUBDOMAIN_SUGGESTED_LABEL" = SRC."SUBDOMAIN_SUGGESTED_LABEL",

      TGT."ROOT_CAUSE_VERDICT" = SRC."ROOT_CAUSE_VERDICT",
      TGT."ROOT_CAUSE_SCORE" = SRC."ROOT_CAUSE_SCORE",
      TGT."ROOT_CAUSE_SUGGESTED_LABEL" = SRC."ROOT_CAUSE_SUGGESTED_LABEL",

      TGT."CONTACT_DRIVER_VERDICT" = SRC."CONTACT_DRIVER_VERDICT",
      TGT."CONTACT_DRIVER_SCORE" = SRC."CONTACT_DRIVER_SCORE",
      TGT."CONTACT_DRIVER_SUGGESTED_LABEL" = SRC."CONTACT_DRIVER_SUGGESTED_LABEL",

      TGT."OVERALL_SCORE" = SRC."OVERALL_SCORE"
    WHEN NOT MATCHED THEN INSERT (
      "CALL_ID",
      "GRADER_RUN_ID",
      "GRADED_AT",

      "CONTACT_TYPE_VERDICT",
      "CONTACT_TYPE_SCORE",
      "CONTACT_TYPE_SUGGESTED_LABEL",

      "DOMAIN_VERDICT",
      "DOMAIN_SCORE",
      "DOMAIN_SUGGESTED_LABEL",

      "SUBDOMAIN_VERDICT",
      "SUBDOMAIN_SCORE",
      "SUBDOMAIN_SUGGESTED_LABEL",

      "ROOT_CAUSE_VERDICT",
      "ROOT_CAUSE_SCORE",
      "ROOT_CAUSE_SUGGESTED_LABEL",

      "CONTACT_DRIVER_VERDICT",
      "CONTACT_DRIVER_SCORE",
      "CONTACT_DRIVER_SUGGESTED_LABEL",

      "OVERALL_SCORE"
    ) VALUES (
      SRC."CALL_ID",
      SRC."GRADER_RUN_ID",
      TRY_TO_TIMESTAMP_NTZ(SRC."GRADED_AT"),

      SRC."CONTACT_TYPE_VERDICT",
      SRC."CONTACT_TYPE_SCORE",
      SRC."CONTACT_TYPE_SUGGESTED_LABEL",

      SRC."DOMAIN_VERDICT",
      SRC."DOMAIN_SCORE",
      SRC."DOMAIN_SUGGESTED_LABEL",

      SRC."SUBDOMAIN_VERDICT",
      SRC."SUBDOMAIN_SCORE",
      SRC."SUBDOMAIN_SUGGESTED_LABEL",

      SRC."ROOT_CAUSE_VERDICT",
      SRC."ROOT_CAUSE_SCORE",
      SRC."ROOT_CAUSE_SUGGESTED_LABEL",

      SRC."CONTACT_DRIVER_VERDICT",
      SRC."CONTACT_DRIVER_SCORE",
      SRC."CONTACT_DRIVER_SUGGESTED_LABEL",

      SRC."OVERALL_SCORE"
    );
    """
    execute_snowflake_multi_query(cfg, sql)

# ...existing code...