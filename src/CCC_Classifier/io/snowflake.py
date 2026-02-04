
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
- merge_results_into_table (MERGE stage into target)

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


def ensure_results_table_exists(cfg: Dict[str, Any], *, result_db: str, result_schema: str, result_table: str) -> None:
    """
    Create base results table (Pass-1 columns). Pass-2 columns are added later by merge_results_into_table.
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


def drop_table_if_exists(cfg: Dict[str, Any], *, db: str, schema: str, table: str) -> None:
    sql = f'DROP TABLE IF EXISTS "{db}"."{schema}"."{table}";'
    execute_snowflake_multi_query(cfg, sql)


def write_stage_and_merge(
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

    ensure_results_table_exists(out_cfg, result_db=result_db, result_schema=result_schema, result_table=result_table)

    merge_results_into_table(
        cfg=out_cfg,
        target_table=result_table,
        stage_table=stage_table,
        id_col=id_col,
    )

    if drop_stage:
        drop_table_if_exists(out_cfg, db=result_db, schema=result_schema, table=stage_table)

    return success, nchunks, nrows, stage_table

def merge_results_into_table(
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
