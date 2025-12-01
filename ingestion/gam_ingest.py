#!/usr/bin/env python3
"""
End-to-end ingestion for Google Ad Manager raw data.

This script:
1) Loads a manually downloaded GAM CSV from the UI (columns match schema.gam_raw).
2) Normalizes column names to snake_case and adds a deterministic row_hash.
3) Loads the data into Postgres with replace/append/upsert options.
4) Optionally refreshes the low-change seed_partner_names_goals table.

Prerequisites
-------------
- Postgres connection env vars: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB.
  Optional overrides: POSTGRES_HOST (default localhost), POSTGRES_PORT (5432),
  TARGET_SCHEMA (default "schema").
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import hashlib

import pandas as pd
import psycopg2
from psycopg2 import sql

RAW_TABLE_NAME: str = "gam_raw"
SEED_GOALS_TABLE_NAME: str = "seed_partner_names_goals"
DEFAULT_SCHEMA: str = os.getenv("TARGET_SCHEMA", "schema")

GAM_RAW_BASE_COLUMNS: List[str] = [
    "order_id",
    "order_name",
    "order_start_date",
    "order_start_date_time",
    "order_end_date",
    "order_end_date_time",
    "line_item_id",
    "line_item_name",
    "line_item_start_date",
    "line_item_start_date_time",
    "line_item_end_date",
    "line_item_end_date_time",
    "line_item_primary_goal_units_absolute",
    "line_item_primary_goal_unit_type_name",
    "line_item_lifetime_impressions",
    "line_item_lifetime_clicks",
    "order_lifetime_clicks",
    "order_lifetime_impressions",
    "line_item_delivery_indicator",
    "creative_name",
    "rendered_creative_size",
    "device_category_name",
    "line_item_priority",
    "line_item_type_name",
    "order_delivery_status_name",
    "advertiser_name",
    "ad_server_impressions",
    "ad_server_clicks",
    "ad_server_ctr",
]

# Target columns in Postgres (base columns plus row_hash for idempotent upserts)
GAM_RAW_TARGET_COLUMNS: List[str] = GAM_RAW_BASE_COLUMNS + ["row_hash"]

SEED_GOALS_COLUMNS: List[str] = [
    "order_name",
    "advertiser_name",
    "impression_goal",
]


def normalize_gam_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the manually downloaded CSV and align headers to gam_raw schema.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]

    missing = [col for col in GAM_RAW_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing expected columns for gam_raw: {', '.join(missing)}"
        )

    df = df[GAM_RAW_BASE_COLUMNS]
    return attach_row_hash(df)


def attach_row_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic row_hash column for idempotent upserts.
    """
    df = df.copy()
    # Normalize values to strings, empty string for NaN/None
    normalized = (
        df[GAM_RAW_BASE_COLUMNS]
        .fillna("")
        .map(lambda v: "" if pd.isna(v) else str(v))
    ) # type: ignore
    df["row_hash"] = normalized.apply(
        lambda row: hashlib.md5("|".join(row.values).encode("utf-8")).hexdigest(),
        axis=1,
    )
    return df


def connect_postgres() -> psycopg2.extensions.connection:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB")

    if not all([user, password, database]):
        raise RuntimeError(
            "POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB must be set."
        )

    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=database,
    )


def copy_dataframe(
    conn: psycopg2.extensions.connection,
    df: pd.DataFrame,
    schema_name: str,
    table_name: str,
    columns: Iterable[str],
    truncate: bool = True,
    upsert: bool = False,
) -> None:
    """
    Bulk copy a DataFrame into Postgres using COPY FROM STDIN.
    """
    with conn.cursor() as cur:
        if truncate and not upsert:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {}.{}")
                .format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                )
            )

        if upsert:
            ensure_row_hash(cur, schema_name, table_name)
            cur.execute(
                sql.SQL("CREATE TEMP TABLE tmp_ingest (LIKE {}.{} INCLUDING DEFAULTS)")
                .format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            copy_sql = sql.SQL(
                "COPY tmp_ingest ({}) FROM STDIN WITH (FORMAT CSV)"
            ).format(sql.SQL(", ").join(sql.Identifier(col) for col in columns))
            with tempfile.TemporaryFile(mode="w+") as tmp:
                df.to_csv(tmp, index=False, header=False)
                tmp.seek(0)
                cur.copy_expert(copy_sql.as_string(cur), tmp)

            insert_sql = sql.SQL(
                """
                INSERT INTO {}.{} ({cols})
                SELECT {cols} FROM tmp_ingest
                ON CONFLICT (row_hash) DO NOTHING
                """
            ).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                cols=sql.SQL(", ").join(sql.Identifier(col) for col in columns),
            )
            cur.execute(insert_sql)
        else:
            copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT CSV)").format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(", ").join(sql.Identifier(col) for col in columns),
            )

            with tempfile.TemporaryFile(mode="w+") as tmp:
                df.to_csv(tmp, index=False, header=False)
                tmp.seek(0)
                cur.copy_expert(copy_sql.as_string(cur), tmp)

    conn.commit()


def ensure_row_hash(
    cur: psycopg2.extensions.cursor,
    schema_name: str,
    table_name: str,
) -> None:
    """
    Ensure row_hash column and unique index exist for idempotent upserts.
    """
    cur.execute(
        sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS row_hash text").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
        )
    )
    cur.execute(
        sql.SQL(
            "CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{} (row_hash)"
        ).format(
            sql.Identifier(f"ux_{table_name}_row_hash"),
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
        )
    )


def load_seed_goals(
    conn: psycopg2.extensions.connection,
    seed_path: Path,
    schema_name: str,
    truncate: bool,
) -> None:
    """
    Load seed_partner_names_goals CSV into Postgres.
    """
    df = pd.read_csv(seed_path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [col for col in SEED_GOALS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Seed file {seed_path} missing columns: {', '.join(missing)}"
        )

    df["impression_goal"] = pd.to_numeric(
        df["impression_goal"], errors="raise"
    ).astype(int)

    copy_dataframe(
        conn=conn,
        df=df[SEED_GOALS_COLUMNS],
        schema_name=schema_name,
        table_name=SEED_GOALS_TABLE_NAME,
        columns=SEED_GOALS_COLUMNS,
        truncate=truncate,
    )


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}. Download a report from the GAM UI and point --input-csv at it."
        )
    logging.info("Using local GAM CSV: %s", csv_path)

    gam_df = normalize_gam_csv(csv_path)
    logging.info("Report pulled with %s rows", len(gam_df))

    if args.skip_db_load:
        logging.info("Skipping Postgres load (per flag).")
        return

    conn = connect_postgres()
    try:
        copy_dataframe(
            conn=conn,
            df=gam_df,
            schema_name=args.schema,
            table_name=RAW_TABLE_NAME,
            columns=GAM_RAW_TARGET_COLUMNS,
            truncate=not args.append,
            upsert=args.upsert,
        )
        logging.info(
            "Loaded %s rows into %s.%s (mode: %s)",
            len(gam_df),
            args.schema,
            RAW_TABLE_NAME,
            "upsert" if args.upsert else ("append" if args.append else "replace"),
        )

        if args.load_seed_goals:
            load_seed_goals(
                conn=conn,
                seed_path=Path(args.seed_goals_path),
                schema_name=args.schema,
                truncate=not args.append,
            )
            logging.info(
                "Loaded seed goals from %s into %s.%s",
                args.seed_goals_path,
                args.schema,
                SEED_GOALS_TABLE_NAME,
            )
    finally:
        conn.close()


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAM ingestion runner.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/gam_report_latest.csv",
        help="Path to a manually downloaded GAM CSV (UI export).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=DEFAULT_SCHEMA,
        help="Target Postgres schema (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-db-load",
        action="store_true",
        help="Download the report but do not load into Postgres.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing tables instead of truncating first.",
    )
    parser.add_argument(
        "--load-seed-goals",
        action="store_true",
        help="Also refresh seed_partner_names_goals from the local CSV.",
    )
    parser.add_argument(
        "--seed-goals-path",
        type=str,
        default="dbt/marketing_dbt/seeds/seed_partner_names_goals.csv",
        help="Path to the seed_partner_names_goals CSV.",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Idempotent load: adds row_hash and inserts new rows only (no truncation).",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    run(cli())
