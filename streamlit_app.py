"""
Streamlit Dashboard for GAM Campaign Health

This app:
- Connects to a Postgres database via st.connection("postgresql", type="sql").
- Reads dbt mart models in the "schema" schema:
    * mart_gam_order_pacing_summary
    * mart_gam_club_portfolio
    * mart_gam_line_item_diagnostics_view
- Provides:
    * Global order pacing overview.
    * Club portfolio view (orders by club).
    * Line-item diagnostics by domain/club.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import subprocess
import os
from typing import List, Optional, Dict, Tuple

import math

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from streamlit.connections.sql_connection import SQLConnection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE: str = "Sports league Ad Campaign Health Dashboard"

# dbt mart schema and table names
MART_SCHEMA: str = "schema"

ORDER_HEALTH_TABLE: str = "mart_gam_order_pacing_summary"
CLUB_PORTFOLIO_TABLE: str = "mart_gam_club_portfolio"
LINE_ITEM_DIAG_TABLE: str = "mart_gam_line_item_diagnostics_view"
LINE_ITEM_CREATIVE_SIZE_TABLE: str = "mart_gam_line_item_creative_sizes_view"
SEED_GOALS_PATH: Path = Path("dbt/marketing_dbt/seeds/seed_partner_names_goals.csv")
DATA_DIR: Path = Path("data")
INGEST_SCRIPT: Path = Path("ingestion/gam_ingest.py")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_sql_connection() -> SQLConnection:
    """
    Create and return a Streamlit SQLConnection to Postgres.

    Returns
    -------
    SQLConnection
        Active SQL connection using the "postgresql" entry in .streamlit/secrets.toml.

    Raises
    ------
    RuntimeError
        If the connection cannot be created.
    """
    try:
        conn: SQLConnection = st.connection("postgresql", type="sql")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not establish a connection to the Postgres database. "
            "Please check your .streamlit/secrets.toml configuration."
        ) from exc

    return conn


def _safe_query_table(
    conn: SQLConnection,
    schema: str,
    table: str,
    ttl_minutes: int = 10,
) -> pd.DataFrame:
    """
    Execute a simple SELECT * FROM schema.table with basic error handling.

    Parameters
    ----------
    conn : SQLConnection
        Active Streamlit SQL connection.
    schema : str
        Database schema name.
    table : str
        Table or view name to query.
    ttl_minutes : int, optional
        Cache time-to-live in minutes for Streamlit connection caching.

    Returns
    -------
    pd.DataFrame
        Resulting DataFrame. Empty DataFrame if query returns no rows.

    Raises
    ------
    RuntimeError
        If the query fails due to a database error.
    """
    qualified_table: str = f"{schema}.{table}"
    query: str = f"SELECT * FROM {qualified_table}"

    try:
        df: pd.DataFrame = conn.query(query, ttl=f"{ttl_minutes}m")
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"Failed to execute query on {qualified_table}: {exc}"
        ) from exc

    if df is None:
        # Streamlit's query should not normally return None, but guard anyway.
        return pd.DataFrame()

    return df


def load_order_pacing(conn: SQLConnection) -> pd.DataFrame:
    """
    Load order-level pacing and health data from mart_gam_order_pacing_summary.

    Adds derived urgency flags:
      - days_to_end: int days from today until schedule_end_date (may be negative)
      - is_ending_soon: True when 0 <= days_to_end <= 7
      - is_urgent: True when is_ending_soon AND pacing_status == 'behind'

    Parameters
    ----------
    conn : SQLConnection
        Active Streamlit SQL connection.

    Returns
    -------
    pd.DataFrame
        DataFrame of order pacing metrics, with basic type coercions and
        urgency flags applied.
    """
    df: pd.DataFrame = _safe_query_table(conn, MART_SCHEMA, ORDER_HEALTH_TABLE)

    if df.empty:
        return df

    # ------------------------------------------------------------------
    # Numeric fields we expect to treat as floats
    # ------------------------------------------------------------------
    numeric_cols: List[str] = [
        "impression_goal",
        # legacy / backwards-compatible aliases are kept defensive
        "delivered_impressions",
        "delivered_clicks",
        "delivered_impressions_pct_of_goal",
        "schedule_days",
        "days_elapsed",
        "schedule_pct_elapsed",
        "actual_impressions_per_day",
        "target_impressions_per_day",
        "pacing_index",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Date fields: keep as pandas Timestamps (NOT .dt.date) so we can safely
    # do datetime arithmetic and handle NaT via pd.isna.
    # ------------------------------------------------------------------
    for date_col in ["schedule_start_date", "schedule_end_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # ------------------------------------------------------------------
    # Convenience label for plotting
    # ------------------------------------------------------------------
    if "order_name" in df.columns and "advertiser_name" in df.columns:
        df["order_label"] = (
            df["order_name"].astype(str)
            + " — "
            + df["advertiser_name"].astype(str)
        )

    # ------------------------------------------------------------------
    # Urgency flags: days_to_end / is_ending_soon / is_urgent
    # ------------------------------------------------------------------
    # Use a Timestamp so arithmetic is Timestamp - Timestamp
    today_ts: pd.Timestamp = pd.Timestamp(date.today())

    # days_to_end: may be negative if order already ended
    if "schedule_end_date" in df.columns:
        def _safe_days_to_end(d: Optional[pd.Timestamp]) -> Optional[int]:
            """
            Safely compute (d - today_ts).days.

            Returns None when d is NaT/None or not a datetime-like value.
            """
            if d is None or pd.isna(d):
                return None
            delta: pd.Timedelta = d - today_ts
            return int(delta.days)

        df["days_to_end"] = df["schedule_end_date"].map(
            _safe_days_to_end  # type: ignore[arg-type]
        )
    else:
        df["days_to_end"] = None

    # Ending soon: within 7 days from today, but not in the past
    df["is_ending_soon"] = df["days_to_end"].map(
        lambda x: (x is not None) and (0 <= x <= 7)
    )

    # Normalise pacing_status for urgency derivation
    if "pacing_status" in df.columns:
        pacing_series: pd.Series = (
            df["pacing_status"]
            .astype(str)
            .str.lower()
        )
    else:
        pacing_series = pd.Series(
            ["unknown"] * len(df),
            index=df.index,
            dtype="object",
        )

    # Urgent: ending soon AND behind pace
    df["is_urgent"] = df["is_ending_soon"] & (pacing_series == "behind")

    # Ensure booleans are cleanly typed
    df["is_ending_soon"] = df["is_ending_soon"].astype(bool)
    df["is_urgent"] = df["is_urgent"].astype(bool)

    return df


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _find_latest_gam_csv(data_dir: Path) -> Optional[Path]:
    """
    Return the newest gam_report_*.csv in the given data directory.
    """
    if not data_dir.exists():
        return None

    try:
        candidates: List[Path] = [
            p
            for p in data_dir.iterdir()
            if p.is_file()
            and p.name.startswith("gam_report_")
            and str(p).lower().endswith(".csv")
        ]
    except OSError:
        return None

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _postgres_env_from_secrets(base_env: Dict[str, str]) -> Dict[str, str]:
    """
    Merge Postgres connection values from st.secrets into the env for ingestion.
    """
    env: Dict[str, str] = base_env.copy()

    pg_secrets: Optional[Dict[str, str]] = None
    connections = st.secrets.get("connections", {})
    if isinstance(connections, dict):
        pg_secrets = connections.get("postgresql")  # type: ignore[assignment]
    if pg_secrets is None and "postgresql" in st.secrets:
        pg_secrets = st.secrets["postgresql"]  # type: ignore[assignment]

    if isinstance(pg_secrets, dict):
        env["POSTGRES_HOST"] = str(
            pg_secrets.get("host") or pg_secrets.get("hostname") or env.get("POSTGRES_HOST", "localhost")
        )
        if "port" in pg_secrets:
            env["POSTGRES_PORT"] = str(pg_secrets["port"])
        if "database" in pg_secrets:
            env["POSTGRES_DB"] = str(pg_secrets["database"])
        if "dbname" in pg_secrets and "POSTGRES_DB" not in env:
            env["POSTGRES_DB"] = str(pg_secrets["dbname"])
        if "username" in pg_secrets:
            env["POSTGRES_USER"] = str(pg_secrets["username"])
        if "user" in pg_secrets and "POSTGRES_USER" not in env:
            env["POSTGRES_USER"] = str(pg_secrets["user"])
        if "password" in pg_secrets:
            env["POSTGRES_PASSWORD"] = str(pg_secrets["password"])

    # Sensible defaults when no host/port are provided
    env.setdefault("POSTGRES_HOST", "postgres")
    env.setdefault("POSTGRES_PORT", "5432")

    return env


def _run_gam_ingestion(csv_path: Path) -> Tuple[bool, str]:
    """
    Execute the gam_ingest.py script for the provided CSV.

    Returns
    -------
    Tuple[bool, str]
        (success flag, combined stdout/stderr)
    """
    cmd: List[str] = [
        "python",
        str(INGEST_SCRIPT),
        "--input-csv",
        str(csv_path),
        "--upsert",
    ]

    env: Dict[str, str] = _postgres_env_from_secrets(os.environ.copy())

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    combined_output: str = "\n".join(
        [completed.stdout.strip(), completed.stderr.strip()]
    ).strip()

    return completed.returncode == 0, combined_output


def render_ingestion_controls() -> None:
    """
    Sidebar controls to run the GAM ingestion flow using the latest CSV in data/.
    """
    st.sidebar.subheader("Ingestion")
    st.sidebar.caption(
        "Finds the newest `gam_report_*.csv` in `data/` and loads it into Postgres."
    )

    status_state: Dict[str, str] = st.session_state.get("ingest_status", {})

    if st.sidebar.button("Update", use_container_width=True):
        latest_csv: Optional[Path] = _find_latest_gam_csv(DATA_DIR)

        if latest_csv is None:
            st.sidebar.error("No GAM CSVs found in data/. Download one first.")
            st.session_state["ingest_status"] = {
                "status": "error",
                "message": "No gam_report_*.csv files found in data/",
            }
            return

        with st.spinner(f"Ingesting {latest_csv.name}..."):
            success, output = _run_gam_ingestion(latest_csv)

        message: str = (
            f"Ingested {latest_csv.name}"
            if success
            else f"Ingestion failed for {latest_csv.name}"
        )
        new_status: Dict[str, str] = {
            "file": latest_csv.name,
            "output": output or "(no output)",
            "status": "success" if success else "error",
            "message": message,
        }
        st.session_state["ingest_status"] = new_status

        if success:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)

    # Persist last result so users can review logs after reruns
    status_state = st.session_state.get("ingest_status", {})
    if status_state:
        status_label: str = status_state.get("status", "info")
        file_label: str = status_state.get("file", "latest file")
        message: str = status_state.get("message", "")
        output_log: str = status_state.get("output", "")

        if status_label == "success":
            st.sidebar.success(f"Last run: {file_label}")
        elif status_label == "error":
            st.sidebar.error(message or f"Last run failed for {file_label}")

        if output_log:
            with st.sidebar.expander("Ingestion log", expanded=False):
                st.code(output_log)



def load_club_portfolio(conn: SQLConnection) -> pd.DataFrame:
    """
    Load club-level portfolio data from mart_gam_club_portfolio.

    Parameters
    ----------
    conn : SQLConnection
        Active Streamlit SQL connection.

    Returns
    -------
    pd.DataFrame
        DataFrame at (club_name, order_id) grain.
    """
    df: pd.DataFrame = _safe_query_table(conn, MART_SCHEMA, CLUB_PORTFOLIO_TABLE)

    if df.empty:
        return df

    numeric_cols: List[str] = [
        "impression_goal",
        "club_delivered_impressions",
        "order_delivered_impressions",
        "club_share_of_order_delivery",
        "order_share_of_club_delivery",
        "order_pacing_index",
        "schedule_days",
        "days_elapsed",
        "schedule_pct_elapsed",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for date_col in ["schedule_start_date", "schedule_end_date", "order_start_date", "order_end_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    return df


def load_line_item_diagnostics(conn: SQLConnection) -> pd.DataFrame:
    """
    Load line-item diagnostics from mart_gam_line_item_diagnostics_view.

    Parameters
    ----------
    conn : SQLConnection
        Active Streamlit SQL connection.

    Returns
    -------
    pd.DataFrame
        DataFrame at (order_id, line_item_id, domain_name) grain.
    """
    df: pd.DataFrame = _safe_query_table(conn, MART_SCHEMA, LINE_ITEM_DIAG_TABLE)

    if df.empty:
        return df

    numeric_cols: List[str] = [
        "line_item_total_impressions",
        "line_item_total_clicks",
        "line_item_window_impressions",
        "line_item_window_clicks",
        "line_item_ctr",
        "domain_impressions",
        "order_delivered_impressions",
        "order_delivered_clicks",
        "impression_goal",
        "seed_impression_goal",
        "order_goal_from_line_items",
        "schedule_days",
        "days_elapsed",
        "schedule_pct_elapsed",
        "actual_impressions_per_day",
        "target_impressions_per_day",
        "order_pacing_index",
        "domain_share_of_line_item",
        "line_item_share_of_order",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    if "schedule_start_date" in df.columns:
        df["schedule_start_date"] = pd.to_datetime(
            df["schedule_start_date"], errors="coerce"
        ).dt.date
    if "schedule_end_date" in df.columns:
        df["schedule_end_date"] = pd.to_datetime(
            df["schedule_end_date"], errors="coerce"
        ).dt.date
    if "order_schedule_start_date" in df.columns:
        df["order_schedule_start_date"] = pd.to_datetime(
            df["order_schedule_start_date"], errors="coerce"
        ).dt.date
    if "order_schedule_end_date" in df.columns:
        df["order_schedule_end_date"] = pd.to_datetime(
            df["order_schedule_end_date"], errors="coerce"
        ).dt.date

    return df


def load_line_item_creative_sizes(conn: SQLConnection) -> pd.DataFrame:
    """
    Load per-line-item creative size breakdown from mart_gam_line_item_creative_sizes_view.
    """
    df: pd.DataFrame = _safe_query_table(conn, MART_SCHEMA, LINE_ITEM_CREATIVE_SIZE_TABLE)

    if df.empty:
        return df

    numeric_cols: List[str] = [
        "creative_size_impressions",
        "creative_size_clicks",
        "creative_size_ctr",
        "creative_size_share_of_line_item",
        "order_reported_impressions",
        "impression_goal",
        "order_size_impressions",
        "order_size_line_items",
        "order_size_avg_impressions_per_line_item",
        "size_share_of_order_delivery",
        "size_pct_of_order_goal",
        "line_item_lifetime_impressions",
        "line_item_lifetime_clicks",
        "line_item_total_impressions",
        "line_item_total_clicks",
        "line_item_ctr",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for date_col in ["schedule_start_date", "schedule_end_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    return df


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def build_sidebar_filters(
    df_orders: pd.DataFrame,
) -> Dict[str, Optional[str]]:
    """
    Build sidebar filters and return the selected values.

    Parameters
    ----------
    df_orders : pd.DataFrame
        Order-level DataFrame (mart_gam_order_pacing_summary).

    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary of selected filter values:
        {
            "advertiser": Optional[str],
            "pacing_status": Optional[str],
            "order_delivery_status": Optional[str]
                - "active" -> Started + Not started
                - "Started"
                - "Not started"
                - "Completed"
                - None -> no status filter,
            "date_range": Optional[Tuple[date, date]]
        }
    """
    st.sidebar.header("Filters")

    advertiser_selection: Optional[str] = None
    pacing_selection: Optional[str] = None
    delivery_status_filter: Optional[str] = None
    date_range_selection: Optional[Tuple[date, date]] = None

    # --------------------------------------------------------------
    # Advertiser filter
    # --------------------------------------------------------------
    if "advertiser_name" in df_orders.columns:
        advertisers: List[str] = (
            df_orders["advertiser_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        advertisers = sorted(advertisers)
        advertisers_with_all: List[str] = ["All advertisers"] + advertisers

        advertiser_choice: str = st.sidebar.selectbox(
            "Advertiser",
            options=advertisers_with_all,
            index=0,
        )

        if advertiser_choice != "All advertisers":
            advertiser_selection = advertiser_choice

    # --------------------------------------------------------------
    # Pacing status filter (if available)
    # --------------------------------------------------------------
    if "pacing_status" in df_orders.columns:
        status_values: List[str] = (
            df_orders["pacing_status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        status_values = sorted(status_values)
        status_with_all: List[str] = ["All pacing statuses"] + status_values

        pacing_choice: str = st.sidebar.selectbox(
            "Pacing Status",
            options=status_with_all,
            index=0,
        )

        if pacing_choice != "All pacing statuses":
            pacing_selection = pacing_choice

    # --------------------------------------------------------------
    # Order delivery status filter (Started / Not started / Completed)
    # Default: "Active (Started + Not started)"
    # --------------------------------------------------------------
    if "order_delivery_status_name" in df_orders.columns:
        delivery_status_choice: str = st.sidebar.selectbox(
            "Order Delivery Status",
            options=[
                "Active (Started + Not started)",  # default
                "Started only",
                "Not started only",
                "Completed only",
                "All statuses",
            ],
            index=0,
            help=(
                "Filter orders by high-level delivery status. "
                "'Active' includes both Started and Not started orders."
            ),
        )

        # Map human-readable choice to an internal filter code
        if delivery_status_choice == "Active (Started + Not started)":
            delivery_status_filter = "active"
        elif delivery_status_choice == "Started only":
            delivery_status_filter = "Started"
        elif delivery_status_choice == "Not started only":
            delivery_status_filter = "Not started"
        elif delivery_status_choice == "Completed only":
            delivery_status_filter = "Completed"
        else:
            # "All statuses" -> no filter applied
            delivery_status_filter = None

    st.sidebar.markdown("---")

    # --------------------------------------------------------------
    # Date range filter (option 3: absolute range picker)
    # Uses schedule_end_date if available, else schedule_start_date.
    # --------------------------------------------------------------
    date_col = "schedule_end_date" if "schedule_end_date" in df_orders.columns else None

    if date_col:
        date_series = pd.to_datetime(df_orders[date_col], errors="coerce").dropna()
        if not date_series.empty:
            min_date: date = date_series.min().date()
            max_date: date = date_series.max().date()
            start_default: date = min_date
            end_default: date = max_date

            # Use session state so button clicks can override the picker value.
            date_key = "schedule_date_range"
            if date_key not in st.session_state:
                st.session_state[date_key] = (start_default, end_default)

            selected_dates = st.sidebar.date_input(
                "Schedule date range",
                value=st.session_state[date_key],
                min_value=min_date,
                max_value=max_date,
                help="Filter orders whose schedule dates fall within this range.",
                key=date_key,
            )
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                date_range_selection = (selected_dates[0], selected_dates[1])

            # Quick range shortcuts (buttons below the picker)
            today_date: date = date.today()

            def _bounded(start: date, end: date) -> Tuple[date, date]:
                return (max(start, min_date), min(end, max_date))

            def _set_range(start: date, end: date) -> None:
                st.session_state[date_key] = (start, end)

            st.sidebar.button(
                "Past 7 days",
                on_click=_set_range,
                args=_bounded(today_date - timedelta(days=6), today_date),
            )
            st.sidebar.button(
                "Past 30 days",
                on_click=_set_range,
                args=_bounded(today_date - timedelta(days=29), today_date),
            )
            st.sidebar.button(
                "Year to date",
                on_click=_set_range,
                args=_bounded(date(today_date.year, 1, 1), today_date),
            )
            st.sidebar.button(
                "All available dates",
                on_click=_set_range,
                args=(start_default, end_default),
            )

    st.sidebar.markdown("---")
    render_ingestion_controls()

    return {
        "advertiser": advertiser_selection,
        "pacing_status": pacing_selection,
        "order_delivery_status": delivery_status_filter,
        "date_range": date_range_selection,
    }


def apply_order_filters(
    df_orders: pd.DataFrame,
    filters: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Apply advertiser, pacing status, and order delivery status filters
    to the order DataFrame.

    Parameters
    ----------
    df_orders : pd.DataFrame
        Original order DataFrame.
    filters : Dict[str, Optional[str]]
        Filter values returned from build_sidebar_filters.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (may be empty).
    """
    if df_orders.empty:
        return df_orders

    df_filtered: pd.DataFrame = df_orders.copy()

    advertiser: Optional[str] = filters.get("advertiser")
    pacing_status: Optional[str] = filters.get("pacing_status")
    delivery_status_filter: Optional[str] = filters.get("order_delivery_status")
    date_range: Optional[Tuple[date, date]] = filters.get("date_range")

    # --------------------------------------------------------------
    # Advertiser filter
    # --------------------------------------------------------------
    if advertiser and "advertiser_name" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["advertiser_name"] == advertiser]

    # --------------------------------------------------------------
    # Pacing status filter
    # --------------------------------------------------------------
    if pacing_status and "pacing_status" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["pacing_status"] == pacing_status]

    # --------------------------------------------------------------
    # Order delivery status filter
    #   - "active" -> Started + Not started
    #   - "Started" / "Not started" / "Completed" -> exact match
    # --------------------------------------------------------------
    if delivery_status_filter and "order_delivery_status_name" in df_filtered.columns:
        if delivery_status_filter == "active":
            df_filtered = df_filtered[
                df_filtered["order_delivery_status_name"].isin(
                    ["Started", "Not started"]
                )
            ]
        else:
            df_filtered = df_filtered[
                df_filtered["order_delivery_status_name"] == delivery_status_filter
            ]

    # --------------------------------------------------------------
    # Date range filter (inclusive) using interval overlap
    # --------------------------------------------------------------
    if date_range:
        start_date, end_date = date_range
        range_start = pd.Timestamp(start_date)
        range_end = pd.Timestamp(end_date)

        start_col = "schedule_start_date" if "schedule_start_date" in df_filtered.columns else None
        end_col = "schedule_end_date" if "schedule_end_date" in df_filtered.columns else None

        mask = pd.Series(True, index=df_filtered.index)

        if start_col and end_col:
            start_series = pd.to_datetime(df_filtered[start_col], errors="coerce")
            end_series = pd.to_datetime(df_filtered[end_col], errors="coerce")
            mask = (start_series <= range_end) & (end_series >= range_start)
        elif end_col:
            end_series = pd.to_datetime(df_filtered[end_col], errors="coerce")
            mask = (end_series >= range_start) & (end_series <= range_end)

        df_filtered = df_filtered[mask]

    return df_filtered



# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def render_order_kpis(df_orders: pd.DataFrame) -> None:
    """
    Render top-level KPIs for order pacing.

    KPIs shown:
      - Active Orders
      - Behind Pace
      - On Track
      - Ahead of Pace
      - Urgent Orders (ending soon & behind)
      - Other Statuses (Too Early / No Goal / Unknown)
      - Delivered vs Goal (overall_pct_goal)

    Parameters
    ----------
    df_orders : pd.DataFrame
        Filtered order DataFrame.
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if df_orders.empty:
        st.info("No orders available for the current filters.")
        return

    # Total number of active orders after all filters
    active_orders: int = len(df_orders)

    # ------------------------------------------------------------------
    # Pacing status breakdown
    # ------------------------------------------------------------------
    behind_count: int = 0
    on_track_count: int = 0
    ahead_count: int = 0
    others_count: int = 0

    if "pacing_status" in df_orders.columns:
        # Normalise pacing_status to lower-case strings to avoid casing issues
        pacing_series: pd.Series = (
            df_orders["pacing_status"]
            .astype(str)
            .str.lower()
        )

        # Core pacing bands
        behind_count = int((pacing_series == "behind").sum())
        on_track_count = int((pacing_series == "on_track").sum())
        ahead_count = int((pacing_series == "ahead").sum())

        core_statuses: set[str] = {"behind", "on_track", "ahead"}

        # "Others" = anything not in the core bands, including:
        #   - too_early
        #   - no_goal
        #   - unknown
        #   - any other non-null, non-core statuses
        others_mask: pd.Series = ~pacing_series.isin(core_statuses)
        others_count = int(others_mask.sum())

    # ------------------------------------------------------------------
    # Urgent orders: ending soon & behind pace
    # ------------------------------------------------------------------
    urgent_count: int = 0
    if "is_urgent" in df_orders.columns:
        urgent_count = int(df_orders["is_urgent"].astype(bool).sum())

    # ------------------------------------------------------------------
    # Portfolio-level delivered vs goal (overall_pct_goal)
    # ------------------------------------------------------------------
    delivered_series: pd.Series = df_orders.get(
        "delivered_impressions",
        pd.Series(dtype="float64"),
    )
    goal_series: pd.Series = df_orders.get(
        "impression_goal",
        pd.Series(dtype="float64"),
    )

    # Coerce to numeric safely
    delivered_numeric: pd.Series = pd.to_numeric(
        delivered_series,
        errors="coerce",
    ).fillna(0.0)
    goal_numeric: pd.Series = pd.to_numeric(
        goal_series,
        errors="coerce",
    ).fillna(0.0)

    total_delivered: float = float(delivered_numeric.sum())
    total_goal: float = float(goal_numeric.sum())

    if total_goal > 0.0:
        overall_pct_goal: float = (total_delivered / total_goal) * 100.0
    else:
        overall_pct_goal = 0.0

    # ------------------------------------------------------------------
    # KPI tiles
    # ------------------------------------------------------------------
    # We now have 7 metrics:
    #   1) Active Orders
    #   2) Behind Pace
    #   3) On Track
    #   4) Ahead of Pace
    #   5) Urgent Orders
    #   6) Other Statuses (Too Early / No Goal / Unknown)
    #   7) Delivered vs Goal
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.metric(
            label="Active Orders",
            value=f"{active_orders:,}",
        )

    with col2:
        st.metric(
            label="Behind Pace",
            value=f"{behind_count:,}",
        )

    with col3:
        st.metric(
            label="On Track",
            value=f"{on_track_count:,}",
        )

    with col4:
        st.metric(
            label="Ahead of Pace",
            value=f"{ahead_count:,}",
        )

    with col5:
        st.metric(
            label="Urgent Orders",
            value=f"{urgent_count:,}",
            help=(
                "Orders that are ending within 7 days and are currently behind pace."
            ),
        )

    with col6:
        st.metric(
            label="Other Statuses",
            value=f"{others_count:,}",
            help=(
                "Includes orders that are too early to assess pacing, "
                "have no impression goal, or have an unknown pacing status."
            ),
        )

    with col7:
        st.metric(
            label="Delivered vs Goal",
            value=f"{overall_pct_goal:.1f}%",
        )

    st.caption(
        "The pacing index measures whether a campaign (or order) is delivering "
        "impressions fast enough to reach its final impression goal by the "
        "scheduled end date."
    )



def render_impressions_vs_goal_by_advertiser(df_orders: pd.DataFrame) -> None:
    """
    Render a bar chart of total delivered impressions vs goal by advertiser.

    - Bars: total delivered impressions per advertiser (sum across orders).
    - T-shaped markers: total impression goals per advertiser (sum across orders).
    - X-axis: advertiser_name.
    - Y-axis: impressions, ordered ascending by delivered impressions.

    Parameters
    ----------
    df_orders : pd.DataFrame
        Filtered order DataFrame (mart_gam_order_pacing_summary).
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    required_cols: List[str] = [
        "advertiser_name",
        "delivered_impressions",
        "impression_goal",
    ]
    missing_cols: List[str] = [c for c in required_cols if c not in df_orders.columns]
    if missing_cols:
        st.info(
            "Cannot render 'Impressions vs Goal by Advertiser' chart because "
            f"these columns are missing: {', '.join(missing_cols)}."
        )
        return

    if df_orders.empty:
        st.info(
            "No data available to render 'Impressions vs Goal by Advertiser' chart."
        )
        return

    # ------------------------------------------------------------------
    # Aggregate to advertiser grain
    # ------------------------------------------------------------------
    df_group: pd.DataFrame = df_orders.copy()

    # Ensure numeric types
    df_group["delivered_impressions"] = pd.to_numeric(
        df_group["delivered_impressions"],
        errors="coerce",
    ).fillna(0.0)

    df_group["impression_goal"] = pd.to_numeric(
        df_group["impression_goal"],
        errors="coerce",
    ).fillna(0.0)

    df_group = (
        df_group.groupby("advertiser_name", dropna=True, as_index=False)
        .agg(
            delivered_impressions=("delivered_impressions", "sum"),
            impression_goal=("impression_goal", "sum"),
        )
    )

    df_group["completion_rate"] = np.where(
        df_group["impression_goal"] > 0,
        df_group["delivered_impressions"] / df_group["impression_goal"],
        np.nan,
    )

    if df_group.empty:
        st.info(
            "No aggregated advertiser data is available to render the chart."
        )
        return

    # Sort ascending by delivered impressions for the x-axis order
    df_group = df_group.sort_values(
        by="delivered_impressions",
        ascending=True,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Determine Y-axis max with headroom
    # ------------------------------------------------------------------
    max_val: float = float(
        max(
            df_group["delivered_impressions"].max(),
            df_group["impression_goal"].max(),
        )
    )
    if not math.isfinite(max_val) or max_val <= 0:
        max_val = 1.0
    y_max: float = max_val * 1.1

    # ------------------------------------------------------------------
    # Build Altair chart
    # ------------------------------------------------------------------
    base = alt.Chart(df_group).encode(
        x=alt.X(
            "advertiser_name:N",
            sort=df_group["advertiser_name"].tolist(),
            axis=alt.Axis(
                title="Advertiser",
                labelAngle=-45,
                labelOverlap=False,
            ),
        ),
        y=alt.Y(
            "delivered_impressions:Q",
            scale=alt.Scale(domain=[0, y_max]),
            axis=alt.Axis(
                title="Delivered Impressions vs Goal",
                format=",.0f",
            ),
        ),
    )

    # Bars for delivered impressions
    bar = base.mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    )

    # T-shaped marker for goals
    tick = base.mark_tick(
        color="#f39c12",
        thickness=2,
        size=30,
    ).encode(
        y="impression_goal:Q",
    )

    label = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-6,
        fontSize=10,
        color="#2c3e50",
    ).encode(
        y="delivered_impressions:Q",
        text=alt.Text("completion_rate:Q", format=".0%"),
    )

    chart = (bar + tick + label).properties(
        height=400,
        title="Impressions Delivered vs Goal by Advertiser",
    )

    st.altair_chart(chart, use_container_width=True)


def render_order_pacing_chart(df_orders: pd.DataFrame) -> None:
    """
    Render a vertical bar chart comparing delivered impressions vs goal by order.

    Bars are coloured by pacing / urgency:
      - behind_urgent (ending soon & behind) -> bright red
      - behind -> dark red
      - on_track -> green
      - ahead -> blue
      - too_early / no_goal / unknown -> grey variants

    Parameters
    ----------
    df_orders : pd.DataFrame
        Filtered order DataFrame.
    """
    required_cols: List[str] = [
        "order_label",
        "delivered_impressions",
        "impression_goal",
    ]
    if any(col not in df_orders.columns for col in required_cols):
        st.info(
            "Cannot render Order Pacing chart because required columns are missing."
        )
        return

    if df_orders.empty:
        st.info("No data available to render the Order Pacing chart.")
        return

    df_plot: pd.DataFrame = df_orders.copy()

    df_plot["delivered_impressions"] = pd.to_numeric(
        df_plot["delivered_impressions"],
        errors="coerce",
    ).fillna(0.0)

    df_plot["impression_goal"] = pd.to_numeric(
        df_plot["impression_goal"],
        errors="coerce",
    ).fillna(0.0)

    df_plot["completion_rate"] = np.where(
        df_plot["impression_goal"] > 0,
        df_plot["delivered_impressions"] / df_plot["impression_goal"],
        np.nan,
    )

    # Sort by pacing_index (worst to best) when available; otherwise by delivered impressions
    if "pacing_index" in df_plot.columns:
        df_plot["pacing_index"] = pd.to_numeric(
            df_plot["pacing_index"],
            errors="coerce",
        )
        df_plot = df_plot.sort_values(
            by="pacing_index",
            ascending=True,  # lower pacing -> more behind -> first
            na_position="last",
            ignore_index=True,
        )
    else:
        df_plot = df_plot.sort_values(
            by="delivered_impressions",
            ascending=True,
            ignore_index=True,
        )

    # Determine y-axis max for a bit of headroom
    max_val: float = float(
        max(df_plot["delivered_impressions"].max(), df_plot["impression_goal"].max())
    )
    if not math.isfinite(max_val) or max_val <= 0.0:
        max_val = 1.0
    y_max: float = max_val * 1.1

    # ------------------------------------------------------------------
    # Derive pacing_urgency_band
    # ------------------------------------------------------------------
    if "pacing_status" in df_plot.columns:
        df_plot["pacing_status"] = (
            df_plot["pacing_status"]
            .astype(str)
            .str.lower()
            .fillna("unknown")
        )

        if "is_urgent" in df_plot.columns:
            df_plot["is_urgent"] = df_plot["is_urgent"].astype(bool)
            df_plot["pacing_urgency_band"] = np.where(
                (df_plot["is_urgent"]) & (df_plot["pacing_status"] == "behind"),
                "behind_urgent",
                df_plot["pacing_status"],
            )
        else:
            df_plot["pacing_urgency_band"] = df_plot["pacing_status"]

        pacing_domain: List[str] = [
            "behind_urgent",
            "behind",
            "on_track",
            "ahead",
            "too_early",
            "no_goal",
            "unknown",
        ]
        pacing_colors: List[str] = [
            "#ff0000",  # behind_urgent -> bright red
            "#f1c40f",  # behind -> yellow
            "#27ae60",  # on_track
            "#2980b9",  # ahead
            "#95a5a6",  # too_early
            "#7f8c8d",  # no_goal
            "#bdc3c7",  # unknown
        ]

        colour_encoding = alt.Color(
            "pacing_urgency_band:N",
            title="Pacing / Urgency",
            scale=alt.Scale(
                domain=pacing_domain,
                range=pacing_colors,
            ),
            legend=alt.Legend(title="Pacing / Urgency"),
        )
    else:
        colour_encoding = alt.value("#2980b9")

    base = alt.Chart(df_plot).encode(
        x=alt.X(
            "order_label:N",
            sort=None,
            axis=alt.Axis(
                title="Order",
                labelAngle=-45,
            ),
        ),
        y=alt.Y(
            "delivered_impressions:Q",
            scale=alt.Scale(domain=[0, y_max]),
            axis=alt.Axis(
                title="Delivered Impressions vs Goal",
                format=",.0f",
            ),
        ),
    )

    # Bars for delivered impressions
    bar = base.mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    ).encode(
        color=colour_encoding,
    )

    # T-shaped marker for goals
    tick = base.mark_tick(
        color="#f39c12",
        thickness=2,
        size=30,
    ).encode(
        y="impression_goal:Q",
    )

    label = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-6,
        fontSize=10,
        color="#2c3e50",
    ).encode(
        y="delivered_impressions:Q",
        text=alt.Text("completion_rate:Q", format=".0%"),
    )

    chart = (bar + tick + label).properties(
        height=400,
    )

    st.altair_chart(chart, use_container_width=True)



def render_order_table(df_orders: pd.DataFrame) -> None:
    """
    Render a detailed order table.

    Adds a human-readable 'urgency_flag' column when urgency information
    is available:

      - '⚠️ Ending soon & behind' when is_urgent == True
      - '' otherwise

    Parameters
    ----------
    df_orders : pd.DataFrame
        Filtered order DataFrame.
    """
    if df_orders.empty:
        st.info("No order rows to display for the current filters.")
        return

    display_cols: List[str] = []

    # We will inject 'urgency_flag' as the first column, if possible
    base_candidates: List[str] = [
        "order_id",
        "order_name",
        "advertiser_name",
        "order_delivery_status_name",
        "schedule_start_date",
        "schedule_end_date",
        "impression_goal",
        "delivered_impressions",
        "delivered_impressions_pct_of_goal",
        "schedule_pct_elapsed",
        "pacing_index",
    ]

    for candidate in base_candidates:
        if candidate in df_orders.columns:
            display_cols.append(candidate)

    df_display: pd.DataFrame = df_orders[display_cols].copy()

    # ------------------------------------------------------------------
    # Urgency flag column
    # ------------------------------------------------------------------
    if "is_urgent" in df_orders.columns:
        # Align index so we pick the correct rows
        urgency_series: pd.Series = df_orders.loc[df_display.index, "is_urgent"].astype(bool)

        df_display.insert(
            0,
            "urgency_flag",
            urgency_series.map(
                lambda x: "⚠️ Ending soon & behind" if x else ""
            ),
        )

    # Format numeric percentages
    if "delivered_impressions_pct_of_goal" in df_display.columns:
        df_display["delivered_impressions_pct_of_goal"] = (
            pd.to_numeric(df_display["delivered_impressions_pct_of_goal"], errors="coerce")
            .fillna(0.0)
            .map(lambda x: f"{x * 100.0:.1f}%" if math.isfinite(x) else "")
        )

    if "schedule_pct_elapsed" in df_display.columns:
        df_display["schedule_pct_elapsed"] = (
            pd.to_numeric(df_display["schedule_pct_elapsed"], errors="coerce")
            .fillna(0.0)
            .map(lambda x: f"{x * 100.0:.1f}%" if math.isfinite(x) else "")
        )

    if "pacing_index" in df_display.columns:
        df_display["pacing_index"] = (
            pd.to_numeric(df_display["pacing_index"], errors="coerce")
            .fillna(0.0)
            .map(lambda x: f"{x:.2f}" if math.isfinite(x) else "")
        )

    st.dataframe(df_display, use_container_width=True)



def build_club_summary(df_club: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mart_gam_club_portfolio to the club grain and compute health metrics.

    The resulting DataFrame is one row per club and includes:
      - club_total_impressions
      - active_orders
      - urgent_orders (ending soon AND behind)
      - impressions split by order pacing status
      - at_risk_share (share of club impressions in 'behind' orders)
      - impression-weighted club_pacing_index
      - top_order_share and top_order_pacing_status
      - club_health_band (healthy / watch / at_risk / unknown)
    """
    if df_club.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Determine club label columns
    # ------------------------------------------------------------------
    group_cols: List[str] = []
    if "club_name" in df_club.columns:
        group_cols.append("club_name")
    if "club_display_name" in df_club.columns:
        group_cols.append("club_display_name")

    if not group_cols:
        # Cannot compute club-level metrics without a club identifier
        return pd.DataFrame()

    df_work: pd.DataFrame = df_club.copy()

    # ------------------------------------------------------------------
    # Ensure numeric types for key metrics
    # ------------------------------------------------------------------
    numeric_cols: List[str] = [
        "club_delivered_impressions",
        "order_pacing_index",
        "order_share_of_club_delivery",
    ]

    for col in numeric_cols:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
        else:
            # If a required numeric column is missing, bail out early
            return pd.DataFrame()

    # Replace missing pacing status with 'unknown' to avoid null handling later
    if "order_pacing_status" in df_work.columns:
        df_work["order_pacing_status"] = (
            df_work["order_pacing_status"]
            .astype(str)
            .fillna("unknown")
        )
    else:
        # If pacing status is not present, treat all as 'unknown'
        df_work["order_pacing_status"] = "unknown"

    # ------------------------------------------------------------------
    # Base aggregation: total impressions, active orders, urgent orders
    # ------------------------------------------------------------------
    agg_dict: Dict[str, Tuple[str, str]] = {
            "club_total_impressions": (
                "club_delivered_impressions",
                "sum",
            ),
            "active_orders": (
                "order_id",
                pd.Series.nunique,
            ),
        }

    # If the mart provides a per-row is_urgent_order flag, aggregate it.
    # This keeps the function robust even if the column isn't there yet.
    if "is_urgent_order" in df_work.columns:
        # Ensure boolean; treat non-True as False
        df_work["is_urgent_order"] = df_work["is_urgent_order"].astype(bool)
        agg_dict["urgent_orders"] = (
            "is_urgent_order",
            "sum",
        )

    df_base: pd.DataFrame = (
        df_work.groupby(group_cols, dropna=True, as_index=False)
        .agg(**agg_dict)
    )

    # Guard against division by zero later
    df_base["club_total_impressions"] = df_base["club_total_impressions"].fillna(0.0)

    if "urgent_orders" not in df_base.columns:
        # If the mart didn't provide is_urgent_order, default to zero
        df_base["urgent_orders"] = 0

    # ------------------------------------------------------------------
    # Impressions split by pacing status (behind / on_track / ahead)
    # ------------------------------------------------------------------
    # Initialize helper columns
    df_work["imps_from_behind"] = np.where(
        df_work["order_pacing_status"] == "behind",
        df_work["club_delivered_impressions"],
        0.0,
    )
    df_work["imps_from_on_track"] = np.where(
        df_work["order_pacing_status"] == "on_track",
        df_work["club_delivered_impressions"],
        0.0,
    )
    df_work["imps_from_ahead"] = np.where(
        df_work["order_pacing_status"] == "ahead",
        df_work["club_delivered_impressions"],
        0.0,
    )

    df_imps_split: pd.DataFrame = (
        df_work.groupby(group_cols, dropna=True, as_index=False)
        .agg(
            imps_from_behind=("imps_from_behind", "sum"),
            imps_from_on_track=("imps_from_on_track", "sum"),
            imps_from_ahead=("imps_from_ahead", "sum"),
        )
    )

    # Merge split impressions into base metrics
    df_summary: pd.DataFrame = df_base.merge(
        df_imps_split,
        on=group_cols,
        how="left",
    )

    for col in ["imps_from_behind", "imps_from_on_track", "imps_from_ahead"]:
        df_summary[col] = df_summary.get(col, 0.0).fillna(0.0)

    # ------------------------------------------------------------------
    # Impression-weighted club pacing index
    # ------------------------------------------------------------------
    df_work["weighted_pacing_imps"] = np.where(
        np.isfinite(df_work["order_pacing_index"]),
        df_work["order_pacing_index"] * df_work["club_delivered_impressions"],
        0.0,
    )
    df_work["valid_pacing_imps"] = np.where(
        np.isfinite(df_work["order_pacing_index"]),
        df_work["club_delivered_impressions"],
        0.0,
    )

    df_pacing: pd.DataFrame = (
        df_work.groupby(group_cols, dropna=True, as_index=False)
        .agg(
            pacing_weighted_imps=("weighted_pacing_imps", "sum"),
            pacing_imps=("valid_pacing_imps", "sum"),
        )
    )

    df_summary = df_summary.merge(df_pacing, on=group_cols, how="left")

    df_summary["club_pacing_index"] = 0.0
    # Compute weighted average where we have non-zero valid impressions
    has_pacing_mask: pd.Series = df_summary["pacing_imps"] > 0.0
    df_summary.loc[has_pacing_mask, "club_pacing_index"] = (
        df_summary.loc[has_pacing_mask, "pacing_weighted_imps"]
        / df_summary.loc[has_pacing_mask, "pacing_imps"]
    )

    # Replace invalid values with NaN for clearer downstream handling
    df_summary["club_pacing_index"] = df_summary["club_pacing_index"].replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Top-order share and pacing status per club
    # ------------------------------------------------------------------
    # Sort so the largest order_share_of_club_delivery appears first for each club
    sort_cols: List[str] = group_cols + ["order_share_of_club_delivery"]
    sort_ascending: List[bool] = [True] * len(group_cols) + [False]

    df_sorted: pd.DataFrame = df_work.sort_values(
        by=sort_cols,
        ascending=sort_ascending,
    )

    # Take the first row per club -> highest order_share_of_club_delivery
    df_top_order: pd.DataFrame = (
        df_sorted.groupby(group_cols, dropna=True, as_index=False)
        .first()
    )

    # Keep only the columns we care about
    df_top_order = df_top_order[
        group_cols
        + [
            "order_share_of_club_delivery",
            "order_pacing_status",
            "order_name",
        ]
    ].rename(
        columns={
            "order_share_of_club_delivery": "top_order_share",
            "order_pacing_status": "top_order_pacing_status",
            "order_name": "top_order_name",
        }
    )

    df_summary = df_summary.merge(df_top_order, on=group_cols, how="left")

    # ------------------------------------------------------------------
    # Derived metrics: at_risk_share and club_health_band
    # ------------------------------------------------------------------
    def _compute_at_risk_share(row: pd.Series) -> float:
        total_imps: float = float(row.get("club_total_impressions", 0.0) or 0.0)
        at_risk_imps: float = float(row.get("imps_from_behind", 0.0) or 0.0)
        if total_imps <= 0.0:
            return 0.0
        return at_risk_imps / total_imps

    df_summary["at_risk_share"] = df_summary.apply(
        _compute_at_risk_share,
        axis=1,
    )

    def _classify_club_health(row: pd.Series) -> str:
        """
        Classify club health using:
          - club_pacing_index
          - at_risk_share
          - top_order_share + top_order_pacing_status
        """
        pacing_idx_raw = row.get("club_pacing_index")
        at_risk_raw = row.get("at_risk_share")
        top_share_raw = row.get("top_order_share")
        top_status_raw = str(row.get("top_order_pacing_status") or "unknown")

        pacing_idx: float = float(pacing_idx_raw) if pacing_idx_raw is not None else float("nan")
        at_risk_share_val: float = float(at_risk_raw) if at_risk_raw is not None else float("nan")
        top_order_share_val: float = float(top_share_raw) if top_share_raw is not None else float("nan")

        # If we lack any meaningful data, classify as unknown
        if not math.isfinite(pacing_idx) and not math.isfinite(at_risk_share_val):
            return "unknown"

        # Inferred thresholds (you can tune these as needed)
        # Healthy when pacing is roughly on target and at-risk volume is low
        if (
            math.isfinite(pacing_idx)
            and pacing_idx >= 0.95
            and math.isfinite(at_risk_share_val)
            and at_risk_share_val <= 0.30
        ):
            return "healthy"

        # At risk if clearly under-pacing or heavily concentrated in behind orders
        if (
            (math.isfinite(pacing_idx) and pacing_idx < 0.80)
            or (math.isfinite(at_risk_share_val) and at_risk_share_val > 0.60)
            or (
                math.isfinite(top_order_share_val)
                and top_order_share_val > 0.50
                and top_status_raw == "behind"
            )
        ):
            return "at_risk"

        # Otherwise, it's somewhere in the middle and should be watched
        return "watch"

    df_summary["club_health_band"] = df_summary.apply(
        _classify_club_health,
        axis=1,
    )

    return df_summary

def render_club_health_overview(df_club: pd.DataFrame) -> None:
    """
    Render a high-level club health overview for the current advertiser scope.

    Uses mart_gam_club_portfolio rows and:
      - Aggregates to one row per club for KPI tiles (via build_club_summary).
      - Builds an explicit pie chart per club:
          * Pie total = total impressions delivered by this club.
          * Slices = impressions by order pacing status (behind / on_track / ahead / other).

    The layout is:
      - Top row: Sports league aggregate club centered.
      - Next row: 3 clubs.
      - Next row: 3 clubs.

    Parameters
    ----------
    df_club : pd.DataFrame
        Club portfolio mart rows already filtered by advertiser (if applicable).
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if df_club.empty:
        st.info("No club-level data available to compute the club health overview.")
        return

    # Build club-level summary for KPI tiles (health bands etc.)
    df_summary: pd.DataFrame = build_club_summary(df_club)
    if df_summary.empty:
        st.info(
            "Club health overview could not be computed because required columns "
            "are missing from mart_gam_club_portfolio."
        )
        return

    # Prefer club_display_name, fall back to club_name if needed
    label_col: str
    if "club_display_name" in df_summary.columns:
        label_col = "club_display_name"
    elif "club_name" in df_summary.columns:
        label_col = "club_name"
    else:
        st.info("Club identifiers are missing from the summary; cannot render overview.")
        return

    st.subheader("Club Health Overview")
    st.caption(
        "Use the in-chart toggle to switch between pacing mix (stacked by pacing status) "
        "and delivery scale (total impressions per club)."
    )

    # ------------------------------------------------------------------
    # Build status-split impressions per club
    # ------------------------------------------------------------------
    df_status: pd.DataFrame = df_club.copy()

    # Ensure label column exists in df_status
    if label_col not in df_status.columns:
        if "club_name" in df_status.columns:
            df_status[label_col] = df_status["club_name"].astype(str)
        else:
            st.info(
                "Cannot render impressions-by-status chart due to missing club labels."
            )
            return

    # Ensure numeric and pacing status fields
    if "club_delivered_impressions" not in df_status.columns:
        st.info(
            "Cannot render impressions-by-status chart because "
            "'club_delivered_impressions' is missing."
        )
        return

    df_status["club_delivered_impressions"] = pd.to_numeric(
        df_status["club_delivered_impressions"],
        errors="coerce",
    ).fillna(0.0)

    if "order_pacing_status" not in df_status.columns:
        df_status["order_pacing_status"] = "unknown"
    else:
        df_status["order_pacing_status"] = (
            df_status["order_pacing_status"]
            .astype(str)
            .fillna("unknown")
        )

    # Aggregate impressions by club + pacing status
    df_status_grouped: pd.DataFrame = (
        df_status.groupby([label_col, "order_pacing_status"], dropna=True, as_index=False)
        .agg(
            club_impressions=("club_delivered_impressions", "sum"),
        )
    )

    if df_status_grouped.empty:
        st.info("No impressions-by-status data available to render the chart.")
        return

    # Compute per-club totals and within-club share (for tooltips)
    df_status_grouped["club_impressions"] = pd.to_numeric(
        df_status_grouped["club_impressions"],
        errors="coerce",
    ).fillna(0.0)

    club_totals: pd.DataFrame = (
        df_status_grouped.groupby(label_col, as_index=False)
        .agg(
            club_total=("club_impressions", "sum"),
        )
    )

    df_status_grouped = df_status_grouped.merge(
        club_totals,
        on=label_col,
        how="left",
    )

    def _compute_share(row: pd.Series) -> float:
        total_val: float = float(row.get("club_total", 0.0) or 0.0)
        part_val: float = float(row.get("club_impressions", 0.0) or 0.0)
        if total_val <= 0.0:
            return 0.0
        return part_val / total_val

    df_status_grouped["impression_share"] = df_status_grouped.apply(
        _compute_share,
        axis=1,
    )

    # Compute each club's behind share for sorting
    behind_share: pd.DataFrame = (
        df_status_grouped[df_status_grouped["order_pacing_status"] == "behind"][
            [label_col, "impression_share"]
        ]
        .rename(columns={"impression_share": "behind_share"})
    )

    df_chart: pd.DataFrame = df_status_grouped.merge(
        behind_share,
        on=label_col,
        how="left",
    )
    df_chart["behind_share"] = df_chart["behind_share"].fillna(0.0)

    # Build sort order: by behind-share descending, with Sports league pinned last
    sort_order: List[str] = (
        df_chart[[label_col, "behind_share"]]
        .dropna(subset=[label_col])
        .drop_duplicates(subset=[label_col])
        .sort_values("behind_share", ascending=False)
    )[label_col].astype(str).tolist()

    sportsleague_label: Optional[str] = None
    for lbl in sort_order:
        lower_lbl = lbl.lower()
        if "sports league" in lower_lbl or lower_lbl == "sportsleague":
            sportsleague_label = lbl
            break

    if sportsleague_label is not None:
        sort_order = [lbl for lbl in sort_order if lbl != sportsleague_label] + [sportsleague_label]

    # Colour mapping for pacing status:
    #   behind   -> red
    #   on_track -> green
    #   ahead    -> blue
    #   others   -> grey variants
    pacing_domain: List[str] = [
        "behind",
        "on_track",
        "ahead",
        "too_early",
        "no_goal",
        "unknown",
    ]
    pacing_colors: List[str] = [
        "#d16f60",  # behind -> softened red
        "#58b67e",  # on_track -> softened green
        "#6b7fdc",  # ahead -> softened blue
        "#b0b7ba",  # too_early -> softer grey
        "#9ba2a6",  # no_goal -> softer dark grey
        "#ced4d7",  # unknown -> lighter grey
    ]

    # Build vertical stacked bar chart sorted by behind-share
    num_clubs: int = df_chart[label_col].dropna().astype(str).nunique()
    chart_width: int = max(400, num_clubs * 80)
    scale_sort: List[str] = (
        club_totals[[label_col, "club_total"]]
        .dropna(subset=[label_col])
        .sort_values("club_total", ascending=False)
    )[label_col].astype(str).tolist()

    # Keep Sports league pinned to the right for both chart views
    sportsleague_scale_label: Optional[str] = None
    for lbl in scale_sort:
        lower_lbl = lbl.lower()
        if "sports league" in lower_lbl or lower_lbl == "sportsleague":
            sportsleague_scale_label = lbl
            break
    if sportsleague_scale_label is not None:
        scale_sort = [lbl for lbl in scale_sort if lbl != sportsleague_scale_label] + [sportsleague_scale_label]

    club_totals_chart = club_totals.copy()
    club_totals_chart["view_mode"] = "scale"

    view_choice: str = st.radio(
        "Club chart view",
        options=["Pacing mix", "Delivery scale"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    show_scale_view: bool = view_choice == "Delivery scale"

    st.subheader("Club Health Overview" if not show_scale_view else "Club Delivery Scale")
    st.caption(
        "Stacked bars show each club's impression mix by pacing status, sorted by the "
        "share of behind-pace orders (red) so the riskiest clubs appear at the top."
        if not show_scale_view
        else "Vertical bars show total delivered impressions per club, sorted to highlight delivery scale differences."
    )

    bar_chart: alt.Chart = (
        alt.Chart(df_chart)
        .mark_bar(cornerRadiusTopLeft=9, cornerRadiusTopRight=9)
        .encode(
            x=alt.X(
                f"{label_col}:N",
                sort=sort_order,
                title="Club",
                scale=alt.Scale(paddingInner=0.35),
            ),
            y=alt.Y(
                "impression_share:Q",
                stack="zero",
                title="Share of Club Impressions",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "order_pacing_status:N",
                title="Order Pacing Status",
                scale=alt.Scale(domain=pacing_domain, range=pacing_colors),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="Club"),
                alt.Tooltip("order_pacing_status:N", title="Pacing"),
                alt.Tooltip("club_impressions:Q", title="Impressions", format=","),
                alt.Tooltip(
                    "impression_share:Q",
                    title="Share of Club",
                    format=".0%",
                ),
                alt.Tooltip("behind_share:Q", title="Behind Share", format=".0%"),
                alt.Tooltip("club_total:Q", title="Club Total", format=","),
            ],
        )
        .properties(width=chart_width, height=360)
    )

    club_totals_chart = club_totals.copy()
    club_totals_chart["view"] = "Delivered impressions"

    scale_chart: alt.Chart = (
        alt.Chart(club_totals_chart)
        .mark_bar(cornerRadiusTopLeft=9, cornerRadiusTopRight=9)
        .encode(
            x=alt.X(
                f"{label_col}:N",
                sort=scale_sort,
                title="Club",
                scale=alt.Scale(paddingInner=0.35),
            ),
            y=alt.Y(
                "club_total:Q",
                title="Delivered Impressions",
            ),
            color=alt.Color(
                "view:N",
                title=None,
                scale=alt.Scale(range=["#6b7fdc"]),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="Club"),
                alt.Tooltip("club_total:Q", title="Delivered Impressions", format=","),
            ],
        )
        .properties(width=chart_width, height=360)
    )

    if show_scale_view:
        st.altair_chart(scale_chart, use_container_width=True)
    else:
        st.altair_chart(bar_chart, use_container_width=True)

def render_club_portfolio(
    df_club: pd.DataFrame,
    advertiser_filter: Optional[str],
) -> Optional[str]:
    """
    Render the Club Portfolio section.

    Parameters
    ----------
    df_club : pd.DataFrame
        Club portfolio mart (mart_gam_club_portfolio).
    advertiser_filter : Optional[str]
        Currently selected advertiser from the sidebar, or None.
    """
    if df_club.empty:
        st.info("No club portfolio data is available.")
        return None

    # ------------------------------------------------------------------
    # Apply advertiser filter (if any)
    # ------------------------------------------------------------------
    df_filtered: pd.DataFrame = df_club.copy()

    if advertiser_filter and "advertiser_name" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["advertiser_name"] == advertiser_filter]

    if df_filtered.empty:
        st.info("No club-level rows match the current advertiser filter.")
        return None

    # ------------------------------------------------------------------
    # High-level club health overview (before drilling into a specific club)
    # ------------------------------------------------------------------
    render_club_health_overview(df_filtered)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Club selector and per-club order contribution chart
    # ------------------------------------------------------------------
    club_label_col: Optional[str] = None
    if "club_display_name" in df_filtered.columns:
        club_label_col = "club_display_name"
    elif "club_name" in df_filtered.columns:
        club_label_col = "club_name"

    if club_label_col is None:
        st.info("Cannot render club-level drill-down because club labels are missing.")
        return None

    # Compute which clubs have urgent orders (if available)
    urgent_club_labels: List[str] = []
    try:
        df_summary: pd.DataFrame = build_club_summary(df_filtered)
        if (
            not df_summary.empty
            and "urgent_orders" in df_summary.columns
            and club_label_col in df_summary.columns
        ):
            urgent_club_labels = (
                df_summary.loc[df_summary["urgent_orders"] > 0, club_label_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
    except Exception as exc:  # noqa: BLE001
        # Fail safe: just don't filter by urgent if something goes wrong
        urgent_club_labels = []

    show_only_urgent: bool = False
    if urgent_club_labels:
        show_only_urgent = st.checkbox(
            "Show only clubs with urgent orders",
            value=False,
            help=(
                "When checked, the club selector only includes clubs that currently "
                "have at least one order that is ending soon and behind pace."
            ),
        )

    # Optionally restrict df_filtered to clubs with urgent orders
    df_club_for_selector: pd.DataFrame = df_filtered.copy()
    if show_only_urgent and urgent_club_labels:
        df_club_for_selector = df_club_for_selector[
            df_club_for_selector[club_label_col].astype(str).isin(urgent_club_labels)
        ]

    if df_club_for_selector.empty:
        st.info("No clubs match the current urgency / advertiser filters.")
        return None

    club_values: List[str] = (
        df_club_for_selector[club_label_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    club_values = sorted(club_values)
    club_options: List[str] = ["All clubs"] + club_values

    selected_club: str = st.selectbox(
        "Club",
        options=club_options,
        index=0,
    )

    df_club_drill: pd.DataFrame = df_club_for_selector.copy()
    if selected_club != "All clubs":
        df_club_drill = df_club_drill[df_club_drill[club_label_col] == selected_club]

    if df_club_drill.empty:
        st.info("No club-level rows match the selected club.")
        return None

    # Return the selected club so diagnostics can use the same filter
    return selected_club


def render_order_share_by_club(
    df_club: pd.DataFrame,
    advertiser_filter: Optional[str],
    selected_club: Optional[str],
) -> None:
    """
    Show each order's share of total delivered impressions for the selected club.

    order_share = order_delivered_impressions / total_delivered_impressions_for_that_club
    """
    if df_club.empty:
        st.info("No club portfolio data is available to compute order share.")
        return

    df_work: pd.DataFrame = df_club.copy()

    # Apply advertiser and club filters to stay aligned with the rest of the dashboard
    if advertiser_filter and "advertiser_name" in df_work.columns:
        df_work = df_work[df_work["advertiser_name"] == advertiser_filter]

    club_label_col: Optional[str] = None
    if "club_display_name" in df_work.columns:
        club_label_col = "club_display_name"
    elif "club_name" in df_work.columns:
        club_label_col = "club_name"

    if club_label_col:
        if selected_club and selected_club != "All clubs":
            df_work = df_work[df_work[club_label_col] == selected_club]
    else:
        st.info("Cannot compute order share because club labels are missing.")
        return

    if df_work.empty:
        st.info("No rows match the current club/advertiser filters for order share.")
        return

    # Use club-level delivered impressions for consistency with the club charts
    delivered_col: Optional[str] = None
    if "club_delivered_impressions" in df_work.columns:
        delivered_col = "club_delivered_impressions"
    elif "order_reported_impressions" in df_work.columns:
        delivered_col = "order_reported_impressions"

    if delivered_col is None:
        st.info(
            "club_delivered_impressions (or a compatible impressions column) is missing from the club mart."
        )
        return

    df_work[delivered_col] = pd.to_numeric(
        df_work[delivered_col],
        errors="coerce",
    ).fillna(0.0)

    # Total delivered per club and share per order
    df_totals = (
        df_work.groupby(club_label_col, as_index=False)
        .agg(total_delivered=(delivered_col, "sum"))
    )

    df_work = df_work.merge(df_totals, on=club_label_col, how="left")
    df_work["order_share_of_club"] = 0.0
    has_total = df_work["total_delivered"] > 0
    # Store as percentage (0-100) instead of decimal (0-1) for proper display
    df_work.loc[has_total, "order_share_of_club"] = (
        df_work.loc[has_total, delivered_col]
        / df_work.loc[has_total, "total_delivered"]
        * 100.0
    )

    display_cols: List[str] = []
    for candidate in [
        club_label_col,
        "order_name",
        "advertiser_name",
        delivered_col,
        "order_share_of_club",
    ]:
        if candidate in df_work.columns:
            display_cols.append(candidate)

    df_display = df_work[display_cols].copy()
    df_display[delivered_col] = pd.to_numeric(
        df_display[delivered_col], errors="coerce"
    ).fillna(0.0)
    df_display["order_share_of_club"] = pd.to_numeric(
        df_display["order_share_of_club"], errors="coerce"
    ).fillna(0.0)

    df_display = df_display.sort_values(
        by="order_share_of_club",
        ascending=False,
        ignore_index=True,
    )

    # Add a totals row for peace of mind
    total_delivered_sum: float = float(df_work[delivered_col].sum())
    total_share_sum: float = float(df_work["order_share_of_club"].sum())
    total_row: Dict[str, float | str] = {
        club_label_col: "Total",  # type: ignore[index]
        "order_name": "",
        "advertiser_name": "",
        delivered_col: total_delivered_sum,
        "order_share_of_club": total_share_sum,
    }
    df_display = pd.concat([df_display, pd.DataFrame([total_row])], ignore_index=True)

    st.subheader("Order Share of Club Impressions")
    st.caption(
        "Share of each order's delivered impressions relative to the selected club's total."
    )
    
    # Configure column formatting - values are already percentages (0-100)
    column_config = {
        "order_share_of_club": st.column_config.NumberColumn(
            "Order Share of Club",
            format="%.2f%%",  # Display with 2 decimal places and % symbol
            help="Percentage of club's total impressions delivered by this order"
        ),
        delivered_col: st.column_config.NumberColumn(
            "Delivered Impressions",
            format="%d",  # Display as integer with comma separators
        )
    }
    
    st.dataframe(
        df_display,
        use_container_width=True,
        column_config=column_config
    )

def render_creative_size_share(
    df_sizes: pd.DataFrame,
    advertiser_filter: Optional[str],
) -> None:
    """
    Show share of impressions by rendered creative size within a selected line item.
    """
    st.subheader("Rendered Creative Size Share")
    st.caption(
        "See creative size distribution from GAM raw. Choose a line item or 'All line items' within the current filters. "
        "Club-level splits are not available because the GAM export does not include size at the domain grain."
    )

    if df_sizes.empty:
        st.info("Creative size mart is empty or unavailable.")
        return

    df_work: pd.DataFrame = df_sizes.copy()

    if advertiser_filter and "advertiser_name" in df_work.columns:
        df_work = df_work[df_work["advertiser_name"] == advertiser_filter]

    if df_work.empty:
        st.info("No creative size rows match the current advertiser filter.")
        return

    # Optional order filter to narrow line item list
    order_label_col: Optional[str] = None
    if "order_name" in df_work.columns and "advertiser_name" in df_work.columns:
        df_work["order_label"] = (
            df_work["order_name"].astype(str)
            + " — "
            + df_work["advertiser_name"].astype(str)
        )
        order_label_col = "order_label"
    elif "order_name" in df_work.columns:
        df_work["order_label"] = df_work["order_name"].astype(str)
        order_label_col = "order_label"

    if order_label_col is not None:
        order_options: List[str] = ["All orders"] + sorted(
            df_work[order_label_col].dropna().astype(str).unique().tolist()
        )
        selected_order: str = st.selectbox(
            "Order (to narrow line items)",
            options=order_options,
            index=0,
            key="rendered_size_order_filter",
        )
        if selected_order != "All orders":
            df_work = df_work[df_work[order_label_col] == selected_order]

    if df_work.empty:
        st.info("No creative size rows match the selected order/advertiser filters.")
        return

    # Require a single line item selection for an accurate share-of-line view
    line_item_label_col: Optional[str] = None
    if "line_item_name" in df_work.columns and "line_item_id" in df_work.columns:
        df_work["line_item_label"] = (
            df_work["line_item_name"].astype(str)
            + " ("
            + df_work["line_item_id"].astype(str)
            + ")"
        )
        line_item_label_col = "line_item_label"
    elif "line_item_id" in df_work.columns:
        df_work["line_item_label"] = df_work["line_item_id"].astype(str)
        line_item_label_col = "line_item_label"

    if line_item_label_col is None:
        st.info("Line item columns are missing; cannot render creative size share.")
        return

    line_item_options: List[str] = sorted(
        df_work[line_item_label_col].dropna().astype(str).unique().tolist()
    )
    if not line_item_options:
        st.info("No line items available to select.")
        return

    selectable_options: List[str] = ["All line items"] + line_item_options
    selected_line_item: str = st.selectbox(
        "Line item",
        options=selectable_options,
        index=0,
        key="rendered_size_line_item_filter",
    )

    if selected_line_item != "All line items":
        df_work = df_work[df_work[line_item_label_col] == selected_line_item]

        if df_work.empty:
            st.info("No creative size rows match the selected line item.")
            return

    # Optional device filter
    device_choice: Optional[str] = None
    if "device_category_name" in df_work.columns:
        device_options: List[str] = (
            df_work["device_category_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        device_options = sorted(device_options)
        device_options_with_all: List[str] = ["All devices"] + device_options
        device_choice = st.selectbox(
            "Device category",
            options=device_options_with_all,
            index=0,
            key="rendered_size_device_filter",
        )
        if device_choice != "All devices":
            df_work = df_work[df_work["device_category_name"] == device_choice]

    if df_work.empty:
        st.info("No creative size rows match the selected line item/device filters.")
        return

    if "rendered_creative_size" not in df_work.columns:
        st.info("rendered_creative_size column is missing; cannot compute creative size share.")
        return

    single_order_context: bool = (
        "order_id" in df_work.columns
        and df_work["order_id"].dropna().nunique() == 1
    )

    # Guard against blank sizes
    before_size_count: int = len(df_work)
    df_work["rendered_creative_size"] = df_work["rendered_creative_size"].astype(str).str.strip()
    df_work = df_work[df_work["rendered_creative_size"] != ""]
    df_work = df_work[~df_work["rendered_creative_size"].isna()]
    dropped_size_rows: int = before_size_count - len(df_work)

    if df_work.empty:
        st.info("All rows have null/blank rendered creative sizes after filtering.")
        return

    impressions_col: Optional[str] = None
    for candidate in ["creative_size_impressions", "impressions"]:
        if candidate in df_work.columns:
            impressions_col = candidate
            break

    if impressions_col is None:
        st.info("No impressions column found for creative sizes.")
        return

    df_work[impressions_col] = pd.to_numeric(df_work[impressions_col], errors="coerce").fillna(0.0)

    # Use line-item total when present (for single line item); otherwise total of selected rows
    total_impressions: float = float(df_work[impressions_col].sum())
    if selected_line_item != "All line items" and "line_item_total_impressions" in df_work.columns:
        total_from_line_item: float = float(
            pd.to_numeric(df_work["line_item_total_impressions"], errors="coerce").fillna(0.0).max()
        )
        if total_from_line_item > 0:
            total_impressions = total_from_line_item

    if total_impressions <= 0.0:
        st.info("Total impressions are zero for the current selection; nothing to chart.")
        return

    df_grouped = (
        df_work.groupby("rendered_creative_size", as_index=False)
        .agg(impressions=(impressions_col, "sum"))
    )

    # Selection-level line item counts/averages (regardless of order scope)
    if "line_item_id" in df_work.columns:
        size_line_items = (
            df_work.groupby("rendered_creative_size", as_index=False)
            .agg(selection_line_items=("line_item_id", "nunique"))
        )
        df_grouped = df_grouped.merge(size_line_items, on="rendered_creative_size", how="left")
        df_grouped["selection_avg_impressions_per_line_item"] = df_grouped.apply(
            lambda row: (row["impressions"] / row["selection_line_items"])
            if pd.notna(row.get("selection_line_items")) and row["selection_line_items"] not in [0, None]
            else None,
            axis=1,
        )
        df_grouped["selection_avg_impressions_per_line_item"] = df_grouped["selection_avg_impressions_per_line_item"].round(2)

    # Order-level metrics (only when a single order is in scope)
    if single_order_context:
        if "order_size_impressions" in df_work.columns:
            order_size = (
                df_work.groupby("rendered_creative_size", as_index=False)
                .agg(order_size_impressions=("order_size_impressions", "max"))
            )
            df_grouped = df_grouped.merge(order_size, on="rendered_creative_size", how="left")

        if "order_size_line_items" in df_work.columns:
            order_size_counts = (
                df_work.groupby("rendered_creative_size", as_index=False)
                .agg(order_size_line_items=("order_size_line_items", "max"))
            )
            df_grouped = df_grouped.merge(order_size_counts, on="rendered_creative_size", how="left")

        if "order_size_avg_impressions_per_line_item" in df_work.columns:
            order_size_avgs = (
                df_work.groupby("rendered_creative_size", as_index=False)
                .agg(order_size_avg_impressions_per_line_item=("order_size_avg_impressions_per_line_item", "max"))
            )
            df_grouped = df_grouped.merge(order_size_avgs, on="rendered_creative_size", how="left")

        for col in ["size_share_of_order_delivery", "size_pct_of_order_goal"]:
            if col in df_work.columns:
                metrics = (
                    df_work.groupby("rendered_creative_size", as_index=False)
                    .agg(**{col: (col, "max")})
                )
                df_grouped = df_grouped.merge(metrics, on="rendered_creative_size", how="left")

    df_grouped["impressions_share"] = df_grouped["impressions"] / total_impressions

    # Prefer order-level share when available; otherwise line-item/selection share
    chart_share_field: str = "impressions_share"
    if single_order_context and "size_share_of_order_delivery" in df_grouped.columns:
        chart_share_field = "size_share_of_order_delivery"

    if df_grouped["impressions"].sum() == 0:
        st.info("Impressions aggregated to zero; cannot render chart.")
        return

    tooltip_fields = [
        alt.Tooltip("rendered_creative_size:N", title="Size"),
        alt.Tooltip("impressions:Q", title="Impressions", format=","),
        alt.Tooltip(
            f"{chart_share_field}:Q",
            title="Share",
            format=".1%",
        ),
    ]
    if single_order_context and "size_pct_of_order_goal" in df_grouped.columns:
        tooltip_fields.append(
            alt.Tooltip(
                "size_pct_of_order_goal:Q",
                title="% of order goal",
                format=".1%",
            )
        )
    if single_order_context and "order_size_line_items" in df_grouped.columns:
        tooltip_fields.append(
            alt.Tooltip(
                "order_size_line_items:Q",
                title="# line items",
                format=",.0f",
            )
        )
    if single_order_context and "order_size_avg_impressions_per_line_item" in df_grouped.columns:
        tooltip_fields.append(
            alt.Tooltip(
                "order_size_avg_impressions_per_line_item:Q",
                title="Avg imps per line item",
                format=",.0f",
            )
        )
    if "selection_line_items" in df_grouped.columns:
        tooltip_fields.append(
            alt.Tooltip(
                "selection_line_items:Q",
                title="# line items (selection)",
                format=",.0f",
            )
        )
    if "selection_avg_impressions_per_line_item" in df_grouped.columns:
        tooltip_fields.append(
            alt.Tooltip(
                "selection_avg_impressions_per_line_item:Q",
                title="Avg imps per line item (selection)",
                format=",.0f",
            )
        )

    chart = (
        alt.Chart(df_grouped)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusBottomLeft=6)
        .encode(
            y=alt.Y(
                "rendered_creative_size:N",
                sort=df_grouped["rendered_creative_size"].tolist(),
                title="Rendered creative size",
            ),
            x=alt.X(
                f"{chart_share_field}:Q",
                title="Share of order impressions" if chart_share_field == "size_share_of_order_delivery" else "Share of line-item impressions",
                axis=alt.Axis(format="%", tickCount=5),
            ),
            color=alt.Color(
                f"{chart_share_field}:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None,
            ),
            tooltip=tooltip_fields,
        )
        .properties(height=380)
    )

    st.altair_chart(chart, use_container_width=True)

    df_table = df_grouped.copy()

    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
    )

    validation_notes: List[str] = []
    if dropped_size_rows > 0:
        validation_notes.append(f"Dropped {dropped_size_rows} rows with blank creative sizes.")

    share_field_for_validation: str = chart_share_field if chart_share_field in df_grouped.columns else "impressions_share"
    share_sum: float = float(df_grouped[share_field_for_validation].sum())
    if not math.isclose(share_sum, 1.0, rel_tol=1e-3):
        validation_notes.append(
            f"Shares sum to {share_sum:.3f} "
            + ("(order-level shares)" if chart_share_field == "size_share_of_order_delivery" else "")
        )

    if validation_notes:
        st.caption(" | ".join(validation_notes))
    else:
        st.caption("Creative size shares sum to 100% for the selected line item.")


def render_advertiser_delivery_by_club_mix(
    df_club: pd.DataFrame,
    advertiser_filter: Optional[str],
) -> None:
    """
    Show each advertiser's distribution across clubs (share of advertiser impressions by club).
    """
    if df_club.empty:
        st.info("No club portfolio data is available to compute advertiser delivery mix.")
        return

    df_work: pd.DataFrame = df_club.copy()

    if advertiser_filter and "advertiser_name" in df_work.columns:
        df_work = df_work[df_work["advertiser_name"] == advertiser_filter]

    if df_work.empty:
        st.info("No rows match the current advertiser filter for delivery mix.")
        return

    club_label_col: Optional[str] = None
    if "club_display_name" in df_work.columns:
        club_label_col = "club_display_name"
    elif "club_name" in df_work.columns:
        club_label_col = "club_name"

    if club_label_col is None:
        st.info("Cannot compute order delivery mix because club labels are missing.")
        return

    # Ensure numeric fields
    for col in ["club_delivered_impressions", "order_reported_impressions"]:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce").fillna(0.0)

    advertiser_label_col: str = "advertiser_name"

    # Aggregate to advertiser x club (removed per request)
    return


def render_seed_goals_tab() -> None:
    """
    Display and edit seed_partner_names_goals.csv, and optionally rerun dbt seed/run.
    """
    st.header("Goals")
    st.subheader("Input updated impression goals in the new goals column")
    st.caption("Edit seed goals and refresh dbt seeds/models.")

    if not SEED_GOALS_PATH.exists():
        st.error(f"Seed file not found at {SEED_GOALS_PATH}.")
        return

    try:
        df_seed: pd.DataFrame = pd.read_csv(SEED_GOALS_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read seed file: {exc}")
        return

    # Ensure expected columns (using impression_goal as provided in seed file)
    if "order_name" not in df_seed.columns or "impression_goal" not in df_seed.columns:
        st.error("Seed file must have 'order_name' and 'impression_goal' columns.")
        return

    df_seed["impression_goal"] = pd.to_numeric(df_seed["impression_goal"], errors="coerce")

    editable_df = df_seed.rename(columns={"impression_goal": "current_goal"}).copy()
    editable_df["new_goal"] = editable_df["current_goal"]

    st.markdown(
        """
        <style>
        [data-testid="stDataEditor"] thead th:nth-child(4),
        [data-testid="stDataEditor"] tbody td:nth-child(4) {
            background-color: #fff6e0;
        }
        [data-testid="stDataEditor"] tbody td:nth-child(4) input {
            border: 1px solid #e8b400;
            box-shadow: 0 0 0 1px #ffe8a3 inset;
        }
        [data-testid="stDataEditor"] tbody td:nth-child(4):hover input,
        [data-testid="stDataEditor"] tbody td:nth-child(4) input:focus {
            box-shadow: 0 0 0 2px #ffcd4d inset, 0 0 8px rgba(255, 189, 46, 0.8);
            transition: box-shadow 120ms ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    edited_df = st.data_editor(
        editable_df[["order_name", "advertiser_name", "current_goal", "new_goal"]],
        num_rows="fixed",
        column_config={
            "order_name": st.column_config.TextColumn("Order Name", disabled=True),
            "advertiser_name": st.column_config.TextColumn("Advertiser", disabled=True),
            "current_goal": st.column_config.NumberColumn(
                "Current Goal", disabled=True
            ),
            "new_goal": st.column_config.NumberColumn(
                "New Goal",
                step=1,
                disabled=False,
                help="Enter the updated target impressions for this order.",
            ),
        },
        use_container_width=True,
        hide_index=True,
        key="seed_goals_editor",
    )

    if st.button("Save and refresh", type="primary"):
        try:
            updated = df_seed.copy()
            updated["impression_goal"] = pd.to_numeric(edited_df["new_goal"], errors="coerce")
            if updated["impression_goal"].isna().any():
                st.error("All new goals must be numeric.")
                return

            updated.to_csv(SEED_GOALS_PATH, index=False)
            st.success("Seed file updated. Running dbt seed and dbt run...")

            # Ensure target path is writable and disable partial parse to avoid PermissionError
            dbt_project_root: Path = SEED_GOALS_PATH.parent.parent
            dbt_target_path: Path = dbt_project_root / "target"
            dbt_target_path.mkdir(parents=True, exist_ok=True)
            env_vars = os.environ.copy()
            env_vars.update(
                {
                    "DBT_TARGET_PATH": str(dbt_target_path),
                    "DBT_PARTIAL_PARSE": "0",
                }
            )

            seed_cmd = ["dbt", "seed"]
            run_cmd = ["dbt", "run"]

            seed_proc = subprocess.run(
                seed_cmd,
                capture_output=True,
                text=True,
                cwd=dbt_project_root,
                env=env_vars,
            )
            if seed_proc.returncode != 0:
                err_msg = seed_proc.stderr or seed_proc.stdout or "Unknown error"
                st.error(f"dbt seed failed (code {seed_proc.returncode}):\n{err_msg}")
                return

            run_proc = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                cwd=dbt_project_root,
                env=env_vars,
            )
            if run_proc.returncode != 0:
                err_msg = run_proc.stderr or run_proc.stdout or "Unknown error"
                st.error(f"dbt run failed (code {run_proc.returncode}):\n{err_msg}")
                return

            st.success("dbt seed and dbt run completed successfully.")
            if seed_proc.stdout:
                st.text(seed_proc.stdout)
            if run_proc.stdout:
                st.text(run_proc.stdout)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to save or refresh dbt: {exc}")


def render_club_performance_panel(
    df_orders: pd.DataFrame,
    df_club: pd.DataFrame,
    selected_club: Optional[str],
) -> None:
    """
    Show bottom-5 under-pacing and top-3 over-performing orders for the selected club.

    Orders are filtered to the selected club (when provided) by matching order_id values
    from the club mart. Falls back to all filtered orders when no club is selected.
    """
    if df_orders.empty:
        st.info("No order data available to compute performance panel.")
        return

    df_work: pd.DataFrame = df_orders.copy()

    # Restrict to the selected club using club mart mappings
    if selected_club and selected_club != "All clubs" and not df_club.empty:
        club_label_col: Optional[str] = None
        if "club_display_name" in df_club.columns:
            club_label_col = "club_display_name"
        elif "club_name" in df_club.columns:
            club_label_col = "club_name"

        if club_label_col and "order_id" in df_club.columns and "order_id" in df_work.columns:
            club_order_ids: List = (
                df_club.loc[
                    df_club[club_label_col] == selected_club,
                    "order_id",
                ]
                .dropna()
                .unique()
                .tolist()
            )
            df_work = df_work[df_work["order_id"].isin(club_order_ids)]

    if df_work.empty:
        st.info("No orders match the selected club for performance panel.")
        return

    # Numeric coercions
    for col in ["pacing_index", "impression_goal", "delivered_impressions"]:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

    # Helper to format display
    display_cols: List[str] = []
    for candidate in [
        "order_name",
        "advertiser_name",
        "pacing_index",
        "delivered_impressions",
        "impression_goal",
        "schedule_pct_elapsed",
    ]:
        if candidate in df_work.columns:
            display_cols.append(candidate)

    # Build under-performers (lowest pacing)
    df_under: pd.DataFrame = (
        df_work.dropna(subset=["pacing_index"])
        .sort_values(by="pacing_index", ascending=True, na_position="last")
        .head(5)
    )

    # Build over-performers (highest pacing)
    df_over: pd.DataFrame = (
        df_work.dropna(subset=["pacing_index"])
        .sort_values(by="pacing_index", ascending=False, na_position="last")
        .head(3)
    )

    def _format_panel(df_slice: pd.DataFrame) -> pd.DataFrame:
        df_disp: pd.DataFrame = df_slice.copy()
        if "pacing_index" in df_disp.columns:
            df_disp["pacing_index"] = df_disp["pacing_index"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
        for col in ["delivered_impressions", "impression_goal"]:
            if col in df_disp.columns:
                df_disp[col] = (
                    pd.to_numeric(df_disp[col], errors="coerce")
                    .fillna(0.0)
                    .map(lambda x: f"{int(x):,}")
                )
        if "schedule_pct_elapsed" in df_disp.columns:
            df_disp["schedule_pct_elapsed"] = (
                pd.to_numeric(df_disp["schedule_pct_elapsed"], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"{x * 100.0:.1f}%")
            )
        return df_disp[display_cols] if display_cols else df_disp

    st.subheader("Performance Spotlight")
    col_under, col_over = st.columns(2)
    with col_under:
        st.markdown("**Bottom-5 Under-pacing Orders (lowest pacing index)**")
        if df_under.empty:
            st.info("No under-pacing orders found for this selection.")
        else:
            st.dataframe(_format_panel(df_under), use_container_width=True)
    with col_over:
        st.markdown("**Top-3 Over-performers (highest pacing index)**")
        if df_over.empty:
            st.info("No over-performing orders found for this selection.")
        else:
            st.dataframe(_format_panel(df_over), use_container_width=True)


def _format_line_item_display(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cleaned/pretty version of the line-item diagnostics table for display.
    """
    excluded_cols: List[str] = [
        "order_id",
        "order_name",
        "advertiser_name",
        "order_pacing_status",
        "domain_share_of_line_item",
        "line_item_share_of_order",
        "order_label",
    ]

    preferred_cols: List[str] = [
        "line_item_id",
        "line_item_name",
        "domain_name",
        "device_category_name",
        "rendered_creative_size",
        "line_item_type_name",
        "line_item_total_impressions",
        "domain_impressions",
        "line_item_ctr",
    ]

    display_cols: List[str] = [
        col for col in preferred_cols if col in df_filtered.columns
    ]

    # If none of the preferred columns exist (unexpected), fall back to any column
    # that is not explicitly excluded so we still render something.
    if not display_cols:
        display_cols = [
            col for col in df_filtered.columns if col not in excluded_cols
        ]

    if not display_cols:
        return pd.DataFrame()

    df_display: pd.DataFrame = df_filtered[display_cols].copy()

    # Format some key numeric fields
    for col in ["line_item_total_impressions", "domain_impressions"]:
        if col in df_display.columns:
            df_display[col] = (
                pd.to_numeric(df_display[col], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"{int(x):,}")
            )

    for pct_col in ["domain_share_of_line_item", "line_item_share_of_order"]:
        if pct_col in df_display.columns:
            df_display[pct_col] = (
                pd.to_numeric(df_display[pct_col], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"{x * 100.0:.1f}%")
            )

    if "line_item_ctr" in df_display.columns:
        ctr_series = pd.to_numeric(df_display["line_item_ctr"], errors="coerce")

        def _fmt_ctr(val: float) -> str:
            if pd.isna(val):
                return "-"
            # Handle both fraction (0–1) and percent (1–100) inputs defensively
            return f"{(val * 100.0):.2f}%" if val <= 1 else f"{val:.2f}%"

        df_display["line_item_ctr"] = ctr_series.apply(_fmt_ctr)

    return df_display


def render_line_item_detail_table(
    df_diag: pd.DataFrame,
    advertiser_filter: Optional[str],
) -> None:
    """
    Render a line-item detail table, filterable by order.
    """
    st.subheader("Line Item Details")

    if df_diag.empty:
        st.info(
            "Line-item diagnostics mart is empty or unavailable. "
            "Run dbt for mart_gam_line_item_diagnostics_view to populate it."
        )
        return

    df_filtered: pd.DataFrame = df_diag.copy()

    if advertiser_filter and "advertiser_name" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["advertiser_name"] == advertiser_filter]

    if df_filtered.empty:
        st.info("No line items match the current advertiser filter.")
        return

    order_label_col: Optional[str] = None
    if "order_name" in df_filtered.columns and "advertiser_name" in df_filtered.columns:
        df_filtered["order_label"] = (
            df_filtered["order_name"].astype(str)
            + " — "
            + df_filtered["advertiser_name"].astype(str)
        )
        order_label_col = "order_label"
    elif "order_id" in df_filtered.columns:
        df_filtered["order_label"] = df_filtered["order_id"].astype(str)
        order_label_col = "order_label"

    if order_label_col is not None:
        order_options: List[str] = (
            ["All orders"]
            + sorted(
                df_filtered[order_label_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        selected_order: str = st.selectbox(
            "Order filter (line items)",
            options=order_options,
            index=0,
        )

        if selected_order != "All orders":
            df_filtered = df_filtered[df_filtered[order_label_col] == selected_order]

    if df_filtered.empty:
        st.info("No line items match the selected order.")
        return

    df_display = _format_line_item_display(df_filtered)
    if df_display.empty:
        st.info("No line-item columns available to display.")
        return

    st.dataframe(df_display, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for the Streamlit app.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.caption(
        "Assess global order pacing, see how orders contribute to each club, "
        "and diagnose underperforming line items by (domain) and device."
    )

    # Connect to Postgres
    try:
        conn: SQLConnection = get_sql_connection()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    # Load order-level data (required for sidebar filters)
    try:
        df_orders: pd.DataFrame = load_order_pacing(conn)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if df_orders.empty:
        st.warning(
            "Order pacing mart (mart_gam_order_pacing_summary) returned no rows. "
            "Ensure dbt has been run and data is available."
        )
        st.stop()

    # Sidebar filters based on order mart
    filters: Dict[str, Optional[str]] = build_sidebar_filters(df_orders)
    df_orders_filtered: pd.DataFrame = apply_order_filters(df_orders, filters)

    # Load club and diagnostics marts
    try:
        df_club: pd.DataFrame = load_club_portfolio(conn)
    except RuntimeError as exc:
        st.warning(str(exc))
        df_club = pd.DataFrame()

    try:
        df_diag: pd.DataFrame = load_line_item_diagnostics(conn)
    except RuntimeError as exc:
        st.warning(str(exc))
        df_diag = pd.DataFrame()

    try:
        df_sizes: pd.DataFrame = load_line_item_creative_sizes(conn)
    except RuntimeError as exc:
        st.warning(str(exc))
        df_sizes = pd.DataFrame()

    # ------------------------------------------------------------------
    # Top-level navigation (replaces st.tabs to keep selection on rerun)
    # ------------------------------------------------------------------
    TAB_ADVERTISER: str = "Advertiser"
    TAB_CLUBS: str = "Club Portfolio (Orders by Club)"
    TAB_CREATIVE_SIZES: str = "Creative Sizes"
    TAB_SEEDS: str = "Goals"

    # Use a radio so the selected "tab" is stored in st.docker_state["main_view"]
    selected_tab: str = st.radio(
        label="View",
        options=[TAB_ADVERTISER, TAB_CLUBS, TAB_CREATIVE_SIZES, TAB_SEEDS],
        index=0,
        horizontal=True,
        key="main_view",
        help="Choose which view of the campaign you want to explore.",
    )

    # ------------------------------------------------------------------
    # Render each logical tab based on selected_tab
    # ------------------------------------------------------------------
    if selected_tab == TAB_ADVERTISER:
        # Impressions vs Goal by Advertiser
        st.subheader("Impressions vs Goal by Advertiser")
        st.markdown("---")
        render_impressions_vs_goal_by_advertiser(df_orders_filtered)

        # Order-level KPIs and pacing chart
        st.markdown("---")
        st.subheader("Orders by Pace")
        render_order_pacing_chart(df_orders_filtered)

        st.markdown("---")
        st.subheader("Order Details")
        render_order_table(df_orders_filtered)

        st.markdown("---")
        render_line_item_detail_table(
            df_diag=df_diag,
            advertiser_filter=filters.get("advertiser"),
        )

    elif selected_tab == TAB_CLUBS:
        # Club Portfolio (Orders by Club)
        selected_club: Optional[str] = render_club_portfolio(
            df_club=df_club,
            advertiser_filter=filters.get("advertiser"),
        )

        # Order delivery mix across clubs (share of order)
        render_advertiser_delivery_by_club_mix(
            df_club=df_club,
            advertiser_filter=filters.get("advertiser"),
        )

        # Performance spotlight (bottom-5 / top-3 by pacing index) respecting filters and club selection
        render_club_performance_panel(
            df_orders=df_orders_filtered,
            df_club=df_club,
            selected_club=selected_club,
        )

        # Order share by club (uses club mart and current filters)
        render_order_share_by_club(
            df_club=df_club,
            advertiser_filter=filters.get("advertiser"),
            selected_club=selected_club,
        )

    elif selected_tab == TAB_CREATIVE_SIZES:
        render_creative_size_share(
            df_sizes=df_sizes,
            advertiser_filter=filters.get("advertiser"),
        )

    elif selected_tab == TAB_SEEDS:
        render_seed_goals_tab()



if __name__ == "__main__":
    main()
