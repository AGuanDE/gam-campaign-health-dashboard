## GAM ingestion runner

Loads manually exported Google Ad Manager reports into `schema.gam_raw` so dbt + Streamlit always have fresh data. It also supports refreshing the low-change `seed_partner_names_goals` mapping.

### Prerequisites
- Manually download a GAM delivery report from the UI (include all fields listed below) and save it locally.
- Postgres connection env vars:
  - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
  - Optional: `POSTGRES_HOST` (default `localhost`), `POSTGRES_PORT` (default `5432`), `TARGET_SCHEMA` (default `schema`)
- Install dependencies: `pip install -r requirements.txt`

### Expected CSV fields
The export should match the warehouse schema (`schema.gam_raw`) exactly:
`ORDER_ID, ORDER_NAME, ORDER_START_DATE, ORDER_START_DATE_TIME, ORDER_END_DATE, ORDER_END_DATE_TIME, LINE_ITEM_ID, LINE_ITEM_NAME, LINE_ITEM_START_DATE, LINE_ITEM_START_DATE_TIME, LINE_ITEM_END_DATE, LINE_ITEM_END_DATE_TIME, LINE_ITEM_PRIMARY_GOAL_UNITS_ABSOLUTE, LINE_ITEM_PRIMARY_GOAL_UNIT_TYPE_NAME, LINE_ITEM_LIFETIME_IMPRESSIONS, LINE_ITEM_LIFETIME_CLICKS, ORDER_LIFETIME_CLICKS, ORDER_LIFETIME_IMPRESSIONS, LINE_ITEM_DELIVERY_INDICATOR, CREATIVE_NAME, RENDERED_CREATIVE_SIZE, DEVICE_CATEGORY_NAME, LINE_ITEM_PRIORITY, LINE_ITEM_TYPE_NAME, ORDER_DELIVERY_STATUS_NAME, ADVERTISER_NAME, AD_SERVER_IMPRESSIONS, AD_SERVER_CLICKS, AD_SERVER_CTR`

### Usage
```
# Load a manually downloaded CSV (recommended: keep --upsert on)
python ingestion/gam_ingest.py \
  --input-csv data/gam_report_01012025_13112025.csv \
  --upsert \
  --load-seed-goals
```

Flags:
- `--input-csv` path to the GAM UI export (default `data/gam_report_latest.csv`).
- `--skip-db-load` to validate the CSV/hash but not write to Postgres.
- `--upsert` to insert only new rows using a `row_hash` unique key (no truncation). Default mode for recurring loads.
- `--append` to keep existing rows instead of truncating (no dedupe).
- `--schema` to override the target schema (default `schema`).
- `--load-seed-goals` to also refresh `schema.seed_partner_names_goals` from `dbt/marketing_dbt/seeds/seed_partner_names_goals.csv` (override via `--seed-goals-path`).

Notes:
- The script normalizes column headers to snake_case before loading; it will error if the CSV is missing expected columns.
- `seed_partner_names_goals` changes infrequently, so you can skip `--load-seed-goals` on most runs and include it only when the CSV updates.
- `--upsert` automatically ensures a `row_hash` column + unique index exist on `schema.gam_raw` and only inserts unseen hashes. This keeps reruns idempotent even when the CSV overlaps with already ingested periods.
