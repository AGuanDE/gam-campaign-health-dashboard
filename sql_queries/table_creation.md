# ==========================================================================
-- 1) Core schema
# ==========================================================================
CREATE SCHEMA IF NOT EXISTS schema;

# ==========================================================================
-- 2) Raw GAM table (matches CSV exactly)
# ==========================================================================
DROP TABLE IF EXISTS schema.gam_raw;

CREATE TABLE schema.gam_raw (
    order_id                               BIGINT,
    order_name                             TEXT,
    order_start_date                       DATE,
    order_start_date_time                  TIMESTAMP WITHOUT TIME ZONE,
    order_end_date                         DATE,
    order_end_date_time                    TIMESTAMP WITHOUT TIME ZONE,
    line_item_id                           BIGINT,
    line_item_name                         TEXT,
    line_item_start_date                   DATE,
    line_item_start_date_time              TIMESTAMP WITHOUT TIME ZONE,
    line_item_end_date                     DATE,
    line_item_end_date_time                TIMESTAMP WITHOUT TIME ZONE,
    line_item_primary_goal_units_absolute  BIGINT,
    line_item_primary_goal_unit_type_name  TEXT,
    line_item_lifetime_impressions         BIGINT,
    line_item_lifetime_clicks              BIGINT,
    order_lifetime_clicks                  BIGINT,
    order_lifetime_impressions             BIGINT,
    line_item_delivery_indicator           NUMERIC(10,4),
    creative_name                          TEXT,
    rendered_creative_size                 TEXT,
    device_category_name                   TEXT,
    line_item_priority                     INTEGER,
    line_item_type_name                    TEXT,
    order_delivery_status_name             TEXT,
    advertiser_name                        TEXT,
    ad_server_impressions                  BIGINT,
    ad_server_clicks                       BIGINT,
    ad_server_ctr                          NUMERIC(10,4),
    row_hash                               TEXT
);

CREATE INDEX IF NOT EXISTS ix_gam_raw_order_line
    ON schema.gam_raw (order_id, line_item_id);

CREATE INDEX IF NOT EXISTS ix_gam_raw_advertiser
    ON schema.gam_raw (advertiser_name);

CREATE UNIQUE INDEX IF NOT EXISTS ux_gam_raw_row_hash
    ON schema.gam_raw (row_hash);

-- =====================================
-- 3) Aggregated tables for Streamlit
-- =====================================

-- 3a) Order-level aggregate table
DROP TABLE IF EXISTS schema.orders;

CREATE TABLE schema.orders (
    order_id                               BIGINT PRIMARY KEY,
    order_name                             TEXT NOT NULL,
    advertiser_name                        TEXT NOT NULL,
    primary_goal_units_absolute            BIGINT,
    primary_goal_unit_type_name            TEXT,
    lifetime_impressions                   BIGINT,
    lifetime_clicks                        BIGINT,
    delivered_impressions                  BIGINT,
    delivered_clicks                       BIGINT,
    delivered_ctr                          NUMERIC(10,4),
    order_delivery_status_name             TEXT
);

-- 3b) Line-item-level aggregate table
DROP TABLE IF EXISTS schema.line_items;

CREATE TABLE schema.line_items (
    line_item_id                           BIGINT PRIMARY KEY,
    line_item_name                         TEXT NOT NULL,
    order_id                               BIGINT NOT NULL,
    order_name                             TEXT NOT NULL,
    advertiser_name                        TEXT NOT NULL,
    primary_goal_units_absolute            BIGINT,
    primary_goal_unit_type_name            TEXT,
    lifetime_impressions                   BIGINT,
    lifetime_clicks                        BIGINT,
    delivered_impressions                  BIGINT,
    delivered_clicks                       BIGINT,
    delivered_ctr                          NUMERIC(10,4),
    line_item_delivery_indicator           NUMERIC(10,4),
    line_item_priority                     INTEGER,
    line_item_type_name                    TEXT,
    order_delivery_status_name             TEXT
);

-- Optional FK if you want integrity checks:
-- ALTER TABLE schema.line_items
--   ADD CONSTRAINT fk_line_items_orders
--   FOREIGN KEY (order_id) REFERENCES schema.orders(order_id);

-- 3c) Order + line-item fact table (for detailed views)
DROP TABLE IF EXISTS schema.delivery_facts;

CREATE TABLE schema.delivery_facts (
    order_id                               BIGINT NOT NULL,
    order_name                             TEXT NOT NULL,
    line_item_id                           BIGINT NOT NULL,
    line_item_name                         TEXT NOT NULL,
    advertiser_name                        TEXT NOT NULL,
    device_category_name                   TEXT,
    creative_name                          TEXT,
    rendered_creative_size                 TEXT,
    primary_goal_units_absolute            BIGINT,
    primary_goal_unit_type_name            TEXT,
    delivered_impressions                  BIGINT,
    delivered_clicks                       BIGINT,
    delivered_ctr                          NUMERIC(10,4),
    line_item_delivery_indicator           NUMERIC(10,4),
    line_item_priority                     INTEGER,
    line_item_type_name                    TEXT,
    order_delivery_status_name             TEXT
);

CREATE INDEX IF NOT EXISTS ix_delivery_facts_order_line
    ON schema.delivery_facts (order_id, line_item_id);





# ==========================================================================
-- load data into gam_raw using the ingestion script (computes row_hash and upserts)
# Example:
#   python ingestion/gam_ingest.py --input-csv data/gam_report_01012025_13112025.csv --upsert
# ==========================================================================

# =================Populate orders, line_items, and delivery_facts====================

  -- Clear any old data (fresh volume, but explicit is fine)
TRUNCATE TABLE schema.orders;
TRUNCATE TABLE schema.line_items;
TRUNCATE TABLE schema.delivery_facts;

-- =====================================
-- A) Populate line_items (1 row per line item)
-- =====================================
INSERT INTO schema.line_items (
    line_item_id,
    line_item_name,
    order_id,
    order_name,
    advertiser_name,
    primary_goal_units_absolute,
    primary_goal_unit_type_name,
    lifetime_impressions,
    lifetime_clicks,
    delivered_impressions,
    delivered_clicks,
    delivered_ctr,
    line_item_delivery_indicator,
    line_item_priority,
    line_item_type_name,
    order_delivery_status_name
)
SELECT
    line_item_id,
    MAX(line_item_name)                         AS line_item_name,
    MAX(order_id)                               AS order_id,
    MAX(order_name)                             AS order_name,
    MAX(advertiser_name)                        AS advertiser_name,
    MAX(line_item_primary_goal_units_absolute)  AS primary_goal_units_absolute,
    MAX(line_item_primary_goal_unit_type_name)  AS primary_goal_unit_type_name,
    MAX(line_item_lifetime_impressions)         AS lifetime_impressions,
    MAX(line_item_lifetime_clicks)              AS lifetime_clicks,
    SUM(ad_server_impressions)                  AS delivered_impressions,
    SUM(ad_server_clicks)                       AS delivered_clicks,
    CASE
        WHEN SUM(ad_server_impressions) > 0
            THEN ROUND(
                (SUM(ad_server_clicks)::NUMERIC
                 / SUM(ad_server_impressions)::NUMERIC) * 100.0,
                4
            )
        ELSE 0
    END                                         AS delivered_ctr,
    MAX(line_item_delivery_indicator)           AS line_item_delivery_indicator,
    MAX(line_item_priority)                     AS line_item_priority,
    MAX(line_item_type_name)                    AS line_item_type_name,
    MAX(order_delivery_status_name)             AS order_delivery_status_name
FROM schema.gam_raw
GROUP BY
    line_item_id;

-- =====================================
-- B) Populate orders (1 row per order)
-- =====================================
INSERT INTO schema.orders (
    order_id,
    order_name,
    advertiser_name,
    primary_goal_units_absolute,
    primary_goal_unit_type_name,
    lifetime_impressions,
    lifetime_clicks,
    delivered_impressions,
    delivered_clicks,
    delivered_ctr,
    order_delivery_status_name
)
SELECT
    order_id,
    MAX(order_name)                             AS order_name,
    MAX(advertiser_name)                        AS advertiser_name,
    SUM(line_item_primary_goal_units_absolute)  AS primary_goal_units_absolute,
    MAX(line_item_primary_goal_unit_type_name)  AS primary_goal_unit_type_name,
    MAX(order_lifetime_impressions)             AS lifetime_impressions,
    MAX(order_lifetime_clicks)                  AS lifetime_clicks,
    SUM(ad_server_impressions)                  AS delivered_impressions,
    SUM(ad_server_clicks)                       AS delivered_clicks,
    CASE
        WHEN SUM(ad_server_impressions) > 0
            THEN ROUND(
                (SUM(ad_server_clicks)::NUMERIC
                 / SUM(ad_server_impressions)::NUMERIC) * 100.0,
                4
            )
        ELSE 0
    END                                         AS delivered_ctr,
    MAX(order_delivery_status_name)             AS order_delivery_status_name
FROM schema.gam_raw
GROUP BY
    order_id;

-- =====================================
-- C) Populate delivery_facts (order + line item grain)
-- =====================================
INSERT INTO schema.delivery_facts (
    order_id,
    order_name,
    line_item_id,
    line_item_name,
    advertiser_name,
    device_category_name,
    creative_name,
    rendered_creative_size,
    primary_goal_units_absolute,
    primary_goal_unit_type_name,
    delivered_impressions,
    delivered_clicks,
    delivered_ctr,
    line_item_delivery_indicator,
    line_item_priority,
    line_item_type_name,
    order_delivery_status_name
)
SELECT
    order_id,
    MAX(order_name)                             AS order_name,
    line_item_id,
    MAX(line_item_name)                         AS line_item_name,
    MAX(advertiser_name)                        AS advertiser_name,
    NULL                                        AS device_category_name,  -- aggregated across devices
    NULL                                        AS creative_name,         -- aggregated across creatives
    NULL                                        AS rendered_creative_size,
    MAX(line_item_primary_goal_units_absolute)  AS primary_goal_units_absolute,
    MAX(line_item_primary_goal_unit_type_name)  AS primary_goal_unit_type_name,
    SUM(ad_server_impressions)                  AS delivered_impressions,
    SUM(ad_server_clicks)                       AS delivered_clicks,
    CASE
        WHEN SUM(ad_server_impressions) > 0
            THEN ROUND(
                (SUM(ad_server_clicks)::NUMERIC
                 / SUM(ad_server_impressions)::NUMERIC) * 100.0,
                4
            )
        ELSE 0
    END                                         AS delivered_ctr,
    MAX(line_item_delivery_indicator)           AS line_item_delivery_indicator,
    MAX(line_item_priority)                     AS line_item_priority,
    MAX(line_item_type_name)                    AS line_item_type_name,
    MAX(order_delivery_status_name)             AS order_delivery_status_name
FROM schema.gam_raw
GROUP BY
    order_id,
    line_item_id;
