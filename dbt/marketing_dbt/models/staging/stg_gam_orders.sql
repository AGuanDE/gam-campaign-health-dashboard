{{ config(
    materialized = 'view'
) }}

-- stg_gam_orders
-- Purpose:
--   Clean, standardize, and expose order-level fields directly from the raw
--   Google Ad Manager export (source: gam.gam_raw).
--   Grain: 1 row per order_id.

WITH base_raw AS (
    SELECT
        -- Identifiers
        order_id,
        order_name,
        advertiser_name,

        -- Order-level dates
        order_start_date,
        order_start_date_time,
        order_end_date,
        order_end_date_time,

        -- Lifetime metrics (as reported by GAM)
        order_lifetime_impressions,
        order_lifetime_clicks,

        -- Delivered metrics
        ad_server_impressions             AS impressions,
        ad_server_clicks                  AS clicks,
        ad_server_ctr                     AS ctr,

        -- Status
        order_delivery_status_name
    FROM {{ source('gam', 'gam_raw') }}
),

aggregated_orders AS (
    SELECT
        -- Grain: 1 row per order_id
        order_id,

        -- Use MAX() for name fields that should be constant per order_id.
        MAX(order_name)              AS order_name,
        MAX(advertiser_name)         AS advertiser_name,

        -- Roll up schedule to an order-level window (dates).
        MIN(order_start_date)        AS schedule_start_date,
        MAX(order_end_date)          AS schedule_end_date,

        -- Optional: keep the timestamp detail as well.
        MIN(order_start_date_time)   AS schedule_start_datetime,
        MAX(order_end_date_time)     AS schedule_end_datetime,

        -- Lifetime metrics as reported by GAM (repeated per row).
        -- Take MAX() to avoid double-counting while preserving the value.
        MAX(order_lifetime_impressions)::bigint AS lifetime_impressions,
        MAX(order_lifetime_clicks)::bigint      AS lifetime_clicks,

        -- Delivered metrics: aggregate AD_SERVER_* across all rows for the order_id.
        SUM(impressions)::bigint     AS impressions,
        SUM(clicks)::bigint          AS clicks,

        -- Recomputed CTR at the order grain.
        CASE
            WHEN SUM(impressions) > 0
            THEN (SUM(clicks)::numeric / SUM(impressions)::numeric) * 100.0
            ELSE NULL
        END                         AS ctr,

        -- Order delivery status (should be constant per order_id; use MAX() as resolver).
        MAX(order_delivery_status_name) AS order_delivery_status_name
    FROM base_raw
    GROUP BY order_id
)

SELECT 
    order_id,
    order_name,
    advertiser_name,
    schedule_start_date,
    schedule_end_date,
    schedule_start_datetime,
    schedule_end_datetime,
    lifetime_impressions,
    lifetime_clicks,
    impressions,
    clicks,
    ctr,
    order_delivery_status_name
FROM aggregated_orders
