{{ config(
    materialized = 'view'
) }}

-- stg_gam_line_items
-- Purpose:
--   Clean, standardize, and expose line-item-level fields directly from the raw
--   Google Ad Manager export (source: gam.gam_raw).
--   Grain: 1 row per line_item_id.

WITH base_raw AS (
    SELECT
        -- Identifiers
        line_item_id,
        line_item_name,
        order_id,
        order_name,
        advertiser_name,

        line_item_start_date,
        line_item_start_date_time,
        line_item_end_date,
        line_item_end_date_time,

        -- Goals --- THESE ARE NOT THE REAL NUMBERS BUT KEEP THEM FOR NOW ---
        line_item_primary_goal_units_absolute,
        line_item_primary_goal_unit_type_name,

        -- Lifetime metrics (as reported by GAM)
        line_item_lifetime_impressions,
        line_item_lifetime_clicks,

        -- Delivered metrics (aggregated AD_SERVER_* metrics)
        ad_server_impressions          AS impressions,
        ad_server_clicks               AS clicks,
        ad_server_ctr                  AS ctr,

        -- Configuration / status
        line_item_delivery_indicator,
        line_item_priority,
        line_item_type_name,
        order_delivery_status_name,

        -- Creative / device breakdown
        creative_name,
        rendered_creative_size,
        device_category_name

    FROM {{ source('gam', 'gam_raw') }}
),

aggregated_line_items AS (
    SELECT
        -- Grain: 1 row per line_item_id
        line_item_id,

        -- Use MAX() for name fields that should be constant per line_item_id.
        MAX(line_item_name)              AS line_item_name,
        MAX(order_id)                    AS order_id,
        MAX(order_name)                  AS order_name,
        MAX(advertiser_name)             AS advertiser_name,

        MIN(line_item_start_date::date)        AS schedule_start_date,
        MAX(line_item_end_date::date)          AS schedule_end_date,

        MIN(line_item_start_date_time::timestamp)   AS schedule_start_datetime,
        MAX(line_item_end_date_time::timestamp)     AS schedule_end_datetime,

        -- Line-item goals (raw, untrusted; keep for reference/comparison)
        MAX(line_item_primary_goal_units_absolute)::bigint AS impression_goal_units_raw,
        MAX(line_item_primary_goal_unit_type_name)         AS impression_goal_unit_type_raw,

        -- Lifetime metrics as reported by GAM (repeated per row).
        -- Take MAX() to avoid double-counting while preserving the value.
        MAX(line_item_lifetime_impressions)::bigint AS lifetime_impressions,
        MAX(line_item_lifetime_clicks)::bigint      AS lifetime_clicks,

        -- Delivered metrics
        SUM(impressions)::bigint       AS impressions,
        SUM(clicks)::bigint           AS clicks,
        
        CASE
            WHEN SUM(impressions) > 0
            THEN SUM(clicks)::numeric / SUM(impressions)::numeric
            ELSE NULL
        END AS ctr,

        -- Configuration / status (should be constant per line item)
        MAX(line_item_delivery_indicator)    AS line_item_delivery_indicator,
        MAX(line_item_priority)::int         AS line_item_priority,
        MAX(line_item_type_name)             AS line_item_type_name,
        MAX(order_delivery_status_name)      AS order_delivery_status_name,

        -- Creative / device:
        -- There can be multiple creatives/devices per line item; we keep a sample value.
        MAX(creative_name)                                 AS creative_name,
        MAX(rendered_creative_size)                        AS rendered_creative_size,
        MAX(device_category_name)                          AS device_category_name

    FROM base_raw
    GROUP BY line_item_id
)   

SELECT
    line_item_id,
    line_item_name,
    order_id,
    order_name,
    advertiser_name,
    schedule_start_date,
    schedule_end_date,
    schedule_start_datetime,
    schedule_end_datetime,
    impression_goal_units_raw,
    impression_goal_unit_type_raw,
    lifetime_impressions,
    lifetime_clicks,
    impressions,
    clicks,
    ctr,
    line_item_delivery_indicator,
    line_item_priority,
    line_item_type_name,
    order_delivery_status_name,
    creative_name,
    rendered_creative_size,
    device_category_name
FROM aggregated_line_items
