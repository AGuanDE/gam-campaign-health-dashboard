{{ config(
    materialized = 'view',
    tags = ['gam', 'staging', 'line_items', 'creative_size']
) }}

-- stg_gam_line_item_creative_sizes
-- Purpose:
--   Aggregate GAM raw export to per-line-item creative sizes (optionally by device).
--   Grain: 1 row per (line_item_id, rendered_creative_size, device_category_name).
--   Provides delivered impressions/clicks and CTR for each size bucket, plus order dates for filtering.

with base as (
    select
        line_item_id,
        line_item_name,
        order_id,
        order_name,
        advertiser_name,

        order_start_date,
        order_end_date,
        order_start_date_time,
        order_end_date_time,

        rendered_creative_size,
        device_category_name,

        ad_server_impressions as impressions,
        ad_server_clicks      as clicks
    from {{ source('gam', 'gam_raw') }}
),

filtered as (
    select *
    from base
    where rendered_creative_size is not null
      and trim(rendered_creative_size) <> ''
),

agg as (
    select
        line_item_id,

        max(line_item_name)       as line_item_name,
        max(order_id)             as order_id,
        max(order_name)           as order_name,
        max(advertiser_name)      as advertiser_name,

        min(order_start_date::date)           as order_start_date,
        max(order_end_date::date)             as order_end_date,
        min(order_start_date_time::timestamp) as order_start_datetime,
        max(order_end_date_time::timestamp)   as order_end_datetime,

        rendered_creative_size,
        device_category_name,

        sum(impressions)::bigint as impressions,
        sum(clicks)::bigint      as clicks,
        case
            when sum(impressions) > 0
                then sum(clicks)::numeric / sum(impressions)::numeric
            else null
        end as ctr
    from filtered
    group by
        line_item_id,
        rendered_creative_size,
        device_category_name
)

select *
from agg
