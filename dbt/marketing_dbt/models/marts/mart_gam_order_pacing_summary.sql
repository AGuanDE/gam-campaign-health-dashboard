{{ config(
    materialized = 'view',
    tags = ['gam', 'order', 'summary']
) }}

-- mart_gam_order_pacing_summary
-- Purpose:
--   Summary mart on top of fct_gam_order_health.
--   Adds a pacing_status classification for dashboard use.

with orders as (
    select
        -- Core identifiers
        order_id,
        order_name,
        advertiser_name,
        order_delivery_status_name,

        -- Schedule / timing fields
        schedule_start_date,
        schedule_end_date,
        schedule_start_datetime,
        schedule_end_datetime,
        schedule_days,
        days_elapsed,
        schedule_pct_elapsed,

        -- Goals
        impression_goal,

        -- Delivery metrics from the fact
        reported_impressions,
        reported_clicks,
        reported_ctr,

        -- Backwards-compatible aliases for dashboard use
        reported_impressions as delivered_impressions,
        reported_clicks      as delivered_clicks,

        -- Pacing / health metrics
        delivered_impressions_pct_of_goal,
        actual_impressions_per_day,
        target_impressions_per_day,
        pacing_index
    from {{ ref('fct_gam_order_health') }}
),

classified as (
    select
        o.*,
        case
            when impression_goal is null or impression_goal <= 0
                then 'no_goal'
            when schedule_pct_elapsed is null or schedule_pct_elapsed <= 0
                then 'too_early'
            when pacing_index is null
                then 'unknown'
            when pacing_index < 0.9
                then 'behind'
            when pacing_index between 0.9 and 1.1
                then 'on_track'
            else 'ahead'
        end as pacing_status
    from orders o
)

select *
from classified