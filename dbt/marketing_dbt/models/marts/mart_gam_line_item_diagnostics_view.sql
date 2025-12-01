{{ config(
    materialized = 'view',
    tags = ['gam', 'line_item', 'diagnostics']
) }}

with base as (
    select
        order_id,
        order_name,
        advertiser_name,

        line_item_id,
        line_item_name,

        schedule_start_date,
        schedule_end_date,
        schedule_start_datetime,
        schedule_end_datetime,

        impression_goal_units_raw,
        impression_goal_unit_type_raw,

        line_item_total_impressions,
        line_item_total_clicks,
        line_item_window_impressions,
        line_item_window_clicks,
        line_item_ctr,

        line_item_delivery_indicator,
        line_item_priority,
        line_item_type_name,
        order_delivery_status_name,

        creative_name,
        rendered_creative_size,
        device_category_name,

        domain_name,
        club_name,
        club_display_name,
        domain_impressions,

        -- aligned with fct_gam_line_item_diagnostics
        order_reported_impressions,
        order_reported_clicks,
        impression_goal,
        order_schedule_start_date,
        order_schedule_end_date,
        schedule_days,
        days_elapsed,
        schedule_pct_elapsed,
        actual_impressions_per_day,
        target_impressions_per_day,
        order_pacing_index,
        domain_share_of_line_item,
        line_item_share_of_order
    from {{ ref('fct_gam_line_item_diagnostics') }}
),

classified as (
    select
        b.*,
        case
            when order_pacing_index is null
                then 'unknown'
            when order_pacing_index < 0.9
                then 'behind'
            when order_pacing_index between 0.9 and 1.1
                then 'on_track'
            else 'ahead'
        end as order_pacing_status
    from base b
)

select *
from classified
