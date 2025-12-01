{{ config(
    materialized = 'view',
    tags = ['gam', 'club', 'summary']
) }}

with club_orders as (
    select
        club_name,
        club_display_name,
        order_id,
        order_name,
        advertiser_name,

        order_delivery_status_name,

        schedule_start_date,
        schedule_end_date,
        schedule_start_datetime,
        schedule_end_datetime,
        schedule_days,
        days_elapsed,
        schedule_pct_elapsed,

        -- Global order goal (no club-specific goals)
        impression_goal,

        club_delivered_impressions,
        order_reported_impressions,
        club_share_of_order_delivery,
        order_share_of_club_delivery,
        order_pacing_index,
        distinct_line_items_on_club
    from {{ ref('fct_gam_club_delivery') }}
),

classified as (
    select
        co.*,

        -- Pacing status per order (based on order_pacing_index)
        case
            when co.order_pacing_index is null
                then 'unknown'
            when co.order_pacing_index < 0.9
                then 'behind'
            when co.order_pacing_index between 0.9 and 1.1
                then 'on_track'
            else 'ahead'
        end as order_pacing_status,

        -- Flags for orders ending soon / urgent orders
        case
            when co.schedule_end_date is not null
                and co.schedule_end_date >= current_date
                and co.schedule_end_date <= current_date + interval '7 days'
            then true
            else false
        end as is_ending_soon_order,

        case
            when co.schedule_end_date is not null
                and co.schedule_end_date >= current_date
                and co.schedule_end_date <= current_date + interval '7 days'
                and co.order_pacing_index is not null
                and co.order_pacing_index < 0.9
            then true
            else false
        end as is_urgent_order,

        case
            when order_share_of_club_delivery is null then 'unknown'
            when order_share_of_club_delivery >= 0.25 then 'major_contributor'
            when order_share_of_club_delivery >= 0.10 then 'medium_contributor'
            else 'minor_contributor'
        end as club_contribution_bucket

    from club_orders co
)

select *
from classified
