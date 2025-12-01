{{ config(
    materialized = 'table',
    tags = ['gam', 'order', 'health']
) }}

-- fct_gam_order_health
-- Purpose:
--   Order-grain fact table for Google Ad Manager.
--   Aggregates delivery to the ORDER level.
--   Uses trusted seed goals when available.
--   Falls back to GAM order-level goals when no seed override is present.
--   Computes pacing and health metrics (percent of goal, pacing index, etc.).

-- ============================================================
-- 1) Base orders (one row per ORDER, with all order-level fields)
-- ============================================================
with orders as (
    select
        o.order_id,
        o.order_name,
        o.advertiser_name,

        o.schedule_start_date,
        o.schedule_end_date,
        o.schedule_start_datetime,
        o.schedule_end_datetime,

        o.impressions as reported_impressions,
        o.clicks      as reported_clicks,
        o.ctr         as reported_ctr,

        o.order_delivery_status_name
    from {{ ref('stg_gam_orders') }} as o
),

-- ============================================================
-- 2) Trusted seed goals
-- ============================================================
seed_goals as (
    select
        sg.order_name,
        sg.advertiser_name,
        sg.impression_goal::bigint as seed_impression_goal
    from {{ ref('seed_partner_names_goals') }} as sg
),

-- ============================================================
-- 3) Join orders to seed goals (order-level goals)
-- ============================================================
order_goals as (
    select
        o.order_id,
        o.order_name,
        o.advertiser_name,
        sg.seed_impression_goal as impression_goal
    from orders o
    left join seed_goals sg
        on  lower(o.order_name)      = lower(sg.order_name)
        and lower(o.advertiser_name) = lower(sg.advertiser_name)
),

-- ============================================================
-- 4) Combine orders + goals into an order-level view
-- ============================================================
order_level as (
    select
        o.order_id,
        o.order_name,
        o.advertiser_name,
        o.order_delivery_status_name,

        o.schedule_start_date,
        o.schedule_end_date,
        o.schedule_start_datetime,
        o.schedule_end_datetime,

        o.reported_impressions,
        o.reported_clicks,
        o.reported_ctr,

        og.impression_goal
    from orders o
    left join order_goals og
        on o.order_id = og.order_id
),

-- ============================================================
-- 5) Pacing + health metrics (per order)
-- ============================================================
final as (
    select
        ol.order_id,
        ol.order_name,
        ol.advertiser_name,
        ol.order_delivery_status_name,

        ol.schedule_start_date,
        ol.schedule_end_date,
        ol.schedule_start_datetime,
        ol.schedule_end_datetime,

        -- Delivery metrics
        ol.reported_impressions,
        ol.reported_clicks,
        ol.reported_ctr,

        ol.impression_goal,

        -- Total scheduled days (inclusive)
        case
            when
                ol.schedule_start_date is not null
                and ol.schedule_end_date   is not null
                and ol.schedule_end_date   >  ol.schedule_start_date
            then (ol.schedule_end_date - ol.schedule_start_date + 1)
            else null
        end::integer as schedule_days,

        -- Days elapsed so far (capped at schedule_end_date, never negative)
        case
            when
                ol.schedule_start_date is not null
                and ol.schedule_end_date   is not null
            then greatest(
                     0,
                     (least(current_date, ol.schedule_end_date) - ol.schedule_start_date + 1)
                 )
            else null
        end as days_elapsed,

        -- Actual impressions per day (over full schedule window)
        case
            when
                ol.schedule_start_date is not null
                and ol.schedule_end_date   is not null
                and (ol.schedule_end_date - ol.schedule_start_date + 1) > 0
            then coalesce(ol.reported_impressions, 0)
                 / (ol.schedule_end_date - ol.schedule_start_date + 1)
            else null
        end as actual_impressions_per_day,

        -- Target impressions per day (goal spread over schedule window)
        case
            when
                ol.schedule_start_date is not null
                and ol.schedule_end_date   is not null
                and (ol.schedule_end_date - ol.schedule_start_date + 1) > 0
            then coalesce(ol.impression_goal, 0)
                 / (ol.schedule_end_date - ol.schedule_start_date + 1)
            else null
        end as target_impressions_per_day
    from order_level ol
)

select
    f.*,

    -- Percent of schedule elapsed (0–1)
    case
        when
            f.days_elapsed   is not null
            and f.schedule_days is not null
            and f.schedule_days > 0
        then f.days_elapsed::numeric / f.schedule_days::numeric
        else null
    end as schedule_pct_elapsed,

    -- Percent of goal delivered (0–1)
    case
        when f.impression_goal is not null and f.impression_goal > 0
        then coalesce(f.reported_impressions, 0)::numeric / f.impression_goal::numeric
        else null
    end as delivered_impressions_pct_of_goal,

    -- Safe pacing index:
    --   (delivered_per_day / target_per_day) / (days_elapsed / schedule_days)
    -- If any guard condition fails, the CASE returns NULL and COALESCE forces 0.
    coalesce(
        case
            when
                f.impression_goal              is not null and f.impression_goal              > 0
                and f.schedule_start_date      is not null
                and f.schedule_end_date        is not null
                and f.target_impressions_per_day is not null and f.target_impressions_per_day > 0
                and f.days_elapsed             is not null and f.days_elapsed             > 0
                and f.schedule_days            is not null and f.schedule_days            > 0
            then
                (f.actual_impressions_per_day::numeric / f.target_impressions_per_day::numeric)
                /
                (f.days_elapsed::numeric / f.schedule_days::numeric)
            else null
        end,
        0
    ) as pacing_index
from final f