{{ config(
    materialized = 'table',
    tags = ['gam', 'line_item', 'diagnostics']
) }}

-- fct_gam_line_item_diagnostics
-- Purpose:
--   Diagnostics fact table at the (order_id, line_item_id, domain_name) grain.
--   Enriches each domain-level row with line-item attributes and club mapping,
--   plus order-level pacing context.
--
-- Grain:
--   One row per (order_id, line_item_id, domain_name).
--
-- Inputs:
--   - stg_gam_line_items
--   - source('gam', 'gam_domain_mapping')
--   - seed_domain_names
--   - fct_gam_order_health

-- ============================================================
-- 1) Base line-item attributes & delivery
-- ============================================================
with line_items as (
    select
        li.line_item_id,
        li.line_item_name,
        li.order_id,
        li.order_name,
        li.advertiser_name,

        li.schedule_start_date,
        li.schedule_end_date,
        li.schedule_start_datetime,
        li.schedule_end_datetime,

        li.impression_goal_units_raw,
        li.impression_goal_unit_type_raw,

        li.lifetime_impressions,
        li.lifetime_clicks,
        li.impressions,
        li.clicks,
        li.ctr,

        li.line_item_delivery_indicator,
        li.line_item_priority,
        li.line_item_type_name,
        li.order_delivery_status_name,

        li.creative_name,
        li.rendered_creative_size,
        li.device_category_name
    from {{ ref('stg_gam_line_items') }} as li
),

-- ============================================================
-- 2) Domain-level delivery mapping
-- ============================================================
domain_mapping as (
    select
        dm.order_id,
        dm.order_name,
        dm.line_item_id,
        dm.line_item_name,
        dm.advertiser_name,

        dm.domain_name,
        dm.ad_server_impressions::bigint as domain_impressions
    from {{ source('gam', 'gam_domain_mapping') }} as dm
),

-- ============================================================
-- 3) Attach line-item attributes to each domain row
-- ============================================================
line_items_with_domain as (
    select
        dm.order_id,
        dm.order_name,
        dm.advertiser_name,

        dm.line_item_id,
        dm.line_item_name,

        li.schedule_start_date,
        li.schedule_end_date,
        li.schedule_start_datetime,
        li.schedule_end_datetime,

        li.impression_goal_units_raw,
        li.impression_goal_unit_type_raw,

        li.lifetime_impressions,
        li.lifetime_clicks,
        li.impressions,
        li.clicks,
        li.ctr as line_item_ctr,

        li.line_item_delivery_indicator,
        li.line_item_priority,
        li.line_item_type_name,
        li.order_delivery_status_name,

        li.creative_name,
        li.rendered_creative_size,
        li.device_category_name,

        dm.domain_name,
        dm.domain_impressions
    from domain_mapping dm
    left join line_items li
        on  dm.order_id     = li.order_id
        and dm.line_item_id = li.line_item_id
),

-- ============================================================
-- 4) Map domain_name -> club_name / club_display_name
-- ============================================================
line_items_with_club as (
    select
        lid.order_id,
        lid.order_name,
        lid.advertiser_name,

        lid.line_item_id,
        lid.line_item_name,

        lid.schedule_start_date,
        lid.schedule_end_date,
        lid.schedule_start_datetime,
        lid.schedule_end_datetime,

        lid.impression_goal_units_raw,
        lid.impression_goal_unit_type_raw,

        lid.lifetime_impressions,
        lid.lifetime_clicks,
        lid.impressions,
        lid.clicks,
        lid.line_item_ctr,

        lid.line_item_delivery_indicator,
        lid.line_item_priority,
        lid.line_item_type_name,
        lid.order_delivery_status_name,

        lid.creative_name,
        lid.rendered_creative_size,
        lid.device_category_name,

        lid.domain_name,
        sdn.club_name,
        sdn.club_display_name,

        lid.domain_impressions
    from line_items_with_domain lid
    left join {{ ref('seed_domain_names') }} as sdn
        on lower(lid.domain_name) = lower(sdn.domain_name)
),

-- ============================================================
-- 5) Bring in global order-level health for context
--    (aligned to fct_gam_order_health)
-- ============================================================
order_health as (
    select
        oh.order_id,
        oh.order_name,
        oh.advertiser_name,

        -- Treat reported_impressions as canonical order-level delivery
        oh.reported_impressions  as order_reported_impressions,
        oh.reported_clicks       as order_reported_clicks,
        oh.impression_goal,

        oh.schedule_start_date   as order_schedule_start_date,
        oh.schedule_end_date     as order_schedule_end_date,
        oh.schedule_days,
        oh.days_elapsed,
        oh.schedule_pct_elapsed,
        oh.actual_impressions_per_day,
        oh.target_impressions_per_day,
        oh.pacing_index          as order_pacing_index
    from {{ ref('fct_gam_order_health') }} as oh
),

-- ============================================================
-- 6) Final diagnostics view
-- ============================================================
final as (
    select
        lic.order_id,
        lic.order_name,
        lic.advertiser_name,

        lic.line_item_id,
        lic.line_item_name,

        lic.schedule_start_date,
        lic.schedule_end_date,
        lic.schedule_start_datetime,
        lic.schedule_end_datetime,

        lic.impression_goal_units_raw,
        lic.impression_goal_unit_type_raw,

        lic.lifetime_impressions      as line_item_total_impressions,
        lic.lifetime_clicks           as line_item_total_clicks,
        lic.impressions               as line_item_window_impressions,
        lic.clicks                    as line_item_window_clicks,
        lic.line_item_ctr,

        lic.line_item_delivery_indicator,
        lic.line_item_priority,
        lic.line_item_type_name,
        lic.order_delivery_status_name,

        lic.creative_name,
        lic.rendered_creative_size,
        lic.device_category_name,

        lic.domain_name,
        lic.club_name,
        lic.club_display_name,

        lic.domain_impressions,

        -- Order-level context:
        oh.order_reported_impressions,
        oh.order_reported_clicks,
        oh.impression_goal,
        oh.order_schedule_start_date,
        oh.order_schedule_end_date,
        oh.schedule_days,
        oh.days_elapsed,
        oh.schedule_pct_elapsed,
        oh.actual_impressions_per_day,
        oh.target_impressions_per_day,
        oh.order_pacing_index,

        -- Diagnostic ratios:

        -- 1) What share of this line item's total impressions are on this domain?
        case
            when lic.lifetime_impressions is not null
                 and lic.lifetime_impressions > 0
            then
                lic.domain_impressions::numeric
                / lic.lifetime_impressions::numeric
            else null
        end as domain_share_of_line_item,

        -- 2) What share of the order's total delivery is coming from this line item?
        case
            when oh.order_reported_impressions is not null
                 and oh.order_reported_impressions > 0
            then
                lic.lifetime_impressions::numeric
                / oh.order_reported_impressions::numeric
            else null
        end as line_item_share_of_order
    from line_items_with_club lic
    left join order_health oh
        on  lic.order_id        = oh.order_id
        and lic.advertiser_name = oh.advertiser_name
)

select *
from final
