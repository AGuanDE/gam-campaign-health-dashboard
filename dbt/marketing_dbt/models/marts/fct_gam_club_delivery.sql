{{ config(
    materialized = 'table',
    tags = ['gam', 'club', 'health']
) }}

-- fct_gam_club_delivery
-- Purpose:
--   Club-grain fact table built from GAM domain mapping.
--   One row per (club_name, order_id).
--   Designed to answer:
--     - Which orders/advertisers are behind or ahead of target (globally)?
--     - Among those, which ones actually touch this club?
--     - How much of an order's delivery is on this club (share of order)?
--     - How much of a club's inventory is from each order (share of club)?
--
-- IMPORTANT:
--   Order-level goals and pacing live in fct_gam_order_health.
--   This model does NOT invent club-specific goals or club pacing.
--   It only provides:
--     - club-level delivery counts
--     - contribution percentages (share of order, share of club)
--     - global order pacing for context.

-- ============================================================
-- 1) Bring in domain-level delivery with mapped clubs
-- ============================================================
with domain_mapping as (
    select
        dm.order_id,
        dm.order_name,
        dm.line_item_id,
        dm.line_item_name,
        dm.advertiser_name,
        dm.domain_name,
        dm.ad_server_impressions
    from {{ source('gam', 'gam_domain_mapping') }} as dm
),

domain_with_club as (
    select
        dm.order_id,
        dm.order_name,
        dm.line_item_id,
        dm.line_item_name,
        dm.advertiser_name,

        dm.domain_name,
        sdn.club_name,
        sdn.club_display_name,

        dm.ad_server_impressions::bigint as ad_server_impressions
    from domain_mapping dm
    left join (
        -- Enforce a single mapping per domain to avoid accidental fan-out
        select distinct on (lower(domain_name))
            lower(domain_name) as domain_name_lower,
            club_name,
            club_display_name
        from {{ ref('seed_domain_names') }}
        order by lower(domain_name), club_name
    ) as sdn
        on lower(dm.domain_name) = sdn.domain_name_lower
),

null_club_check as (
    select count(*) as null_count
    from domain_with_club
    where club_name is null
       or club_display_name is null
),

-- Guard: force the model to fail fast if any null clubs are present
null_club_guard as (
    select case when null_count > 0 then 1 / 0 else 0 end as must_fail
    from null_club_check
),

-- ============================================================
-- 2) Aggregate to (club, order) grain
-- ============================================================
club_order_agg as (
    select
        dwc.club_name,
        dwc.club_display_name,

        dwc.order_id,
        dwc.order_name,
        dwc.advertiser_name,

        sum(dwc.ad_server_impressions)::bigint as club_delivered_impressions,
        count(distinct dwc.line_item_id)        as distinct_line_items_on_club
    from domain_with_club dwc
    group by
        dwc.club_name,
        dwc.club_display_name,
        dwc.order_id,
        dwc.order_name,
        dwc.advertiser_name
),

-- ============================================================
-- 3) Compute total impressions per club
-- ============================================================
club_totals as (
    select
        coa.club_name,
        coa.club_display_name,
        sum(coa.club_delivered_impressions)::bigint as total_club_impressions
    from club_order_agg coa
    group by
        coa.club_name,
        coa.club_display_name
),

-- ============================================================
-- 4) Bring in global order-level health
--    (aligned to fct_gam_order_health)
-- ============================================================
order_health as (
    select
        oh.order_id,
        oh.order_name,
        oh.advertiser_name,

        oh.order_delivery_status_name,

        oh.schedule_start_date,
        oh.schedule_end_date,
        oh.schedule_start_datetime,
        oh.schedule_end_datetime,

        -- Treat reported_impressions as the canonical total delivered
        oh.reported_impressions  as order_reported_impressions,
        oh.reported_clicks       as order_reported_clicks,
        oh.reported_ctr,

        oh.impression_goal,

        oh.schedule_days,
        oh.days_elapsed,
        oh.schedule_pct_elapsed,
        oh.actual_impressions_per_day,
        oh.target_impressions_per_day,

        oh.pacing_index          as order_pacing_index
    from {{ ref('fct_gam_order_health') }} as oh
),

-- ============================================================
-- 5) Combine club-level delivery with global order health
--    and compute contribution metrics:
--      - club_share_of_order_delivery
--      - order_share_of_club_delivery
-- ============================================================
final as (
    select
        coa.club_name,
        coa.club_display_name,

        coa.order_id,
        coa.order_name,
        coa.advertiser_name,

        oh.order_delivery_status_name,

        -- Schedule / pacing from the global order
        oh.schedule_start_date,
        oh.schedule_end_date,
        oh.schedule_start_datetime,
        oh.schedule_end_datetime,
        oh.schedule_days,
        oh.days_elapsed,
        oh.schedule_pct_elapsed,
        oh.order_pacing_index,

        -- Delivery metrics
        coa.club_delivered_impressions,
        oh.order_reported_impressions,
        oh.order_reported_clicks,

        -- Goal metrics (global, not club-specific)
        oh.impression_goal,

        -- How much of the order's delivered impressions are on this club?
        case
            when oh.order_reported_impressions is not null
                 and oh.order_reported_impressions > 0
            then
                coa.club_delivered_impressions::numeric
                / oh.order_reported_impressions::numeric
            else null
        end as club_share_of_order_delivery,

        -- How much of the club's total impressions are from this order?
        case
            when ct.total_club_impressions is not null
                 and ct.total_club_impressions > 0
            then
                coa.club_delivered_impressions::numeric
                / ct.total_club_impressions::numeric
            else null
        end as order_share_of_club_delivery,

        -- Count of distinct line items from this order on this club
        coa.distinct_line_items_on_club
    from club_order_agg coa
    left join club_totals ct
        on  coa.club_name         = ct.club_name
        and coa.club_display_name = ct.club_display_name
    left join order_health oh
        on coa.order_id = oh.order_id
    -- Ensure null-club guard is evaluated (and fails if null_count > 0)
    cross join null_club_guard
)

select *
from final
