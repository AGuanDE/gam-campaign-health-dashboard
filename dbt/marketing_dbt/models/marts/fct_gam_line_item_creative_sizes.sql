{{ config(
    materialized = 'table',
    tags = ['gam', 'line_item', 'creative_size']
) }}

-- fct_gam_line_item_creative_sizes
-- Purpose:
--   Per-line-item creative size breakdown with delivery and share-of-line metrics.
--   Grain: (order_id, line_item_id, rendered_creative_size, device_category_name).

with size_breakdown as (
    select
        s.line_item_id,
        s.line_item_name,
        s.order_id,
        s.order_name,
        s.advertiser_name,

        s.order_start_date,
        s.order_end_date,
        s.order_start_datetime,
        s.order_end_datetime,

        s.rendered_creative_size,
        s.device_category_name,

        s.impressions as creative_size_impressions,
        s.clicks      as creative_size_clicks,
        s.ctr         as creative_size_ctr
    from {{ ref('stg_gam_line_item_creative_sizes') }} as s
),

order_health as (
    select
        oh.order_id,
        oh.order_name,
        oh.advertiser_name,
        oh.reported_impressions as order_reported_impressions,
        oh.impression_goal
    from {{ ref('fct_gam_order_health') }} as oh
),

size_order_stats as (
    select
        sb.order_id,
        sb.order_name,
        sb.advertiser_name,
        sb.rendered_creative_size,

        sum(sb.creative_size_impressions)::numeric as order_size_impressions,
        count(distinct sb.line_item_id)            as order_size_line_items,
        case
            when count(distinct sb.line_item_id) > 0
                then sum(sb.creative_size_impressions)::numeric / count(distinct sb.line_item_id)
            else null
        end as order_size_avg_impressions_per_line_item
    from size_breakdown sb
    group by
        sb.order_id,
        sb.order_name,
        sb.advertiser_name,
        sb.rendered_creative_size
),

line_item_totals as (
    select
        li.line_item_id,
        li.order_id,
        li.order_name,
        li.advertiser_name,

        li.impressions                 as line_item_total_impressions,
        li.clicks                      as line_item_total_clicks,
        li.ctr                         as line_item_ctr,

        li.line_item_priority,
        li.line_item_type_name,
        li.order_delivery_status_name
    from {{ ref('stg_gam_line_items') }} as li
)

select
    sb.line_item_id,
    sb.line_item_name,
    sb.order_id,
    sb.order_name,
    sb.advertiser_name,

    sb.order_start_date,
    sb.order_end_date,
    sb.order_start_datetime,
    sb.order_end_datetime,

    li.line_item_total_impressions,
    li.line_item_total_clicks,
    li.line_item_ctr,

    li.line_item_priority,
    li.line_item_type_name,
    li.order_delivery_status_name,

    sb.rendered_creative_size,
    sb.device_category_name,
    sb.creative_size_impressions,
    sb.creative_size_clicks,
    sb.creative_size_ctr,

    case
        when li.line_item_total_impressions is not null
             and li.line_item_total_impressions > 0
            then sb.creative_size_impressions::numeric
                 / li.line_item_total_impressions::numeric
        else null
    end as creative_size_share_of_line_item,

    -- Order-level context
    oh.order_reported_impressions,
    oh.impression_goal,

    sos.order_size_impressions,
    sos.order_size_line_items,
    sos.order_size_avg_impressions_per_line_item,

    -- Share of the order's delivered impressions that this creative size represents
    case
        when oh.order_reported_impressions is not null
             and oh.order_reported_impressions > 0
        then
            sos.order_size_impressions
            / oh.order_reported_impressions::numeric
        else null
    end as size_share_of_order_delivery,

    -- Portion of the order's impression goal delivered by this creative size
    case
        when oh.impression_goal is not null
             and oh.impression_goal > 0
        then
            sos.order_size_impressions
            / oh.impression_goal::numeric
        else null
    end as size_pct_of_order_goal
from size_breakdown sb
left join line_item_totals li
    on sb.line_item_id = li.line_item_id
left join order_health oh
    on sb.order_id = oh.order_id
left join size_order_stats sos
    on  sb.order_id = sos.order_id
    and sb.rendered_creative_size = sos.rendered_creative_size
