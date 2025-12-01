{{ config(
    materialized = 'view',
    tags = ['gam', 'line_item', 'creative_size']
) }}

select *
from {{ ref('fct_gam_line_item_creative_sizes') }}
