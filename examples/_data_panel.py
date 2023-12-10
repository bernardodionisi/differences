from __future__ import annotations

import polars as pl

import differences
import datetime

data = differences.Dataset(
    {
        "entities": ["e1", "e1", "e1", "e2", "e2", "e3"],
        "times": [
            datetime.date(2011, 1, 1),
            datetime.date(2011, 1, 1),
            datetime.date(2011, 5, 1),
            datetime.date(2012, 12, 1),
            datetime.date(2013, 1, 30),
            datetime.date(2005, 12, 1)
        ],
        "n": [1, 2, 3, 4, 5, 6],
        "s": ["a", "b", "c", "d", "e", "f"],
        "group": ["y", "n", "y", "y", "n", "y"]
    },
    name="for panel"
)

data_panel = differences.DataPanel(data, entity_column="entities")

data_panel._info

data_panel._info_add(check={"a": "a"})



check = data_panel.add_entity_agg(pl.col("n") + pl.col("n").max().alias("max"))
check = check.add_entity_agg(pl.col("n").max().alias("max"))

check.meta_agg()
check.build_panel()
check._agg
agg: pl.Expr = check._agg["entity"]["aggs"][0]
agg.meta.tree_format()
check._has_aggregations()

data = differences.DataPanel(
    {
        "entities": ["e1", "e1", "e1", "e2", "e2", "e3"],
        "times": [
            datetime.date(2011, 1, 1),
            datetime.date(2011, 1, 1),
            datetime.date(2011, 5, 1),
            datetime.date(2012, 12, 1),
            datetime.date(2013, 1, 30),
            datetime.date(2005, 12, 1)
        ],
        "n": [1, 2, 3, 4, 5, 6],
        "s": ["a", "b", "c", "d", "e", "f"],
        "group": ["y", "n", "y", "y", "n", "y"]
    },
    entity_column="entities"
)
data

data.build_panel()

data = data.add_entity_agg(
    pl.col("n").mean(), pl.col("n").mean().alias("mean2")
).add_entity_time_agg(abc=pl.col("group").first())

data.meta_agg
data = data.add_entity_agg(pl.col("n").first())
data.meta_agg(tree_format=True)

data._meta_output_names()
data._agg
data.collect()

[pl.col("c").mean(), pl.col("c").mean()]
data.collect().build_panel().collect()
#
# # ----------------------------------------------------------------------
#
# data = (
#     data
#     .set_entity_time_agg(
#         pl.col("n").mean()
#     )
#     .build_panel_data(relative_end=3, end=2020)
#     .sort('entities')
# )
#
# data.get_variables().collect()
#
# # ----------------------------------------------------------------------
#
#
# data.data.pipe(differences.to_datapanel(entity_column="entities", time_column="times"))
