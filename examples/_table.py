from __future__ import annotations

import polars as pl

from differences.report.table import Table

# ----------------------------------------------------------------------

data = pl.DataFrame({
    "coefficient": ["b1", "a2"],
    "estimate": [0.1, 0.2],
    "lower": [0, 0.1],
    "upper": [0.2, 0.3]
})

id(data)

check = {"a": data}

check["a"] is data

table_0 = Table(
    data=data,
    name="table-0",

    id_column="coefficient",
    estimate_column="estimate",
    group="1"
)

table_0._info_add(statistics={"R squared": 0.2})

table_1 = Table(
    data=data,
    name="table-1",

    id_column="coefficient",
    estimate_column="estimate",

)

table_1._info_add(statistics={"R squared": 0.2})

check = table_0.concat_other(table_1)

data = check.data.with_columns(pl.col("lower").cast(pl.Utf8))





# ----------------------------------------------------------------------

table_0._info_update_polars_config(tbl_rows=1)

table_0._polars_config

table_0.sort_by(as_numeric=True)
table_0.insert_zeros([5, 4, 3]).sort_id(as_numeric=True)

table_0.group_columns

# table.plot().show()
# table.add_reference("d").plot().show()
# table.filter_coefficients(values=["a", "d"]).plot().show()

# ----------------------------------------------------------------------

data = pl.DataFrame({
    "coefficient": ["a", "b", "a", "b"],
    "estimate": [0.1, 0.2, 0.3, 0.4],
    "lower": [0, 0.1, 0.2, 0.3],
    "upper": [0.2, 0.3, 0.4, 0.5],
    "sample": [1, 1, 2, 2]
})

try:
    Table(
        data=data,
        id_column="coefficient",
        estimate_column="estimate",
        # group_columns="sample"
    )
except ValueError:
    print("correct")

table_1 = Table(
    data=data,
    id_column="coefficient",
    estimate_column="estimate",
    group_columns="sample"
)

# ----------------------------------------------------------------------

data = pl.DataFrame({
    "coefficient": ["t-1", "t1", "t-1", "t1"],
    "estimate": [0.1, 0.2, 0.3, 0.4],
    "lower": [0, 0.1, 0.2, 0.3],
    "upper": [0.2, 0.3, 0.4, 0.5],
    "sample": [1, 1, 2, 2]
})

table_2 = Table(
    data=data,
    name="table-2",
    id_column="coefficient",
    estimate_column="estimate",
    group_columns="sample"
)

table_2 = table_2.insert_zeros()

# table.plot().show()

# ----------------------------------------------------------------------

data = pl.DataFrame({
    "coefficient": ["t-1", "t1", "a", "t-1", "t1", "a"],
    "estimate": [0.1, 0.2, 1, 0.3, 0.4, 1],
    "lower": [0, 0.1, 0, 0.2, 0.3, 0],
    "upper": [0.2, 0.3, 2, 0.4, 0.5, 2],
    "sample": [1, 1, 1, 2, 2, 2]
})

table_3 = Table(
    data=data,
    name="table-3",
    id_column="coefficient",
    estimate_column="estimate",
    group_columns="sample"
)
#
# table.add_reference(['ref-0', "ref-1"])
# table.filter_coefficients(starts_with="t").add_reference(0)

check = table_0.concat_other([table_2, table_3])

# check.group_columns
# data = pl.DataFrame({
#     "coefficient": ["A", "B"],
#     "estimate": [0.1, 0.2],
#     "lower": [0, 0.1],
#     "upper": [0.2, 0.3],
# })
#
# table_3 = Table(
#     data=data,
#     table_name="table-4",
#     id_column="coefficient",
#     est_column="estimate",
# )
#
# check.concat_other(table_3)
#
# table_3.group_columns
# table_3.concat_other(check)
#
# check._info
# table_3.group_columns


# ----------------------------------------------------------------------

import polars as pl

data = pl.DataFrame({
    "coefficient": ["b1", "a2"],
    "estimate": [0.1, 0.2],
    "se": [1, 2],
    "lower": [0, 0.1],
    "upper": [0.2, 0.3]
})

check = (
    data
    .with_row_count()
    .melt(id_vars=["coefficient", "row_nr"], value_vars=["estimate", "se"])
    .with_row_count(name="row_nr_2")
    .sort(["row_nr", "row_nr_2"])
    .drop(["row_nr", "row_nr_2"])
)

# ----------------------------------------------------------------------

import pandas as pd

check = check.to_pandas().set_index("coefficient")

multi_index_columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])
check.columns = pd.MultiIndex.from_tuples(multi_index_columns, names=["letters", "numbers"])

print(check.to_latex())
print(check.to_html())


import polars as pl

from differences.report.table import Table

data = pl.DataFrame({
    "coefficient": ["a", "b"],
    "estimate": [0.1, 0.2],
    "lower": [0, 0.1],
    "upper": [0.2, 0.3]
})

table = Table(
    data=data,
    id_column="coefficient",
    estimate_column="estimate",
)

import lets_plot as lp

plot = (
    table
    .plot(
        labs_kws=dict(
            title="Event Study",
            x="X Title",
            y="Y Title",
            caption="aaa"
        ),
        geom_hline_kws=dict(yintercept=0, linetype=3),
        geom_vline_kws=dict(xintercept=0.5, linetype=3),
        geom_text_kws=dict(x=0.8, y=0.35, label='Example\nText', size=6),
        ggsize_kws=dict(width=1.5, height=1.5),
        # coord_flip_kws=dict(),
        theme_kws=dict(plot_caption=lp.element_text(color="red", size=20, hjust=0)),
    )
    # + lp.labs(axis_title_x=lp.element_text("red"))
    # + lp.theme(axis_title_x=lp.element_text("red"))
)

plot.show()
