"""
This script is meant to give a brief introduction to the Dataset object.

Ideally this script should correspond to the code examples in the documentation basic-usage
"""
import polars as pl
import differences
import datetime

data = differences.Dataset(
    {
        "numbers": [1, 2, 3],
        "letters": ["a", "b", "c"],
        "dates": [
            datetime.date(2010, 1, 1),
            datetime.date(2011, 1, 1),
            datetime.date(2012, 1, 1)]
    },
    name="example-dataset"  # we can give the dataset a name
)

# ------------------------- add information ----------------------------

# data = data.filter(pl.col(""))

data = data.info_add(
    descriptions={
        "data description": "This is an example Dataset",
        "column_descriptions": {"col1": "description of col 1"}
    },
)

from differences import Info

new_info = Info("Various Info")

new_info.update(
    statistics={"R2": 0.4, "R2-Adj": 0.3},
    fixed_effects={"firm": "YES", "year": "YES", "product": "NO"}
)

print(str(new_info))

# inplace operation
new_info._config_repr(fmt_type=2, n_columns=2)
print(str(new_info))

data.info_add(new_info)
data.filter(pl.col("letters").eq("a"))
# data._data
# data._info
