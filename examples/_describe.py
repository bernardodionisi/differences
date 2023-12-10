from __future__ import annotations

from differences.report.describe.describe_new import describe_numeric
from differences.report.describe.describe_new import plot_unicode_distribution

import polars as pl
from datetime import date
import numpy as np
import pandas as pd

data = pl.DataFrame(
    {
        "c": [True, False, True],
        "d": [None, "b", "c"],
        "group": [1, 1, 2],
        "f": [date(2020, 1, 1),
              date(2021, 1, 1),
              date(2022, 1, 1)],
        "b": [4, None, None],

        "a": [1.111, 2.8, 3.91409],

    }
)
hist = plot_unicode_distribution(data, n_bins=9)

out = describe_numeric(data=data, by="group")

out.with_columns(pl.col("column").map_dict(hist))

array = np.random.exponential(scale=1 / 0.1, size=1000)
data_pl = pl.DataFrame({"values": array})

data_pd = pd.DataFrame({"values": array})

counts, edges = np.histogram(data_pl['values'].to_numpy(), bins=10)

data_pl['values'].hist(bin_count=9)

# data.select([pl.col("a").mean().alias("mean"), pl.col('a').n_unique().alias("n_unique")])
# print(data.__repr__())
#
# with pl.Config(tbl_cols=20, tbl_width_chars=200):
#     print(describe(data))

# -------------------------------------------

from differences.report.describe.describe_new import plot_unicode_distribution


# _data_describe_utf8(data_describe).rename({"column": ""})

# ----------------------------------------------------------------------

import numpy as np

lambda_param = 0.00001

array = np.random.exponential(scale=1 / lambda_param, size=1000)

arr = np.random.uniform(size=10)

arr = np.random.binomial(1, 0.7, size=10000)

plot_unicode_distribution(arr, n_bins=9)

# Example usage with the previous data
plot_distribution(data['values'])

# Example usage with the previous data
plot_distribution(data['values'])
