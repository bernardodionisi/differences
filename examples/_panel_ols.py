import pandas as pd
import statsmodels.api as sm

from linearmodels.panel import PanelOLS
from linearmodels.datasets import wage_panel

from time import perf_counter

data = wage_panel.load()
year = pd.Categorical(data.year)
data = data.set_index(["nr", "year"])
data["year"] = year

print(wage_panel.DESCR)
print(data.head())

exog_vars = ["expersq", "union", "married", "year"]
exog = sm.add_constant(data[exog_vars])

start = perf_counter()
mod = PanelOLS(data.lwage, exog, entity_effects=True)
fe_res = mod.fit()

end = perf_counter()

print(f"lineramodels: {end - start}")

# ----------------------------------------------------------------------

import polars as pl
from differences.models.lm.panel_ols import panel_ols_fit

data = (
    pl.DataFrame(
        data
        .drop(columns=["year"])
        .reset_index()
    )
    .with_columns(
        pl.col("year").cast(pl.Utf8).cast(pl.Categorical).alias("year_cat"),
        pl.lit(1).alias("Intercept")
    )
)

#
# panel_ols_fit(
#     data=data,
#     y_column="lwage",
#     x_columns=["expersq", "union", "married", "year_cat"],
#     entity_column='nr',
#     time_column='year',
# )


start = perf_counter()

res = panel_ols_fit(
    data=data,
    y_column="lwage",
    x_columns=["Intercept", "expersq", "union", "married", "year_cat"],
    entity_column='nr',
    time_column='year',
    entity_fixed_effects=True,
)

end = perf_counter()

print(f"wrapper: {end - start}")

res.coefficient_table().sort_by(by_estimate=True, descending=True)
