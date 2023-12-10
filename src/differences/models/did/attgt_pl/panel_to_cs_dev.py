from __future__ import annotations

from typing import Literal

import polars as pl


def is_two_periods_panel():
    pass


def panel_2_cross_section(
        data: pl.DataFrame,
        time_column: str,
        y_column: str,
        x: list,  # x_base? but what about "universal" when base_period > time?
        x_delta: list,
        base_period: int = None,
        time: int = None,
):
    """Converts a balanced panel, with 2 periods, into a cross-section"""

    time_name = data.index.names[1]  # 'time'

    # maybe make this an argument instead of extracting it within the function
    times = data.select(time_column).unique()

    if time is None or base_period is None:
        base_period, time = min(_times), max(_times)

    mask_time = times_values == time

    #  y[time] - y[base_period]
    y_delta = data[y_name].loc[mask_time].values - data[y_name].loc[~mask_time].values

    if x_delta:  # get delta Xs if requested: x[time] - x[base_period]
        # only time varying covariates
        tvc_delta = (
                data[x_delta].loc[mask_time].values - data[x_delta].loc[~mask_time].values
        )

    # keep vars in the earliest between time and base_period: x[time] or x[base_period]
    data = (
        data.loc[mask_time].copy()
        if time < base_period
        else data.loc[~mask_time].copy()
    )

    data[y_name] = y_delta

    if x_delta:
        (
            delta_to_create,
            create_idx,
            delta_to_replace,
            replace_idx,
        ) = delta_col_to_create(x_delta=x_delta, x_base=x_base, return_idx=True)

        if (
                delta_to_create
        ):  # if x is in both delta and base then add delta_ cols to data
            data[[f"delta_{c}" for c in delta_to_create]] = tvc_delta[:, create_idx]

        if (
                delta_to_replace
        ):  # replace base cols with delta, if these cols are not in x_base

            data[delta_to_replace] = tvc_delta[:, replace_idx]

        # warning: delta_to_replace cols will have the original name but they will be deltas
        # however the user has not access to this data

    return data


data = pl.DataFrame(
    {"int": [20, 30, 25, 3345], "int2": [20, 30, 25, 3345], "entity": ["a", "a", "b", "b"],
     "time": [1, 2, 1, 2]}
).sort(["entity", "time"], descending=[False, True])

delta_columns = [pl.col(i).diff().over("entity").alias(f"delta_{i}") for i in ["int", "int2"]]
delta_columns.extend(["int"])

data.select(delta_columns)


class PanelAsCrossSection:

    def __init__(
            self,
            data: pl.DataFrame,
            entity_column: str,
            time_column: str,
    ):
        self.data = data
        self.entity_column = entity_column
        self.time_column = time_column

        self.y_column = None
        self.x_column = None

    def panel_2_cross_section(
            self,
            y_column: str,
            x: str | list[str],
            x_delta: str | list[str]
    ) -> pl.DataFrame:

        if not isinstance(x, list):
            x = [x]
        elif x is None:
            x = []

        if not isinstance(x, list):
            x_delta = [x_delta]
        elif x_delta is None:
            x_delta = []

        delta_columns = [
            pl.col(i).diff().over(self.entity_column).alias(f"delta_{i}")
            for i in [y_column] + x_delta
        ]

        delta_columns.extend(x)

        return self.data.select(delta_columns)


def get_cross_section_columns(
        entity_column: str,
        y_column: str,
        x: str | list[str],
        x_delta: str | list[str]
) -> list[str | pl.Expr]:
    """Cross-section columns to keep/create."""

    if not isinstance(x, list):
        x = [x]
    elif x is None:
        x = []

    if not isinstance(x_delta, list):
        x_delta = [x_delta]
    elif x_delta is None:
        x_delta = []

    delta_columns = [
        pl.col(i).diff().over(entity_column).alias(f"delta_{i}")
        for i in [y_column] + x_delta
    ]

    # add base columns
    delta_columns.extend(x)

    return delta_columns

# ----------------------------------------------------------------------
