from __future__ import annotations

from typing import Literal

import polars as pl


def get_relative_periods_data(
        data: pl.DataFrame,  # event_data
        entity_column: str,
        cohort_column: str,
        reference_period: int = None,
        dosage_column: str = None,
        start: int = None,
        end: int = None,
        bin_endpoints: Literal["start", "end", "both"] | bool = "both",
        min_max_time_columns: tuple[str, str] = ("_min_panel_time", "_max_panel_time"),
        time_column_name: str = "time",
):
    data = _get_relative_periods_dosages(
        data=data,
        cohort_column=cohort_column,
        dosage_column=dosage_column,
        min_max_time_columns=min_max_time_columns,
        time_column_name=time_column_name
    )

    relative_periods = sorted(
        [item for item in data.columns if item.startswith('relative_period::')],
        key=lambda x: int(x.split('::')[-1])
    )

    # ------------------------------------------------------------------

    # if there are multiple events per entity
    data = (
        data
        .group_by([entity_column, time_column_name], maintain_order=True)
        .agg([pl.col(i).sum() for i in relative_periods])
    )

    # ------------------------------------------------------------------

    if isinstance(bin_endpoints, bool):
        if bin_endpoints:
            bin_endpoints = "both"

    data = _restrict_start_end_and_bin_endpoints(
        data=data,
        start=start,
        end=end,
        entity_column=entity_column,
        time_column=time_column_name,
        bin_endpoints=bin_endpoints
    )

    # ------------------------------------------------------------------

    if reference_period is not None:
        reference_period_column = f"relative_period::{reference_period}"
        if reference_period_column not in data.columns:
            raise ValueError(f"Invalid 'reference_period': {reference_period}")
        data = data.drop(columns=[reference_period_column])

    return data.sort([entity_column, time_column_name])


def _get_relative_periods_dosages(
        data: pl.DataFrame,
        cohort_column: str,
        dosage_column: str | None,
        min_max_time_columns: tuple[str, str],
        time_column_name: str,
):
    data = _get_relative_periods_dummies(
        data=data,
        cohort_column=cohort_column,
        min_max_time_columns=min_max_time_columns,
        time_column_name=time_column_name
    )

    relative_periods = sorted(
        [item for item in data.columns if item.startswith('relative_period::')],
        key=lambda x: int(x.split('::')[-1])
    )

    if dosage_column:
        data = data.with_columns([
            pl.col(i).mul(pl.col(dosage_column)) for i in relative_periods
        ])

    return data


def _get_relative_periods_dummies(
        data: pl.DataFrame,
        cohort_column: str,
        min_max_time_columns: tuple[str, str],
        time_column_name: str,
):
    min_time_column: str = min_max_time_columns[0]
    max_time_column: str = min_max_time_columns[1]

    data = (
        data
        .with_columns(
            pl.int_ranges(
                pl.col(min_time_column), pl.col(max_time_column) + 1
            ).alias(time_column_name)
        )
        .select(pl.exclude([min_time_column, max_time_column]))
        .explode(time_column_name)
        .with_columns((pl.col(time_column_name) - pl.col(cohort_column)).alias("relative_period"))
        .to_dummies("relative_period", separator="::")
    )

    return data


def _restrict_start_end_and_bin_endpoints(
        data: pl.DataFrame,
        start: int | None,
        end: int | None,
        entity_column: str,
        time_column: str,
        bin_endpoints: Literal["start", "end", "both"],
):
    relative_periods = sorted(
        [item for item in data.columns if item.startswith('relative_period::')],
        key=lambda x: int(x.split('::')[-1])
    )

    # get the names of the columns before-start OR after-end
    new_relative_periods = _restrict_relative_periods(
        start=start, end=end, relative_periods=relative_periods
    )

    print('new_relative_periods', new_relative_periods)
    print('relative_periods', relative_periods)

    if len(new_relative_periods) != len(relative_periods):
        # drop the columns that are: before-start OR after-end

        print("dropping columns")
        data = data.drop(
            columns=[i for i in relative_periods if i not in new_relative_periods]
        )

    # only bins if requested, and renames the columns <>
    data = _bin_start_end(
        data=data,
        entity_column=entity_column,
        time_column=time_column,
        bin_endpoints=bin_endpoints,
        relative_periods=relative_periods,
        new_relative_periods=new_relative_periods
    )

    return data


def _bin_start_end(
        data: pl.DataFrame,
        entity_column: str,
        time_column: str,
        bin_endpoints: Literal["start", "end", "both"],
        relative_periods: list[str],
        new_relative_periods: dict,
):
    # ------------------------------------------------------------------

    print(bin_endpoints)
    if bin_endpoints in ["start", "both"]:
        print('In')
        start_column = list(new_relative_periods.keys())[0]

        if start_column != relative_periods[0]:
            print("BINNNG DATA")
            data = (
                data
                .sort([entity_column, time_column], descending=[True, True])
                .with_columns(pl.col(start_column).cumsum().over(entity_column))
            )

            data = data.rename({start_column: new_relative_periods[start_column]})

            print(data)

    # ------------------------------------------------------------------

    if bin_endpoints in ["end", "both"]:

        end_column = list(new_relative_periods.keys())[-1]

        if end_column != relative_periods[-1]:
            data = (
                data
                .sort([entity_column, time_column], descending=[True, False])
                .with_columns(pl.col(end_column).cumsum().over(entity_column))
            )

            data = data.rename({end_column: new_relative_periods[end_column]})

    return data


def _restrict_relative_periods(
        start: int | None,
        end: int | None,
        relative_periods: list[str]
) -> dict[str, str]:
    """
    Restrict and modify a list of relative periods based on given start and end values.

    Parameters
    ----------
    start : int
        Starting relative period. If None, starts from the first period.
    end : int
        Ending relative period. If None, ends at the last period.
    relative_periods : list[str]
        A list of relative periods as strings, e.g., ["relative_period::-1", "relative_period::0"].

    Returns
    -------
    dict[str, str]
        A dictionary where the keys are the original period strings from the list and the values
        are the modified period strings. The modification involves substituting the number
        and placing `≤` before the number for the start if it's greater than the minimum
        original start. Similarly, placing `≥` before the number for the end if it's less than
        the maximum original end.
    """

    # Sort the list based on the integer value after 'relative_period::'
    sorted_periods = sorted(relative_periods, key=lambda x: int(x.split('::')[-1]))

    if start is None:
        start_period = sorted_periods[0]
    else:
        if start > -1:
            raise ValueError("start should be <= -1")

        start_period = f'relative_period::{start}'

    if end is None:
        end_period = sorted_periods[-1]
    else:
        if end < 1:
            raise ValueError("start should be >= 1")

        end_period = f'relative_period::{end}'

    start_index = sorted_periods.index(start_period)
    end_index = sorted_periods.index(end_period) + 1  # Include the end_period

    restricted_list = sorted_periods[start_index:end_index]

    # Build the dictionary
    out = {}
    for period in restricted_list:
        value = period
        if period == start_period and period != sorted_periods[0]:
            value = f"relative_period::≤ {start}"

        elif period == end_period and period != sorted_periods[-1]:
            value = f"relative_period::≥ {end}"

        out[period] = value

    return out
