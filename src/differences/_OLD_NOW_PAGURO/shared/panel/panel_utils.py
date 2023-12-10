from __future__ import annotations

import datetime
import warnings

import polars as pl

from differences._OLD_NOW_PAGURO.shared.dates import (
    date_as_int, int_as_date, date_as_int_single, PeriodFrequency
)


def raise_reserved_column_names_error(
        data: pl.DataFrame,
        columns: list
):
    """utility for data_es, silently may use PandasDF"""
    inv_names = [c for c in data.columns if c in columns]
    if inv_names:
        raise ReservedColumnNames(
            f"The data can't contain columns names: {inv_names}. These names are used internally."
        )


class ReservedColumnNames(Exception):
    pass


def raise_invalid_entity_time_data_error(
        data: pl.DataFrame,
        entity_column: str,
        time_column: str,
):
    if entity_column not in data.columns:
        raise InvalidEntityTime(f"'entity_column': '{entity_column}' not in data.")

    if time_column not in data.columns:
        raise InvalidEntityTime(f"'time_column': '{time_column}' not in data.")

    if not data[time_column].is_integer():
        raise InvalidEntityTime(
            f"The values in 'time_column': '{time_column}' must be integers.")

    if not (data[entity_column].is_integer() or (data[entity_column].dtype is pl.Utf8)):
        raise InvalidEntityTime(
            f"The values in 'entity_column': '{entity_column}' must be either "
            f"integers or strings.")

    # raise error if the data is not a panel or a repeated cross-section
    if data.select([entity_column, time_column]).n_unique() != len(data):
        raise InvalidData(
            "The data provided is not valid.\n"
            "The data should be unique at the entity-time level.\n"
            "\t- If the data is a panel it should contain entities "
            "repeated for multiple time periods.\n"
            "\t- If the data is a repeated cross-section, entities should be observed one time "
            "period each.\n"
        )


class InvalidEntityTime(Exception):
    pass


class InvalidData(Exception):
    pass


# ----------------------------------------------------------------------

def determine_if_panel_or_rcs(
        data: pl.DataFrame,
        entity_column: str,
        time_column: str,
) -> tuple[bool, bool, bool]:
    """utility for data_es"""

    is_rcs_data = is_repeated_cross_section(data=data, entity_column=entity_column)
    is_panel_data = not is_rcs_data

    # ------------------- is balanced panel? -------------------

    if is_panel_data:
        is_balanced_panel = is_panel_balanced(data, time_column=time_column)
    else:
        is_balanced_panel = None

    return is_panel_data, is_balanced_panel, is_rcs_data


def is_repeated_cross_section(data: pl.DataFrame, entity_column: str) -> bool:
    """utility for data_es"""

    return data.select(entity_column).n_unique() == len(data)


def is_panel_balanced(data: pl.DataFrame, time_column: str) -> bool:
    """utility for data_es
    checks if the panel is balanced: all units observed the same periods"""

    # is_entity_time_balanced
    time_counts = (
        data
        .select(time_column)
        .group_by(time_column)
        .count()
        .select("count")
        .to_series()
        .unique()
    )
    return len(time_counts) == 1


# ------------------------- to panel data ------------------------------

def build_panel_data(
        data: pl.LazyFrame | pl.DataFrame,

        entity_column: str,
        time_column: str,
        period_frequency: PeriodFrequency,

        *,

        start: int | str | datetime.date | None,
        end: int | str | datetime.date | None,
        relative_end: int | None,

        with_date: bool,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Construct a panel dataset from an unbalanced panel of entity - date

    Group
    -----
        general

    Parameters
    ----------
    data
    entity_column
    time_column
    period_frequency
    start
    end
    relative_end
    with_date
    """

    if relative_end is not None and end is None:
        # must use end when using relative_end
        warnings.warn(
            "Care must be used when setting 'relative_end' without setting 'end'\n"
            "You may end up with a panel spanning periods too far into the future; "
            "periods in which your variables may not be observed yet."
        )

    if period_frequency != "year" and (isinstance(start, int) or isinstance(end, int)):
        warnings.warn(
            "\nYou have used an integer for the start and/or end arguments.\n"
            "The integer needs to correspond to the integer conversion of the date.\n"
            "If you want the start/end to be from a specific date, but you do not want to "
            "convert that date to an integer, please use a string or a datetime.date, "
            "for example if you use end='2018-05' the time-series for each entity will "
            "end on May 2015."
        )

    # convert start, end to integers in case they are str | datetime.date
    start = date_as_int_single(date=start, period_frequency=period_frequency)
    end = date_as_int_single(date=end, period_frequency=period_frequency)

    if data.schema[time_column] is pl.Date:
        # convert time_column from pl.Date to integer
        data = data.with_columns([
            date_as_int(time_column, period_frequency=period_frequency)
        ])

    # --------------------- panel data ---------------------------------

    # select only entity and time columns (the variables will be joined after)
    data = (
        data
        .select([entity_column, time_column])
        .unique()
    )

    # convert data to panel
    data = data_to_panel(
        data=data,
        entity_column=entity_column,
        time_column=time_column,
        start=start,
        end=end,
        relative_end=relative_end,
    )

    data = data.drop_nulls()  # nulls when first year of entity > end date

    if relative_end is not None and end is not None:
        # if the relative end overshoots: goes up to years in which we don't have data
        data = data.filter(pl.col(time_column) <= end)

    if with_date:  # add a pl.Date column corresponding to the chosen period frequency
        data = (
            data
            .with_columns([
                int_as_date(time_column, period_frequency=period_frequency)
                .alias(f"{time_column}_as_date")
            ])
        )

    return data


def data_to_panel(
        data: pl.LazyFrame,
        entity_column: str,
        time_column: str,
        start: int | None,
        end: int | None,
        relative_end: int | None
) -> pl.LazyFrame | pl.DataFrame:
    """
    Converts the given data to a panel data format.

    Parameters
    ----------
    data : pl.DataFrame
        The data to be converted.
    entity_column : str
        The name of the column that contains the entities.
    time_column : str
        The name of the column that contains the time periods.
    start : int or None
        The start of the time period.
    end : int or None
        The end of the time period.
    relative_end : int or None
        The relative end of the time period.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        The converted data in panel format.

    Notes
    -----
    1. The function first collapses the data to find the minimum and
        maximum dates for each entity by calling `entity_time_min_max`.
    2. If no `start` date is provided, the function uses the first date
        in `time_column` as the `start` date.
    3. If neither an `end` date nor a `relative_end` date is provided,
        the function uses the last date in the data as the `end` date.
    4. The function then generates a range of date integers using the
        `generate_date_int_range` function. This function returns a range
        of integers that correspond to the time periods from the `start`
        to the `end` (or `relative_end` if specified).
    5. The `date_ranges` are then added as a new column to the DataFrame
        using the `with_columns` method. The DataFrame is then exploded
        using the `explode` method, which expands lists in rows to create
        new rows, one for each element in the list.
    6. Finally, the function selects the necessary columns
        (`entity_column` and `date`) and renames the `date` column
        to `time_column` to produce the final panel data.

    In this format, each row in the DataFrame corresponds to a
    particular entity at a specific time period. This kind of data
    structure is often used in econometrics and other fields that
    analyze temporal data.
    """
    # collapse data to have min-max dates for each entity
    data = entity_time_min_max(
        data=data,
        entity_column=entity_column,
        time_column=time_column
    )

    # start from the first date in time_column if no start date is provided

    if start is None:
        start = time_column + '_min'

    # if no end or relative and date, then use the last date in the data
    if end is None and relative_end is None:
        end = time_column + '_max'

    date_ranges = generate_date_int_range(start=start, end=end, relative_end=relative_end)

    data = (
        data
        .with_columns(date=date_ranges)
        .explode('date')
        .select([entity_column, 'date'])
        .rename({'date': time_column})
    )

    return data


def entity_time_min_max(
        data: pl.LazyFrame | pl.DataFrame,
        entity_column: str,
        time_column: str
) -> pl.LazyFrame | pl.DataFrame:
    """
    Generates a DataFrame with minimum and maximum time for each entity.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        The input data to perform the operation on. It can be either a
        Polars DataFrame or a LazyFrame.
    entity_column : str
        The name of the entity column.
    time_column : str
        The name of the time column.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with each entity alongside its minimum and maximum time.
    """
    data = (
        data
        .drop_nulls([entity_column])
        .group_by([entity_column])
        .agg([
            pl.col(time_column).min().suffix('_min'),
            pl.col(time_column).max().suffix('_max'),
        ])

    )
    return data


def generate_date_int_range(
        start: str | int,  # either integer or column name
        end: str | int | None,
        relative_end: int = None
) -> pl.Expr | list:
    """
    Generates a range of integers representing dates.

    Parameters
    ----------
    start : str or int
        The starting date as an integer or a column name.
    end : str or int
        The ending date as an integer or a column name.
    relative_end : int, optional
        The number of periods relative to the starting dates.

    Returns
    -------
    date_range : pl.Expr or list
        A list or a Polars expression representing the range of dates.

    Raises
    ------
    ValueError
        If 'end' is not an integer or a column name, or if 'start' and
        'end' are neither str nor int.
    """

    # if relative_end is not None:
    #     end = None  # end should always be passed

    if isinstance(start, str):
        # column with starting dates
        start = pl.col(start)

    elif not isinstance(start, int):
        raise ValueError("'start' must be either an integer or a column name")

    if end is None:
        if relative_end is None:
            # this case is handled before reaching this function
            raise ValueError("Something went wrong. 'end' and 'relative_end' can't both be None")
        end = start + relative_end  # is start is str, it's column + relative_time

    elif isinstance(end, str):
        end = pl.col(end)

    elif isinstance(end, int):
        # if an end and a relative_end are provided, use the relative end
        # (data will then be filtered to make sure all time observations are < end)
        if relative_end is not None:
            end = start + relative_end  # is start is str, it's column + relative_time
    else:
        raise ValueError("'end' must be either an integer, a column name or None")

    if isinstance(start, int) and isinstance(end, int):
        if end < start:
            raise ValueError("start should be <= end")

    int_date_range = pl.int_ranges(start, end + 1)

    return int_date_range


# --------------------------- some descriptives ------------------------

def percentage_nulls(data: pl.DataFrame):
    return {
        k: round(v[0] / len(data), 3)
        for k, v in data.null_count().to_dict().items()
        if v[0] > 0
    }


def get_time_varying_columns(
        data: pl.DataFrame | pl.LazyFrame,
        entity_column: str,
        time_column: str,
        column: str | list[str] = None
):
    """Find which columns vary within entity across time"""
    # time varying dict
    if isinstance(column, str):
        column = [column]
    elif column is None:
        column = pl.exclude(time_column)

    data = (
        data
        .select(column)
        .group_by(entity_column)
        .n_unique()
        .select(pl.exclude(entity_column))
        .max()
    )

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    # a dictionary with each column and True (if time varying) False (if time invariant)
    return {k: v[0] > 1 for k, v in data.to_dict().items()}


def null_count_by_column(
        data: pl.DataFrame | pl.LazyFrame,  # panel_data
        by: str | bool | None,
        columns: str | list[str] | None,
        share: bool,
        time_column: str = None
):
    # todo: do not show columns without nulls?

    if isinstance(data, pl.DataFrame):
        data = data.lazy()

    if columns is None:
        nulls_in_columns = pl.col("*")
    else:
        nulls_in_columns = pl.col(columns)

    agg = (
        (nulls_in_columns.null_count() / nulls_in_columns.count()) if share
        else nulls_in_columns.null_count()
    )
    if isinstance(by, bool):
        if not by:
            return data.select(agg)

    print(time_column)
    if time_column is None and by is None:
        raise ValueError("Missing value for 'by' argument")

    elif by is None:
        # default by is 'time_column'
        by = time_column

    data = data.group_by(by).agg(agg)

    return data.sort(by).collect()


# ----------------------------------------------------------------------

def OLD_extract_entity_time(
        data: pl.DataFrame,  # | PandasDF
        entity_column: str = None,
        time_column: str = None,
) -> tuple[pl.DataFrame, str, str]:
    """utility for data_es, silently may use PandasDF"""

    # todo: move pandas related stuff somewhere else, like before the __init__ of DataES

    if entity_column is None or time_column is None:

        # if isinstance(data, PandasDF): # not importing pandas, may error if something
        if not isinstance(data, pl.DataFrame):  # instead of referencing PandasDF directly

            # unresolved references are ok, as long as PandasDF is passed
            # and not a third thing
            if data.index.nlevels != 2:
                raise ValueError(
                    "Either:\n"
                    "\t- provide 'entity_column' and 'time_column'"
                    "\t- or data must be a MultiIndex PandasDF where "
                    "the first index level contains the identifier for the entities "
                    "and the second index the identifiers for calendar time."
                )

            # if a multiindex dataframe use the indices for entity and time
            entity_column, time_column = data.index.names

            data = pl.from_pandas(data.reset_index())

        elif isinstance(data, pl.DataFrame):
            raise ValueError("Missing arguments for: 'entity_column' and 'time_column'")

    else:
        # if isinstance(data, PandasDF):
        if isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data)

    return data, entity_column, time_column
