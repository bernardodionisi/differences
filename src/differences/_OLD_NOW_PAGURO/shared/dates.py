from __future__ import annotations

from typing import Literal

import polars as pl
import datetime

__all__ = [
    "PeriodFrequency",
    "date_to_int_column",
    "int_to_date_column",
    "date_as_int",
    "int_as_date",
    "date_as_int_single"
]

PeriodFrequency = Literal["year", "semester", "quarter", "month", "week", "day"]


def date_to_int_column(
        data: pl.DataFrame | pl.LazyFrame,
        time_column: str,
        period_frequency: PeriodFrequency
) -> pl.DataFrame | pl.LazyFrame:
    return data.with_columns(date_as_int(time_column, period_frequency=period_frequency))


def int_to_date_column(
        data: pl.DataFrame | pl.LazyFrame,
        time_column: str,
        period_frequency: PeriodFrequency
) -> pl.DataFrame | pl.LazyFrame:
    return data.with_columns(int_as_date(time_column, period_frequency=period_frequency))


def date_as_int(
        date: str | pl.Expr | datetime.date,
        period_frequency: PeriodFrequency  # or Callable?
) -> pl.Expr | int:
    """
    Convert date to integer.

    Group
    -----
        general

    Parameters
    ----------
    date
    period_frequency

    Returns
    -------
    pl.Expr | int

    """
    if isinstance(date, str):
        date = pl.col(date)

    func_dict = {
        "year": year_as_int,
        "semester": semester_as_int,
        "quarter": quarter_as_int,
        "month": month_as_int,
        "week": week_as_int,
        "day": day_as_int,
    }

    func = func_dict.get(period_frequency)
    return func(date)


def int_as_date(
        date: str | pl.Expr | int,
        period_frequency: PeriodFrequency  # or Callable?
) -> pl.Expr | datetime.date:
    """
    Convert integer to date.

    Group
    -----
        general

    Parameters
    ----------
    date
    period_frequency

    Returns
    -------
    pl.Expr | datetime.date

    """

    if isinstance(date, str):
        date = pl.col(date)

    func_dict = {
        "year": int_as_year,
        "semester": int_as_semester,
        "quarter": int_as_quarter,
        "month": int_as_month,
        "week": int_as_week,
        "day": int_as_day,
    }

    func = func_dict.get(period_frequency)
    return func(date)


def date_as_int_single(
        date: int | str | datetime.date,
        period_frequency: PeriodFrequency
) -> int:
    """convert to integer"""

    # if an integer is passed the same integer is returned (no conversion happens)

    # this function just adds the help of using string formatted dates and converts them to dates
    # using string_to_date, otherwise calling date_as_int with a string, the string is
    # supposed to be the name of a column

    if isinstance(date, str):
        date = string_to_date(date)  # converted to datetime.date

    if isinstance(date, datetime.date):
        date = date_as_int(date, period_frequency=period_frequency)

    # if it is an integer, the integer will be returned, may not be what the user intended
    return date


# ----------------------------------------------------------------------


# year

def year_as_int(date: pl.Expr | datetime.date) -> pl.Expr | int:
    if isinstance(date, datetime.date):
        return date.year
    return date.dt.year()


def int_as_year(date: pl.Expr | int) -> pl.Expr | datetime.date:
    if isinstance(date, int):
        return datetime.date(date, 1, 1)
    return pl.date(date, 1, 1)


# semester

def semester_as_int(date: pl.Expr | datetime.date) -> pl.Expr | int:
    if isinstance(date, datetime.date):
        return date.year * 2 + (date.month - 1) // 6

    return date.dt.year() * 2 + (date.dt.month() - 1) // 6


def int_as_semester(date: pl.Expr | int) -> pl.Expr | datetime.date:
    date_tu = date // 2, (date % 2) * 6 + 1, 1
    return datetime.date(*date_tu) if isinstance(date, int) else pl.date(*date_tu)


# quarter

def quarter_as_int(date: pl.Expr | datetime.date) -> pl.Expr | int:
    if isinstance(date, datetime.date):
        return date.year * 4 + (date.month - 1) // 3

    return date.dt.year() * 4 + (date.dt.quarter() - 1)


def int_as_quarter(date: pl.Expr | int) -> pl.Expr | datetime.date:
    date_tu = date // 4, (date % 4) * 3 + 1, 1
    return datetime.date(*date_tu) if isinstance(date, int) else pl.date(*date_tu)


# month

def month_as_int(date: pl.Expr | datetime.date) -> pl.Expr | int:
    if isinstance(date, datetime.date):
        return date.year * 12 + (date.month - 1)

    return date.dt.year() * 12 + (date.dt.month() - 1)


def int_as_month(date: pl.Expr | int) -> pl.Expr | datetime.date:
    date_tu = date // 12, (date % 12) + 1, 1
    return datetime.date(*date_tu) if isinstance(date, int) else pl.date(*date_tu)


# week

# using UNIX epoch: January 1, 1970, at 00:00:00

assert pl.DataFrame(
    {"unix_date": [datetime.date(1970, 1, 1)]}
).with_columns(pl.col("unix_date").to_physical()).row(0)[0] == 0


def week_as_int(date: pl.Expr | datetime.date) -> pl.Expr | int:
    if isinstance(date, datetime.date):
        return (date - datetime.date(1970, 1, 1)).days // 7

    # if pl.Expr
    # return (date - datetime.date(1970, 1, 1)).dt.days() // 7
    return date.to_physical() // 7


def int_as_week(date: pl.Expr | int) -> pl.Expr | datetime.date:
    if isinstance(date, int):
        # same base 1, 1, 1 as week_as_int
        return datetime.date(1970, 1, 1) + datetime.timedelta(weeks=date)

    raise ValueError("Issue with duration? https://github.com/pola-rs/polars/issues/10868")
    # return pl.date(1970, 1, 1) + pl.duration(weeks=date)


# day

def day_as_int(date: pl.Expr | datetime.date) -> pl.Expr | int:
    if isinstance(date, datetime.date):
        return (date - datetime.date(1970, 1, 1)).days

    # if pl.Expr
    # return (date - datetime.date(1970, 1, 1)).dt.days()
    return date.to_physical()


def int_as_day(date: pl.Expr | int) -> pl.Expr | datetime.date:
    if isinstance(date, int):
        return datetime.date(1970, 1, 1) + datetime.timedelta(days=date)

    # "Issue with duration? https://github.com/pola-rs/polars/issues/10868"
    # return pl.date(1970, 1, 1) + pl.duration(days=date)
    return date.cast(pl.Date)


# ----------------------------------------------------------------------

def string_to_date(date: str) -> datetime.date:
    formats = ["%Y", "%Y-%m", "%Y-%m-%d", "%Y/%m", "%Y/%m/%d"]  # acceptable formats

    for date_format in formats:
        try:
            # todo: decide end or start of the string?
            return datetime.datetime.strptime(date, date_format).date()
        except ValueError:
            pass  # format didn't work, try next one

    raise ValueError(f"Input {date} does not match any known format: {formats}")
