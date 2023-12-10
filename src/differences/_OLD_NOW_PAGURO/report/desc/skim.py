from __future__ import annotations

from typing import Literal, Sequence, Iterable, Mapping

import polars as pl

from polars import selectors as cs
from polars.utils.various import parse_percentiles


def describe_numeric(
        data: pl.LazyFrame | pl.DataFrame,
        *,
        by: str | Iterable = None,
        include_dtype: bool = False,
        exclude_columns: list[str] | str = None,
) -> pl.DataFrame:
    return _describe(
        data=data,
        by=by,
        include_dtype=include_dtype,
        exclude_columns=exclude_columns,
        which="numeric",
    )


def _describe(
        data: pl.LazyFrame | pl.DataFrame,
        *,
        by: str | Iterable,
        include_dtype: bool,
        which: Literal["numeric"],
        exclude_columns: list[str] | None,
) -> pl.DataFrame:
    """Describe function to be used across datatypes"""

    data = data.lazy()

    metrics, sel, columns = _get_metrics(
        data=data, by=by, which=which, exclude_columns=exclude_columns
    )

    data_schema = data.schema  # use in case one wants dtypes among descriptives

    if by is None:
        by = []
    elif isinstance(by, str):
        by = [by]

    if by:
        data = data.group_by(by).agg(sel)
    else:
        data = data.select(sel)

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    cols_dict = {
        c: [f"{i}:{c}" for i in metrics] for c in columns  # columns does not include by
    }

    n_by = len(by)

    # for each column, create a separate DataFrame (of descriptive stats)
    out = []
    for k, v in cols_dict.items():
        v = by + v  # groups at the end, for readability?

        out.append(
            data
            .select(v)
            .rename(dict(zip(v, [i.split(":")[0] for i in v])))  # rename the metrics columns
            .insert_at_idx(n_by, pl.Series([k] * data.shape[0]).alias("column"))
        )

    data = pl.concat(out, how="vertical_relaxed")
    if by:
        data = data.sort(by)

    if include_dtype:
        _insert_dtype(data_describe=data, data_schema=data_schema, position=n_by + 1)

    return data


# ------------------------ expressions ---------------------------------

def _expr_describe_numeric(
        selector: pl.selectors._selector_proxy_ | None,
        percentiles: Sequence[float] | float | None = (0.25, 0.5, 0.75),
) -> tuple:
    if selector is None:
        selector = cs.numeric()

    metrics = ["count", "null_count", "mean", "std", "min"]

    percentile_exprs = []
    for p in parse_percentiles(percentiles):
        p_name = f"{p:.0%}"

        percentile_exprs.append(selector.quantile(p).prefix(f"{p_name}:"))

        metrics.append(p_name)

    metrics.append("max")

    sel = [
        selector.count().prefix("count:"),
        selector.null_count().prefix("null_count:"),
        selector.mean().prefix("mean:"),
        selector.std().prefix("std:"),
        selector.min().prefix("min:"),
        *percentile_exprs,
        selector.max().prefix("max:"),
    ]

    return metrics, sel


# ----------------------------------------------------------------------

def _get_type_selector(
        which: Literal["numeric"] = "numeric"
):
    if which == "numeric":
        return cs.numeric()
    else:
        raise ValueError(f"Invalid which='{which}'")


def _get_col_selectors(
        selector: pl.selectors._selector_proxy_ | None,
        which: Literal["numeric"] = "numeric"
) -> tuple:
    if which == "numeric":
        metrics, sel = _expr_describe_numeric(selector)
        return metrics, sel
    else:
        raise ValueError(f"Invalid which='{which}'")


def _get_metrics(
        data: pl.DataFrame | pl.LazyFrame,
        by: str | list[str],
        which: Literal["numeric"],
        exclude_columns: list[str] | str | None
):
    selector = _get_type_selector(which=which)

    if by is not None:
        if not isinstance(by, str):
            by = "|".join(by)

        selector -= cs.matches(by)

    if exclude_columns is not None:
        if not isinstance(exclude_columns, str):
            exclude_columns = "|".join(exclude_columns)

        selector -= cs.matches(exclude_columns)

    metrics, sel = _get_col_selectors(selector, which=which)

    # ------------------------------------------------------------------

    columns = data.select(selector).columns

    return metrics, sel, columns


# ----------------------------------------------------------------------


def _insert_dtype(data_describe: pl.DataFrame, data_schema: Mapping, position: int):
    """Insert the column datatype as a column in te describe frame"""
    col_dtypes = (
        data_describe
        .select(
            pl.col("column").map_dict({
                k: str(v) for k, v in data_schema.items()
            }).alias("dtype")
        )
        .to_series()
    )

    data_describe = data_describe.insert_at_idx(position, col_dtypes)
    return data_describe


# ----------------------------------------------------------------------

def plot_unicode_distribution(data: pl.DataFrame, n_bins: int = None) -> dict:
    out = {}
    for i in data.columns:
        if not data[i].is_numeric():
            continue
        out[i] = _plot_unicode_distribution(arr=data[i], n_bins=n_bins)

        # todo: by
    return out


def _plot_unicode_distribution(arr, n_bins: int = None):
    """
    Generates a text-based distribution plot for a Polars Series.

    The idea for this feature has been sourced from a package called 'skimpy'
    """
    # todo: polars only implementation
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Please install numpy to use 'plot_unicode_distribution'")

    symbols = " ▁▂▃▄▅▆▇█"

    # Drop NA values
    if isinstance(arr, pl.Series):
        arr = arr.drop_nulls().to_numpy()  # pandas Series

    if n_bins is None:
        # Calculate the number of bins using the Freedman-Diaconis rule
        iqr = np.subtract(*np.percentile(arr, [75, 25]))
        if iqr == 0:
            n_bins = 1
        else:
            bin_width = 2 * (iqr / (len(arr) ** (1 / 3)))
            max_bins = len(symbols)
            n_bins = max(2, min(max_bins, int((arr.max() - arr.min()) / bin_width)))
    else:
        if n_bins > len(symbols):
            raise ValueError(f"max n_bins should be {len(symbols)}")

    # Compute the histogram using the calculated number of bins
    counts, edges = np.histogram(arr, bins=n_bins)

    # Map histogram bin counts to symbols
    max_count = max(counts) if counts.any() else 1
    normalized_counts = (counts / max_count * (len(symbols) - 1)).astype(int)
    distribution_plot = "│" + ''.join(symbols[count] for count in normalized_counts) + "│"

    return distribution_plot

