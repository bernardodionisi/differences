from __future__ import annotations

import polars as pl
from polars import selectors as cs


def _data_to_utf8(data: pl.DataFrame, round: int = 2) -> pl.DataFrame:
    """Replace numeric values with strings"""

    def _expr_string_float_to_int(
            columns: list
    ) -> list[pl.Expr]:
        return [
            pl.when(pl.col(i).str.contains(r'^\d+\.0+$'))
            .then(
                pl.col(i).str.split(".")
                .list.first()
            )
            .otherwise(pl.col(i))
            for i in columns
        ]

    data = data.with_columns(cs.float().round(round))

    data = (
        data
        .cast(pl.Utf8)
        .fill_null("")
        .with_columns(_expr_string_float_to_int(data.columns))
    )
    return data
