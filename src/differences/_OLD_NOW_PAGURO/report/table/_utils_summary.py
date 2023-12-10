from __future__ import annotations

import polars as pl


def wide_table_header(table) -> pl.DataFrame:
    """

    Parameters
    ----------
    table: Table

    Returns
    -------

    """
    out: pl.DataFrame = (
        table
        ._data
        .select(table.group_columns)
        .unique()
        .cast(pl.Utf8)
        .fill_null("")
    )
    n_rows = out.shape[0]
    return out.transpose(
        include_header=True,
        column_names=[f"({i})" for i in range(n_rows)]
    )


def sort_dataframe_from_list(
        data: pl.DataFrame, columns: str | list[str], values: list[str]) -> pl.DataFrame:
    with pl.StringCache():
        _ = pl.Series(values).cast(pl.Categorical)
        return data.sort(pl.col(columns).cast(pl.Categorical).to_physical())


def to_wide_data_dict(
        table,
        in_parenthesis: str | list[str] = None
) -> dict[str, pl.DataFrame]:
    """

    Parameters
    ----------
    table: Table
    in_parenthesis

    Returns
    -------

    """
    data = table._data

    index_column, group_columns = table.id_column, table.group_columns

    value_columns = table.estimate_column
    if in_parenthesis is None:
        value_columns = [value_columns, in_parenthesis]

    main_data, header_data, row_sorter = [], [], []

    for i, (h, df) in enumerate(
            data.partition_by(group_columns, as_dict=True).items()
    ):
        column_name = f"({i})"

        # --------------------------------------------------------------

        _header = (
            pl.DataFrame({"idx": group_columns, column_name: h})
            .cast(pl.Utf8)
            .fill_null("")
        )

        header_data.append(_header)

        # --------------------------------------------------------------

        _temp = df.melt(
            id_vars=index_column,
            value_vars=value_columns,
            variable_name="VARIABLE",
            value_name=column_name
        )

        # maintain_order is set to True: data here should be small, come from Table
        for i in _temp.select(index_column).to_series().unique(maintain_order=True):
            if i not in row_sorter:
                row_sorter.append(i)

        main_data.append(_temp)

    # ------------------------------------------------------------------

    header_data = sort_dataframe_from_list(
        data=pl.concat(header_data, how="align"),
        columns="idx",
        values=group_columns
    )

    main_data = (
        sort_dataframe_from_list(
            data=pl.concat(main_data, how="align"),
            columns=[index_column, "VARIABLE"],
            values=row_sorter + value_columns
        )
    )

    main_data = main_data.drop(columns="VARIABLE")

    return {"header": header_data, "data": main_data}
