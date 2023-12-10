from __future__ import annotations

import polars as pl
from polars import ComputeError


def add_reference_coefficient(
        data: pl.DataFrame,
        id_column: str,
        estimate_column: str,
        group_columns: str | list[str] = None,
        reference_id: int | str | list[int] | list[str] = None,
) -> pl.DataFrame:
    if not group_columns:
        if len(data) != data[id_column].n_unique():
            raise ValueError(
                "'data' contains duplicate IDs, please provide 'group_column'"
            )

        out = _add_reference_coefficient_single(
            data=data,
            id_column=id_column,
            estimate_column=estimate_column,
            reference=reference_id
        )
    else:
        if isinstance(group_columns, str):
            group_columns = [group_columns]

        out = []
        for group, group_data in data.group_by(group_columns):
            group_data = _add_reference_coefficient_single(
                data=group_data,
                id_column=id_column,
                estimate_column=estimate_column,
                reference=reference_id
            )

            group_data = (
                group_data
                .with_columns([pl.col(col).fill_null(g) for col, g in zip(group_columns, group)])
            )

            out.append(group_data)

        out = pl.concat(out)

    return out


def _add_reference_coefficient_single(
        data: pl.DataFrame,
        id_column: str,
        estimate_column: str,
        reference: int | str | list[int] | list[str] = None,
) -> pl.DataFrame:
    reference = _identify_reference(
        data=data,
        id_column=id_column,
        reference=reference
    )

    data = _concat_data_with_reference(
        data=data,
        id_column=id_column,
        reference=reference,
        est_column=estimate_column,
    )

    return data


def _identify_reference(
        data: pl.DataFrame,
        id_column: str,
        reference: list[int] | list[str] | None,
) -> list:
    if reference is None:
        data = data.with_columns(pl.col(id_column).alias(f"temp_{id_column}"))

        # automatic filling of reference only if coefficient_column 'is' an integer

        if not (data[id_column].dtype in pl.INTEGER_DTYPES):

            try:  # try to replace all the strings among the coefficients and only keep the ints

                # todo: make sure there are no empty rows
                data = (
                    data
                    .with_columns(
                        pl.col(f"temp_{id_column}")
                        .str.replace_all("[^0-9-]", "")
                        .cast(pl.Int32)
                        .alias(f"temp_{id_column}")
                    )
                )

            except ComputeError:  # unable to cast to integer

                raise ValueError(
                    f"Missing 'reference'.\n"
                    f"When '{id_column}' does not contain integers "
                    f"you must provide a 'reference'"
                )

        reference = []
        for i in range(
                data[f"temp_{id_column}"].min(),
                data[f"temp_{id_column}"].max() + 1
        ):
            if i not in data[f"temp_{id_column}"].unique():
                reference.append(i)

    else:

        if not isinstance(reference, list):
            reference = [reference]

    return reference


def _concat_data_with_reference(
        data: pl.DataFrame,
        id_column: str,
        est_column: str,
        reference: list[int] | list[str],
):
    reference = pl.DataFrame({id_column: reference})

    is_integer = reference[id_column].is_integer()

    reference = (
        reference
        .with_columns(pl.col(id_column).cast(data[id_column].dtype))
    )

    data = (
        pl.concat([data, reference], how="diagonal")
        .with_columns(pl.col(est_column).fill_null(0))  # 0 for reference_id <----
    )

    if is_integer:
        data = _sort_as_numeric(
            data=data, id_column=id_column, by=[id_column]
        )

    return data


def _sort_as_numeric(
        data: pl.DataFrame,
        id_column: str,
        by: list[str] | None,
        **kwargs
) -> pl.DataFrame:
    if by is None:
        by = id_column

    try:
        data = (
            data
            .with_columns(pl.col(by).alias(f"temp_{by}"))
            .with_columns(
                pl.col(f"temp_{by}")
                .str.replace_all("[^0-9-]", "")
                .cast(pl.Int32)
                .alias(f"temp_{by}")
            )
            .sort(f"temp_{by}", **kwargs)
            .drop(f"temp_{by}")
        )

        return data

    except Exception as e:
        # if there are some value containing no numbers this won't work
        raise e


def identify_est_and_index_columns(
        data: pl.DataFrame,
        id_column: str | None,
        estimate_column: str | None,
) -> tuple[str, str]:
    if estimate_column is None:
        # use the first float as the 'est_column', if no 'est_column' was provided

        float_column = [k for k, v in data.schema.items() if v in pl.FLOAT_DTYPES]
        if float_column:
            estimate_column = float_column[0]

    # if can't automatically identify 'est_column' don't try identifying 'id_column'
    # since the est_column may be integers

    if not estimate_column:
        raise TypeError("missing a keyword-only argument: 'est_column'")

    if id_column is None:
        # look for 'id_column' among the int/str columns

        str_int_column = [
            k for k, v in data.schema.items()
            if (v is pl.Utf8) or (v in pl.INTEGER_DTYPES)
        ]

        if str_int_column:
            if len(str_int_column) == 1:
                id_column = str_int_column[0]
            else:
                # if there is more than one, take the closest to the left to the 'est_column'
                # if there is no closest to the left, take the closest to the right to 'est_column'
                id_column = find_id_column(
                    columns=data.columns,
                    est_column=estimate_column,
                    candidate_id_columns=str_int_column
                )

    if not id_column:
        raise TypeError("missing a keyword-only argument: 'id_column'")

    return id_column, estimate_column


def find_id_column(
        columns: list[str],
        est_column: str,
        candidate_id_columns: list[str]
) -> str | None:
    if est_column in columns:
        index = columns.index(est_column)

        # First, iterate over the items to the left of est_column
        for i in range(index - 1, -1, -1):  # Go from (index - 1) to 0, in reverse order
            if columns[i] in candidate_id_columns:
                return columns[i]

        # If no valid items found on the left, iterate over the items to the right
        for i in range(index + 1, len(columns)):
            if columns[i] in candidate_id_columns:
                return columns[i]


# ------------------------- from pandas --------------------------------

def from_pandas_remove_nested_index_from_dataframe(
        data,  # PandasDF
        with_title: bool = True  # highest column index used as title
) -> tuple:  # tuple[str | None, PandasDF] | PandasDF

    data = data.copy()

    cols = list(data)

    if isinstance(cols[0], tuple):
        title = cols[0][0]
        data.columns = [c[-1] for c in cols]
    else:
        title = None

    if data.index.names[0] is not None:
        data = data.reset_index()

    if with_title:
        return title, data

    return data


def camel_to_snake(s: str, separator: str = "_", capitalize: bool = False):
    import re

    # Handle sequences of capital letters as a single word
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1' + separator + r'\2', s)

    # Convert remaining camel case boundaries to the separator
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1' + separator + r'\2', s)

    if capitalize:
        # Capitalize the first letter of each word if required
        s = s.title()
    else:
        # Convert to lowercase
        s = s.lower()

    return s
