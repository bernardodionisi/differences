"""
Add group lines to polars frame representation
"""


def get_first_char_by_line(s):
    return [i[0] for i in s.split("\n") if not i[0].isalpha()]


def find_change_indices(lst):
    if not lst:
        return []

    indices = []
    prev_elem = lst[0]

    for i, elem in enumerate(lst[1:], 1):  # Start with index 1
        if elem != prev_elem:
            indices.append(i)  # Append index where change occurred
            prev_elem = elem

    return indices


import polars as pl

pl.Config.set_tbl_cols(27)
columns = 'abcdefghilmnopqrstuvz'
polars_data = pl.DataFrame(dict(zip(columns, columns)))

find_change_indices(get_first_char_by_line(polars_data.__repr__()))

print(pl.from_repr(polars_data.__repr__()))


def _repr_describe_split_by(data: pl.DataFrame, by: str) -> str:
    data_repr = _repr_list(data=data, remove_shape=True, remove_dtypes=True)

    index_values = _repr_get_value_separator_index(data_repr=data_repr)

    pl_header = data_repr[:index_values + 1]
    pl_values = data_repr[index_values + 1:]

    group_separator = "|" + "─" * (len(data_repr[0]) - 2) + "|"

    for idx in (data.select(
            (pl.col(by).ne(pl.col(by).shift(1))).arg_true()).to_series()
    ):
        pl_values.insert(idx, group_separator)

    return "\n".join(pl_header + pl_values)


import polars as pl

pl.Config.set_tbl_cols(27)
columns = 'abcdefghilmnopqrstuvz'
polars_data = pl.DataFrame(dict(zip(columns, columns)))


def get_data_repr_group(data: pl.DataFrame, no_header: bool = True):
    no_header_config = {}
    if no_header:
        no_header_config = dict(tbl_hide_column_names=True)

    with pl.Config(
            tbl_hide_dataframe_shape=True,
            set_tbl_formatting="UTF8_BORDERS_ONLY",
            tbl_hide_column_data_types=True,
            tbl_hide_dtype_separator=True,
            **no_header_config
    ):
        data_repr = data.__repr__()

    return data_repr


df = pl.DataFrame(
    {
        "a": ["a", "b", "a", "b", "c"],
        "b": [1, 2, 1, 3, 3],
        "cdef": [5, 4, 3, 2, 1],
    }
)

out = []
for idx, (k, d) in enumerate(df.group_by("a")):
    if idx == 0:
        out.append(get_data_repr_group(df, False))
    else:
        out.append(get_data_repr_group(df))

    print(k, d)

print(out[1])
data
