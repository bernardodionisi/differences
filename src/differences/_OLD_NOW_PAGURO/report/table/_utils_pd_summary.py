try:
    import pandas as pd
except ImportError as e:
    raise ImportError("Please install pandas to access Summary's functionality")

import polars as pl


def frames_to_pandas(
        *frames: pl.DataFrame,
        header_names: bool = True,
        index_name: bool = False
) -> pd.DataFrame:
    """Use pandas in case we want to use multiindex columns"""

    if len(set([tuple(i.columns) for i in frames])) > 1:
        raise ValueError("The frames must have the same columns")

    fc = frames[0].columns[0]

    data: pd.DataFrame = pl.concat(frames[1:], how="diagonal").to_pandas().set_index(fc)

    if not index_name:
        data.index.name = None

    # first frame must be the header
    data.columns = _pl_rows_to_multiindex(data=frames[0], header_names=header_names)
    return data


def _pl_rows_to_multiindex(
        data: pl.DataFrame,
        header_names: bool = True
) -> pd.MultiIndex:
    """
    Create multiindex from polars dataframe
    Parameters
    ----------
    data
        header data
    header_names

    Returns
    -------

    """
    # data = (
    #     pl.DataFrame({
    #         "table": [1, 1, 2],
    #         "group": ["a", "b", "a"]}
    #     ).transpose(include_header=True,
    #                 column_names=['(1)', '(2)', '(3)']
    #                 )
    # )

    # df = pd.DataFrame({"coef": [1, 2, 3]}).T
    # df.columns = pl_rows_to_multiindex(data)

    names = data.select(data.columns[0]).to_series().to_list()

    data = data.select(data.columns[1:])
    header = data.rows()

    if header_names:
        # header.insert(0, tuple(temp.columns))
        header.append(tuple(data.columns))
        names.append(None)

    return pd.MultiIndex.from_tuples(list(zip(*header)), names=names)

# data = (
#     pl.DataFrame({
#         "table": [1, 1, 2],
#         "group": ["a", "b", "a"]}
#     ).transpose(include_header=True,
#                 column_names=['(1)', '(2)', '(3)']
#                 )
# )
#
# df = pl.from_pandas(pd.DataFrame({"coef": ["a", 1, 2, 3]}).T)
# df.columns = data.columns
#
# check = frames_to_pandas(data, df, index_name=True)
