def initialize_att_gt_dict(
        data: PandasDF,
        data_matrix: PandasDF,
        sample_split_column: Callable | str | dict | None,
        had_nulls_dropped: bool
) -> None:
    """

    used to be called '_create_result_dict'

    creates self._result_dict dictionary

    if sample_split_column:
      self._result_dict = {sample_name: {'sample_mask': np mask}
    else:
      self._result_dict = {'full_sample': {}}

    user should not _interact wth this dictionary
    """

    if isinstance(sample_split_column, str):

        # 'out' used to be named '_result_dict'
        out = insert_sample_mask_into_att_gt_dict(
            data=(
                data
                if not had_nulls_dropped
                else data[sample_split_column].loc[
                    lambda x: x.index.isin(data_matrix.index)
                ]
            ),
            sample_column=sample_split_column,
        )


    else:  # self._result_dict is None
        out = {"full_sample": {}}

    return out  # att_gt_dict



def insert_sample_mask_into_att_gt_dict(
        data: PandasDF,
        sample_column: str = None,
):
    """finds the masks for the data given split_sample_by

    returns a dict {'name of sample': data mask for sample}"""

    if sample_column is None:
        return None

    elif isinstance(sample_column, str):
        return {
            f"{sample_column}::{i}": {
                "sample_mask": np.array(data[sample_column] == i)
            }
            for i in data[sample_column].dropna().unique()  # for each value in the sample col
        }

    else:
        raise ValueError("invalid 'split_sample_by'")

