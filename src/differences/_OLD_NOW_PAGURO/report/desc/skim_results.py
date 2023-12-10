from __future__ import annotations

import polars as pl

from differences.report._switch_data import _DataInfoSwitch


class SkimResults(_DataInfoSwitch):
    """

    Group
    -----
        report
    """
    def __init__(self, data_dict: dict[str, pl.DataFrame]):
        super().__init__(data_dict=data_dict)
