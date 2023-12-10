from __future__ import annotations

from typing import Literal
import linearmodels
import polars as pl


# the representation of this function is like the compare method from linearmodels
# https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.results.compare.html#linearmodels.panel.results.compare
# https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.results.PanelModelComparison.html#linearmodels.panel.results.PanelModelComparison
# https://bashtage.github.io/linearmodels/panel/examples/examples.html#Comparing-models


class PanelModels:
    """
    A factory class for linearmodels' panel-models

    Group
    -----
        factory
    """

    def __init__(
            self,
            data: pl.DataFrame,
            entity_column: str,
            time_column: str
    ):
        self.data = data
        self.entity_column = entity_column
        self.time_column = time_column

        self._results = []

    def ols(
            self,

            return_type: Literal["self", "model", "result"] = "self"
    ) -> linearmodels.panel.results.PanelResults:
        return self

    def to_table(self):
        pass

    def _compare(self):
        pass
