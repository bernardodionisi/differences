from __future__ import annotations
import polars as pl


# from ATTgt -> initialize a list of DataGT
# I need a formula method that takes the full data, returns the augmented model matrix
# after the model matrix is returned some checks need to be done:
# is_panel (not changed), is_balanced_panel (may have changed depending on 'nulls')

class _DataGT:
    def __init__(
            self,
            data: pl.DataFrame | pl.LazyFrame,  # group-time data
            time: int,
            cohort: int,
            dosage: str | int = None
    ):
        pass

    def _to_cross_section(self):
        pass
