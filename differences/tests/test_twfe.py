from __future__ import annotations

from altair import HConcatChart
from pandas import DataFrame

from differences import TWFE, simulate_data

panel_data = simulate_data(nentity=20)  # generate data


def test_basic_twfe():
    twfe = TWFE(
        data=panel_data,
        cohort_name="cohort",
    )

    res = twfe.fit(
        formula="y",
        cluster_names="entity",
    )

    assert isinstance(res, DataFrame)

    assert len(res)

    assert isinstance(twfe.plot(), HConcatChart)
