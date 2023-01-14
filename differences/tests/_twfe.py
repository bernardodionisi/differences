from pandas import DataFrame
from altair import HConcatChart
from differences import simulate_data, TWFE

panel_data = simulate_data(nentity=20)  # generate data

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
