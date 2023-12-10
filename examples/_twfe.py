from differences.models import TWFE
from differences.datasets import load_data

dataset = load_data.mpdta()

twfe = TWFE(
    formula=f"{dataset.y_column} ~ - 1",
    data=dataset.data(to_polars=True),

    entity_column=dataset.entity_column,
    time_column=dataset.time_column,
    cohort_column=dataset.cohort_column
)

res = twfe.fit(vcov=dataset.entity_column, reference_period=0)

res.coefficient_table().insert_zeros(0)


import polars as pl

from differences._OLD_NOW_PAGURO.shared import get_relative_periods_data

data = pl.DataFrame({
    "entity": [1],
    "cohort": [2003],
    "dosage": [0.2],
    "_min_panel_data": [2000],
    "_max_panel_data": [2006]
})

get_relative_periods_data(
    data,
    entity_column="entity",
    cohort_column="cohort",
    # dosage_column="dosage",
    start=-2,
    end=3,
    bin_endpoints="both"
)

get_relative_periods_data(
    data,
    entity_column="entity",
    cohort_column="cohort",
    reference_period=-2,
    start=-2,
    end=3,
    bin_endpoints="both"
)

get_relative_periods_data(
    data,
    entity_column="entity",
    cohort_column="cohort",
    dosage_column="dosage",
    start=-2,
    end=3,
    bin_endpoints="both"
)
# ----------------------------------------------------------------------

data = pl.DataFrame({
    "entity": [1, 1, 1],
    "cohort": [2003, 2004, 2006],
    "dosage": [0.2, -0.1, 0.3],
    "_min_panel_data": [2000, 2000, 2000],
    "_max_panel_data": [2010, 2010, 2010]
})

get_relative_periods_data(
    data,
    entity_column="entity",
    cohort_column="cohort",
    dosage_column="dosage",
    start=-3,
    end=3,
    bin_endpoints="both"
)

get_relative_periods_data(
    data,
    entity_column="entity",
    cohort_column="cohort",
    # dosage_column="dosage",
    start=-3,
    end=3,
    bin_endpoints="both"
)

# ----------------------------------------------------------------------


data = pl.DataFrame({
    "entity": [1, 1, 2, 3, 4],
    "cohort": [1990, 1992, 2000, 2001, 2004],
    "dosage": [0.1, 0.2, 0.3, 0.4, 0.5],
    "_min_panel_data": [1987, 1987, 1995, 1998, 2000],
    "_max_panel_data": [2000, 2000, 2005, 2005, 2008]
})

get_relative_periods_data(
    data,
    entity_column="entity",
    cohort_column="cohort",
    dosage_column="dosage",
    start=-3,
    end=3,
    bin_endpoints="both"
)
