import polars as pl
from differences.datasets import load_data
from differences._OLD_NOW_PAGURO.shared.event.data_es import DataES

dataset = load_data.mpdta(
    weights_column="random_weights_by_entity",
    x_formula="~ lpop"
)

data = dataset.data(to_polars=True)

data_es = DataES(
    data=data,
    entity_column=dataset.entity_column,
    time_column=dataset.time_column,
    cohort_column=dataset.cohort_column,
    event_data=None,
    dosage_column=None,
)

data_es._info

data_es.format_info_repr(n_columns=4)

print(boxed_str("a"))

with pl.Config(tbl_rows=1):
    print(data_es)

data_es._is_balanced_panel

group_times = data_es._filter_group_times("varying")

data._filter_by(group_times[-1]['expr']['never_treated'])
data._filter_by(group_times[-1]['expr']['not_yet_treated'])
