from itertools import product

from differences.datasets import load_data
from differences.models import ATTgt

dataset = load_data.mpdta(
    weights_column="random_weights_by_entity",
    x_formula="~ lpop"
)

# dataset = load_data.rcs_multi_period(
#     weights_column="random_weights",
#     x_formula="~ 1"
# )

data_panel_params = list(product(
    *[
        ["mpdta"],
        [None, "random_weights_varying"],  # weights_column
        ["~ 1", "~ lpop"]  # x_formula
    ])
)

data_panel_rc = list(product(
    *[
        ["rcs_multi_period"],
        [None, "random_weights_varying"],  # weights_column
        ["~ 1", "~ X"]  # x_formula
    ])
)

data_params = data_panel_params + data_panel_rc

params = list(product(
    *[
        ["dr-mle", "reg", "std_ipw-mle"],  # est_method
        ["universal", "varying"],  # base_period
        ["never_treated", "not_yet_treated"],  # control_group
        [0, 1, 2],  # anticipation
    ])
)

est_method, base_period, control_group, anticipation = ("reg", "universal", "never_treated", 0)

# ------------------------ python --------------------------------------


att_gt = ATTgt(
    data=dataset.data(),
    cohort_column=dataset.cohort_column,
    base_period=base_period,
    anticipation=anticipation
)

py_result = att_gt.fit(
    formula=dataset.formula,
    control_group=control_group,
    est_method=est_method,
    weights_column=dataset.weights_column,
    progress_bar=False,
    # alpha=0.01
)

# aggregation: simple
py_simple = py_result.aggregate("simple", False)

# aggregation: event
py_event = py_result.aggregate("event", False)
py_event_overall = py_result.aggregate("event", True)

# aggregation: cohort
py_cohort = py_result.aggregate("cohort", False)
py_cohort_overall = py_result.aggregate("cohort", True)

# aggregation: time
py_time = py_result.aggregate("time", False)
py_time_overall = py_result.aggregate("time", True)

# ----------------------------------------------------------------------

from differences.report.table import Table

table = Table.from_pandas(py_cohort, estimate_column="ATT")

table.data

py_result.to_pandas()
overall_table = Table.from_pandas(py_result.to_pandas())
overall_table = overall_table.concat_other([
    Table.from_pandas(py_event_overall),
    Table.from_pandas(py_cohort_overall),
    Table.from_pandas(py_time_overall),
])

overall_table = overall_table.str_replace(" Aggregation", "").str_replace("Simple",
                                                                                  "Overall")

import lets_plot as lp

overall_table.plot(
    coord_flip_kws={},
    theme_kws=dict(axis_text_y=lp.element_text(margin=lp.margin(t=10)),
                   axis_line_y=lp.element_line(size=1))
).show()
