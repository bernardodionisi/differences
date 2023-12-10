import numpy as np
import warnings

from itertools import product
from tqdm import tqdm

from differences.datasets import load_data
from differences.models import ATTgt

from arpoon.rother._rdid import attgt as r_attgt

warnings.filterwarnings("ignore", module="arpoon")
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# ------------------------ python --------------------------------------

est_method, base_period, control_group, anticipation = ("reg", "universal", "never_treated", 0)

# for data_name, weights_column, x_formula in tqdm(data_params):
#     dataset = getattr(load_data, data_name)(weights_column=weights_column, x_formula=x_formula)

# for est_method, base_period, control_group, anticipation in params:

att_gt = ATTgt(
    data=dataset.data(),
    cohort_column=dataset.cohort_column,
    base_period=base_period,
    anticipation=anticipation
)

# python
py_result = att_gt.fit(
    formula=dataset.formula,
    control_group=control_group,
    est_method=est_method,
    weights_column=dataset.weights_column,
    progress_bar=False,
    # alpha=0.01
)

# R
r_did = r_attgt.ATTgt(
    data=dataset.data(fill_cohort_column=0, reset_index=True),
)

r_result = r_did.attgt(
    idname=dataset.entity_column,
    tname=dataset.time_column,
    gname=dataset.cohort_column,
    yname=dataset.y_column,
    xformla=dataset.x_formula,
    panel=dataset.is_panel,
    weightsname=dataset.weights_column,

    anticipation=anticipation,
    base_period=base_period,
    control_group=control_group.replace("_", ""),
    est_method={"reg": "reg", "dr-mle": "dr", "std_ipw-mle": "ipw"}.get(est_method),

    allow_unbalanced_panel=False,
    bstrap=False,
    # alp=0.01
)

# aggregation: simple

# R aggregation
r_did.aggte("simple")  # <--- not stored here
r_simple = r_did.aggte_overall()  # <--- simple is stored here

# python aggregation
py_simple = py_result.aggregate("simple", False)

# aggregation: event

# R
r_event = r_did.aggte("dynamic")
r_event_overall = r_did.aggte_overall()

# python
py_event = py_result.aggregate("event", False)
py_event_overall = py_result.aggregate("event", True)

# aggregation: cohort

# R
r_cohort = r_did.aggte("group")
r_cohort_overall = r_did.aggte_overall()

# python
py_cohort = py_result.aggregate("cohort", False)
py_cohort_overall = py_result.aggregate("cohort", True)

# aggregation: time

# R
r_time = r_did.aggte("calendar")
r_time_overall = r_did.aggte_overall()

# python
py_time = py_result.aggregate("time", False)
py_time_overall = py_result.aggregate("time", True)

# ----------------------------------------------------------------------

for r_res, py_res in [
    (r_result, py_result.to_pandas()),
    (r_simple, py_simple),
    (r_event, py_event),
    (r_event_overall, py_event_overall),
    (r_cohort, py_cohort),
    # (r_cohort_overall, py_cohort_overall),
    (r_time, py_time),
    (r_time_overall, py_time_overall),
]:
    assert np.allclose(
        r_res["att"].dropna().to_numpy(),
        py_res.no_index(with_title=False)["ATT"].dropna().to_numpy(),
    ), f"\nATT for\n\tR:\n{r_res}\n\tpython:\n{py_res}\n\nare not equal\n"

    assert np.allclose(
        r_res["se"].dropna().to_numpy(),
        py_res.no_index(with_title=False)["std_error"].dropna().to_numpy(),
    ), f"\nstd_error for\n\tR:\n{r_res}\n\tpython:\n{py_res}\n\nare not equal\n"
