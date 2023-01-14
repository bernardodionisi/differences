from __future__ import annotations

import pandas as pd

pd.set_option("display.max_rows", 6)

from differences import ATTgt, load_data
from differences.tests._utility import get_r_did_result

data_name = "rc_multi_period"

params_tuple = (0, "varying", "nevertreated", "dr-mle", "random_weights")
anticipation, base_period, control_group, est_method, weights_name = params_tuple

R_result = get_r_did_result(
    data_name=data_name,
    params_tuple=params_tuple,
    result_type="att_gt",
)

data_dict = getattr(load_data, data_name)()

att_gt = ATTgt(
    data=data_dict["data"],
    cohort_name=data_dict["cohort_name"],
    anticipation=anticipation,
    base_period=base_period,
)

att_gt.fit(
    formula=f"{data_dict['y_name']}",
    control_group=control_group,
    weights_name=weights_name,
    est_method=est_method,
)

agg_type = "time"
att_gt.aggregate(agg_type, overall=True)

get_r_did_result(
    data_name=data_name,
    params_tuple=params_tuple,
    result_type="overall",
    agg_type=agg_type,
)
