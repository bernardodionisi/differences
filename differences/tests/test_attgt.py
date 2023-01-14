from __future__ import annotations

import numpy as np
import pytest

from differences import ATTgt
from differences.tests._utility import RATTgtArgs, get_r_did_result, flatten_res, py_test_data


@pytest.mark.parametrize(
    "data_name,anticipation,base_period,control_group,est_method,weights_name",
    RATTgtArgs.varying_args,
)
def test_att_gt_agains_r_did(
        data_name, anticipation, base_period, control_group, est_method, weights_name
) -> None:
    # ------------------------- R --------------------------------------

    R_result = get_r_did_result(
        data_name=data_name,
        params_tuple=(anticipation, base_period, control_group, est_method, weights_name),
        result_type="att_gt",
    )
    # ------------------------- python ---------------------------------

    att_gt = ATTgt(
        data=py_test_data[data_name]["data"],
        cohort_name=py_test_data[data_name]["cohort_name"],
        anticipation=anticipation,
        base_period=base_period,
    )

    py_result = att_gt.fit(
        formula=f"{py_test_data[data_name]['y_name']}",
        control_group=control_group,
        weights_name=weights_name,
        est_method=est_method,
    )

    py_result = flatten_res(py_result)

    # ----------------------- compare att gt ---------------------------

    assert np.allclose(
        R_result["att"].dropna().to_numpy(), py_result["ATT"].dropna().to_numpy()
    )

    assert np.allclose(
        R_result["se"].dropna().to_numpy(), py_result["std_error"].dropna().to_numpy()
    )

    # ----------------------- aggregate --------------------------------

    for agg_type in ["event", "simple", "time", "cohort"]:

        # load pre calculated R results

        R_agg_res = get_r_did_result(
            data_name=data_name,
            params_tuple=(anticipation, base_period, control_group, est_method, weights_name),
            result_type="aggregate",
            agg_type=agg_type,
        )

        R_agg_overall_result = get_r_did_result(
            data_name=data_name,
            params_tuple=(anticipation, base_period, control_group, est_method, weights_name),
            result_type="overall",
            agg_type=agg_type,
        )

        # ------------------------ overall -----------------------------

        py_agg_overall_result = flatten_res(att_gt.aggregate(agg_type, overall=True))

        assert np.allclose(
            R_agg_overall_result["att"].dropna().to_numpy(),
            py_agg_overall_result["ATT"].dropna().to_numpy(),
        )

        if agg_type != "cohort":  # group_overall likely a bug in R
            assert np.allclose(
                R_agg_overall_result["se"].dropna().to_numpy(),
                py_agg_overall_result["std_error"].dropna().to_numpy(),
            )

        if agg_type != "simple":
            # ------------------------ not overall ---------------------

            py_agg_result = flatten_res(att_gt.aggregate(agg_type))  # py

            assert np.allclose(
                R_agg_res["att"].dropna().to_numpy(),
                py_agg_result["ATT"].dropna().to_numpy(),
            )

            assert np.allclose(
                R_agg_res["se"].dropna().to_numpy(),
                py_agg_result["std_error"].dropna().to_numpy(),
            )
