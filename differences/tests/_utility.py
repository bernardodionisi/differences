from __future__ import annotations

import pickle
from itertools import product
from os.path import join
from typing import NamedTuple

from pandas import DataFrame

from differences.datasets import load_data

py_test_data = {
    "mpdta": load_data.mpdta(),
    "rc_multi_period": load_data.rc_multi_period(),
}


def flatten_res(df):
    df.columns = [c[2] for c in list(df)]
    return df.reset_index()


class RATTgtArgs(NamedTuple):
    aggregation_type = ["simple", "dynamic", "group", "calendar"]

    python_agg_types = {
        "group": "cohort",
        "dynamic": "event",
        "calendar": "time",
        "simple": "simple",
    }
    r_est_method = {"dr-mle": "dr", "reg": "reg", "std_ipw-mle": "ipw"}

    dataset_names = ["mpdta", "rc_multi_period"]

    varying_args = list(
        product(
            ["mpdta", "rc_multi_period"],
            [0, 1, 2],  # anticipation
            ["varying", "universal"],  # base_period
            ["nevertreated", "notyettreated"],  # control_group
            ["dr-mle", "std_ipw-mle", "reg"],  # est method
            [None, "random_weights"],  # weights
        )
    )


def read_r_did_results():
    with open(
        f"{join(load_data._get_path(__file__), 'r_did_results')}", "rb"
    ) as handle:
        res = pickle.load(handle)
    return res


r_did_results = read_r_did_results()


def get_r_did_result(
    data_name: str,  # 'mpdta', 'rc_multi_period'
    params_tuple: tuple,  # anticipation, base_period, control_group, est_method, weights
    result_type: str,  # 'att_gt', 'aggregate', 'overall'
    agg_type: str = None,  # 'simple', 'dynamic', 'group', 'calendar'
):
    res = r_did_results[data_name][params_tuple][result_type]

    if result_type in ["aggregate", "overall"]:
        if agg_type is None:
            raise ValueError(
                "need to provide an agg_type: 'simple', 'dynamic', 'group', 'calendar'"
            )
        res = res[agg_type]

    return DataFrame.from_dict(res)
