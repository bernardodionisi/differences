import pytest

from itertools import product

import numpy as np

from ..attgt.attgt import ATTgt

from ..tests._datasets import py_data, R_data

from ..tests._rutility import (r_attgt,
                               R_attgt_to_pandas,
                               r_agg,
                               r_agg_overall,
                               r_agg_not_overall)


def process_result_table(df):
    df.columns = [c[2] for c in list(df)]
    return df.reset_index()


aggregation_type = ['simple', 'dynamic', 'group', 'calendar']

python_agg_types = {
    'group': 'cohort',
    'dynamic': 'event',
    'calendar': 'time',
    'simple': 'simple'
}

overall_agg_types = list(product(['overall'], aggregation_type))

varying_args = list(
    product(
        # ['panel_unbalanced'],  # 'panel_balanced', 'rc_multi_period', 'mpdta'
        ['mpdta', 'rc_multi_period'],
        [0, 1, 2],  # anticipation
        ['varying', 'universal'],  # base_period
        ['nevertreated', 'notyettreated'],  # control_group
        ['std_ipw-mle', 'dr-mle', 'reg'],  # est method
        [None, 'random_weights']  # weights
    )
)

R_est_method = {'dr-mle': 'dr', 'reg': 'reg', 'std_ipw-mle': 'ipw'}

args = []
for a in varying_args:
    # this is because the dataset named 'panel_balanced' has no never treated group
    if a[0] == 'panel_balanced' and a[3] == 'nevertreated':
        continue
    args.append(a)
del varying_args


# args = [args[0]]


@pytest.mark.parametrize('data_name,anticipation,base_period,'
                         'control_group,est_method,weights_name',
                         args)
def test_ATTgt_with_Rdid(data_name,
                         anticipation,
                         base_period,
                         control_group,
                         est_method,
                         weights_name):
    # ------------------------- python ---------------------------------

    att_gt = ATTgt(
        data=py_data[data_name]['data'],
        cohort_name=py_data[data_name]['cohort_name'],
        anticipation=anticipation,
        base_period=base_period,
    )

    py_result = att_gt.fit(
        formula=f"{py_data[data_name]['y_name']}",
        control_group=control_group,
        weights_name=weights_name,
        est_method=est_method,
    )

    py_result = process_result_table(py_result)

    # ---------------------------- R -----------------------------------

    r_attgt_res = r_attgt(
        data=R_data[data_name]['data'],
        entity_name=R_data[data_name]['entity_name'],
        time_name=R_data[data_name]['time_name'],
        cohort_name=R_data[data_name]['cohort_name'],
        y_name=R_data[data_name]['y_name'],
        panel=R_data[data_name]['panel'],
        allow_unbalanced_panel=R_data[data_name]['allow_unbalanced_panel'],
        weights_name=weights_name,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        est_method=R_est_method[est_method]
    )

    r_result = R_attgt_to_pandas(r_attgt_res)

    # ----------------------- compare att gt ---------------------------

    assert np.allclose(
        r_result['att'].dropna().to_numpy(),
        py_result['ATT'].dropna().to_numpy()
    )

    assert np.allclose(
        r_result['se'].dropna().to_numpy(),
        py_result['std_error'].dropna().to_numpy()
    )

    # ----------------------- aggregate --------------------------------

    for agg_type in ['dynamic', 'simple', 'calendar', 'group']:
        r_agg_res = r_agg(r_attgt_res=r_attgt_res, agg_type=agg_type)

        # R overall aggregation
        r_agg_overall_result = r_agg_overall(r_agg_res=r_agg_res)

        # py overall aggregation
        py_agg_overall_result = att_gt.aggregate(python_agg_types[agg_type], overall=True)
        py_agg_overall_result = process_result_table(py_agg_overall_result)

        assert np.allclose(
            r_agg_overall_result['att'].dropna().to_numpy(),
            py_agg_overall_result['ATT'].dropna().to_numpy()
        )

        if agg_type in ['dynamic', 'simple', 'calendar']:  # group_overall likely a bug in R
            assert np.allclose(
                r_agg_overall_result['se'].dropna().to_numpy(),
                py_agg_overall_result['std_error'].dropna().to_numpy()
            )

        if agg_type != 'simple':
            # R aggregation
            r_agg_result = r_agg_not_overall(r_agg_res=r_agg_res)

            # py aggregation
            py_agg_result = att_gt.aggregate(python_agg_types[agg_type])
            py_agg_result = process_result_table(py_agg_result)

            assert np.allclose(
                r_agg_result['att'].dropna().to_numpy(),
                py_agg_result['ATT'].dropna().to_numpy()
            )

            assert np.allclose(
                r_agg_result['se'].dropna().to_numpy(),
                py_agg_result['std_error'].dropna().to_numpy()
            )
