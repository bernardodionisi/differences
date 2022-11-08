import numpy as np
from numpy import ndarray

import warnings

from ..did import att_if
from ..did.utility import wols_out_reg
from ..did.pscore_cal import pscore_ipt, pscore_mle


# ------------------------- improved -----------------------------------

def drdid_panel_improved(endog: ndarray,
                         exog: ndarray,
                         treated: ndarray,
                         weights: ndarray,
                         **post
                         ):
    pscore_fit, *_ = pscore_ipt(treated=treated,
                                exog=exog,
                                weights=weights)

    # compute the outcome regression for the control group using wols
    out_delta = wols_out_reg(endog=endog,
                             exog=exog,
                             weights=weights,
                             mask=treated == 0,  # control group
                             pscore_fit=pscore_fit)

    att, influence_func = att_if.dr_improved_panel_att_if(
        endog=endog[:, None],
        treated=treated[:, None],
        weights=weights[:, None],
        out_delta=out_delta[:, None],
        ps_fit=pscore_fit[:, None],
    )

    return {'att': att, 'influence_func': influence_func}


def drdid_rc_improved(endog: ndarray,
                      exog: ndarray,
                      treated: ndarray,
                      weights: ndarray,
                      post: ndarray
                      ):
    pscore_fit, *_ = pscore_ipt(treated=treated,
                                exog=exog,
                                weights=weights,
                                tol=None)

    out_delta = []
    for idx, _mask in enumerate([(treated == 0) & (post == 0),
                                 (treated == 0) & (post == 1),
                                 (treated == 1) & (post == 0),
                                 (treated == 1) & (post == 1)]):
        out_delta.append(
            wols_out_reg(endog=endog,
                         exog=exog,
                         weights=weights,
                         mask=_mask,
                         pscore_fit=pscore_fit if idx <= 1 else None
                         )
        )

    out_delta = np.stack(out_delta, axis=1)

    att, influence_func = att_if.dr_improved_rc_att_if(
        endog=endog[:, None],
        treated=treated[:, None],
        weights=weights[:, None],
        out_delta=out_delta,
        ps_fit=pscore_fit[:, None],
        post=post[:, None]
    )

    return {'att': att, 'influence_func': influence_func}


# --------------------------- traditional ------------------------------


def drdid_panel_traditional(endog: ndarray,
                            exog: ndarray,
                            treated: ndarray,
                            weights: ndarray,
                            **post
                            ):
    ps_fit, ps_cov = pscore_mle(treated=treated,
                                exog=exog,
                                weights=weights
                                )

    out_delta = wols_out_reg(endog=endog,
                             exog=exog,
                             weights=weights,
                             mask=treated == 0
                             )

    att, influence_func = att_if.dr_traditional_panel_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        out_delta=out_delta[:, None],
        ps_fit=ps_fit[:, None],
        pscore_cov_params=ps_cov,
    )

    return {'att': att, 'influence_func': influence_func}


def drdid_rc_traditional(endog: ndarray,
                         exog: ndarray,
                         treated: ndarray,
                         weights: ndarray,
                         post: ndarray
                         ):
    ps_fit, ps_cov = pscore_mle(treated=treated,
                                exog=exog,
                                weights=weights
                                )

    out_delta = []
    for _mask in [(treated == 0) & (post == 0),
                  (treated == 0) & (post == 1),
                  (treated == 1) & (post == 0),
                  (treated == 1) & (post == 1)]:
        out_delta.append(
            wols_out_reg(endog=endog,
                         exog=exog,
                         weights=weights,
                         mask=_mask
                         )
        )

    out_delta = np.stack(out_delta, axis=1)

    att, influence_func = att_if.dr_traditional_rc_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        out_delta=out_delta,
        ps_fit=ps_fit[:, None],
        pscore_cov_params=ps_cov,
        post=post[:, None]
    )

    return {'att': att, 'influence_func': influence_func}


# ------------------------- regression ---------------------------------


def reg_did_panel(endog: ndarray,
                  exog: ndarray,
                  treated: ndarray,
                  weights: ndarray,
                  **post
                  ):
    out_delta = wols_out_reg(endog=endog,
                             exog=exog,
                             weights=weights,
                             mask=treated == 0,  # control group
                             )

    att, influence_func = att_if.reg_panel_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        out_delta=out_delta[:, None]
    )

    return {'att': att, 'influence_func': influence_func}


def reg_did_rc(endog: ndarray,
               exog: ndarray,
               treated: ndarray,
               weights: ndarray,
               post: ndarray
               ):
    out_delta = []
    for _mask in [(treated == 0) & (post == 0),
                  (treated == 0) & (post == 1)]:
        out_delta.append(
            wols_out_reg(endog=endog,
                         exog=exog,
                         weights=weights,
                         mask=_mask
                         )
        )

    out_delta = np.stack(out_delta, axis=1)

    att, influence_func = att_if.reg_rc_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        out_delta=out_delta,
        post=post[:, None]
    )

    return {'att': att, 'influence_func': influence_func}


# ------------------------------ ipw -----------------------------------

def ipw_did_panel(endog: ndarray,
                  exog: ndarray,
                  treated: ndarray,
                  weights: ndarray,
                  **post
                  ):
    ps_fit, ps_cov = pscore_mle(treated=treated,
                                exog=exog,
                                weights=weights
                                )

    att, influence_func = att_if.ipw_panel_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        ps_fit=ps_fit[:, None],
        pscore_cov_params=ps_cov,
    )
    return {'att': att, 'influence_func': influence_func}


def ipw_did_rc(endog: ndarray,
               exog: ndarray,
               treated: ndarray,
               weights: ndarray,
               post: ndarray
               ):
    ps_fit, ps_cov = pscore_mle(treated=treated,
                                exog=exog,
                                weights=weights
                                )

    att, influence_func = att_if.ipw_rc_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        ps_fit=ps_fit[:, None],
        pscore_cov_params=ps_cov,
        post=post[:, None]
    )

    return {'att': att, 'influence_func': influence_func}


def std_ipw_did_panel(endog: ndarray,
                      exog: ndarray,
                      treated: ndarray,
                      weights: ndarray,
                      **post
                      ):
    ps_fit, ps_cov = pscore_mle(treated=treated,
                                exog=exog,
                                weights=weights
                                )

    att, influence_func = att_if.std_ipw_panel_att_if(
        endog=endog[:, None],
        exog=exog,
        treated=treated[:, None],
        weights=weights[:, None],
        ps_fit=ps_fit[:, None],
        pscore_cov_params=ps_cov,
    )
    return {'att': att, 'influence_func': influence_func}


def std_ipw_did_rc(endog: ndarray,
                   exog: ndarray,
                   treated: ndarray,
                   weights: ndarray,
                   post: ndarray
                   ):
    ps_fit, ps_cov = pscore_mle(treated=treated,
                                exog=exog,
                                weights=weights
                                )

    # todo: look into the warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        att, influence_func = att_if.std_ipw_rc_att_if(
            endog=endog[:, None],
            exog=exog,
            treated=treated[:, None],
            weights=weights[:, None],
            ps_fit=ps_fit[:, None],
            pscore_cov_params=ps_cov,
            post=post[:, None]
        )

    return {'att': att, 'influence_func': influence_func}


# --------------------------- all --------------------------------------

did_cal_funcs = {
    ('dr', 'ipt', 'panel'): drdid_panel_improved,
    ('dr', 'ipt', 'rc'): drdid_rc_improved,

    ('dr', 'mle', 'panel'): drdid_panel_traditional,
    ('dr', 'mle', 'rc'): drdid_rc_traditional,

    ('reg', None, 'panel'): reg_did_panel,
    ('reg', None, 'rc'): reg_did_rc,

    # ('ipw', 'mle', 'panel'): did_cal.ipw_did_panel,
    # ('ipw', 'mle', 'rc'): did_cal.ipw_did_rc,

    ('std_ipw', 'mle', 'panel'): std_ipw_did_panel,
    ('std_ipw', 'mle', 'rc'): std_ipw_did_rc
}


def get_method_tuple(estimator: str,
                     pscore_est_method: str,
                     panel_type: str):
    """
    helper to make sure estimator, pscore_est_method, panel_type can be used together

    Parameters
    ----------
    estimator: str
        either 'reg', 'dr' or 'ipw'
    pscore_est_method: str
        either 'mle' or 'ipt'
    panel_type
        either 'panel' or 'rc'

    Returns
    -------
    tuple
        estimator, pscore_est_method, panel_type

    """

    if estimator == 'reg' and pscore_est_method is not None:
        raise ValueError("pscore not used for the 'reg' estimator, "
                         "set 'pscore_est_method' to None")

    if estimator in ['ipw', 'std_ipw'] and pscore_est_method == 'ipt':
        raise ValueError("pscore = 'ipt' not used for the 'ipw' estimator, "
                         "set 'pscore_est_method' to 'mle'")

    if estimator in ['dr', 'ipw'] and pscore_est_method is None:
        raise ValueError('please specify pscore')

    return estimator, pscore_est_method, panel_type
