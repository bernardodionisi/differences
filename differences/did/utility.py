import numpy as np
import statsmodels.api as sm


# ---------------------- outcome regression ----------------------------

def wols_out_reg(endog: np.ndarray,
                 exog: np.ndarray,
                 weights: np.ndarray,
                 mask: np.ndarray,
                 pscore_fit: np.ndarray = None):
    """computes the outcome regression for the control group using wols"""
    if pscore_fit is not None:
        weights = (weights * pscore_fit / (1 - pscore_fit))

    _wls = sm.WLS(endog=endog[mask],
                  exog=exog[mask],
                  weights=weights[mask]
                  ).fit()

    wols_betas = _wls.params
    out_delta = wols_betas @ exog.T

    return out_delta


# --------------------------- various ----------------------------------


def _analytical_standard_error(_att, _inf_func, n_obs):
    se_dr_att = np.std(_inf_func, ddof=1) / np.sqrt(n_obs)

    uci = _att + 1.96 * se_dr_att
    lci = _att - 1.96 * se_dr_att

    return {'_att': _att,
            'se_dr_att': se_dr_att,
            'lci': lci,
            'uci': uci}
