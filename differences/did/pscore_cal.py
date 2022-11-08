import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# mle: used for drdid traditional
# ----------------------------------------------------------------------


def pscore_mle(treated: np.ndarray,
               exog: np.ndarray,
               weights: np.ndarray,
               _freq_weights_flag: bool = False):
    if _freq_weights_flag:
        weights_arg = {'freq_weights': weights}
    else:
        weights_arg = {'var_weights': weights}

    ps_mle = sm.GLM(treated,
                    exog,
                    family=sm.families.Binomial(),
                    **weights_arg
                    ).fit()

    ps_fit = ps_mle.predict()
    ps_fit = np.where(ps_fit > 1 - 1e-16, 1, ps_fit)

    # pscore_mle.params
    return ps_fit, ps_mle.cov_params()


# ----------------------------------------------------------------------
# ipt: pscore based on inverse probability of tilting
# ----------------------------------------------------------------------


def loss_ps_ipt_fun(gam: np.ndarray,
                    treated: np.ndarray,
                    exog: np.ndarray,
                    iw: np.ndarray):
    ps_ind = exog @ gam
    exp_ps_ind = np.exp(ps_ind)

    arg_D = np.where(np.array(treated) == 0)[0]
    ps_ind[arg_D] = -exp_ps_ind[arg_D]

    val = - np.mean(ps_ind * iw)

    return val


def loss_ps_ipt_grad(gam: np.ndarray,
                     treated: np.ndarray,
                     exog: np.ndarray,
                     iw: np.ndarray):
    exp_ps_ind = np.exp(exog @ gam)
    arg_D = np.where(np.array(treated) == 0)[0]

    ones = np.ones(len(exp_ps_ind))
    ones[arg_D] = -exp_ps_ind[arg_D]

    grad = - np.mean((ones * iw)[:, None] * exog, axis=0)
    return grad


def loss_ps_ipt_hess(gam: np.ndarray,
                     treated: np.ndarray,
                     exog: np.ndarray,
                     iw: np.ndarray):
    exp_ps_ind = np.exp(exog @ gam)
    arg_D = np.where(np.array(treated) == 0)[0]

    zeros = np.zeros(len(exp_ps_ind))
    zeros[arg_D] = -exp_ps_ind[arg_D]

    hess = - (exog.T @ ((zeros * iw)[:, None] * exog)) / len(exog)
    return hess


def pscore_ipt(treated: np.ndarray,
               exog: np.ndarray,
               weights: np.ndarray,
               _freq_weights_flag: bool = False,
               tol: float = None,  # 1e-10
               display_minimize: bool = True):
    if _freq_weights_flag:
        weights_arg = {'freq_weights': weights}
    else:
        weights_arg = {'var_weights': weights}

    gamma_model = sm.GLM(treated,
                         exog,
                         family=sm.families.Binomial(),
                         **weights_arg
                         ).fit()

    gam = gamma_model.params
    gam = np.array(gam)

    # method: trust region on loss_ps_cal_fun
    trust_exact_res = minimize(fun=loss_ps_ipt_fun,
                               x0=gam,
                               args=(treated, exog, weights),
                               method='trust-exact',
                               jac=loss_ps_ipt_grad,
                               hess=loss_ps_ipt_hess,
                               options={'maxiter': 1000, 'disp': display_minimize},
                               tol=tol
                               )
    if not trust_exact_res.success:
        raise ValueError("trust algorithm did not converge when estimating propensity score")

    # gamma_cal == res.x
    # pscore_index = res.x @ exog.T

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pscore = sigmoid(trust_exact_res.x @ exog.T)

    pscore = np.where(pscore > 1 - 1e-16, 1, pscore)

    return pscore,

# if algorithm doesn't converge, update like in Graham et al


# todo: add the other loss function + other minimization in case
#  res.success is not successful

# ----------------------------------------------------------------------
