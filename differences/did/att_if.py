# computing ATT (g,t) and influence functions

import numpy as np


# ------------------------- drdid traditional --------------------------


def dr_traditional_panel_att_if(endog: np.ndarray,  # [:, None]
                                exog: np.ndarray,
                                treated: np.ndarray,  # [:, None]
                                weights: np.ndarray,  # [:, None]
                                out_delta: np.ndarray,  # [:, None]
                                ps_fit: np.ndarray,  # [:, None]
                                pscore_cov_params: np.ndarray,
                                **kwargs):
    n_obs = exog.shape[0]

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------

    w_treat = weights * treated
    w_cont = weights * ps_fit * ((1 - treated) / (1 - ps_fit))

    dr_att_treat = w_treat * (endog - out_delta)
    dr_att_cont = w_cont * (endog - out_delta)

    eta_treat = np.mean(dr_att_treat) / np.mean(w_treat)
    eta_cont = np.mean(dr_att_cont) / np.mean(w_cont)

    dr_att = eta_treat - eta_cont

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # first, the influence function of the nuisance functions
    # asymptotic linear representation of OLS parameters

    weights_ols = weights * (1 - treated)
    wols_x = weights_ols * exog
    wols_eX = weights_ols * (endog - out_delta) * exog

    XpX_inv = np.linalg.inv((wols_x.T @ exog) / n_obs)
    asy_lin_rep_wols = wols_eX @ XpX_inv.T

    # asymptotic linear representation of logit's beta's
    score_ps = (weights * (treated - ps_fit)) * exog

    Hessian_ps = pscore_cov_params * n_obs
    asy_lin_rep_ps = score_ps @ Hessian_ps.T

    # now, the influence function of the "treat" component
    # leading term of the influence function: no estimation effect

    inf_treat_1 = (dr_att_treat - w_treat * eta_treat)

    # Estimation effect from beta hat
    # Derivative matrix (k x 1 vector)
    M1 = np.mean((w_treat * exog), axis=0)

    # now get the influence function related to the estimation effect
    # related to beta's
    inf_treat_2 = asy_lin_rep_wols @ M1[:, None]

    # Influence function for the treated component
    inf_treat = (inf_treat_1 - inf_treat_2) / np.mean(w_treat)

    # ------------------------------------------------------------------

    # now, get the influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont_1 = (dr_att_cont - w_cont * eta_cont)

    # Estimation effect from gamma hat (pscore)
    # Derivative matrix (k x 1 vector)
    M2 = np.mean(w_cont * (endog - out_delta - eta_cont) * exog, axis=0)

    # Now the influence function related to estimation effect of pscores
    inf_cont_2 = asy_lin_rep_ps @ M2[:, None]

    # Estimation Effect from beta hat (weighted OLS)
    M3 = np.mean(w_cont * exog, axis=0)

    # Now the influence function related to estimation effect of regressions
    inf_cont_3 = asy_lin_rep_wols @ M3[:, None]

    # Influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / np.mean(w_cont)

    # get the influence function of the DR estimator (put all pieces together)
    dr_att_inf_func = inf_treat - inf_control

    return dr_att, dr_att_inf_func


def dr_traditional_rc_att_if(endog: np.ndarray,  # [:, None]
                             exog: np.ndarray,
                             treated: np.ndarray,  # [:, None]
                             weights: np.ndarray,  # [:, None]
                             out_delta: np.ndarray,  # n x 4
                             ps_fit: np.ndarray,  # [:, None]
                             pscore_cov_params: np.ndarray,
                             post: np.ndarray,  # [:, None]
                             **kwargs):
    n_obs = exog.shape[0]
    out_y_cont_pre, out_y_cont_post, \
    out_y_treat_pre, out_y_treat_post = np.split(out_delta, 4, axis=1)

    # ------------------------------------------------------------------

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------

    # first, the weights
    w_treat_pre = weights * treated * (1 - post)
    w_treat_post = weights * treated * post
    w_cont_pre = weights * ps_fit * (1 - treated) * (1 - post) / (1 - ps_fit)
    w_cont_post = weights * ps_fit * (1 - treated) * post / (1 - ps_fit)

    w_d = weights * treated
    w_dt1 = weights * treated * post
    w_dt0 = weights * treated * (1 - post)

    # elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * (endog - out_y_cont) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (endog - out_y_cont) / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * (endog - out_y_cont) / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * (endog - out_y_cont) / np.mean(w_cont_post)

    # extra elements for the locally efficient DRDID
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_dt0)

    # estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    att_d_post = np.mean(eta_d_post)
    att_dt1_post = np.mean(eta_dt1_post)
    att_d_pre = np.mean(eta_d_pre)
    att_dt0_pre = np.mean(eta_dt0_pre)

    # ATT estimator
    dr_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre) + (
            att_d_post - att_dt1_post) - (
                     att_d_pre - att_dt0_pre)

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # first, the influence function of the nuisance functions

    # asymptotic linear representation of OLS parameters in pre-period, control group
    asy_lin_rep_ols_pre = asymptotic_lin_repr(endog=endog,
                                              exog=exog,
                                              n_obs=n_obs,
                                              weights=(weights * (1 - treated) * (1 - post)),
                                              out_y=out_y_cont_pre)

    # asymptotic linear representation of OLS parameters in post-period, control group
    asy_lin_rep_ols_post = asymptotic_lin_repr(endog=endog,
                                               exog=exog,
                                               n_obs=n_obs,
                                               weights=(weights * (1 - treated) * post),
                                               out_y=out_y_cont_post)

    # asymptotic linear representation of OLS parameters in pre-period, treated
    asy_lin_rep_ols_pre_treat = asymptotic_lin_repr(endog=endog,
                                                    exog=exog,
                                                    n_obs=n_obs,
                                                    weights=(weights * treated * (1 - post)),
                                                    out_y=out_y_treat_pre)

    # asymptotic linear representation of OLS parameters in post-period, treated
    asy_lin_rep_ols_post_treat = asymptotic_lin_repr(endog=endog,
                                                     exog=exog,
                                                     n_obs=n_obs,
                                                     weights=(weights * treated * post),
                                                     out_y=out_y_treat_post)

    # asymptotic linear representation of logit's beta's
    score_ps = weights * (treated - ps_fit) * exog
    Hessian_ps = pscore_cov_params * n_obs
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # ------------------------------------------------------------------
    # now, the influence function of the "treat" component
    # leading term of the influence function: no estimation effect
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)

    # estimation effect from beta hat from post and pre-periods
    # derivative matrix (k x 1 vector)
    M1_post = - np.mean(w_treat_post * post * exog, axis=0)[:, None] / np.mean(w_treat_post)
    M1_pre = - np.mean(w_treat_pre * (1 - post) * exog, axis=0)[:, None] / np.mean(w_treat_pre)

    # now get the influence function related to the estimation
    # effect related to beta's
    inf_treat_or_post = asy_lin_rep_ols_post @ M1_post
    inf_treat_or_pre = asy_lin_rep_ols_pre @ M1_pre

    inf_treat_or = inf_treat_or_post + inf_treat_or_pre

    # influence function for the treated component
    inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or

    # ------------------------------------------------------------------
    # now, get the influence function of control component
    # leading term of the influence function: no estimation effect from
    # nuisance parameters
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

    # estimation effect from gamma hat (pscore)

    # derivative matrix (k x 1 vector)

    M2_pre = (np.mean(w_cont_pre * (endog - out_y_cont - att_cont_pre) * exog, axis=0)[:, None]
              / np.mean(w_cont_pre))

    M2_post = (np.mean(w_cont_post * (endog - out_y_cont - att_cont_post) * exog, axis=0)[:, None]
               / np.mean(w_cont_post))

    # now the influence function related to estimation effect of pscores
    inf_cont_ps = asy_lin_rep_ps @ (M2_post - M2_pre)

    # estimation effect from beta hat from post and pre-periods

    # derivative matrix (k x 1 vector)

    M3_post = - np.mean(w_cont_post * post * exog, axis=0)[:, None] / np.mean(w_cont_post)
    M3_pre = - np.mean(w_cont_pre * (1 - post) * exog, axis=0)[:, None] / np.mean(w_cont_pre)

    # now get the influence function related to the estimation effect related to beta's
    inf_cont_or_post = asy_lin_rep_ols_post @ M3_post
    inf_cont_or_pre = asy_lin_rep_ols_pre @ M3_pre

    inf_cont_or = inf_cont_or_post + inf_cont_or_pre

    # influence function for the control component
    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

    # ------------------------------------------------------------------
    # get the influence function of the inefficient DR estimator (put all pieces together)
    dr_att_inf_func1 = inf_treat - inf_cont
    # ------------------------------------------------------------------

    # now, we only need to get the influence function of the adjustment terms
    # first, the terms as if all OR parameters were known
    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)

    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # now the estimation effect of the OR coefficients
    mom_post = np.mean((w_d / np.mean(w_d) - w_dt1 / np.mean(w_dt1)) * exog, axis=0)[:, None]
    mom_pre = np.mean((w_d / np.mean(w_d) - w_dt0 / np.mean(w_dt0)) * exog, axis=0)[:, None]

    inf_or_post = (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
    inf_or_pre = (asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre) @ mom_pre

    inf_or = inf_or_post - inf_or_pre

    # ------------------------------------------------------------------
    # get the influence function of the locally efficient DR estimator
    # (put all pieces together)

    dr_att_inf_func = dr_att_inf_func1 + inf_eff + inf_or

    # ------------------------------------------------------------------

    return dr_att, dr_att_inf_func


# ------------------------- drdid improved -----------------------------

def dr_improved_panel_att_if(endog: np.ndarray,  # [:, None]
                             treated: np.ndarray,  # [:, None]  D
                             weights: np.ndarray,  # [:, None]
                             out_delta: np.ndarray,  # [:, None]
                             ps_fit: np.ndarray,  # [:, None]
                             **kwargs):
    dr_att_summand_num = (1 - (1 - treated) / (1 - ps_fit)) * (endog - out_delta)
    dr_att = np.mean(weights * dr_att_summand_num) / np.mean(treated * weights)

    # get the influence function to compute standard error
    dr_att_inf_func = ((weights * (dr_att_summand_num - treated * dr_att)) /
                       np.mean(treated * weights))

    return dr_att, dr_att_inf_func


def dr_improved_rc_att_if(endog: np.ndarray,  # [:, None]
                          treated: np.ndarray,  # [:, None]
                          weights: np.ndarray,  # [:, None]
                          out_delta: np.ndarray,  # n x 4
                          ps_fit: np.ndarray,  # [:, None]
                          post: np.ndarray,  # [:, None]
                          **kwargs):
    out_y_cont_pre, out_y_cont_post, \
    out_y_treat_pre, out_y_treat_post = np.split(out_delta, 4, axis=1)

    # ------------------------------------------------------------------

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------

    # first, the weights
    w_treat_pre = weights * treated * (1 - post)
    w_treat_post = weights * treated * post
    w_cont_pre = weights * ps_fit * (1 - treated) * (1 - post) / (1 - ps_fit)
    w_cont_post = weights * ps_fit * (1 - treated) * post / (1 - ps_fit)

    w_d = weights * treated
    w_dt1 = weights * treated * post
    w_dt0 = weights * treated * (1 - post)

    # elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * (endog - out_y_cont) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (endog - out_y_cont) / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * (endog - out_y_cont) / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * (endog - out_y_cont) / np.mean(w_cont_post)

    # extra elements for the locally efficient DRDID
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_dt0)

    # estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    att_d_post = np.mean(eta_d_post)
    att_dt1_post = np.mean(eta_dt1_post)
    att_d_pre = np.mean(eta_d_pre)
    att_dt0_pre = np.mean(eta_dt0_pre)

    # ATT estimator
    dr_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre) + \
             (att_d_post - att_dt1_post) - (att_d_pre - att_dt0_pre)

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------
    # now, the influence function of the "treat" component

    # leading term of the influence function: no estimation effect
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)

    # influence function for the treated component
    inf_treat = inf_treat_post - inf_treat_pre
    # ------------------------------------------------------------------
    # now, get the influence function of control component

    # leading term of the influence function: no estimation effect from nuisance parameters
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

    # influence function for the control component
    inf_cont = inf_cont_post - inf_cont_pre
    # ------------------------------------------------------------------

    # get the influence function of the DR estimator (put all pieces together)
    dr_att_inf_func1 = inf_treat - inf_cont

    # ------------------------------------------------------------------
    # now, we only need to get the influence function of the adjustment terms
    # first, the terms as if all OR parameters were known

    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # ------------------------------------------------------------------
    # get the influence function of the locally efficient DR estimator (put all pieces together)
    dr_att_inf_func = dr_att_inf_func1 + inf_eff

    return dr_att, dr_att_inf_func


# ---------------------- did outcome regression ------------------------

def reg_panel_att_if(endog: np.ndarray,
                     exog: np.ndarray,
                     weights: np.ndarray,
                     treated: np.ndarray,
                     out_delta: np.ndarray,
                     **kwargs):
    n_obs = len(exog)

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------

    # compute the OR-DID estimator
    # first, the weights
    w_treat = weights * treated
    w_cont = weights * treated

    reg_att_treat = w_treat * endog
    reg_att_cont = w_cont * out_delta

    eta_treat = np.mean(reg_att_treat) / np.mean(w_treat)
    eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)

    reg_att = eta_treat - eta_cont

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # first, the influence function of the nuisance functions

    # asymptotic linear representation of OLS parameters

    asy_lin_rep_ols = asymptotic_lin_repr(endog=endog,
                                          exog=exog,
                                          n_obs=n_obs,
                                          weights=weights * (1 - treated),
                                          out_y=out_delta)

    # ------------------------------------------------------------------
    # Now, the influence function of the "treat" component
    # Leading term of the influence function

    inf_treat = (reg_att_treat - w_treat * eta_treat) / np.mean(w_treat)

    # ------------------------------------------------------------------

    # get the influence function of control component
    # leading term of the influence function: no estimation effect
    inf_cont_1 = (reg_att_cont - w_cont * eta_cont)

    # estimation effect from beta hat (OLS using only controls)
    # derivative matrix (k x 1 vector)

    M1 = np.mean((w_cont * exog), axis=0)

    # Now get the influence function related to the estimation effect
    # related to beta's

    inf_cont_2 = asy_lin_rep_ols @ M1[:, None]

    # Influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2) / np.mean(w_cont)

    # ------------------------------------------------------------------
    # get the influence function of the DR estimator
    # (put all pieces together)
    reg_att_inf_func = (inf_treat - inf_control)

    return reg_att, reg_att_inf_func


def reg_rc_att_if(endog: np.ndarray,
                  exog: np.ndarray,
                  weights: np.ndarray,
                  treated: np.ndarray,
                  out_delta: np.ndarray,
                  post: np.ndarray,
                  **kwargs):
    n_obs = exog.shape[0]
    out_y_pre, out_y_post = np.split(out_delta, 2, axis=1)

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------
    # compute the OR DID estimators

    # first, the weights
    w_treat_pre = weights * treated * (1 - post)
    w_treat_post = weights * treated * post
    w_cont = weights * treated

    reg_att_treat_pre = w_treat_pre * endog
    reg_att_treat_post = w_treat_post * endog
    reg_att_cont = w_cont * (out_y_post - out_y_pre)

    eta_treat_pre = np.mean(reg_att_treat_pre) / np.mean(w_treat_pre)
    eta_treat_post = np.mean(reg_att_treat_post) / np.mean(w_treat_post)
    eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)

    reg_att = (eta_treat_post - eta_treat_pre) - eta_cont

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # First, the influence function of the nuisance functions
    # Asymptotic linear representation of OLS parameters in pre-period

    asy_lin_rep_ols_pre = asymptotic_lin_repr(endog=endog,
                                              exog=exog,
                                              n_obs=n_obs,
                                              weights=weights * (1 - treated) * (1 - post),
                                              out_y=out_y_pre)

    # Asymptotic linear representation of OLS parameters in post-period

    asy_lin_rep_ols_post = asymptotic_lin_repr(endog=endog,
                                               exog=exog,
                                               n_obs=n_obs,
                                               weights=weights * (1 - treated) * post,
                                               out_y=out_y_post)

    # ------------------------------------------------------------------
    # Now, the influence function of the "treat" component
    # Leading term of the influence function
    inf_treat_pre = (reg_att_treat_pre - w_treat_pre * eta_treat_pre) / np.mean(w_treat_pre)
    inf_treat_post = (reg_att_treat_post - w_treat_post * eta_treat_post) / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre
    # ------------------------------------------------------------------

    # Now, get the influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont_1 = (reg_att_cont - w_cont * eta_cont)
    # Estimation effect from beta hat (OLS using only controls)
    # Derivative matrix (k x 1 vector)

    M1 = np.mean((w_cont * exog), axis=0)

    # Now get the influence function related to the estimation effect
    # related to beta's in post-treatment
    inf_cont_2_post = asy_lin_rep_ols_post @ M1[:, None]

    # Now get the influence function related to the estimation effect
    # related to beta's in pre-treatment
    inf_cont_2_pre = asy_lin_rep_ols_pre @ M1[:, None]

    # Influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2_post - inf_cont_2_pre) / np.mean(w_cont)

    # ------------------------------------------------------------------

    # get the influence function of the DR estimator (put all pieces together)
    reg_att_inf_func = (inf_treat - inf_control)

    return reg_att, reg_att_inf_func


# ---------------------------- ipw did ---------------------------------


def ipw_panel_att_if(endog: np.ndarray,  # [:, None]
                     exog: np.ndarray,
                     treated: np.ndarray,  # [:, None]  D
                     weights: np.ndarray,  # [:, None]
                     ps_fit: np.ndarray,  # [:, None]
                     pscore_cov_params: np.ndarray,
                     **kwargs):
    n_obs = exog.shape[0]

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------
    # compute IPW estimator

    # first, the weights

    w_treat = weights * treated
    w_cont = weights * ps_fit * (1 - treated) / (1 - ps_fit)

    att_treat = w_treat * endog
    att_cont = w_cont * endog

    eta_treat = np.mean(att_treat) / np.mean(weights * treated)
    eta_cont = np.mean(att_cont) / np.mean(weights * treated)

    ipw_att = eta_treat - eta_cont

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # asymptotic linear representation of logit's beta's
    score_ps = weights * (treated - ps_fit) * exog
    Hessian_ps = pscore_cov_params * n_obs
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # ------------------------------------------------------------------
    # now, get the influence function of control component

    # leading term of the influence function: no estimation effect
    att_lin1 = att_treat - att_cont

    # derivative matrix (k x 1 vector)
    mom_logit = np.mean((att_cont * exog), axis=0)[:, None]

    # now the influence function related to estimation effect of pscores
    att_lin2 = asy_lin_rep_ps @ mom_logit

    # get the influence function of the DR estimator (put all pieces together)

    att_inf_func = ((att_lin1 - att_lin2 - weights * treated * ipw_att) /
                    np.mean(weights * treated))

    return ipw_att, att_inf_func


def std_ipw_panel_att_if(endog: np.ndarray,  # [:, None]
                         exog: np.ndarray,
                         treated: np.ndarray,  # [:, None]
                         weights: np.ndarray,  # [:, None]
                         ps_fit: np.ndarray,  # [:, None]
                         pscore_cov_params: np.ndarray,
                         **kwargs):
    n_obs = exog.shape[0]

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------

    # compute IPW estimator
    # first, the weights

    w_treat = weights * treated
    w_cont = weights * ps_fit * (1 - treated) / (1 - ps_fit)

    att_treat = w_treat * endog
    att_cont = w_cont * endog

    eta_treat = np.mean(att_treat) / np.mean(w_treat)
    eta_cont = np.mean(att_cont) / np.mean(w_cont)

    ipw_att = eta_treat - eta_cont

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # asymptotic linear representation of logit's beta's
    score_ps = weights * (treated - ps_fit) * exog
    Hessian_ps = pscore_cov_params * n_obs
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # ------------------------------------------------------------------
    # now, the influence function of the "treat" component

    # leading term of the influence function: no estimation effect

    inf_treat = (att_treat - w_treat * eta_treat) / np.mean(w_treat)

    # now, get the influence function of control component
    # leading term of the influence function: no estimation effect

    inf_cont_1 = (att_cont - w_cont * eta_cont)
    # estimation effect from gamma hat (pscore)

    # derivative matrix (k x 1 vector)
    M2 = np.mean(w_cont * (endog - eta_cont) * exog, axis=0)[:, None]

    # now the influence function related to estimation effect of pscores
    inf_cont_2 = asy_lin_rep_ps @ M2

    # influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2) / np.mean(w_cont)

    # get the influence function of the DR estimator (put all pieces together)
    att_inf_func = inf_treat - inf_control

    return ipw_att, att_inf_func


def ipw_rc_att_if(endog: np.ndarray,  # [:, None]
                  exog: np.ndarray,
                  treated: np.ndarray,  # [:, None]
                  weights: np.ndarray,  # [:, None]
                  ps_fit: np.ndarray,  # [:, None]
                  pscore_cov_params: np.ndarray,
                  post: np.ndarray,
                  **kwargs):
    n_obs = exog.shape[0]

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------
    # compute IPW estimator

    # first, the weights
    w_treat_pre = weights * treated * (1 - post)
    w_treat_post = weights * treated * post
    w_cont_pre = weights * ps_fit * (1 - treated) * (1 - post) / (1 - ps_fit)
    w_cont_post = weights * ps_fit * (1 - treated) * post / (1 - ps_fit)

    Pi_hat = np.mean(weights * treated)
    lambda_hat = np.mean(weights * post)
    one_minus_lambda_hat = np.mean(weights * (1 - post))

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * endog / (Pi_hat * one_minus_lambda_hat)
    eta_treat_post = w_treat_post * endog / (Pi_hat * lambda_hat)
    eta_cont_pre = w_cont_pre * endog / (Pi_hat * one_minus_lambda_hat)
    eta_cont_post = w_cont_post * endog / (Pi_hat * lambda_hat)

    # Estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    # ATT estimator
    ipw_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre)

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # asymptotic linear representation of logit's beta's
    score_ps = weights * (treated - ps_fit) * exog
    Hessian_ps = pscore_cov_params * n_obs
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # ------------------------------------------------------------------
    # influence function of the treated components

    inf_treat_post1 = eta_treat_post - att_treat_post
    inf_treat_post2 = - (weights * treated - Pi_hat) * att_treat_post / Pi_hat
    inf_treat_post3 = - (weights * post - lambda_hat) * att_treat_post / lambda_hat
    inf_treat_post = inf_treat_post1 + inf_treat_post2 + inf_treat_post3

    inf_treat_pre1 = eta_treat_pre - att_treat_pre
    inf_treat_pre2 = - (weights * treated - Pi_hat) * att_treat_pre / Pi_hat
    inf_treat_pre3 = - ((weights * (1 - post) - one_minus_lambda_hat) * att_treat_pre
                        / one_minus_lambda_hat)
    inf_treat_pret = inf_treat_pre1 + inf_treat_pre2 + inf_treat_pre3

    # Now, get the influence function of control component
    # First, terms of the inf_ funct as if pscore was known
    inf_cont_post1 = eta_cont_post - att_cont_post
    inf_cont_post2 = - (weights * treated - Pi_hat) * att_cont_post / Pi_hat
    inf_cont_post3 = - (weights * post - lambda_hat) * att_cont_post / lambda_hat
    inf_cont_post = inf_cont_post1 + inf_cont_post2 + inf_cont_post3

    inf_cont_pre1 = eta_cont_pre - att_cont_pre

    inf_cont_pre2 = - (weights * treated - Pi_hat) * att_cont_pre / Pi_hat

    inf_cont_pre3 = - ((weights * (1 - post) - one_minus_lambda_hat) * att_cont_pre
                       / one_minus_lambda_hat)
    inf_cont_pret = inf_cont_pre1 + inf_cont_pre2 + inf_cont_pre3

    # estimation effect from the propensity score parametes

    # derivative matrix (k x 1 vector)

    mom_logit_pre = np.mean(- eta_cont_pre * exog, axis=0)[:, None]
    mom_logit_post = np.mean(- eta_cont_post * exog, axis=0)[:, None]

    # now the influence function related to estimation effect of pscores

    inf_logit = asy_lin_rep_ps @ (mom_logit_post - mom_logit_pre)

    # get the influence function of the DR estimator (put all pieces together)
    att_inf_func = (inf_treat_post - inf_treat_pret) - (inf_cont_post - inf_cont_pret) + inf_logit

    return ipw_att, att_inf_func


def std_ipw_rc_att_if(endog: np.ndarray,  # [:, None]
                      exog: np.ndarray,
                      treated: np.ndarray,  # [:, None]
                      weights: np.ndarray,  # [:, None]
                      ps_fit: np.ndarray,  # [:, None]
                      pscore_cov_params: np.ndarray,
                      post: np.ndarray,
                      **kwargs):
    n_obs = exog.shape[0]

    # ------------------------------------------------------------------
    # att
    # ------------------------------------------------------------------
    # compute IPW estimator

    # First, the weights
    w_treat_pre = weights * treated * (1 - post)
    w_treat_post = weights * treated * post
    w_cont_pre = weights * ps_fit * (1 - treated) * (1 - post) / (1 - ps_fit)
    w_cont_post = weights * ps_fit * (1 - treated) * post / (1 - ps_fit)

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * endog / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * endog / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * endog / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * endog / np.mean(w_cont_post)

    # Estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    # ATT estimator
    ipw_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre)

    # ------------------------------------------------------------------
    # influence function
    # ------------------------------------------------------------------

    # asymptotic linear representation of logit's beta's
    score_ps = weights * (treated - ps_fit) * exog
    Hessian_ps = pscore_cov_params * n_obs
    asy_lin_rep_ps = score_ps @ Hessian_ps

    # ------------------------------------------------------------------

    # now, the influence function of the "treat" component

    # leading term of the influence function: no estimation effect

    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    # now, get the influence function of control component

    # leading term of the influence function: no estimation effect

    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)
    inf_cont = inf_cont_post - inf_cont_pre

    # estimation effect from gamma hat (pscore)

    # derivative matrix (k x 1 vector)

    M2_pre = (np.mean(w_cont_pre * (endog - att_cont_pre) * exog, axis=0)[:, None] /
              np.mean(w_cont_pre)
              )

    M2_post = (np.mean(w_cont_post * (endog - att_cont_post) * exog, axis=0)[:, None] /
               np.mean(w_cont_post))

    # now the influence function related to estimation effect of pscores
    inf_cont_ps = asy_lin_rep_ps @ (M2_post - M2_pre)

    # influence function for the control component
    inf_cont = inf_cont + inf_cont_ps

    # get the influence function of the DR estimator (put all pieces together)
    att_inf_func = inf_treat - inf_cont

    return ipw_att, att_inf_func


# --------------------------- helpers ----------------------------------

def asymptotic_lin_repr(endog,
                        exog,
                        n_obs,
                        weights,
                        out_y):
    wols_x = weights * exog
    wols_eX = weights * (endog - out_y) * exog

    XpX_inv = np.linalg.inv((wols_x.T @ exog) / n_obs)

    return wols_eX @ XpX_inv
