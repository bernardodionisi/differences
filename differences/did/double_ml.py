# the code in this module is sourced and adapted from
# https://github.com/vveitch/causality-tutorials/blob/main/difference_in_differences.ipynb

# THIS MODULE is A WORK IN PROGRESS

import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import (
    KFold,
    StratifiedKFold)

# ----------------- conditional expected outcome -----------------------


def make_outcome_model(random_seed=123):
    """sklearn model for the conditional expected outcome

    for later use in k-folding"""

    # return LinearRegression()
    return RandomForestRegressor(
        random_state=random_seed,
        n_estimators=100,
        max_depth=2
    )


# --------------------- propensity score -------------------------------


def make_pscore_model():
    """model for the propensity score"""

    #  return LogisticRegression(max_iter=1000)

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=2
    )


# ------------------------- cross fitting ------------------------------


def predict_treatment(
        exog: ndarray,
        treated: ndarray,
        n_splits: int,
        pscore_model: BaseEstimator = None,
        random_seed=123):
    """
    K fold cross-fitting for the model predicting the treatment, for each unit

    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make pred for each data point in fold j

    Parameters
    ----------
    make_model
        function that returns sklearn model (which implements fit and predict_prob)
    exog
        covariates
    treated
        array of treatments
    n_splits
        number of splits to use
    pscore_model

    Returns
    -------
    an array containing the predictions
    """

    pred = np.full_like(treated, np.nan, dtype=float)

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed
    )

    for train_idx, test_idx in kf.split(exog, treated):

        if pscore_model is None:
            g = make_pscore_model()
        else:
            g = pscore_model

        g.fit(exog[train_idx], treated[train_idx])

        # get pred for split
        pred[test_idx] = g.predict_proba(exog[test_idx])[:, 1]

    assert np.isnan(pred).sum() == 0

    return pred


def predict_outcome(
        endog: ndarray,
        exog: ndarray,
        treated: ndarray,
        n_splits: int,
        outcome_type: str,
        outcome_model: BaseEstimator = None,
        random_seed=123):
    """
    K fold cross-fitting for the model predicting the outcome, for each unit

    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make pred for each data point in fold j

    Parameters
    ----------
    make_model
        function that returns sklearn model
        (that implements fit and either predict_prob or predict)
    endog
        array of outcomes
    exog
        covariates
    treated
        array of treatments
    n_splits
        number of splits to use
    outcome_type
        type of outcome, "binary" or "continuous"
    outcome_model

    Returns
    -------
    2 arrays containing the pred for all units untreated, all units treated
    """

    pred_0 = np.full_like(treated, np.nan, dtype=float)
    pred_1 = np.full_like(endog, np.nan, dtype=float)

    if outcome_type == 'binary':
        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed
        )

    elif outcome_type == 'continuous':
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed
        )

    for train_idx, test_idx in kf.split(exog, endog):

        if outcome_model is None:
            q = make_outcome_model()
        else:
            q = outcome_model

        # include the treatment as input feature
        q.fit(np.c_[exog, treated][train_idx], endog[train_idx])

        if outcome_type == 'binary':
            pred_0[test_idx] = q.predict_proba(
                np.c_[exog, np.zeros(exog.shape[0])][test_idx])[:, 1]

            pred_1[test_idx] = q.predict_proba(
                np.c_[exog, np.ones(exog.shape[0])][test_idx])[:, 1]

        elif outcome_type == 'continuous':
            pred_0[test_idx] = q.predict(
                np.c_[exog, np.zeros(exog.shape[0])][test_idx])

            pred_1[test_idx] = q.predict(
                np.c_[exog, np.ones(exog.shape[0])][test_idx])

    assert np.isnan(pred_0).sum() == 0
    assert np.isnan(pred_1).sum() == 0

    return pred_0, pred_1


# ----------------------------------------------------------------------
# Combine predicted values and data into estimate of ATT

def aiptw_att_if(outcome_0,
                 pscore,
                 endog: ndarray,
                 treated: ndarray,
                 prob_t: ndarray = None,
                 **kwargs):
    """
    DoubleML estimator for the ATT

    ATT specific scores,
    see equation 3.9 of https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf
    """

    if prob_t is None:
        prob_t = treated.mean()  # estimate marginal probability of treatment

    tau_hat = (treated * (endog - outcome_0) - (1 - treated) *
               (pscore / (1 - pscore)) * (endog - outcome_0)
               ).mean() / prob_t

    scores = (treated * (endog - outcome_0) - (1 - treated) *
              (pscore / (1 - pscore)) * (endog - outcome_0)
              - tau_hat * treated) / prob_t

    # std_hat = np.std(scores) / np.sqrt(n observations)
    # f"The estimate is {tau_hat} ± {1.96 * std_hat}"

    return {'att': tau_hat, 'influence_func': scores}


# ----------------------------------------------------------------------


def aiptw_double_ml_did_panel(endog: ndarray,
                              exog: ndarray,
                              treated: ndarray,
                              pscore_model: BaseEstimator = None,
                              outcome_model: BaseEstimator = None,
                              outcome_type: str = 'continuous',  # 'binary'
                              **other):
    pscore = predict_treatment(
        pscore_model=pscore_model,
        exog=exog,
        treated=treated,
        n_splits=10
    )

    outcome_0, outcome_1 = predict_outcome(
        outcome_model=outcome_model,
        endog=endog,
        exog=exog,
        treated=treated,
        n_splits=10,
        outcome_type=outcome_type
    )

    res = aiptw_att_if(
        outcome_0=outcome_0,
        pscore=pscore,
        endog=endog,
        treated=treated)

    return res
