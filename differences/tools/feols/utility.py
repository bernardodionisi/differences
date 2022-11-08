import numpy as np

import pandas as pd
from pandas import DataFrame, Series

from scipy import stats

from formulaic import model_matrix

from linearmodels import AbsorbingLS

from pyhdfe.utilities import identify_singletons, Groups


# from linearmodels.panel.utility import in_2core_graph, check_absorbed


# ------------------------- data matrix --------------------------------


def build_model_matrix(data: DataFrame,
                       formula: str,

                       categories_data: DataFrame = None,
                       dummy_fe: list = None,

                       zeros_as_nas: bool = False,
                       ensure_full_rank: bool = False
                       ) -> tuple[DataFrame, DataFrame]:
    """

    prepare controls + fixed effects matrices

    only two-way fixed effects can be absorbed at this time with PanelOLS
    feed the other fixed effects as dummies
    (need to handle singletons first)

    for now: if one wants to estimate the fixed effects, only one can
    be absorbed, so absorb one, and feed the other as dummies.

    need to change this at some point

    """

    # if formula is '-1' it returns idxs to join to dummy_fe
    y_data_matrix, x_data_matrix = model_matrix(spec=formula, data=data)

    if dummy_fe:
        if categories_data is None:
            raise ValueError('missing categories_data: '
                             'dataframe with fe as categories')

        model_dummies = model_matrix(
            spec=f"-1 + {'+'.join(dummy_fe)}",  # formulaic converts categories to dummies
            data=categories_data,
            ensure_full_rank=ensure_full_rank
        ).replace(0, (np.nan if zeros_as_nas else 0))  # nas in place of 0s

        x_data_matrix = x_data_matrix.join(model_dummies)
        # todo: missing values in the categories?

    return y_data_matrix, x_data_matrix


# ------------------------- singletons ---------------------------------

def eliminate_singletons_form_data(fe_data: DataFrame) -> DataFrame:
    """
    subsets the data eliminating all the singletons

    todo: add in_2core_graph

    Parameters
    ----------
    fe_data

    Returns
    -------

    """

    # temp rename the index: some cols may have the same name, groupby would not work
    major_idx, minor_idx = fe_data.index.names
    fe_data.index.names = [f'{major_idx}_tmp', f'{minor_idx}_tmp']

    col_sing = [True]  # [True, ...] element is True if col has singleton

    while any(col_sing):
        col_sing = []
        for col in list(fe_data):
            # group-by fe column and generate mask for non-singletons
            not_sing = fe_data.groupby([col], observed=True)[col].count() > 1

            # eliminate singletons
            fe_data = fe_data[fe_data[col].isin(not_sing[not_sing].index)]

            # if this round has singletons, go another time around
            col_sing.append(not not_sing.all())

    return fe_data


def find_singletons(categorical_data: DataFrame,
                    fe: list) -> np.ndarray:
    """
    finds singletons in a dataframe of categories.

    identify_singletons from pyhdfe
    """

    ids = np.atleast_2d(
        pd.concat([categorical_data[c].cat.codes for c in fe], axis=1)
    )

    return identify_singletons([Groups(i) for i in ids.T])


# -------------------------- clusters ----------------------------------


def find_nested_fe(categorical_data: DataFrame,
                   fe: list,
                   cluster_names: list) -> list:
    """finds the fixed effects that are nested within cluster_names"""

    # temp rename the index: some cols may have the same name, groupby would not work
    entity_name, time_name = categorical_data.index.names
    categorical_data.index.names = [f'{entity_name}_tmp', f'{time_name}_tmp']

    nested_fe = []
    for f in fe:
        for c in cluster_names:
            if f == c:
                nested_fe.append(f)
            else:
                if categorical_data.groupby(f)[c].nunique().max() == 1:
                    nested_fe.append(f)

    # rename index with original name
    categorical_data.index.names = [entity_name, time_name]

    return nested_fe


# ------------------- fixed effects & clusters -------------------------

def get_categories(data: DataFrame,
                   cols_categories: list,
                   idx_categories: list = None
                   ) -> DataFrame:
    if idx_categories:
        return (data
                .index.to_frame()
                [idx_categories]
                .join(data[cols_categories])
                .astype('category')
                )
    else:
        return data[cols_categories].astype('category')


# ----------------------- post estimation ------------------------------


def estimate_absorbed_fixed_effects(mod, res) -> DataFrame:
    """
    extract estimate of absorbed fixed effects

    Parameters
    ----------
    mod
    res

    Returns
    -------

    """

    if isinstance(mod, AbsorbingLS):
        # can be done with res.estimated_effects
        estimated_fe = (mod.dependent.ndarray.flatten()
                        - mod.exog.pandas[res.params.index].to_numpy() @ res.params
                        - res.resids.to_numpy()
                        )
        estimated_fe = pd.DataFrame(estimated_fe,
                                    columns=['estimated_effects'],
                                    index=mod.dependent.pandas.index)
    else:  # PanelOLS
        estimated_fe = res.estimated_effects

    return estimated_fe


def estimated_effects(estimated_fe: DataFrame,
                      absorb: DataFrame,
                      drop_duplicates: bool = True,
                      original_dtype: bool = False) -> DataFrame:
    """
    aligns the estimated effects with the fixed effects columns

    for example if the estimated effects are for state and year
    the result will be a dataframe with state, year, estimated_effects

    if drop_duplicates is True unique values of state, year
    if original_dtype is True, the data of the fixed effects will have
    the original data type, for example if state was a string will be a
    string not a category

    """

    effects = (
        absorb
        .join(estimated_fe)
        .reset_index(drop=True)
    )

    if drop_duplicates:
        effects = effects.drop_duplicates(list(absorb)).reset_index(drop=True)

    if original_dtype:
        for c in list(absorb):
            effects[c] = effects[c].astype(effects[c].cat.categories.dtype)

    # if the est effects refer to an index var make that var the index
    absorbed_fe_idx = [c for c in absorb.columns if c in absorb.index.names]
    # absorbed_fe_cols = [c for c in absorb.columns if c not in absorbed_fe_idx]

    # if absorbed_fe_idx:
    #     effects.set_index(absorbed_fe_idx, inplace=True)

    absorbed_names = list(absorb)
    effects.set_index(absorbed_names, inplace=True)

    return effects


# ----------------------------- vcv ------------------------------------

# great reference: https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html

def sum_variance(vcv, w=None):
    """lincom of vcv"""
    vcv = np.array(vcv)
    w = w if w is not None else np.ones((vcv.shape[0], 1))
    return (w.T @ vcv @ w).item(0)


# functionality below is an adaptation of linearmodels'
# https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/covariance.html


def eps(y, x, params):
    """model residuals"""
    return y - x @ params


def s2(eps, nobs, extra_df) -> float:
    """error variance"""
    scale = nobs / (nobs - extra_df)
    return scale * ((eps.T @ eps) / nobs)


def cov(x, s2):
    """estimated covariance"""
    out = s2 * np.linalg.inv(x.T @ x)
    return (out + out.T) / 2


def cov_cluster(z, clusters):
    num_clusters = len(np.unique(clusters))

    sort_args = np.argsort(clusters)
    clusters = clusters[sort_args]
    locs = np.where(np.r_[True, clusters[:-1] != clusters[1:], True])[0]

    z = z[sort_args]
    n, k = z.shape

    s = np.zeros((k, k))

    for i in range(num_clusters):
        st, en = locs[i], locs[i + 1]
        z_bar = z[st:en].sum(axis=0)[:, None]
        s += z_bar @ z_bar.T

    s /= n
    return s


# calculate clustered vcv

def one_way_clustered_vcv(y: np.ndarray,  # residualized
                          X: np.ndarray,  # residualized
                          params,  # params from res of linear model
                          clusters: np.ndarray,
                          dof_a: int,
                          dof_m: int):
    """todo: twoway

    https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/covariance.html#ClusteredCovariance

    """

    clusters = np.array(clusters)
    nobs = X.shape[0]

    xpxi = np.linalg.inv(X.T @ X / nobs)

    epsilon = eps(y, X, params.to_numpy())
    xe = X * epsilon[:, None]

    xeex = cov_cluster(xe, clusters)

    # debias
    n_clusters = np.unique(clusters).shape[0]

    xeex *= (n_clusters / (n_clusters - 1))

    scale = (nobs - 1) / (nobs - dof_a - dof_m)  # as in stata reghdfe
    xeex *= scale

    out = (xpxi @ xeex @ xpxi) / nobs

    vcv = (out + out.T) / 2

    return pd.DataFrame(vcv, columns=params.index, index=params.index)


# -------------------------- report info -------------------------------

def get_t_stats(params: Series, std_errors: Series) -> Series:
    """Parameter t-statistics"""
    return Series(params / std_errors, name='t_stat')


def get_p_value(params: Series,
                std_errors: Series,
                t_stats: Series = None,
                resid: DataFrame = None,  # todo: resid, with debiased
                ) -> Series:
    """
    from: https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/results.html#PanelResults
    Parameter p-vals. Uses t(df_resid) if ``debiased`` is True, else normal
    """
    if t_stats is None:
        abs_tstats = np.abs(get_t_stats(params, std_errors))
    else:
        abs_tstats = np.abs(t_stats)

    if resid is not None:
        pv = 2 * (1 - stats.t.cdf(abs_tstats, resid))
    else:
        pv = 2 * (1 - stats.norm.cdf(abs_tstats))

    return Series(pv, index=params.index, name='p_value')


def get_conf_int(params: Series,
                 std_errors: Series,
                 resid: DataFrame = None,  # todo: resid, with debiased
                 alpha: float = 0.05) -> DataFrame:
    """
    confidence interval construction

    from:
    https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/results.html#PanelResults

    Notes
    -----
    Uses a t(df_resid) if ``debiased`` is True, else normal.
    """
    ci_quantiles = [alpha / 2, 1 - alpha / 2]

    if resid is not None:  # if self._debiased
        cvals = stats.t.ppf(ci_quantiles, resid)
    else:
        cvals = stats.norm.ppf(ci_quantiles)

    names = params.index
    params = np.asarray(params)[:, None]

    ci = params + np.asarray(std_errors)[:, None] * cvals[None, :]
    return DataFrame(ci, index=names, columns=['lower', 'upper'])
