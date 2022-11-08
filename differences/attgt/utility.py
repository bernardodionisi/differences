from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Index

from collections import namedtuple
from typing import Callable, NamedTuple

from warnings import warn

from scipy.stats import chi2
from scipy.sparse import hstack, issparse, csr_array


# ------------------ output namedtuples --------------------------------


class ATTgtResult(NamedTuple):
    cohort: int
    base_period: int
    time: int
    stratum: int | float | str | None

    anticipation: int
    control_group: str

    n_sample: int
    n_total: int

    cohort_dummy: csr_array | np.ndarray | None
    cohort_stratum_dummy: csr_array | np.ndarray | None
    stratum_dummy: csr_array | np.ndarray | None

    exception: Exception | None

    ATT: float | None
    influence_func: csr_array | np.ndarray | None

    std_error: float = None
    lower: float = None
    upper: float = None,
    boot_iterations: int = None


def aggregation_namedtuple(*args,
                           type_of_aggregation: str,
                           overall: bool = False,
                           stratum: str | int | float = None):
    """
    create a namedtuple for the aggregation results

    Parameters
    ----------
    args
        values for the att, inf_func, (std_error, lower, upper)
    type_of_aggregation
    overall
    stratum

    Returns
    -------
    """
    args = list(args)

    include_fields = [
        'ATT',
        'influence_func',
        'std_error',
        'lower',
        'upper',
        'boot_iterations'
    ]

    if stratum is not None:
        include_fields.insert(0, 'stratum')
        args.insert(0, stratum)

    missing = len(include_fields) - len(args)
    if missing:  # fill missing with NAs
        missing = 4  # std_error, lower, upper, boot_iterations
        args = *args, *[np.NaN] * missing

    if type_of_aggregation == 'simple' or overall:
        name = f'{type_of_aggregation.title()}Aggregation'
        if overall:
            name += 'Overall'

        nt = namedtuple(name, include_fields)
        return nt(*args)

    option_map = {
        'cohort': 'cohort',
        'time': 'time',
        'event': 'relative_period'
    }
    if stratum is not None:
        include_fields.insert(1, option_map[type_of_aggregation])
    else:
        include_fields.insert(0, option_map[type_of_aggregation])

    # need an additional field for cohort, relative period or time
    nt = namedtuple(f'{type_of_aggregation.title()}Aggregation', include_fields)

    return nt(*args)


def get_agg_attr_name(type_of_aggregation: str = None,
                      overall: bool = False) -> str:
    if (not overall and type_of_aggregation) or type_of_aggregation == 'simple':
        attr_name = type_of_aggregation
    elif overall and type_of_aggregation:
        attr_name = f'{type_of_aggregation}_overall'
    else:  # type_of_aggregation is None
        attr_name = 'att_gt'  # an attr of the _Difference class only

    return attr_name


# ------------------ filter cohort times dict ---------------------------

def filter_gt_dict(group_time: list[dict],
                   filter_gt: dict
                   ) -> list[dict]:
    """
    filters the cohorts-times-strata to run the estimation by

    this filter is applied before calculating the ATT, it is called in .fit()

    Parameters
    ----------
    group_time:
        dictionary of cohorts-times-strata
    filter_gt:
        user input filter as a dictionary

    Returns
    -------

    """

    for i in ['cohort', 'time']:
        gt = filter_gt.get(i)
        if gt is not None:
            if isinstance(gt, int):
                gt = [gt]
            group_time = [d for d in group_time if d[i] in gt]

        ct_start = filter_gt.get(f'{i}_start')
        if ct_start is not None:
            group_time = [d for d in group_time if d[i] >= ct_start]

        ct_end = filter_gt.get(f'{i}_end')
        if ct_end is not None:
            group_time = [d for d in group_time if d[i] <= ct_end]

    event = filter_gt.get('event')
    if event is not None:
        if isinstance(event, int):
            event = [event]

        group_time = [d for d in group_time
                      if d['time'] - d['cohort'] in event]

    event_start = filter_gt.get('event_start')
    if event_start is not None:
        group_time = [d for d in group_time
                      if d['time'] - d['cohort'] >= event_start]

    event_end = filter_gt.get('event_end')
    if event_end is not None:
        group_time = [d for d in group_time
                      if d['time'] - d['cohort'] <= event_end]

    return group_time


# ----------------- varying vs universal base period -------------------

def varying_base_period(cohort_ar: np.ndarray,
                        time_ar: np.ndarray,
                        anticipation: int,
                        ) -> list[dict]:
    """
    returns a list with all the cohort-times to be calculated

    using a varying base period
    """
    if anticipation < 0:
        raise ValueError("'anticipation' must be >= 0")

    cbt = []
    for cohort in cohort_ar:
        for t_idx, time in enumerate(time_ar[:-1]):

            bp_idx = t_idx

            # if time in post-treatment
            if time_ar[t_idx + 1] >= cohort:

                # fix base_period 'anticipation' times before treat
                fix_base_idx = np.argwhere(time_ar < (cohort - anticipation))

                if len(fix_base_idx) == 0:
                    break
                else:
                    bp_idx = fix_base_idx[-1, 0]

            cbt.append(
                {'cohort': int(cohort),
                 'base_period': time_ar[bp_idx],
                 'time': time_ar[t_idx + 1]}
            )

    return cbt


def universal_base_period(cohort_ar: np.ndarray,
                          time_ar: np.ndarray,
                          anticipation: int,
                          ) -> list[dict]:
    """
    returns a list of dict with all the cohort--base_period-times

    using a universal base period
    """
    if anticipation < 0:
        raise ValueError("'anticipation' must be >= 0")

    cbt = []
    for cohort in cohort_ar:
        for t_idx, time in enumerate(time_ar):

            try:
                bp_idx = np.argwhere(time_ar < (cohort - anticipation))[-1, 0]
            except IndexError:
                continue

            c, b, t = int(cohort), time_ar[bp_idx], time_ar[t_idx]

            cbt.append(
                {'cohort': c,
                 'base_period': b,
                 'time': t}
            )

    return cbt


# ------------------ filter the output namedtuples ---------------------


def filter_ntl(ntl: list[namedtuple],
               cohort: int | list[int] = None,
               time: int | list[int] = None,
               stratum: int | float | str | list[int | float | str] = None,
               relative_period: int | list[int] = None,

               cohort_start_end: tuple[int | None, int | None] = None,
               time_start_end: tuple[int | None, int | None] = None,
               stratum_start_end: tuple[int | float | None, int | float | None] = None,
               relative_period_start_end: tuple[int | None, int | None] = None,

               post: bool = False,
               pre: bool = False,
               non_zero_influence_func: bool = False
               ) -> list[namedtuple]:
    f = [nt for nt in ntl if ~np.isnan(nt.ATT)]  # only keep gt with non-missing att

    # filter by cohort

    if cohort is not None:
        if isinstance(cohort, list):
            f = [nt for nt in f if nt.cohort in cohort]
        else:
            f = [nt for nt in f if nt.cohort == cohort]

    if cohort_start_end is not None:

        cohort_start, cohort_end = cohort_start_end

        if cohort_start is not None:
            f = [nt for nt in f if nt.cohort >= cohort_start]

        if cohort_end is not None:
            f = [nt for nt in f if nt.cohort <= cohort_end]

    # filter by time

    if time is not None:
        if isinstance(time, list):
            f = [nt for nt in f if nt.time in time]
        else:
            f = [nt for nt in f if nt.time == time]

    if time_start_end is not None:

        time_start, time_end = time_start_end

        if time_start is not None:
            f = [nt for nt in f if nt.time >= time_start]

        if time_end is not None:
            f = [nt for nt in f if nt.time <= time_end]

    # filter by stratum

    if stratum is not None:
        if isinstance(stratum, list):
            f = [nt for nt in f if nt.stratum in stratum]
        else:
            f = [nt for nt in f if nt.stratum == stratum]

    if stratum_start_end is not None:
        stratum_start, stratum_end = stratum_start_end

        if stratum_start is not None:
            f = [nt for nt in f if nt.stratum >= stratum_start]

        if stratum_end is not None:
            f = [nt for nt in f if nt.stratum <= stratum_end]

    # filter by relative_period

    if relative_period is not None:
        if isinstance(relative_period, list):
            f = [nt for nt in f if nt.time - nt.cohort in relative_period]
        else:
            f = [nt for nt in f if nt.time - nt.cohort == relative_period]

    if relative_period_start_end is not None:
        relative_period_start, relative_period_end = relative_period_start_end

        if relative_period_start is not None:
            f = [nt for nt in f if nt.time - nt.cohort >= relative_period_start]

        if relative_period_end is not None:
            f = [nt for nt in f if nt.time - nt.cohort <= relative_period_end]

    if post:  # keep if post-event
        f = [nt for nt in f if nt.time >= nt.cohort]

    if pre:  # keep if pre-event
        f = [nt for nt in f if nt.time < nt.cohort]

    # filter by influence_func
    #   last condition because it's not a comprehension

    if non_zero_influence_func:
        non_zero_if = []
        for nt in f:
            if nt.influence_func is not None:
                if issparse(nt.influence_func):
                    if nt.influence_func.nnz:  # has at least 1 non 0 el
                        non_zero_if.append(nt)
                else:  # not sparse
                    if not np.all((nt.influence_func == 0)):
                        non_zero_if.append(nt)
        return non_zero_if

    return f


def get_att_from_ntl(ntl: list[namedtuple]) -> list[float]:
    return [nt.ATT for nt in ntl]


def get_if_from_ntl(ntl: list[namedtuple]) -> list:
    return [nt.influence_func for nt in ntl]


def get_cohorts_from_ntl(ntl: list[namedtuple]) -> list[int]:
    return [nt.cohort for nt in ntl]


# -------------------- inf funcs matrix --------------------------------

def stack_influence_funcs(ntl: list[namedtuple],
                          return_idx: bool = False):
    influence_func_is_sparse = False
    for r in ntl:
        if r.influence_func is not None:
            influence_func_is_sparse = issparse(r.influence_func)
            break

    if influence_func_is_sparse:
        inf_funcs = hstack([r.influence_func for r in ntl
                            if r.influence_func is not None])
        inf_funcs = inf_funcs.toarray()  # faster mboot if dense matrix
    else:
        inf_funcs = np.stack([r.influence_func for r in ntl
                              if r.influence_func is not None], axis=1)

    if return_idx:
        # indexes for the non-missing influence_func
        not_nan_idx = np.array([i for i, r in enumerate(ntl)
                                if r.influence_func is not None])

        return inf_funcs, not_nan_idx

    return inf_funcs


# ------------------------- idxs ---------------------------------------


def get_idx_of_values(main_list: list | Index,
                      list_of_values: list):
    return [idx for idx, v in enumerate(main_list) if v in list_of_values]


def _idx_to_arr(idx_array,
                length):
    arr = np.zeros(length)
    arr[idx_array] = 1
    return arr


def get_sparse_array(nrows: int,
                     ary: np.ndarray | int,
                     fill_idx: np.ndarray | list,
                     sparse_func: Callable):
    ary_sp = sparse_func((nrows, 1))
    ary_sp[fill_idx] = ary

    ary_sp = ary_sp.tocsr()

    return ary_sp


# ---------------------- standard errors -------------------------------


def get_std_errors_from_if(inf_funcs):
    n = inf_funcs.shape[0]
    vcv = (inf_funcs.T @ inf_funcs) / n

    if issparse(vcv):
        return np.sqrt(np.diag(vcv.toarray()) / n)

    return np.sqrt(np.diag(vcv) / n)


def get_vcv_from_if(inf_funcs, return_n_obs=False):
    n = inf_funcs.shape[0]
    vcv = (inf_funcs.T @ inf_funcs) / n

    if issparse(vcv):
        return vcv.toarray()

    if return_n_obs:
        return vcv, n

    return vcv


def get_std_errors_from_vcv(vcv, n):
    if issparse(vcv):
        return np.sqrt(np.diag(vcv.toarray()) / n)

    return np.sqrt(np.diag(vcv) / n)


def get_se_from_single_inf_func(inf_func):
    if issparse(inf_func):
        return np.sqrt(np.mean(inf_func.power(2)) / inf_func.shape[0])

    return np.sqrt(np.mean(inf_func ** 2) / len(inf_func))


# ------------------------ wald pre test -------------------------------

def wald_pre_test(ntl: list[namedtuple]):
    # pre-treatment periods results
    pre_ntl = filter_ntl(
        ntl=ntl,
        pre=True,
        non_zero_influence_func=True)

    # pseudo-att
    pre_att = np.array(get_att_from_ntl(pre_ntl))

    # vcv
    pre_vcv, n = get_vcv_from_if(
        inf_funcs=stack_influence_funcs(pre_ntl),
        return_n_obs=True)

    w = n * pre_att.T @ np.linalg.inv(pre_vcv) @ pre_att
    q = len(pre_ntl)  # restrictions

    # p_value for pre_test of parallel trends assumption
    w_p_value = 1 - chi2.cdf(w, q)

    return {'W': round(w, 6),
            'p_value': round(w_p_value, 6)}


# ------------------ pre processing args: est_method -------------------

est_method_map = {'dr': 'dr-mle', 'ipw': 'ipw-mle', 'std_ipw': 'std_ipw-mle'}
possible_est_methods = ['dr-mle', 'dr-ipt', 'ipw-mle', 'std_ipw-mle', 'reg']


def preprocess_est_method(est_method) -> tuple:
    if est_method in est_method_map.keys():
        est_method = est_method_map[est_method]

    if est_method not in possible_est_methods:
        raise ValueError(f"Invalid est_method. 'est_method' must be one of {possible_est_methods}")

    if est_method == 'reg':
        est_method_pscore = None
    else:
        est_method, est_method_pscore = est_method.split('-')

    return est_method, est_method_pscore


# ------------------ pre processing args: base_delta -------------------

# TODO: make sure Intercept is not excluded when needed
def preprocess_base_delta(base_delta: str | list | dict,
                          x_covariates: list,
                          time_varying_x: list,
                          intercept_name: str = 'Intercept') -> tuple:
    """
    extract x_base & x_delta from x_covariates

    Examples
    --------

    x_covariates = ['Intercept', 'a', 'b', 'c_tv', 'd_tv']
    time_varying_x = ['c_tv', 'd_tv']

    preprocess_base_delta('base', x_covariates, time_varying_x)
    preprocess_base_delta('delta', x_covariates, time_varying_x)
    preprocess_base_delta(['delta'], x_covariates, time_varying_x)

    x_covariates, x_base, x_delta = preprocess_base_delta(
        base_delta={'base': ['d_tv', 'b'], 'delta': ['c_tv', 'a']},
        x_covariates=x_covariates,
        time_varying_x=['c_tv', 'd_tv'])

    Parameters
    ----------
    base_delta
    x_covariates
    time_varying_x
    intercept_name

    Returns
    -------

    """
    has_intercept = bool([i for i in x_covariates if i == intercept_name])

    x_covariates = [i for i in x_covariates if i != intercept_name]

    if isinstance(base_delta, str) or isinstance(base_delta, list):
        if 'base' in base_delta and 'delta' in base_delta:
            # all the covariates used as base and all the time varying covariates used as delta
            x_base, x_delta = x_covariates, time_varying_x

        elif 'base' in base_delta:
            x_base, x_delta = x_covariates, []

        else:  # delta only
            x_base, x_delta = [], time_varying_x

            dropped_cols = [i for i in x_covariates
                            if i not in time_varying_x]
            if dropped_cols:
                warn(f'if base_delta = delta only time varying covariates will be used: '
                     f'{dropped_cols}, which were generated by x_formula, were ignored')

    else:  # if a dictionary is passed

        x_base = base_delta.get('base')
        x_delta = base_delta.get('delta')

        if x_base and x_delta is None:
            # use only the requested base + all the time varying as delta

            x_delta = time_varying_x

            missing_cols = [i for i in x_covariates if i not in x_base + x_delta]

        elif x_delta and x_base is None:
            # use only the requested delta + all the covariates as base

            x_base = x_covariates

            missing_cols = []

        else:  # both base and delta
            non_tv_x_delta = [i for i in x_delta if i not in time_varying_x]

            if non_tv_x_delta:
                warn(f'when specifying base_delta the list for "delta" was restricted and '
                     f'{non_tv_x_delta}, which were generated by x_formula, '
                     f'were ignored because non-time-varying')

            x_delta = [i for i in x_delta if i in time_varying_x]

            missing_cols = [i for i in x_covariates if i not in x_base + x_delta]

        if missing_cols:
            warn(f'when specifying base_delta the analysis was restricted '
                 f'to the requested columns and {missing_cols}, '
                 f'which were generated by x_formula, have been ignored')

        non_existing_cols = [i for i in x_base + x_delta if i not in x_covariates]
        if non_existing_cols:
            raise ValueError(f'{non_existing_cols} missing from data matrix')

    if has_intercept:
        x_base = [intercept_name] + x_base
        x_covariates = [intercept_name] + x_covariates

    # new covariates may be less than initial covariates
    x_covariates = [i for i in x_covariates if i in set(x_base + x_delta)]

    return x_covariates, x_base, x_delta


# ------------------ pre processing args: cluster_var ------------------


def preprocess_fit_cluster_arg(cluster_var: list | str,
                               entity_name: str,
                               true_rc: bool) -> tuple[str | None, bool]:
    """determines the variable t cluster on when cluster is requested from fit (ATTgt)"""

    cluster_by_entity = True  # default when cluster_var=None

    if cluster_var is not None:

        if not cluster_var:  # don't cluster by entity if empty list or any falsy (but None)
            cluster_by_entity = False

        else:
            if isinstance(cluster_var, str):
                cluster_var = [cluster_var]

            if entity_name in cluster_var:
                cluster_var = [c for c in cluster_var if c != entity_name]

            if cluster_var:  # there is an extra cluster_var
                if len(cluster_var) > 1:
                    raise ValueError(f'only 2 cluster_var allowed where one is {entity_name}')
                cluster_var = cluster_var[0]  # turn into a sting

            else:
                cluster_var = None

    if true_rc:
        return cluster_var, False

    return cluster_var, cluster_by_entity


def preprocess_aggregate_cluster_arg(cluster_var: list | str,
                                     entity_name: str,
                                     cluster_by_entity: bool) -> str | None:
    """determines the variable t cluster on when cluster is requested from aggregate"""
    if cluster_var is not None:
        if isinstance(cluster_var, str):
            cluster_var = [cluster_var]

        if entity_name in cluster_var:
            if not cluster_by_entity:
                # should change this and allow clustering by entity in aggregate
                # even if no clustering in fit
                raise ValueError('cluster_var: should cluster by entity in .fit() '
                                 'in order to cluster by entity in .aggregate()')
            cluster_var = [c for c in cluster_var if c != entity_name]

        if cluster_var:  # there is an extra cluster_var
            if len(cluster_var) > 1:
                raise ValueError(f'only 2 cluster_var allowed where one is {entity_name}')
            cluster_var = cluster_var[0]  # turn into a sting

        else:
            cluster_var = None

    return cluster_var


# ------------------------ extract results -----------------------------

def _extract_dict_strata_ntl(result_dict: dict,
                             attr_name: str,
                             strata: list,
                             items: bool = False):
    if attr_name == 'att_gt':

        out_dict = {
            g: [nt for nt
                in result_dict['full_sample']['ATTgt_ntl']
                if nt.stratum == g]
            for g in strata
        }

    else:

        out_dict = {
            g: [nt for nt
                in getattr(result_dict['full_sample']['aggregate_inst'], attr_name)
                if nt.stratum == g]
            for g in strata
        }

    if items:
        return list(out_dict.items())  # change with abc

    return out_dict


def _extract_dict_samples_ntl(result_dict: dict,
                              attr_name: str,
                              sample_names: list,
                              items: bool = False):
    if attr_name == 'att_gt':

        out_dict = {
            s: result_dict[s]['ATTgt_ntl']
            for s in sample_names
        }

    else:

        out_dict = {
            s: getattr(result_dict[s]['aggregate_inst'], attr_name)
            for s in sample_names
        }

    if items:
        return list(out_dict.items())

    return out_dict


def _extract_dict_sg_ntl(result_dict: dict,
                         attr_name: str,
                         sample_names: list,
                         strata: list,
                         dict_keys: str = None,
                         items: bool = False):
    if dict_keys == 's':

        if attr_name == 'att_gt':

            out_dict = {
                s: {
                    g: [nt for nt
                        in result_dict[s]['ATTgt_ntl']
                        if nt.stratum == g]
                    for g in strata
                }
                for s in sample_names
            }

        else:
            out_dict = {
                s: {
                    g: [nt for nt
                        in getattr(result_dict[s]['aggregate_inst'], attr_name)
                        if nt.stratum == g]
                    for g in strata
                }
                for s in sample_names
            }

    elif dict_keys == 'g':

        if attr_name == 'att_gt':
            out_dict = {
                g: {
                    s: [nt for nt
                        in result_dict[s]['ATTgt_ntl']
                        if nt.stratum == g]
                    for s in sample_names
                }
                for g in strata
            }
        else:
            out_dict = {
                g: {
                    s: [nt for nt
                        in getattr(result_dict[s]['aggregate_inst'], attr_name)
                        if nt.stratum == g]
                    for s in sample_names
                }
                for g in strata
            }
    else:
        if attr_name == 'att_gt':

            out_dict = {
                (s, g): [nt for nt
                         in result_dict[s]['ATTgt_ntl']
                         if nt.stratum == g]
                for g in strata
                for s in sample_names
            }
        else:
            out_dict = {
                (s, g): [nt for nt
                         in getattr(result_dict[s]['aggregate_inst'], attr_name)
                         if nt.stratum == g]
                for g in strata
                for s in sample_names
            }

    if items and dict_keys:
        out_dict = {k: list(v.items()) for k, v in out_dict.items()}

    return out_dict


def extract_dict_ntl(result_dict: dict,
                     type_of_aggregation: str = None,
                     overall: bool = False,
                     sample_names: list = None,
                     strata: list = None,
                     items: bool = False,
                     dict_keys: str = None,
                     ) -> dict:
    """extract sample/stratum dictionaries"""
    if not sample_names:
        sample_names = list(result_dict.keys())

    s_0 = sample_names[0]

    if sample_names == ['full_sample']:
        sample_names = None

    attr_name = get_agg_attr_name(
        type_of_aggregation=type_of_aggregation,
        overall=overall
    )

    if type_of_aggregation is not None:
        if getattr(result_dict[s_0]['aggregate_inst'], attr_name) is None:
            raise AttributeError(f'{attr_name} unavailable')

    if sample_names and strata:

        out_dict = _extract_dict_sg_ntl(
            result_dict=result_dict,
            attr_name=attr_name,
            sample_names=sample_names,
            strata=strata,
            items=items,
            dict_keys=dict_keys
        )

    elif sample_names and not strata:

        out_dict = _extract_dict_samples_ntl(
            result_dict=result_dict,
            attr_name=attr_name,
            sample_names=sample_names,
            items=items
        )

    elif not sample_names and strata:

        out_dict = _extract_dict_strata_ntl(
            result_dict=result_dict,
            attr_name=attr_name,
            strata=strata,
            items=items
        )

    else:  # no sample_names and no strata

        # return ATTct base: only full sample & no treatment strata

        if attr_name == 'att_gt':
            out_dict = {
                'full_sample': result_dict['full_sample']['ATTgt_ntl']
            }
        else:
            out_dict = {
                'full_sample': getattr(result_dict['full_sample']['aggregate_inst'], attr_name)
            }

    return out_dict


def extract_dict_ntl_for_difference(result_dict: dict,
                                    type_of_aggregation: str = None,
                                    overall: bool = False,

                                    difference_strata: list = None,
                                    iterate_samples: list = None,

                                    difference_samples: list = None,
                                    iterate_strata: list = None
                                    ) -> list | dict[str, list]:
    """
    extract results to be subtracted

    either
        difference_strata, iterate_samples
    or
        difference_samples, iterate_strata

    iterate_samples & iterate_strata can be empty [],
    while difference_strata and difference_samples, must contain 2 elements

    Parameters
    ----------
    result_dict
    type_of_aggregation
    overall
    difference_strata
    iterate_samples
    difference_samples
    iterate_strata

    Returns
    -------
    if no iteration needed:
        a list of 2 tuples

    if iteration needed:
        a dictionary of {'stratum or sample name': list[2 tuples]}

    """

    if difference_samples:
        if len(difference_samples) != 2:
            raise ValueError('len(difference_samples) must be == 2')

        out_dict = extract_dict_ntl(
            result_dict=result_dict,
            type_of_aggregation=type_of_aggregation,
            overall=overall,
            dict_keys='g',
            sample_names=difference_samples,
            strata=iterate_strata,  # can be empty
            items=True
        )

        # out_dict is a list of 2 tuples.
        # each tuple 1st element is the sample name and the 2nd elements the ntl

    elif difference_strata:
        if len(difference_strata) != 2:
            raise ValueError('len(difference_strata) must be == 2')

        out_dict = extract_dict_ntl(
            result_dict=result_dict,
            type_of_aggregation=type_of_aggregation,
            overall=overall,
            dict_keys='s',
            sample_names=iterate_samples,
            strata=difference_strata,  # can be empty
            items=True
        )

    else:
        raise ValueError('must contain either difference_samples or difference_strata')

    return out_dict


def list_fs_ntl(x: list[tuple] | dict):
    """helper to get the ntl from list or dict, where ntl is the second el of each tuple
    [('a', [1, 2]), ('b', [4, 5])]

    returns:     [ ( [1, 2], [4, 5] ) ]

    {'k1': [('a', [1, 2]), ('b', [4, 5])],
    'k2': [('a', [11, 12]), ('b', [14, 15])]}

    returns:     [ ( [1, 2], [4, 5] ), ( [11, 12], [14, 15] ) ]

    """

    if isinstance(x, list):
        return [(t[1] for t in x)]
    else:  # dict
        return [(t[1] for t in v) for v in x.values()]


# ------------------------ output dataframes ---------------------------


def output_dict_to_dataframe(result: dict[str, list[namedtuple]],
                             stratum: bool = False,
                             date_map: dict | None = None,
                             add_info: bool = False
                             ) -> DataFrame:
    output = []
    for k, ntl in result.items():

        out_df, name_table, se_info, conf_info = ntl_to_dataframe(ntl=ntl, stratum=stratum)

        if k != 'full_sample':
            if isinstance(k, tuple):
                k = k[0]  # sample is in the first element of tuple: see extract_dict_ntl

            out_df.insert(loc=0, column='sample_name', value=k)

        output.append(out_df)

    output = pd.concat(output, axis=0).reset_index(drop=True)

    if add_info:
        post = np.where(output['cohort'] < output['time'], 'post', 'pre')
        output.insert(0, 'post', post)

    output = replace_dates(out_df=output, date_map=date_map)

    output = out_df_index(
        out_df=output,
        name_table=name_table,
        se_info=se_info,
        conf_info=conf_info
    )
    return output


def replace_dates(out_df: DataFrame, date_map: dict | None = None):
    if date_map is not None:
        for i in ['cohort', 'base_period', 'time']:
            if i not in list(out_df):
                continue
            out_df[i] = out_df[i].map(date_map)
    return out_df


def out_df_index(out_df, name_table, se_info, conf_info):
    # row index

    col_idx = [
        'sample_name',
        'stratum',
        'difference_between',
        'cohort',
        'base_period',
        'time',
        'relative_period'
    ]

    col_idx = [i for i in col_idx if i in list(out_df)]

    if col_idx:
        out_df.set_index([i for i in col_idx if i in list(out_df)], inplace=True)

    # column index

    new_cols = list(out_df)
    name_table_cols = [name_table] * len(new_cols)
    std_error_cols = [''] * len(new_cols)

    std_error_cols[new_cols.index('std_error')] = se_info
    std_error_cols[new_cols.index('lower'):] = [conf_info] * 3

    out_df.columns = pd.MultiIndex.from_tuples(
        zip(name_table_cols, std_error_cols, new_cols))

    return out_df


def ntl_to_dataframe(ntl: list,
                     name_table: str = None,
                     stratum: bool = False
                     ):
    """(results to a dataframe) for nice output"""

    if name_table is None:
        name_table = type(ntl[0]).__name__

    boot_iterations = getattr(ntl[0], 'boot_iterations')

    if boot_iterations:
        se_info, conf_info = 'bootstrap', 'simult. conf. band'
    else:
        se_info, conf_info = 'analytic', 'pointwise conf. band'

    exclude_fields = [
        # 'base_period',
        'stratum',
        'influence_func',
        'anticipation',
        'control_group',
        'n_sample',
        'n_total',

        'cohort_dummy',
        'cohort_stratum_dummy',
        'stratum_dummy',

        'exception',
        'boot_iterations'
    ]

    include_fields = [f for f in ntl[0]._fields if f not in exclude_fields]

    if stratum:
        include_fields.insert(0, 'stratum')

    values = map(lambda x: (getattr(x, v) for v in include_fields), ntl)
    out_df = DataFrame(values, columns=include_fields)

    out_df = fix_std_error_cols(out_df=out_df)

    return out_df, name_table, se_info, conf_info


def fix_std_error_cols(out_df: DataFrame):
    out_df['std_error'] = np.where(
        (out_df['std_error'] == 0), np.nan, out_df['std_error'])

    out_df['lower'] = np.where(
        (out_df['lower'] == 0), np.nan, out_df['lower'])

    out_df['upper'] = np.where(
        (out_df['upper'] == 0), np.nan, out_df['upper'])

    out_df['zero_not_in_cband'] = np.where(
        np.sign(out_df['lower']) * np.sign(out_df['upper']) > 0, '*', '')

    return out_df
