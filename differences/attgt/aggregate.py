from __future__ import annotations

import numpy as np
from pandas import DataFrame

from collections import namedtuple
from scipy.sparse import hstack, csr_matrix, issparse

from ..attgt.utility import (filter_ntl,
                             aggregation_namedtuple,
                             get_cohorts_from_ntl,
                             get_att_from_ntl,
                             get_agg_attr_name)

from ..attgt.attgt_cal import get_standard_errors

__all__ = ['_AggregateGT']


class _AggregateGT:
    def __init__(self,
                 ntl: list[namedtuple],

                 weights: np.ndarray,
                 strata: list[float | int | str] = None,
                 cluster_groups: np.ndarray = None,

                 alpha: float = 0.05,

                 boot_iterations: int = 0,
                 random_state: int = None,

                 n_jobs_boot: int = 1,
                 backend_boot: str = 'loky',

                 ):

        self.ntl = ntl

        self.weights = weights
        self.strata = strata

        self.cohort_dummy_flag = True if self.strata is None else False

        self.cluster_groups = cluster_groups

        self.alpha = alpha
        self.boot_iterations = boot_iterations
        self.random_state = random_state

        self.n_jobs_boot = n_jobs_boot
        self.backend_boot = backend_boot

        self.cohort = None
        self.cohort_overall = None

    def _get_standard_errors(self, ntl: list[namedtuple]) -> list[namedtuple]:

        return get_standard_errors(
            ntl=ntl,
            cluster_groups=self.cluster_groups,
            alpha=self.alpha,
            boot_iterations=self.boot_iterations,
            random_state=self.random_state,
            backend_boot=self.backend_boot,
            n_jobs_boot=self.n_jobs_boot
        )

    def aggregate(self,
                  type_of_aggregation: str,
                  overall: bool
                  ) -> list[namedtuple]:

        if type_of_aggregation == 'cohort':

            if self.cohort is None:
                self.cohort = cohort_agg(
                    ntl=self.ntl,
                    weights=self.weights,
                    strata=self.strata,
                    cohort_dummy_flag=self.cohort_dummy_flag
                )

            if overall:
                self.cohort_overall = cohort_overall_agg(
                    ntl=self.ntl,
                    cohort_agg_ntl=self.cohort,
                    weights=self.weights,
                    strata=self.strata,
                    cohort_dummy_flag=self.cohort_dummy_flag
                )

                self.cohort_overall = self._get_standard_errors(
                    ntl=self.cohort_overall
                )

                return self.cohort_overall

            self.cohort = self._get_standard_errors(
                ntl=self.cohort
            )
            return self.cohort

        # other aggregations: simple, event, time

        attr_name = get_agg_attr_name(
            type_of_aggregation=type_of_aggregation,
            overall=overall
        )

        res_ntl = getattr(self, type_of_aggregation, None)

        if res_ntl is None:
            agg_func = globals().get(f"{type_of_aggregation}_agg")

            res_ntl = agg_func(
                ntl=self.ntl,
                weights=self.weights,
                cohort_dummy_flag=self.cohort_dummy_flag,
                strata=self.strata
            )

            setattr(self, type_of_aggregation, res_ntl)

        if overall and type_of_aggregation != 'simple':
            agg_overall_func = globals().get(f'{attr_name}_agg')

            res_overall_ntl = agg_overall_func(
                ntl=res_ntl,
                strata=self.strata
            )

            res_overall_ntl = self._get_standard_errors(
                ntl=res_overall_ntl
            )

            setattr(self, attr_name, res_overall_ntl)

            return res_overall_ntl

        res_ntl = self._get_standard_errors(
            ntl=res_ntl
        )
        setattr(self, type_of_aggregation, res_ntl)

        return res_ntl


# --------------------------- simple -----------------------------------

def simple_agg(ntl: list[namedtuple],
               weights: np.ndarray,
               cohort_dummy_flag: bool,
               strata: list[float | int | str] = None
               ) -> list[namedtuple]:
    """
    simple aggregation

    Parameters
    ----------
    ntl
        ntl of att gt

    weights

    cohort_dummy_flag
        whether to use the cohort dummy or the cohort_stratum dummy in the aggregation

    strata
        list of strata used

    Returns
    -------

    """

    ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    output = []
    for ntsl in ntll:
        d_name = getattr(ntsl[0], 'stratum', None)

        ntl_post = filter_ntl(
            ntl=ntsl,
            post=True
        )

        att, influence_func = get_att_and_if(
            ntl=ntl_post,
            weights=weights,
            cohort_dummy_flag=cohort_dummy_flag
        )

        output.append(
            aggregation_namedtuple(
                att,
                influence_func,
                type_of_aggregation='simple',
                stratum=d_name
            )
        )

    return output


# ------------------------ cohort aggregation ---------------------------

def cohort_agg(ntl: list[namedtuple],
               weights: np.ndarray,
               cohort_dummy_flag: bool,
               strata: list[float | int | str] = None
               ) -> list[namedtuple]:
    ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    output = []
    for ntsl in ntll:
        d_name = getattr(ntsl[0], 'stratum', None)

        post_ntl = filter_ntl(ntl=ntsl, post=True)

        # sorted unique cohorts in post event
        cohorts_list = sorted(set([c.cohort for c in post_ntl]))

        for c in cohorts_list:
            # results for specified cohort
            post_cohort_ntl = filter_ntl(
                ntl=post_ntl,
                cohort=c
            )

            # att: extract ATT from ntl, and average
            att = np.mean(get_att_from_ntl(post_cohort_ntl))

            shares = get_shares_from_ntl(
                ntl=post_cohort_ntl,
                weights=weights,
                cohort_dummy_flag=cohort_dummy_flag,
                repeated=True
            )

            s_ = shares / np.sum(shares)

            # influence function
            influence_func = get_agg_influence_func(
                ntl=post_cohort_ntl,
                weights_agg=s_
            )

            output.append(
                aggregation_namedtuple(
                    c,
                    att,
                    influence_func,
                    type_of_aggregation='cohort',
                    stratum=d_name
                )
            )

    return output


def cohort_overall_agg(ntl: list[namedtuple],  # need it for the cohort shares
                       cohort_agg_ntl: list,
                       weights: np.ndarray,
                       cohort_dummy_flag: bool,
                       strata: list[float | int | str] = None
                       ) -> list[namedtuple]:
    ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    cohort_agg_ntll = separate_strata_ntl(ntl=cohort_agg_ntl, strata=strata)

    output = []
    for idx, ntsl in enumerate(ntll):
        d_name = getattr(ntsl[0], 'stratum', None)

        post_ntl = filter_ntl(ntl=ntsl, post=True)

        # dictionary {cohort: share}
        shares, dummies = get_shares_from_ntl(
            ntl=post_ntl,
            cohort_dummy_flag=cohort_dummy_flag,
            repeated=False,
            return_dummies=True,
            weights=weights
        )

        s_ = shares / np.sum(shares)

        # att
        # att = np.sum(g_shares * g_att) / np.sum(g_shares)
        att = np.sum(s_ * [i.ATT for i in cohort_agg_ntll[idx]])

        _wif = get_wif(
            dummies=dummies,
            shares=shares
        )

        # influence function
        influence_func = get_agg_influence_func(
            ntl=cohort_agg_ntll[idx],
            weights_agg=s_,
            wif=_wif
        )

        output.append(
            aggregation_namedtuple(
                att,
                influence_func,
                type_of_aggregation='cohort',
                overall=True,
                stratum=d_name
            )
        )

    return output


# --------------------- event study aggregation ------------------------


def event_agg(ntl: list[namedtuple],
              weights: np.ndarray,
              cohort_dummy_flag: bool,
              strata: list[float | int | str] = None
              ) -> list[namedtuple]:
    ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    output = []
    for ntsl in ntll:
        d_name = getattr(ntsl[0], 'stratum', None)

        relative_time_list = sorted(
            set([nt.time - nt.cohort for nt in filter_ntl(ntsl)]))

        for rt in relative_time_list:
            # filter
            event_ntl = filter_ntl(
                ntl=ntsl,
                relative_period=rt
            )

            att, influence_func = get_att_and_if(
                ntl=event_ntl,
                weights=weights,
                cohort_dummy_flag=cohort_dummy_flag
            )

            output.append(
                aggregation_namedtuple(
                    rt,
                    att,
                    influence_func,
                    type_of_aggregation='event',
                    stratum=d_name
                )
            )

    return output


def event_overall_agg(ntl: list[namedtuple],  # event_agg_ntl
                      strata: list[float | int | str] = None,
                      ) -> list[namedtuple]:
    event_agg_ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    output = []
    for da_ntsl in event_agg_ntll:
        d_name = getattr(da_ntsl[0], 'stratum', None)

        post_da_ntsl = [i for i in da_ntsl if i.relative_period >= 0]

        att = np.mean([i.ATT for i in post_da_ntsl])

        weights_agg = np.repeat(1 / len(post_da_ntsl), len(post_da_ntsl))

        influence_func = get_agg_influence_func(
            ntl=post_da_ntsl,
            weights_agg=weights_agg
        )

        output.append(
            aggregation_namedtuple(
                att,
                influence_func,
                type_of_aggregation='event',
                overall=True,
                stratum=d_name
            )
        )
    return output


# ---------------------- time aggregation --------------------------


def time_agg(ntl: list,
             weights: np.ndarray,
             cohort_dummy_flag: bool,
             strata: list[float | int | str] = None
             ) -> list[namedtuple]:
    ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    output = []
    for ntsl in ntll:
        d_name = getattr(ntsl[0], 'stratum', None)

        # first treatment period
        min_cohort = min(get_cohorts_from_ntl(ntsl))

        # drop time periods where no one is treated yet
        times = sorted(
            set([nt.time for nt
                 in filter_ntl(ntsl, time_start_end=(min_cohort, None))])
        )

        for t in times:
            # fileter result: post treatment + time t
            time_ntl = filter_ntl(
                ntl=ntsl,
                time=t,
                post=True
            )

            att, influence_func = get_att_and_if(
                ntl=time_ntl,
                weights=weights,
                cohort_dummy_flag=cohort_dummy_flag
            )

            output.append(
                aggregation_namedtuple(
                    t,
                    att,
                    influence_func,
                    type_of_aggregation='time',
                    stratum=d_name
                )
            )

    return output


def time_overall_agg(ntl: list,  # time_agg_ntl
                     strata: list[float | int | str] = None,
                     ) -> list[namedtuple]:
    time_agg_ntll = separate_strata_ntl(ntl=ntl, strata=strata)

    output = []
    for ca_ntsl in time_agg_ntll:
        d_name = getattr(ca_ntsl[0], 'stratum', None)

        att = np.mean([i.ATT for i in ca_ntsl])

        weights_agg = np.repeat(1 / len(ca_ntsl), len(ca_ntsl))

        influence_func = get_agg_influence_func(
            ntl=ca_ntsl,
            weights_agg=weights_agg
        )

        output.append(
            aggregation_namedtuple(
                att,
                influence_func,
                type_of_aggregation='time',
                overall=True,
                stratum=d_name
            )
        )

    return output


# ----------------------------- helpers --------------------------------

def separate_strata_ntl(ntl: list[namedtuple],
                        strata: list[float | int | str] = None
                        ) -> list[list[namedtuple]]:
    if strata is None:
        ntll = [ntl]
    else:
        ntll = [[nt for nt in ntl if nt.stratum == d] for d in strata]

    return ntll


def get_att_and_if(ntl: list,
                   weights: np.ndarray,
                   cohort_dummy_flag: bool,
                   ) -> tuple:
    """
    main function to return the ATT and the influence function

    the output will depend on the level of aggregation chosen, the input
    list must differ depending on the level of aggregation.

    used for simple, event and time aggregations

    Parameters
    ----------
    ntl
        list[namedtuple], subset of the main result of list of ATTgt
    weights
    cohort_dummy_flag

    Returns
    -------

    """

    shares, dummies = get_shares_from_ntl(
        ntl=ntl,
        return_dummies=True,
        weights=weights,
        cohort_dummy_flag=cohort_dummy_flag
    )

    wif = get_wif(dummies=dummies, shares=shares)

    gxs_ = shares / np.sum(shares)

    # att
    att = np.sum(gxs_ * [i.ATT for i in ntl])

    # influence function
    influence_func = get_agg_influence_func(
        ntl=ntl,
        weights_agg=gxs_,
        wif=wif
    )

    return att, influence_func


# ----------------- aggregate influence function -----------------------


def get_agg_influence_func(ntl: list[namedtuple],
                           weights_agg: np.ndarray,
                           wif: np.ndarray = None):
    """
    get influence function for particular aggregate parameters

    Parameters
    ----------
    ntl
    weights_agg
    wif

    Returns
    -------

    """
    inf_funcs = [i.influence_func for i in ntl]

    if issparse(inf_funcs[0]):
        inf_funcs = hstack(inf_funcs)
        inf_funcs = inf_funcs.toarray()  # faster mboot if dense matrix
    else:
        inf_funcs = np.stack(inf_funcs, axis=1)

    inf_funcs = inf_funcs @ weights_agg

    if wif is not None:
        inf_funcs += wif @ np.array([i.ATT for i in ntl])

    return inf_funcs


# weights are added in get_cohort_shares_from_ntl
def get_wif(dummies: csr_matrix,
            shares: np.ndarray):
    if_1 = (dummies - shares) / np.sum(shares)

    if_2 = (np.sum((dummies - np.array(shares)), axis=1)[:, None] @
            np.array((shares / (np.sum(shares) ** 2)))[:, None].T)

    wif = if_1 - if_2

    return wif


# --------------------- user provided weights --------------------------

# weights: used in csdid but related to aggregations
# go in get_cohort_shares_from_ntl

def get_weights(data: DataFrame,
                weights_name: str,
                entity_level: bool = False):
    if not entity_level:
        return data[[weights_name]].to_numpy()

    # warning: if the weights vary within entity, here keeping only the first weight

    entity_name = data.index.names[0]
    return data.groupby(entity_name)[[weights_name]].first().to_numpy()


# warning: aggregations with weights in unbalanced panels give slightly different results than R


# ----------------------- shares & dummies -----------------------------

def get_shares_from_ntl(ntl: list[namedtuple],
                        weights: np.ndarray,
                        cohort_dummy_flag: bool = True,
                        repeated: bool = True,
                        return_dummies: bool = False,
                        ):
    dummies = get_dummies_from_ntl(
        ntl=ntl,
        repeated=repeated,
        cohort_dummy_flag=cohort_dummy_flag
    )

    dummies = dummies.toarray() * weights

    # cohort shares ordered as in the input ntl list
    shares = np.asarray(np.mean(dummies, axis=0)).flatten()

    if return_dummies:
        return shares, dummies

    return shares


def get_dummies_from_ntl(ntl: list[namedtuple],
                         repeated: bool = True,
                         cohort_dummy_flag: bool = True):
    """
    returns an array of cohort dummies

    if repeated is set to true the array will have d x t columns,
    ordered as the input ntl
    (which would be the resulting ntl from ATTgt)
    """
    if cohort_dummy_flag:
        if repeated:
            return hstack([i.cohort_dummy for i in ntl])

        return hstack([i.cohort_dummy for i in unique_dummies_in_ntl(ntl=ntl)])

    else:
        if repeated:
            return hstack([i.cohort_stratum_dummy for i in ntl])

        return hstack([i.cohort_stratum_dummy
                       for i in unique_dummies_in_ntl(ntl=ntl,
                                                      cohort_dummy_flag=cohort_dummy_flag)]
                      )


def unique_dummies_in_ntl(ntl: list[namedtuple],
                          cohort_dummy_flag: bool = True
                          ) -> list[namedtuple]:
    """
    keeps only one result namedtuple per cohort

    needed to get unique nt.cohort_dummy_flag,
    for the cohort shares and cohort dummies

    """
    output = []

    if cohort_dummy_flag:
        d = []
        for nt in ntl:
            if nt.cohort not in d:
                output.append(nt)
                d.append(nt.cohort)
    else:
        gx = []
        for nt in ntl:
            if (nt.cohort, nt.stratum) not in gx:
                output.append(nt)
                gx.append((nt.cohort, nt.stratum))
    return output
