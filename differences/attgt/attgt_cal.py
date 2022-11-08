from __future__ import annotations

import numpy as np
from pandas import DataFrame, Index

from scipy.stats import norm
from scipy.sparse import csr_array, lil_array

from collections import namedtuple
from typing import Callable

from tqdm import tqdm

from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import get_reusable_executor

from ..tools.utility import tqdm_joblib
from ..tools.panel_utility import panel_2_cross_section_diff

from ..attgt.mboot import mboot
from ..attgt.utility import (get_std_errors_from_if,
                             get_sparse_array,
                             stack_influence_funcs,
                             ATTgtResult)


# todo: FIX important, for unbalanced panels the cohort dummy and the rest has, when cluster var
# ------------ att and influence function for a single ct --------------

def did_single_gt(data: DataFrame,
                  entities: Index,
                  entity_name: str,
                  n_total: int,

                  y_name: str,

                  cohort_name: str,
                  strata_name: str,

                  weights_name: str,

                  x_covariates: list,
                  x_base: list,
                  x_delta: list,

                  anticipation: int,
                  control_group: str,

                  is_panel: bool,
                  is_balanced_panel: bool,
                  cluster_by_entity: bool,

                  att_ct_func: Callable,

                  # will be the iterable to iterate over
                  cohort: int,
                  base_period: int,
                  time: int,
                  stratum: str | float | int = None

                  ) -> namedtuple:
    """
    computes the ATT for a single cohort-time

    main function to compute ATT
    both for panel and repeated cross-sections

    it is called by joblib to be run in parallel for each cohort-time

    --------------------------------------------------------------------

    idx of the treated observations

    - balanced: need to make cross-section first
    - unbalanced: ? either keep data as is or the idx of the unique entities treated
    - repeated cross-section: data as is

    """
    cohort_stratum_dummy, stratum_dummy = None, None

    # ----------------- 0 ATT for reference period ---------------------

    if base_period == time:  # and 'universal'

        # if a stratum (for example intensity of treat) then treated are subset of cohort
        # stratum name should not vary within entity
        treat_info_list = [cohort_name,
                           strata_name] if strata_name is not None else cohort_name

        # cluster_by_entity should be False for true rc (would be redundant)
        if is_panel or cluster_by_entity:
            data = data.groupby(entity_name)[treat_info_list].first().reset_index()

        cohort_dummy = csr_array((data[cohort_name] == cohort).astype(int).to_numpy()[:, None])

        if strata_name is not None:
            cohort_stratum_mask = (data[cohort_name] == cohort) & (
                    data[strata_name] == stratum)

            cohort_stratum_dummy = csr_array(cohort_stratum_mask.astype(int).to_numpy()[:, None])
            stratum_dummy = csr_array(
                (data[strata_name] == stratum).astype(int).to_numpy()[:, None])

        return ATTgtResult(
            cohort=cohort,
            base_period=base_period,
            time=time,
            stratum=stratum,

            anticipation=anticipation,
            control_group=control_group,

            n_sample=np.nan,
            n_total=n_total,

            cohort_dummy=cohort_dummy,
            cohort_stratum_dummy=cohort_stratum_dummy,
            stratum_dummy=stratum_dummy,

            exception=None,

            ATT=0,
            influence_func=csr_array((n_total, 1))
        )

    # --------------------- treated / control --------------------------

    if strata_name is None:  # classic ct
        mask_treated = (data[cohort_name] == cohort).values
    else:
        mask_treated = ((data[cohort_name] == cohort) & (data[strata_name] == stratum)).values

    mask_control = (data[cohort_name].isnull()).values  # never treated

    # not yet treated
    if control_group == 'not_yet_treated':
        maxt_ = max(time, base_period)

        mask_control = mask_control | ((~mask_treated)
                                       & (data[cohort_name] > maxt_ + anticipation).values
                                       )

    # ---------------- pre / post (base period / time) -----------------

    times_values = data.index.get_level_values(1)
    mask_pre_post = (times_values == base_period) | (times_values == time)  # pre or post

    # must be treated or control & must be observed pre or post
    mask_pre_post = (mask_treated | mask_control) & mask_pre_post

    # ------------------------------------------------------------------

    # cluster_by_entity should be False for true rc (would be redundant)
    if not is_balanced_panel and cluster_by_entity:  # panels (as rc) but entity level if

        # cohort_dummy needs to be defined before masking for unbalanced panels,
        # some treated may be dropped when masked because they may not have pre or post,
        # or both pre-post (in case balance_2x2)

        treated_entities = data[mask_treated].index.get_level_values(0).unique()
        mask_treated_entities = np.isin(entities, treated_entities)

        # entity level - dummy for the treated entities: need this for the aggregations
        if strata_name is None:
            cohort_dummy = csr_array(mask_treated_entities.astype(int)[:, None])
        else:
            cohort_dummy, stratum_dummy = get_cohort_stratum_dummies(
                data=data,
                entities=entities,
                cohort_name=cohort_name,
                cohort=cohort,
                strata_name=strata_name,
                stratum=stratum)

            cohort_stratum_dummy = csr_array(mask_treated_entities.astype(int)[:, None])

    # ------------------------------------------------------------------

    data = (data
            [mask_pre_post]
            .assign(_treated=lambda x: mask_treated[mask_pre_post].astype(int),
                    _control=lambda x: mask_control[mask_pre_post].astype(int))
            )

    if is_panel and not is_balanced_panel:  # unbalanced panel: balance it 2x2
        mask_balance_2x2 = data.index.get_level_values(0).duplicated(keep=False)
        n_dropped_entities = np.sum(~mask_balance_2x2)
        data = data[mask_balance_2x2].copy()

    # ------------------------------------------------------------------

    # Q: shouldn't n_sample, for unbalanced (as rc): in elif cluster_by_entity, be sample_entities?
    # sample_entities = data.index.get_level_values(0).nunique()
    # given that n_totals is number of entities?

    n_sample = len(data)  # unbalanced (as rc) + repeated cross-section

    if is_panel:  # originally balanced panel + unbalanced-balance_2x2

        # automatically clustered by entity
        treated_entities = data[lambda x: x['_treated'] == 1].index.get_level_values(0).unique()
        mask_treated_entities = np.isin(entities, treated_entities)

        if is_balanced_panel:
            # dummy for the treated entities (obs level): need this for the aggregations
            if strata_name is None:
                cohort_dummy = csr_array(mask_treated_entities.astype(int)[:, None])
            else:
                cohort_dummy, stratum_dummy = get_cohort_stratum_dummies(
                    data=data,
                    entities=entities,
                    cohort_name=cohort_name,
                    cohort=cohort,
                    strata_name=strata_name,
                    stratum=stratum)

                cohort_stratum_dummy = csr_array(mask_treated_entities.astype(int)[:, None])

        control_entities = data[lambda x: x['_control'] == 1].index.get_level_values(0).unique()
        mask_control_entities = np.isin(entities, control_entities)

        tc_bool = np.array(mask_treated_entities | mask_control_entities)

        sample_idx = np.argwhere(tc_bool).flatten()  # to populate influence_func
        n_sample = np.sum(tc_bool)  # overwrites the n_sample = len(data) above

        data = panel_2_cross_section_diff(
            data=data,
            y_name=y_name,
            x_base=x_base,
            x_delta=x_delta,
            base_period=base_period,
            time=time,
        )

    else:
        data['post'] = (data.index.get_level_values(1) == time).astype(int)  # post dummy

        if cluster_by_entity:  # panel (as rc) but entity level if

            # entity level - to populate influence_func
            sample_entities = data.index.get_level_values(0)
            sample_idx = np.argwhere(np.isin(entities, sample_entities)).flatten()
            # sample_idx = [idx for idx, e in enumerate(entities) if e in sample_entities]

            # obs level - generate a temp-id (start at 0) to cohort the if by entity
            entity_strata = data.groupby(entity_name).ngroup().to_numpy()

        else:  # repeated cross-section (could be panel as rc)
            # -- all at obs level --

            sample_idx = np.argwhere(mask_pre_post).flatten()  # to populate influence_func

            if strata_name is None:
                cohort_dummy = csr_array(mask_treated.astype(int)[:, None])  # obs level (rc/as rc)
            else:
                # obs level (rc/as rc)

                cohort_dummy, stratum_dummy = get_cohort_stratum_dummies(
                    data=data,
                    entities=entities,
                    cohort_name=cohort_name,
                    cohort=cohort,
                    strata_name=strata_name,
                    stratum=stratum,
                    repeated_cross_section=True
                )

                # cohort_stratum_dummy as cohort_dummy when strata_name is not None
                cohort_stratum_dummy = csr_array(mask_treated.astype(int)[:, None])

    # ---------------------- compute att_ct ----------------------------

    try:

        drdid_res = att_ct_func(
            endog=data[y_name].to_numpy(),
            exog=data[x_covariates].to_numpy(),
            treated=data['_treated'].to_numpy(),
            weights=data[weights_name].to_numpy(),
            post=data['post'].to_numpy() if not is_panel else None
        )

    except Exception as ex:
        return ATTgtResult(
            cohort=cohort,
            base_period=base_period,
            time=time,
            stratum=stratum,

            anticipation=anticipation,
            control_group=control_group,

            n_sample=n_sample,
            n_total=n_total,

            cohort_dummy=None,
            cohort_stratum_dummy=None,
            stratum_dummy=None,

            exception=ex,

            ATT=np.NaN,  # att, need nan to do some operations
            influence_func=None
        )

    # ----------------------- influence function -----------------------

    influence_func = drdid_res['influence_func'].flatten()

    influence_func *= (n_total / n_sample)

    if not is_panel and cluster_by_entity:
        influence_func = np.bincount(entity_strata, weights=influence_func)

    influence_func = get_sparse_array(
        nrows=n_total,
        ary=influence_func,
        fill_idx=sample_idx,
        sparse_func=lil_array)

    return ATTgtResult(
        cohort=cohort,
        base_period=base_period,
        time=time,
        stratum=stratum,

        anticipation=anticipation,
        control_group=control_group,

        n_sample=n_sample,
        n_total=n_total,

        cohort_dummy=cohort_dummy,
        cohort_stratum_dummy=cohort_stratum_dummy,
        stratum_dummy=stratum_dummy,

        exception=None,

        ATT=drdid_res['att'],
        influence_func=influence_func
    )


# --------------------- ATT & standard errors --------------------------


def get_att_gt(group_time: list[dict],

               data: DataFrame,

               y_name: str,

               cohort_name: str,

               is_panel: bool,
               is_balanced_panel: bool,
               cluster_by_entity: bool,

               x_covariates: list,
               x_base: list,
               x_delta: list,

               strata_name: str = None,
               weights_name: str = None,

               control_group: str = 'never_treated',
               anticipation: int = 0,

               att_function_ct: Callable = None,

               n_jobs_ct: int = 1,
               backend_ct: str = 'loky',

               progress_bar: bool = True,
               sample_name: str = None,
               release_workers: bool = True,

               ) -> list[namedtuple]:
    if control_group not in ['never_treated', 'not_yet_treated']:
        raise ValueError("'control_group' must be either set to either"
                         "'never_treated' or 'not_yet_treated'")

    entities = data.index.get_level_values(0).unique()
    entity_name = data.index.names[0]

    # is_panel is False if the user requested as_repeated_cross_section
    n_total = entities.nunique() if is_panel or cluster_by_entity else len(data)  # total n obs

    # needed for the progress bar
    jobs = cpu_count() + 1 - n_jobs_ct if n_jobs_ct < 0 else n_jobs_ct
    sample_name = f'for {sample_name} ' if sample_name else ''

    with tqdm_joblib(
            tqdm(disable=not progress_bar,
                 desc=f'Computing ATTgt {sample_name}[workers={jobs}]',
                 total=len(group_time),
                 bar_format='{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}'
                 )
    ):

        # compute the ct att. order of parameters must conform to did_single_gt()
        res_ntl = Parallel(n_jobs=n_jobs_ct,
                           backend=backend_ct,
                           )(
            delayed(
                did_single_gt)(

                data,
                entities,
                entity_name,
                n_total,

                y_name,
                cohort_name,
                strata_name,

                weights_name,

                x_covariates,
                x_base,
                x_delta,

                anticipation,
                control_group,

                is_panel,
                is_balanced_panel,
                cluster_by_entity,

                att_function_ct,

                **gt
            )
            for gt in group_time
        )

    if release_workers:
        get_reusable_executor().shutdown(wait=True)

    return res_ntl


def get_standard_errors(ntl: list[namedtuple],

                        cluster_groups: np.ndarray = None,

                        alpha: float = 0.05,
                        boot_iterations: int = 0,
                        random_state: int = None,

                        backend_boot: str = 'loky',
                        n_jobs_boot: int = -1,

                        progress_bar: bool = True,
                        sample_name: str = None,
                        release_workers: bool = True
                        ) -> list[namedtuple]:
    if boot_iterations < 0:
        raise ValueError("'boot_iterations' must be >= 0. "
                         "If boot_iterations=0, analytic standard errors are computed")

    # influence funcs + idx for not nan cols
    inf_funcs, not_nan_idx = stack_influence_funcs(ntl, return_idx=True)

    # create an empty array to populate with the standard errors
    se_array = np.empty(len(ntl))
    se_array[:] = np.NaN

    if boot_iterations:

        # get the standard errors
        mboot_res = mboot(
            inf_funcs=inf_funcs,
            cluster_groups=cluster_groups,
            alpha=alpha,
            boot_iterations=boot_iterations,
            random_state=random_state,

            boot_backend=backend_boot,
            boot_n_jobs=n_jobs_boot,

            progress_bar=progress_bar,
            sample_name=sample_name,
            release_workers=release_workers
        )

        # populate empty se array
        se_array[not_nan_idx] = mboot_res['se']
        cval = mboot_res['crit_val']

    else:  # analytic standard errors

        # analytic standard errors
        analytic_se = get_std_errors_from_if(inf_funcs)

        # populate empty se array
        se_array[not_nan_idx] = analytic_se

        cval = norm.ppf(1 - alpha / 2)

    # add standard errors & cband to namedtuples
    out = [nt._replace(

        std_error=se_array[nt_idx],
        lower=nt.ATT - se_array[nt_idx] * cval,
        upper=nt.ATT + se_array[nt_idx] * cval,
        boot_iterations=boot_iterations
    )
        for nt_idx, nt in enumerate(ntl)]

    return out


def get_cohort_stratum_dummies(data: DataFrame,
                               entities: Index | list,
                               cohort_name: str,
                               cohort: int,
                               strata_name: str,
                               stratum: str | int | float,
                               repeated_cross_section: bool = False
                               ) -> tuple:
    """helper for did_single_gt"""
    if repeated_cross_section:

        # ------------------- cohort dummy ------------------------------

        cohort_dummy = csr_array((data[cohort_name] == cohort).astype(int)[:, None])

        # ------------------- stratum dummy ------------------------------

        stratum_dummy = csr_array((data[strata_name] == stratum).astype(int)[:, None])

    else:
        # ------------------- cohort dummy ------------------------------

        cohort_entities = (
            data  # this is only pre/post & control/treated
            .loc[lambda x: x[cohort_name] == cohort]
            .index.get_level_values(0)
            .unique()
        )

        mask_cohort_entities = np.isin(entities, cohort_entities)

        cohort_dummy = csr_array(mask_cohort_entities.astype(int)[:, None])

        # ------------------- stratum dummy ------------------------------

        stratum_entities = (
            data  # this is only pre/post & control/treated
            .loc[lambda x: x[strata_name] == stratum]
            .index.get_level_values(0)
            .unique()
        )

        mask_stratum_entities = np.isin(entities, stratum_entities)

        stratum_dummy = csr_array(mask_stratum_entities.astype(int)[:, None])

    return cohort_dummy, stratum_dummy
