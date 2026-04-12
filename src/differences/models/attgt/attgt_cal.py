from __future__ import annotations

import warnings
from collections import namedtuple
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from joblib import Parallel, cpu_count, delayed
from joblib.externals.loky import get_reusable_executor

from pandas import Index

from scipy.sparse import csr_array, lil_array
from scipy.stats import norm
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationWarning,
)

from differences.models.attgt.mboot import mboot

from differences.models.attgt.utility_ntl import (
    ATTgtElements,
    get_sparse_array,
    get_std_errors_from_if,
    stack_influence_funcs
)

from differences.models.attgt.data_validator.panel_utility import panel_2_cross_section_diff
from differences.tools.shared import tqdm_joblib


# todo: FIX important, for unbalanced panels the cohort dummy and the rest has, when cluster var
# ------------ att and influence function for a single ct --------------

def did_single_gt(
        data: pd.DataFrame,
        entities: Index,
        entity_column: str,  # name of the index column
        n_total: int,
        y_column: str,
        cohort_column: str,
        strata_column: str,
        weights_column: str,
        x_covariates: list,
        x_base: list,
        x_delta: list,
        anticipation: int,
        control_group: str,
        is_panel: bool,
        is_balanced_panel: bool,
        cluster_by_entity: bool,
        att_gt_func: Callable,
        files_path: Path | str | None,

        # will iterate over the following:
        cohort: int,
        base_period: int,
        time: int,
        stratum: str | float | int = None,

) -> namedtuple:
    """
    Computes the Average Treatment Effect on the Treated (ATT)
    for a given cohort, time, and stratum.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing information about the entities, treatments, and other covariates.
    entities : Index
        Index of the entities considered in the study.
    entity_column : str
        Name of the index column representing entities.
    n_total : int
        Total number of entities in the dataset.
    y_column : str
        Name of the column containing the dependent variable.
    cohort_column : str
        Name of the column indicating cohort grouping.
    strata_column : str
        Name of the column indicating strata within cohorts.
    weights_column : str
        Name of the column containing weights for observations.
    x_covariates : list
        List of covariates to be included in the model.
    x_base : list
        List of baseline covariates.
    x_delta : list
        List of delta covariates.
    anticipation : int
        The anticipation period for treatment.
    control_group : str
        Control group definition.
    is_panel : bool
        Indicates if the data is a panel dataset (as opposed to a repeated cross-section).
    is_balanced_panel : bool
        Indicates if the panel data is balanced.
    cluster_by_entity : bool
        If true, clusters by entity.
    att_gt_func : Callable
        A function to compute the group-time ATT.
    files_path : Path | tuple[Path, bool]
        Directory to save the data split.
        tuple not implemented ! If files_path is a tuple, the first element must be
        directory to save the data split and the second element a boolean indicating
        whether to perform the estimation of the ATT. If True, the ATTgt will be estimated,
        if False then this function will return None, and no estimation is performed.
    cohort : int
        Cohort identifier for the analysis.
    base_period : int
        Reference period for the analysis.
    time : int
        Time period for the analysis.
    stratum : str, float, int, optional
        Stratum identifier for the analysis (default is None).

    Returns
    -------
    namedtuple
        A named tuple containing the results of the ATT computation. This includes:
            - cohort: int, Cohort identifier
            - base_period: int, Base period
            - time: int, Time period
            - stratum: str | float | int, Stratum identifier
            - anticipation: int, Anticipation period
            - control_group: str, Control group
            - n_sample: int, Sample size
            - n_total: int, Total entities
            - cohort_dummy: sparse matrix, Cohort dummy
            - cohort_stratum_dummy: sparse matrix, Cohort stratum dummy
            - stratum_dummy: sparse matrix, Stratum dummy
            - exception: Exception, Any exception raised
            - ATT: float, Computed ATT
            - influence_func: sparse matrix, Influence function

    Notes
    -----
    This function is designed to be run in parallel for each cohort-time using joblib.
    It handles various scenarios such as balanced panels, unbalanced panels, and repeated
    cross-sections, depending on the provided flags.
    """

    # it is called by joblib to be run in parallel for each cohort-time
    #
    # idx of the treated observations
    #
    # - balanced: need to make cross-section first
    # - unbalanced: ? either keep data as is or the idx of the unique entities treated
    # - repeated cross-section: data as is

    # todo: estimation_flag is not used, for now the estimation is always run
    #   would need to reorganize the ATTgt.fit() to not run the estimation on demand
    files_path, estimation_flag = process_files_path(files_path=files_path)

    # ------------------------------------------------------------------

    cohort_stratum_dummy, stratum_dummy = None, None

    # ----------------- 0 ATT for reference period ---------------------

    if base_period == time:  # and 'universal'

        # if a stratum (for example intensity of treat) then treated are subset of cohort
        # stratum name should not vary within entity
        treat_info_list = (
            [cohort_column, strata_column] if strata_column is not None else cohort_column
        )

        # cluster_by_entity should be False for true rc (would be redundant)
        if is_panel or cluster_by_entity:
            data = data.groupby(entity_column)[treat_info_list].first().reset_index()

        cohort_dummy = csr_array(
            (data[cohort_column] == cohort).astype(int).to_numpy()[:, None]
        )

        if strata_column is not None:
            cohort_stratum_mask = (data[cohort_column] == cohort) & (
                    data[strata_column] == stratum
            )

            cohort_stratum_dummy = csr_array(
                cohort_stratum_mask.astype(int).to_numpy()[:, None]
            )
            stratum_dummy = csr_array(
                (data[strata_column] == stratum).astype(int).to_numpy()[:, None]
            )

        # not writing parquet here for the reference period

        return ATTgtElements(  # <---  returning here !
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
            ATT=0,  # <--- 0 ATT for reference period
            influence_func=csr_array((n_total, 1)),
        )

    # --------------------- treated / control --------------------------

    if strata_column is None:  # cohort-time (no multivalued treatement)
        mask_treated = (data[cohort_column] == cohort).values
    else:
        mask_treated = (
                (data[cohort_column] == cohort) & (data[strata_column] == stratum)
        ).values

    mask_control = (data[cohort_column].isnull()).values  # never treated

    # not yet treated
    if control_group == "not_yet_treated":
        maxt_ = max(time, base_period)

        mask_control = mask_control | (
                (~mask_treated) & (data[cohort_column] > maxt_ + anticipation).values
        )

    # ---------------- pre / post (base period / time) -----------------

    times_values = data.index.get_level_values(1)
    mask_pre_post = (times_values == base_period) | (
            times_values == time
    )  # pre or post

    # must be treated or control & must be observed pre or post
    mask_pre_post = (mask_treated | mask_control) & mask_pre_post

    # ------------------------------------------------------------------

    # cluster_by_entity should be False for true rc (would be redundant)
    if (
            not is_balanced_panel and cluster_by_entity
    ):  # panels (as rc) but entity level if

        # cohort_dummy needs to be defined before masking for unbalanced panels,
        # some treated may be dropped when masked because they may not have pre or post,
        # or both pre-post (in case balance_2x2)

        treated_entities = data[mask_treated].index.get_level_values(0).unique()
        mask_treated_entities = np.isin(entities, treated_entities)

        # entity level - dummy for the treated entities: need this for the aggregations
        if strata_column is None:
            cohort_dummy = csr_array(mask_treated_entities.astype(int)[:, None])
        else:
            cohort_dummy, stratum_dummy = get_cohort_stratum_dummies(
                data=data,
                entities=entities,
                cohort_column=cohort_column,
                cohort=cohort,
                strata_column=strata_column,
                stratum=stratum,
            )

            cohort_stratum_dummy = csr_array(mask_treated_entities.astype(int)[:, None])

    # ------------------------------------------------------------------

    data = data[mask_pre_post].assign(
        _treated=lambda x: mask_treated[mask_pre_post].astype(int),
        _control=lambda x: mask_control[mask_pre_post].astype(int),
    )

    if is_panel and not is_balanced_panel:  # unbalanced panel: balance it 2x2
        # print("MAKING THE UNBALANCED PANEL BALANCED")

        mask_balance_2x2 = data.index.get_level_values(0).duplicated(keep=False)

        # np.sum(~mask_balance_2x2)
        data = data[mask_balance_2x2].copy()

    # ------------------------------------------------------------------

    # Q: shouldn't n_sample, for unbalanced (as rc): in elif cluster_by_entity, be sample_entities?
    # sample_entities = data.index.get_level_values(0).nunique()
    # given that n_totals is number of entities?

    n_sample = len(data)  # unbalanced (as rc) + repeated cross-section

    if is_panel:  # originally balanced panel + unbalanced-balance_2x2

        # automatically clustered by entity
        treated_entities = (
            data[lambda x: x["_treated"] == 1].index.get_level_values(0).unique()
        )
        mask_treated_entities = np.isin(entities, treated_entities)

        if is_balanced_panel:
            # dummy for the treated entities (obs level): need this for the aggregations
            if strata_column is None:
                cohort_dummy = csr_array(mask_treated_entities.astype(int)[:, None])
            else:
                cohort_dummy, stratum_dummy = get_cohort_stratum_dummies(
                    data=data,
                    entities=entities,
                    cohort_column=cohort_column,
                    cohort=cohort,
                    strata_column=strata_column,
                    stratum=stratum,
                )

                cohort_stratum_dummy = csr_array(
                    mask_treated_entities.astype(int)[:, None]
                )

        control_entities = (
            data[lambda x: x["_control"] == 1].index.get_level_values(0).unique()
        )
        mask_control_entities = np.isin(entities, control_entities)

        tc_bool = np.array(mask_treated_entities | mask_control_entities)

        sample_idx = np.argwhere(tc_bool).flatten()  # to populate influence_func
        n_sample = np.sum(tc_bool)  # overwrites the n_sample = len(data) above

        # print("CALLING PANEL 2 CROSS SECTION")
        data = panel_2_cross_section_diff(
            data=data,
            y_name=y_column,
            x_base=x_base,
            x_delta=x_delta,
            base_period=base_period,
            time=time,
        )

    else:

        # print("AS REPEATED CROSS SECTION")

        data["post"] = (data.index.get_level_values(1) == time).astype(
            int
        )  # post dummy

        if cluster_by_entity:  # panel (as rc) but entity level if

            # entity level - to populate influence_func
            sample_entities = data.index.get_level_values(0)
            sample_idx = np.argwhere(np.isin(entities, sample_entities)).flatten()
            # sample_idx = [idx for idx, e in enumerate(entities) if e in sample_entities]

            # obs level - generate a temp-id (start at 0) to cohort the if by entity
            entity_strata = data.groupby(entity_column).ngroup().to_numpy()

        else:  # repeated cross-section (could be panel as rc)
            # -- all at obs level --

            sample_idx = np.argwhere(
                mask_pre_post
            ).flatten()  # to populate influence_func

            if strata_column is None:
                cohort_dummy = csr_array(
                    mask_treated.astype(int)[:, None]
                )  # obs level (rc/as rc)
            else:
                # obs level (rc/as rc)

                cohort_dummy, stratum_dummy = get_cohort_stratum_dummies(
                    data=data,
                    entities=entities,
                    cohort_column=cohort_column,
                    cohort=cohort,
                    strata_column=strata_column,
                    stratum=stratum,
                    repeated_cross_section=True,
                )

                # cohort_stratum_dummy as cohort_dummy when strata_name is not None
                cohort_stratum_dummy = csr_array(mask_treated.astype(int)[:, None])

    # ------------------------ save data -------------------------------

    # write the group-time data in a parquet file in the files_path directory
    if isinstance(files_path, Path):
        # in case the data is a panel balanced (
        # or balanced2x2, then the data has been converted to
        # a cross-section by panel_2_cross_section_diff)

        save_gt_data(
            files_path=files_path,
            data=data,
            cohort=cohort,
            base_period=base_period,
            time=time,
            stratum=stratum,
        )

    # ---------------------- compute att_ct ----------------------------

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)

            # todo: check warnings

            drdid_res = att_gt_func(
                endog=data[y_column].to_numpy(),
                exog=data[x_covariates].to_numpy(),
                treated=data["_treated"].to_numpy(),
                weights=data[weights_column].to_numpy(),
                post=data["post"].to_numpy() if not is_panel else None,
            )

    except Exception as ex:
        # not writing a parquet file here?
        return ATTgtElements(  # <---  returning here !
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
            ATT=np.nan,  # att, need nan to do some operations
            influence_func=None,
        )

    # ----------------------- influence function -----------------------

    influence_func = drdid_res["influence_func"].flatten()

    influence_func *= n_total / n_sample

    if not is_panel and cluster_by_entity:
        influence_func = np.bincount(entity_strata, weights=influence_func)

    influence_func = get_sparse_array(
        nrows=n_total, ary=influence_func, fill_idx=sample_idx, sparse_func=lil_array
    )

    # print("The column name for the y is", y_name)
    # print("The sum of the y is", data[y_name].sum())

    return ATTgtElements(  # <---  returning here !
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
        ATT=drdid_res["att"],
        influence_func=influence_func,
    )


def process_files_path(
        files_path: Path | tuple[Path | str, bool] | str | None
) -> tuple[Path, bool] | tuple[None, bool]:
    if isinstance(files_path, str):
        files_path = Path(files_path)

    if isinstance(files_path, Path) or files_path is None:
        return files_path, True  # files_path, estimation_flag

    elif isinstance(files_path, tuple):

        path = files_path[0]
        if isinstance(path, str):
            path = Path(path)

        if not isinstance(path, Path):
            raise ValueError("The first element of the files_path tuple must be a Path")

        if not isinstance(files_path[1], bool):
            raise ValueError(
                "The second element of the files_path tuple must be a bool, "
                "indicating whether to perform the estimation"
            )
        return path, files_path[1]  # files_path, estimation_flag
    else:
        raise ValueError("files_path must be either a Path or a tuple[Path, bool]")


def save_gt_data(
        files_path: Path,
        data: pd.DataFrame,
        cohort: int,
        base_period: int,
        time: int,
        stratum: int | str | None
) -> None:
    if isinstance(files_path, Path):
        if stratum is None:
            file_name = f"cohort-{cohort}_base_period-{base_period}_time-{time}.parquet"
        else:
            file_name = f"cohort-{cohort}_base_period-{base_period}_" \
                        f"time-{time}_stratum-{stratum}.parquet"

        # save the used data
        data.reset_index().to_parquet(files_path / file_name)


def check_empty_directory(path: str | Path | None):
    if path is not None:
        p = Path(path)

        if not p.is_dir():
            raise ValueError(f"{path} is not a directory.")
        if any(p.iterdir()):
            raise ValueError(
                f"{path} is not an empty directory. "
                f"Only an empty directory is allowed."
            )


# --------------------- ATT & standard errors --------------------------


def get_att_gt(
        group_time: list[dict],
        data: pd.DataFrame,
        y_column: str,
        cohort_column: str,
        is_panel: bool,
        is_balanced_panel: bool,
        cluster_by_entity: bool,
        x_covariates: list,
        x_base: list,
        x_delta: list,
        strata_column: str = None,
        weights_column: str = None,
        control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",
        anticipation: int = 0,
        att_function_ct: Callable = None,
        n_jobs_gt: int = 1,
        backend_ct: str = "loky",
        progress_bar: bool = True,
        sample_name: str = None,

        files_path: Path | str = None,

        release_workers: bool = True,
) -> list[namedtuple]:
    if control_group not in ["never_treated", "not_yet_treated"]:
        raise ValueError(
            "'control_group' must be either set to either"
            "'never_treated' or 'not_yet_treated'"
        )

    entities = data.index.get_level_values(0).unique()
    entity_column = data.index.names[0]

    # is_panel is False if the user requested as_repeated_cross_section
    n_total = (
        entities.nunique() if is_panel or cluster_by_entity else len(data)
    )  # total n obs

    # needed for the progress bar
    jobs = cpu_count() + 1 - n_jobs_gt if n_jobs_gt < 0 else n_jobs_gt
    sample_name = f"for {sample_name} " if sample_name else ""

    tqdm_object = tqdm(
        disable=not progress_bar,
        desc=f"Computing ATTgt {sample_name}[workers={jobs}]",
        total=len(group_time),
        bar_format="{desc:<30}{percentage:3.0f}%|{bar:20}{r_bar}",
    )

    if jobs > 1:  # tqdm_joblib doesn't work anymore for jobs=1
        with tqdm_joblib(tqdm_object):

            # compute the ct att. order of parameters must conform to did_single_gt()
            res_ntl = Parallel(n_jobs=jobs, backend=backend_ct, )(
                delayed(did_single_gt)(

                    data=data,

                    entities=entities,
                    entity_column=entity_column,

                    n_total=n_total,

                    y_column=y_column,
                    cohort_column=cohort_column,
                    strata_column=strata_column,
                    weights_column=weights_column,

                    x_covariates=x_covariates,
                    x_base=x_base,
                    x_delta=x_delta,

                    anticipation=anticipation,
                    control_group=control_group,

                    is_panel=is_panel,
                    is_balanced_panel=is_balanced_panel,

                    cluster_by_entity=cluster_by_entity,

                    att_gt_func=att_function_ct,

                    files_path=files_path,

                    **gt,
                )
                for gt in group_time
            )

        if release_workers:
            get_reusable_executor().shutdown(wait=True)

    else:  # n_jobs=1

        with tqdm_joblib(tqdm_object):

            res_ntl = []
            for gt in group_time:
                res_ntl.append(
                    did_single_gt(
                        data=data,
                        entities=entities,
                        entity_column=entity_column,
                        n_total=n_total,
                        y_column=y_column,
                        cohort_column=cohort_column,
                        strata_column=strata_column,
                        weights_column=weights_column,
                        x_covariates=x_covariates,
                        x_base=x_base,
                        x_delta=x_delta,
                        anticipation=anticipation,
                        control_group=control_group,
                        is_panel=is_panel,
                        is_balanced_panel=is_balanced_panel,
                        cluster_by_entity=cluster_by_entity,
                        att_gt_func=att_function_ct,
                        files_path=files_path,

                        **gt,
                    )
                )

                tqdm_object.update(1)

    return res_ntl


def get_standard_errors(
        ntl: list[namedtuple],
        cluster_groups: np.ndarray = None,
        alpha: float = 0.05,
        boot_iterations: int = 0,
        random_state: int = None,
        backend_boot: str = "loky",
        n_jobs_boot: int = -1,
        progress_bar: bool = True,
        sample_name: str = None,
        release_workers: bool = True,
) -> list[namedtuple]:
    if boot_iterations < 0:
        raise ValueError(
            "'boot_iterations' must be >= 0. "
            "If boot_iterations=0, analytic standard errors are computed"
        )

    # influence funcs + idx for not nan cols
    inf_funcs, not_nan_idx = stack_influence_funcs(ntl, return_idx=True)

    # create an empty array to populate with the standard errors
    se_array = np.empty(len(ntl))
    se_array[:] = np.nan

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
            release_workers=release_workers,
        )

        # populate empty se array
        se_array[not_nan_idx] = mboot_res["se"]
        cval = mboot_res["crit_val"]

    else:  # analytic standard errors

        # analytic standard errors
        analytic_se = get_std_errors_from_if(inf_funcs)

        # populate empty se array
        se_array[not_nan_idx] = analytic_se

        cval = norm.ppf(1 - alpha / 2)

    # add standard errors & cband to namedtuples
    out = [
        nt._replace(
            std_error=se_array[nt_idx],
            lower=nt.ATT - se_array[nt_idx] * cval,
            upper=nt.ATT + se_array[nt_idx] * cval,
            boot_iterations=boot_iterations,
        )
        for nt_idx, nt in enumerate(ntl)
    ]

    return out


def get_cohort_stratum_dummies(
        data: pd.DataFrame,
        entities: Index | list,
        cohort_column: str,
        cohort: int,
        strata_column: str,
        stratum: str | int | float,
        repeated_cross_section: bool = False,
) -> tuple:
    """helper for did_single_gt"""
    if repeated_cross_section:

        # ------------------- cohort dummy ------------------------------

        cohort_dummy = csr_array((data[cohort_column] == cohort).astype(int)[:, None])

        # ------------------- stratum dummy ------------------------------

        stratum_dummy = csr_array((data[strata_column] == stratum).astype(int)[:, None])

    else:
        # ------------------- cohort dummy ------------------------------

        cohort_entities = (
            data.loc[  # this is only pre/post & control/treated
                lambda x: x[cohort_column] == cohort
            ]
            .index.get_level_values(0)
            .unique()
        )

        mask_cohort_entities = np.isin(entities, cohort_entities)

        cohort_dummy = csr_array(mask_cohort_entities.astype(int)[:, None])

        # ------------------- stratum dummy ------------------------------

        stratum_entities = (
            data.loc[  # this is only pre/post & control/treated
                lambda x: x[strata_column] == stratum
            ]
            .index.get_level_values(0)
            .unique()
        )

        mask_stratum_entities = np.isin(entities, stratum_entities)

        stratum_dummy = csr_array(mask_stratum_entities.astype(int)[:, None])

    return cohort_dummy, stratum_dummy
