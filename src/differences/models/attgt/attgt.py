from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from formulaic import model_matrix
import pandas as pd

from differences.models.attgt.results import ATTgtResult, \
    insert_sample_mask_into_att_gt_dict
from differences.models.did.did_cal import did_cal_funcs, get_method_tuple

from differences.models.attgt.data_validator.panel_utility import (
    find_time_varying_covars,
    is_panel_balanced
)
from differences.models.attgt.data_validator.data_validator import _ValiDIData

from differences.models.attgt.aggregate import _AggregateGT, get_weights

from differences.models.attgt.attgt_cal import (
    get_att_gt,
    get_standard_errors,
    check_empty_directory

)

from differences.models.attgt.difference import (
    _Difference,
    difference_ntl_to_dataframe,
    get_ds_masks,
    preprocess_difference
)
from differences.models.attgt.mboot import get_cluster_groups

from differences.models.attgt.utility_ntl import (
    extract_dict_ntl,
    extract_dict_ntl_for_difference,
    output_dict_to_dataframe,
    preprocess_est_method,
    preprocess_fit_cluster_arg,
    wald_pre_test,
    group_time_base_period
)

from differences.models.attgt.utils.utils import pre_process_base_delta, \
    delta_col_to_create
from differences.models.attgt.utils.group_time import filter_group_time_list_of_dicts


class ATTgt:
    """
    Group-Time ATT

    ATTgt is designed for difference-in-differences with

    - balanced panels, unbalanced panels or repeated cross-section
    - two or multiple periods
    - fixed or staggered treatment timing
    - binary or multi-valued treatment
    - heterogeneous treatment effects

    The Group in Group-Time indicates a treatment group which is a set of
    entities that underwent an event/treatment. A Group is typically defined by the
    cohort of treatment: the time an entity underwent a treatment event.

    :class:`differences.ATTgt` allows for Group to be defined also by an additional
    level of treatment, beside cohort, where entities many undergo a treatment
    at different time but also at different, discrete, intensities/dosage;
    we will refer to this case as multi-valued treatment.
    """

    data = _ValiDIData()
    """
    Data
    
    Group
    -----
        Attributes
    
    Order
    -----
        21
    """

    def __init__(
            self,
            data: pd.DataFrame,
            cohort_column: str,
            dosage_column: str = None,  # extra treatment information
            base_period: Literal["varying", "universal"] = "varying",  # or 'universal'
            anticipation: int = 0,
    ):
        """
        initialize attgt

        Parameters
        ----------
        data
            .. code-block:: python

                df = df.set_index(['entity', 'time'])

            where *df* is the dataframe to use, *'entity'* should be replaced with the
            name of the entity column and *'time'* should be replaced with
            the name of the time column.

        cohort_column: str
            cohort name

        base_period: str, default: ``"varying"``

            - ``"universal"``

            - ``"varying"``

        anticipation: *int*, default: ``0``
            The number of time periods before participating in the treatment where units can
            anticipate participating in the treatment, and therefore it can affect their untreated
            potential outcomes

        dosage_column: str, default: ``None``
            The name of the column to be used in case of multi-valued treatment, used to calculate
            cohort-time-stratum ATT.

            If stratum name is ``None``, fit() will return cohort-time ATT.
        """

        # from now on 'base_period' is called 'base_period_type'
        self.base_period_type = base_period

        if self.base_period_type not in ["varying", "universal"]:
            raise ValueError(
                "'base_period' must be either set to either " "'varying' or 'universal'"
            )

        self.cohort_column = cohort_column
        self.dosage_column = dosage_column

        self.anticipation = anticipation
        self.copy_data = True  # maybe make an option to the user

        self.data = data

        self._data_matrix = None

        self._result_dict = None
        self._sample_names = None

        self._control_group = None
        self._y = None
        self._x_formula = None

        self._x_covariates = []
        self._time_varying_x = []  # time varying covariates
        self._base_delta = "base"
        self._x_base_delta = {}

        self._weights = None
        self._weights_column = None

        self._dropped_nas_flag = None  # True if dropped obs do to NaNs

        self._cluster_var_fit = None
        self._cluster_by_entity = None

        self._method_tuple = ()

        self._aggregate_cflag = None

        self._difference_inst = None
        self._difference_ntl = None

        self._agg_types = ["simple", "cohort", "event", "time"]

        self._aggregate_locals = None
        self._boot_iterations_difference = None
        self._as_rcs = None

        # set self.group_time
        self.filter_gt(feasible=False, base_period=base_period)

    # def __repr__(self):
    #     pass

    @property
    def _times(self) -> np.ndarray:
        return np.array(
            sorted(self.data.index.get_level_values(self._time_name).unique())
        )

    @property
    def _cohorts(self) -> np.ndarray:
        cohorts = np.array(sorted(self.data[self.cohort_column].dropna().unique()))
        return cohorts[cohorts > self._times[0] + self.anticipation]

    @property
    def _dosages(self) -> list:
        if self.dosage_column is not None:
            dosages = []

            for d in sorted(self.data[self.dosage_column].dropna().unique()):

                if isinstance(d, str):
                    pass
                elif int(d) == d:
                    d = int(d)
                else:
                    pass

                dosages.append(d)

            return dosages

    @cached_property
    def _feasible_gt(self):
        """time, cohort, stratum"""
        info = (
            [self.cohort_column, self.dosage_column]
            if self.dosage_column
            else [self.cohort_column]
        )

        # time is the second level of the index
        feasible = (
            self.data[info]
            .reset_index()  # here we make time (and entity) columns
            .drop(columns=[
                self._entity_name])  # drop entity: just keep time-cohort-(stratum)
            .drop_duplicates()
            .dropna()  # drop never-treated
        )
        feasible[self.cohort_column] = feasible[self.cohort_column].astype(int)

        return {tuple(i) for i in feasible.to_records(index=False)}

    def filter_gt(
            self,
            base_period: Literal["varying", "universal"],
            feasible: bool = False,
            *,

            relative_period: int | list[int] = None,
            relative_period_start: int = None,
            relative_period_end: int = None,

            cohort: int | list[int] = None,
            cohort_start: int = None,
            cohort_end: int = None,

            time: int | list[int] = None,
            time_start: int = None,
            time_end: int = None,

    ) -> list[dict]:
        """
        Filter cohort-base_period-time combinations based on specified
        criteria and base period type.

        Group
        -----
            Setup

        The method filters the group time data based on the provided criteria and the
        selected base period handling method. Two base period methods are available:
        "varying" and "universal". If feasible is set to True, only combinations of time
        and cohort present in the pipeline are included.

        Parameters
        ----------
        feasible : bool, optional
            Whether to include only feasible combinations of time and
            cohort based on the pipeline. Default is False.
        base_period : Literal["varying", "universal"], optional
            The base period handling method. It can be either "varying" or "universal".
            If not specified, it uses the `base_period_type` attribute set during initialization.
        relative_period : int | list[int], optional
            Specific relative period values to filter for inclusion. Default is None.
        relative_period_start : int, optional
            Minimum relative period value to include. Default is None.
        relative_period_end : int, optional
            Maximum relative period value to include. Default is None.
        cohort : int | list[int], optional
            Specific cohort values to filter for inclusion. Default is None.
        cohort_start : int, optional
            Minimum cohort value to include. Default is None.
        cohort_end : int, optional
            Maximum cohort value to include. Default is None.
        time : int | list[int], optional
            Specific time values to filter for inclusion. Default is None.
        time_start : int, optional
            Minimum time value to include. Default is None.
        time_end : int, optional
            Maximum time value to include. Default is None.
        save : bool, optional
            Whether to save the filtered group time and replace the base period
            type in the object's state. Default is False.

        Returns
        -------
        list[dict]
            A list of dictionaries representing cohort-base_period-time combinations.
            Each dictionary's keys include: ``cohort``, ``base_period``, ``time``, and
            optionally ``stratum`` if the `strata_column` attribute is defined.

        Raises
        ------
        ValueError
            If 'base_period' is neither 'varying' nor 'universal'.

        See Also
        --------
        varying_base_period : Function for handling varying base period.
        universal_base_period : Function for handling universal base period.

        """

        if base_period is None:
            base_period = self.base_period_type  # use base_period_type set in __init__

        group_time = group_time_base_period(
            base_period=base_period,
            cohorts=self._cohorts,
            times=self._times,
            anticipation=self.anticipation
        )

        if self.dosage_column is None:
            if feasible:
                group_time = [
                    d
                    for d in group_time
                    if (d["time"], d["cohort"]) in self._feasible_gt
                       or (d["base_period"] == d["time"])
                ]

        else:
            group_time = [{**d, "stratum": xt} for xt in self._dosages for d in
                          group_time]

            if feasible:
                group_time = [
                    d
                    for d in group_time
                    if (d["time"], d["cohort"], d["stratum"]) in self._feasible_gt
                       or (d["base_period"] == d["time"])
                ]

        # todo: if balance panel if filter? should already be done if balance 2*2

        group_time = filter_group_time_list_of_dicts(
            group_times=group_time,
            relative_period=relative_period,
            relative_period_start=relative_period_start,
            relative_period_end=relative_period_end,
            cohort=cohort,
            cohort_start=cohort_start,
            cohort_end=cohort_end,
            time=time,
            time_start=time_start,
            time_end=time_end,
        )

        self._group_time = group_time
        self.base_period_type = base_period  # replace self.base_period_type set in __init__

        return group_time

    @property
    def group_time(self):
        """
        List of group times

        Group
        -----
            Info

        Returns
        -------

        """
        return self._group_time

    def _create_data_matrix(self, is_panel: bool):

        y_matrix = None

        if is_panel:  # both balance and unbalanced will be passed to p2c
            # keep Xs separate to look for time varying covariates (needed for base-delta)
            y_matrix, self._data_matrix = model_matrix(
                spec=f"{self._y} ~ {self._x_formula}", data=self.data
            )

        else:
            self._data_matrix = model_matrix(
                spec=f"{self._y} + {self._x_formula}", data=self.data
            )

        self._data_matrix = pd.DataFrame(self._data_matrix)

        self._dropped_nas_flag = len(self.data) != len(self._data_matrix)

        return y_matrix

    def _preprocess_covariates(
            self, is_panel: bool, base_delta: str | list | dict, y_matrix
    ) -> None:

        if is_panel:  # y_matrix is available only if is_panel

            self._x_covariates = list(
                self._data_matrix
            )  # includes y if in specification

            self._x_base, self._x_delta = self._x_covariates, []

            if ("delta" in base_delta) or isinstance(base_delta, dict):

                # identify time varying Xs (only if needed: when delta is requested)
                self._time_varying_x = find_time_varying_covars(
                    data=self._data_matrix,
                    # do not include y (in case it is controlled for)
                    covariates=[c for c in list(self._data_matrix) if c != self._y],
                )

                self._x_covariates, self._x_base, self._x_delta = pre_process_base_delta(
                    base_delta=base_delta,
                    x_columns=self._x_covariates,
                    x_time_varying_columns=self._time_varying_x,
                )

                delta_to_create, _ = delta_col_to_create(
                    x_delta=self._x_delta, x_base=self._x_base
                )

                if delta_to_create:  # these will be created by p2c later
                    self._x_covariates.extend([f"delta_{c}" for c in delta_to_create])

            # need to copy y as delta_y in case y is among the controls (as base period y),
            # if y is among the controls it is added by x_formula to _data_matrix
            self._data_matrix[f"delta_{self._y}"] = y_matrix
            self._y = f"delta_{self._y}"

        else:  # repeated_cross_section (or panel as rc, is_panel is set to False)

            # list all Xs generated by model_matrix; excluding y which was included in x_formula
            self._x_covariates = [x for x in list(self._data_matrix) if x != self._y]
            self._x_base, self._x_delta = self._x_covariates, []

        self._x_base_delta = {"base": self._x_base, "delta": self._x_delta}

    def _create_result_dict(
            self, sample_split_column: Callable | str | None
    ) -> None:
        """
        creates self._result_dict dictionary

        if sample_split_column:
          self._result_dict = {sample_name: {'sample_mask': np mask}
        else:
          self._result_dict = {'full_sample': {}}

        user should not _interact wth this dictionary
        """

        if isinstance(sample_split_column, str):
            self._result_dict = insert_sample_mask_into_att_gt_dict(
                data=(
                    self.data
                    if not self._dropped_nas_flag
                    else self.data[sample_split_column].loc[
                        lambda x: x.index.isin(self._data_matrix.index)
                    ]
                ),
                split_sample_by=sample_split_column,
            )

        elif isinstance(sample_split_column, Callable):
            # as it is set up we don't know the col name

            # this should be more efficient, if split_sample_by is a callable
            # need a way to get the col name: possibly change input to dict: {'name': callable}
            self._result_dict = insert_sample_mask_into_att_gt_dict(
                data=(
                    self.data
                    if not self._dropped_nas_flag
                    else self.data.loc[lambda x: x.index.isin(self._data_matrix.index)]
                ),
                split_sample_by=sample_split_column,
            )

        else:  # self._result_dict is None
            self._result_dict = {"full_sample": {}}

    def _get_clusters_for_difference(
            self,
            cluster_var: list | str | None,
            difference_samples: list,
            data_mask: np.ndarray,
            iterate_samples: list,
    ) -> np.ndarray | dict:

        # clusters
        cluster_groups = None
        if cluster_var:

            if difference_samples:
                cluster_groups = get_cluster_groups(
                    data=self._data_matrix[cluster_var].loc[data_mask],
                    cluster_var=cluster_var,
                )

            elif iterate_samples:
                cluster_groups = {
                    s: get_cluster_groups(
                        data=self._data_matrix[cluster_var].loc[
                            self._result_dict[s]["sample_mask"]
                        ],
                        cluster_var=cluster_var,
                    )
                    for s in self.sample_names
                }

            else:  # main case, no samples splits or treat strata
                cluster_groups = get_cluster_groups(
                    data=self._data_matrix[cluster_var], cluster_var=cluster_var
                )

        return cluster_groups

    # att gt
    def fit(
            self,
            formula: str,
            est_method: Literal["reg", "dr", "dr-ipt", "std_ipw-mle", "std_ipw"] |
                        Callable = "reg",
            control_group: Literal[
                "never_treated", "not_yet_treated"] = "never_treated",
            as_repeated_cross_section: bool = None,
            base_delta: str | list | dict = "base",
            *,
            weights_column: str = None,
            sample_split_column: Callable | str | dict = None,
            cluster_var: list | str = None,
            alpha: float = 0.05,
            boot_iterations: int = 0,
            random_state: int = None,
            n_jobs: int = 1,
            backend: str = "loky",
            progress_bar: bool = True,
            files_path: Path | str = None
    ) -> ATTgtResult:
        """
        Computes the cohort-time-(stratum) average treatment effects:

        effects for each cohort, in each time, (for each stratum).

        Group
        -----
            Estimation

        Parameters
        ----------
        formula: str
            Wilkinson model_matrix for the outcome variable and covariates

            If no covariates the model_matrix must contain only the name of the outcome variable

            .. code-block:: python

                # example with covariates
                model_matrix = 'y ~ a + b + a:b'

                # example without covariates
                model_matrix = 'y'

            Formulas are implemented using
            `formulaic <https://matthewwardrop.github.io/formulaic/>`_,
            refer to its documentation for additional details.

        weights_column: *str* | None, default: ``None``

            The name of the column containing the sampling weights.
            If None, all observations have same weights.

        control_group: *str*, default: ``"never_treated"``

            - ``"never_treated"``

            - ``"not_yet_treated"``

        base_delta: *str* | *list* | *dict*, default: ``"base"``

            Use base period values for covariates and/or delta values, i.e. the change in value,
            between the value of covariates at *time* and the value at *base period*.

            Available options are:

            - ``"base"``
                the value of :underline:`each` covariate is set to its *base period* value

            - ``"delta"``
                the value of :underline:`each` time-varying covariate is set to the delta.
                Time-constant covariates included through *x_formula* are dropped, and a warning
                issued.

            - ``["base", "delta"]`` or ``"base_delta"``
                the value of :underline:`each` covariate is set to its *base period* value, and
                the value of :underline:`each` time-varying covariate is set to the delta.

            - ``{'base': ['a', 'b', ..]}``
                the value of the :underline:`specified` covariates is set to its *base period*
                value, and the value of :underline:`each` time-varying covariate is set to
                the delta. A warning is issued if *x_formula* included
                time-constant covariates that are not included in *base_delta*.

            - ``{'delta': ['c', 'd', ..]}``
                the value of :underline:`each` covariate is set to its *base period* value, and
                the value of the :underline:`specified` time-varying covariates
                is set to the delta. If the covariates included in *'delta'* are not
                time-varying they will be removed from the list.

            - ``{'base': ['a', 'b', ..], 'delta': ['c', 'd', ..]}``
                the value of the :underline:`specified` covariates
                is set to its *base period* value, and
                the value of the :underline:`specified` time-varying covariates is set to
                the delta. A warning is issued if *x_formula* included time-constant covariates
                that are not included in *'delta'*. If the covariates included in 'delta' are not
                time-varying they will be removed from the list.

        est_method: *str*, default: ``"dr-mle"``

            - ``"dr-mle"`` or ``"dr"``
                for locally efficient doubly robust DiD estimator,
                with logistic propensity score model for the probability of being treated

            - ``"dr-ipt"``
                for locally efficient doubly robust DiD estimator,
                with propensity score estimated using the inverse probability tilting

            - ``"reg"``
                for outcome regression DiD estimator

            - ``"std_ipw-mle"`` or ``"std_ipw"``
                for standardized inverse probability weighted DiD estimator,
                with logistic propensity score model for the probability of being treated

        as_repeated_cross_section: *bool* | None, default: ``None``

        boot_iterations: *int*, default: ``0``

        random_state: *int* | None, default: ``None``

        alpha: *float*, default: ``0.05``

            The significance level.

        cluster_var: *str* | *list* | *None*, default: ``None``

        sample_split_column: *str* | *Callable* | None, default: ``None``
            The name of the column along which to split the data, or a function which takes the
            data and returns a sample mask for a binary split, for example:

            .. code-block:: python

                lambda: x = x['column name'] >= x['column name'].median()

            The estimation of the ATT will be run separately for each specified sample;
            used for heterogeneity analysis.

        n_jobs: *int*, default: ``1``
            The maximum number of concurrently running jobs. If -1 all CPUs are used.

            If ≠ 1, concurrent jobs will be run for two separate tasks:

            - computing the cohort-time ATT; each cohort-time is assigned to a job

            - computing the bootstrap; the influence function is split into n_jobs parts and the
              boostrap is computed concurrently for each part

            Parallelization is implemented using
            `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_,
            refer to its documentation for additional details on n_jobs.

        backend: *int*, default: ``"loky"``
            Parallelization backend implementation.

            Parallelization is implemented using
            `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_,
            refer to its documentation for additional details on backend.

        progress_bar: *bool*, default: ``True``
            If True, a progress bar will display the progress over the cohort-times iterations
            and/or the iterations over the number of boostrap concurrent splits
            (not the bootstrap iterations).

        files_path

        Returns
        -------
        A pd.DataFrame with the group time ATTs
        """
        check_empty_directory(path=files_path)

        # ------------------ some pre processing -----------------------

        if "~" in formula:
            y, x_formula = formula.split("~")
            y, x_formula = y.strip(), x_formula.strip()
        else:
            y, x_formula = formula, None

        self._y = y
        self._y_name = y

        self._x_formula = "1" if x_formula is None else x_formula

        self._control_group = control_group

        if self._no_never_treated_flag and self._control_group == "never_treated":
            raise ValueError(
                "there is no never treated entity. "
                "use not_yet_treated as control cohort"
            )

        # --------------------------------------------------------------

        is_panel = self.is_panel
        is_balanced_panel = getattr(self, "is_balanced_panel", False)

        # True: default repeated cs for unbalanced panel
        self._as_rcs = as_repeated_cross_section
        self._as_rcs = (
            True if not is_balanced_panel and (self._as_rcs is None or self._as_rcs)
            else False
        )

        true_repeated_cross_section = not is_panel

        if self._as_rcs:  # estimate panels as repeated cross-section
            # do not modify self.is_balanced_panel or self.is_panel
            is_panel, is_balanced_panel = False, False

        # -------------------- cluster_var -----------------------------

        # defaults entity cluster if not specified
        self._cluster_var_fit, self._cluster_by_entity = preprocess_fit_cluster_arg(
            cluster_var=cluster_var,
            entity_name=self._entity_name,
            true_rc=true_repeated_cross_section,
        )

        # ----------------- data matrix from model_matrix -------------------

        y_matrix = self._create_data_matrix(is_panel=is_panel)

        if is_panel and self._dropped_nas_flag:
            # balance of panel may have changed after dropping NaNs
            is_balanced_panel = is_panel_balanced(self._data_matrix)

        # --------------------- time-varying Xs ------------------------

        self._preprocess_covariates(
            is_panel=is_panel, base_delta=base_delta, y_matrix=y_matrix
        )

        # ------------ keep additional cols in data_matrix -------------

        add_cols = [self.cohort_column, self.dosage_column, weights_column, cluster_var]
        add_cols = [c for c in add_cols if c is not None]

        self._data_matrix[add_cols] = (
            self.data[add_cols]
            .loc[lambda x: x.index.isin(self._data_matrix.index)]
        )

        # ------------------------ weights -----------------------------

        self._weights_column = weights_column
        if weights_column is None:
            self._data_matrix["_w"] = 1
            self._weights_column = "_w"

        # ----------------------- _result_dict -------------------------

        self._create_result_dict(sample_split_column=sample_split_column)

        # ---------- select function to compute the ATT and if ---------

        self.est_method = est_method
        if isinstance(est_method, str):
            est_method, est_method_pscore = preprocess_est_method(est_method=est_method)

            self._method_tuple = get_method_tuple(
                est_method, est_method_pscore, "panel" if is_panel else "rc"
            )

            att_function_ct = did_cal_funcs[self._method_tuple]  # select did est method
        elif isinstance(est_method, Callable):
            # document how the user provided att_function_ct should look like
            att_function_ct = est_method

        else:
            raise ValueError("invalid est_method")

        # --------------------------------------------------------------

        self._aggregate_cflag = None  # make sure the aggregations are reset to None

        n_sample_names = len(self.sample_names)

        for s_idx, s in enumerate(self.sample_names):

            # clusters
            cluster_groups = None
            if cluster_var:
                cluster_groups = get_cluster_groups(
                    data=(
                        self._data_matrix[cluster_var]
                        if s == "full_sample"
                        else self._data_matrix[cluster_var].loc[
                            self._result_dict[s]["sample_mask"]
                        ]
                    ),
                    cluster_var=cluster_var,
                )

            # att
            res = get_att_gt(
                data=(
                    self._data_matrix
                    if s == "full_sample"
                    else self._data_matrix[self._result_dict[s]["sample_mask"]].copy()
                ),
                y_column=self._y,
                cohort_column=self.cohort_column,
                strata_column=self.dosage_column,
                group_time=self.group_time,
                weights_column=self._weights_column,
                control_group=self._control_group,
                anticipation=self.anticipation,
                x_covariates=self._x_covariates,
                x_base=self._x_base,
                x_delta=self._x_delta,
                is_panel=is_panel,
                is_balanced_panel=is_balanced_panel,
                cluster_by_entity=self._cluster_by_entity,
                att_function_ct=att_function_ct,
                backend_ct=backend,
                n_jobs_gt=n_jobs,
                progress_bar=progress_bar,
                sample_name=s if s != "full_sample" else None,  # just for progress_bar
                files_path=files_path,
                release_workers=(
                        not bool(boot_iterations) and (s_idx + 1 == n_sample_names)
                ),
            )

            # standard errors & ci/cbands
            res = get_standard_errors(
                ntl=res,
                cluster_groups=cluster_groups,
                alpha=alpha,
                boot_iterations=boot_iterations,
                random_state=random_state,
                n_jobs_boot=n_jobs,
                backend_boot=backend,
                progress_bar=progress_bar,
                sample_name=s if s != "full_sample" else None,
                release_workers=s_idx == n_sample_names,
            )

            self._result_dict[s]["ATTgt_ntl"] = res

        self._fit_res = output_dict_to_dataframe(
            extract_dict_ntl(self._result_dict),
            stratum=bool(self._dosages),
            date_map=self._map_datetime,
        )
        # return self._fit_res

        return ATTgtResult(
            att_gt=self._result_dict,
            data_matrix=self._data_matrix,
            dosages=self._dosages,
            weights_column=self._weights_column,
            cluster_by_entity=self._cluster_by_entity
        )

    def aggregate(
            self,
            type_of_aggregation: Literal[
                                     "simple", "cohort", "event", "time"] | None = "simple",
            overall: bool = False,
            difference: bool | list | dict[str, list] = False,
            *,
            alpha: float = 0.05,
            cluster_var: list | str = None,
            boot_iterations: int = 0,
            random_state: int = None,
            n_jobs: int = 1,
            backend: str = "loky",
    ) -> pd.DataFrame:
        """
        Aggregate the ATTgt

        Group
        -----
            Estimation

        Parameters
        ----------
        type_of_aggregation: *str* | None, default: ``None``

            - ``"simple"``
                to calculate the weighted average of all cohort-time average treatment effects,
                with weights proportional to the cohort size.

            - ``"event"``
                to calculate the average effects in each relative period:
                periods relative to the treatment; as in an event study.

            - ``"cohort"``
                to calculate the average treatment effect in each cohort.

            - ``"time"``
                to calculate the average treatment effect in each time.

        overall: *bool*, default: ``False``
            calculates the average effect within each type_of_aggregation.

            - if type_of_aggregation is set to ``"event"``
                to calculate the average effect of the treatment across positive relative periods

            - if type_of_aggregation is set to ``"cohort"``
                to calculate the average effect of the treatment across cohorts

            - if type_of_aggregation is set to ``"time"``
                to calculate the average effect of the treatment across time times

        difference: *bool* | *list* | *dict*, default: ``False``
            take the difference of the estimates

            Available options are:

            - ``True``
                to calculate the difference between 2 samples or 2 strata of treatments

            .. note::
                - Samples difference: if the estimation is run on 2 samples and more than
                  2 strata,
                  the estimates for the two samples will be subtracted, as long as there are no
                  strata that have the same names as the samples, in that case use
                  a dictionary as indicated below

                - strata difference: if the estimation is run on 2 strata and more than
                  2 samples,
                  the estimates for the two strata will be subtracted, as long as there are no
                  samples that have the same names as the strata, in that case use
                  a dictionary as indicated below

            - ``[sample-0, sample-1]`` or ``[stratum-A, stratum-B]``
                to calculate the difference between 2 samples listed in the argument or
                the 2 strata of treatments listed in the argument

            .. note::
                - Samples difference: if there are strata with the same name as the two samples
                  listed, use a dictionary as indicated below

                - strata difference: if there are samples with the same name as the two strata
                  listed, use a dictionary as indicated below

            - ``{'strata': [stratum-A, stratum-B]}`` or
              ``{'sample_names': [sample-0, sample-1]}``

        alpha: *float*, default: ``0.05``

            The significance level.

        cluster_var: *str* | *list* | *None*, default: ``None``
            cluster variables

        boot_iterations: *int*, default: ``0``
            bootstrap iterations

        random_state: *int* | None, default: ``None``
            seed for bootstrap

        n_jobs: *int*, default: ``1``
            The maximum number of concurrently running jobs. If -1 all CPUs are used.

            If ≠ 1, concurrent jobs will be run for:

            - computing the bootstrap; the influence function is split into n_jobs parts and the
              boostrap is computed concurrently for each part

            Parallelization is implemented using
            `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_,
            refer to its documentation for additional details on n_jobs.

        backend: *int*, default: ``"loky"``
            Parallelization backend implementation.

            Parallelization is implemented using
            `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_,
            refer to its documentation for additional details on backend.

        Returns
        -------
        A pd.DataFrame with the requested aggregation

        """

        # --------------------------------------------------------------

        if self._result_dict is None:
            raise RuntimeError(
                "call fit() to estimate group-time ATTs before calling aggregate()"
            )

        if type_of_aggregation not in self._agg_types and not difference:
            warnings.warn(
                f"{type_of_aggregation} is not a valid 'type_of_aggregation'. "
                f"If you want to aggregate the ATTgt please select one of {self._agg_types}.\n"
                f"Returning ATTgt ..."
            )

            return self._fit_res

        if isinstance(
                cluster_var, str
        ):  # entity cluster is automatic, exclude from list
            cluster_var = [
                c for c in [cluster_var] if c != self._data_matrix.index.names[0]
            ]

        if isinstance(difference, dict):
            correct = {"d": "strata", "s": "sample_names"}
            for k, v in [(k[0], k) for k in difference.keys()]:
                difference[correct[k]] = difference.pop(v)

        # -------------------------- cache -----------------------------

        # decide when to un-cache the aggregation class
        _locals_aggregate = tuple(
            (k, v)
            for k, v in locals().items()
            if k not in ["self", "type_of_aggregation", "overall"]
        )

        if _locals_aggregate != self._aggregate_locals:
            self._aggregate_cflag = None
            self._aggregate_locals = _locals_aggregate

        # if difference, no need to bootstrap the standard errors for the aggregation
        self._boot_iterations_difference = boot_iterations
        if difference:
            boot_iterations = 0

        # ---------- instances of _Aggregategt for each sample --------

        if type_of_aggregation is not None:

            if self._aggregate_cflag is None:  # set to None in .fit()
                self._aggregate_cflag = True

                for s in self.sample_names:

                    if self._result_dict[s].get("weights") is None:
                        self._result_dict[s]["weights"] = get_weights(
                            data=(
                                self._data_matrix[[self._weights_column]]
                                if s == "full_sample"
                                else self._data_matrix[[self._weights_column]].loc[
                                    self._result_dict[s]["sample_mask"]
                                ]
                            ),
                            weights_name=self._weights_column,
                            entity_level=self._cluster_by_entity,
                        )

                    cluster_groups = None
                    if cluster_var:
                        cluster_groups = get_cluster_groups(
                            data=(
                                self._data_matrix[cluster_var]
                                if s == "full_sample"
                                else self._data_matrix[cluster_var].loc[
                                    self._result_dict[s]["sample_mask"]
                                ]
                            ),
                            cluster_var=cluster_var,
                        )

                    # save the instance of the aggregation class in the result dict
                    self._result_dict[s]["aggregate_inst"] = _AggregateGT(
                        ntl=self._result_dict[s]["ATTgt_ntl"],
                        weights=self._result_dict[s]["weights"],
                        strata=self._dosages,
                        cluster_groups=cluster_groups,
                        alpha=alpha,
                        boot_iterations=boot_iterations,
                        random_state=random_state,
                        backend_boot=backend,
                        n_jobs_boot=n_jobs,
                    )

            # ------ perform aggregation within _Aggregategt ----------

            for s in self.sample_names:
                self._result_dict[s]["aggregate_inst"].aggregate(
                    type_of_aggregation=type_of_aggregation,
                    overall=overall,
                )

        # ----------------- differences between ATTs -------------------

        # differences between
        #   - strata
        #   - sample splits
        #   - sample splits & strata

        if difference:

            # pre-process difference input: strata, samples
            difference_between = preprocess_difference(
                difference=difference,
                sample_names=self._sample_names,
                strata=self._dosages,
            )

            # if isinstance(difference_between, tuple):
            #     pass

            # else:  # dict
            difference_strata, iterate_samples = [
                difference_between.get(key)
                for key in ["difference_strata", "iterate_samples"]
            ]

            difference_samples, iterate_strata = [
                difference_between.get(key)
                for key in ["difference_samples", "iterate_strata"]
            ]

            data_mask, sample_masks = None, None
            if difference_samples:  # if the difference is between samples, get masks

                # data_mask: numpy array masking the full data to filter the two samples
                # sample_masks: list of np arrays with mask for each sample (resized)
                data_mask, sample_masks = get_ds_masks(
                    result_dict=self._result_dict,
                    difference=difference_samples,
                    entity_index=self._data_matrix.index.get_level_values(0),
                    cluster_by_entity=self._cluster_by_entity,
                )

            cluster_groups = self._get_clusters_for_difference(
                cluster_var=cluster_var,
                difference_samples=difference_samples,
                data_mask=data_mask,
                iterate_samples=iterate_samples,
            )

            # set up data for difference
            diff_pairs_ntls = extract_dict_ntl_for_difference(
                result_dict=self._result_dict,
                type_of_aggregation=type_of_aggregation,
                overall=overall,
                difference_samples=difference_samples,
                iterate_strata=iterate_strata,
                difference_strata=difference_strata,
                iterate_samples=iterate_samples,
            )

            self._difference_inst = _Difference(
                alpha=alpha,
                boot_iterations=self._boot_iterations_difference,
                random_state=random_state,
                n_jobs_boot=n_jobs,
                backend_boot=backend,
            )

            # get difference between estimates
            self._difference_ntl = self._difference_inst.get_difference(
                diff_pairs_ntls=diff_pairs_ntls,
                sample_masks=sample_masks,  # need it when subtracting two samples
                cluster_groups=cluster_groups,
                type_of_aggregation=type_of_aggregation,
                overall=overall,
                iterating_samples=bool(iterate_samples)
                # false if iter strata or only one sample
            )

            output = difference_ntl_to_dataframe(
                ntl=self._difference_ntl,
            )

            return output

        # aggregation if no difference requested

        output = extract_dict_ntl(
            result_dict=self._result_dict,
            type_of_aggregation=type_of_aggregation,
            overall=overall,
            sample_names=self.sample_names,
        )

        output = output_dict_to_dataframe(
            output,
            stratum=bool(self._dosages),
        )

        return output

    @property
    def sample_names(self):
        """
        Names of the sample splits

        Group
        -----
            Info

        Returns
        -------

        """
        try:
            self._sample_names = list(self._result_dict.keys())
        except AttributeError:
            return self._sample_names

        return self._sample_names

    @property
    def wald_pre_test(self):
        """
        Wald pre-test

        Group
        -----
            Info

        Returns
        -------

        """
        if self.dosage_column is not None:
            # todo
            return None

        if self.sample_names is None:
            raise RuntimeError(".fit() must be called before accessing Wald pre-test")

        _wald_pre_test = {}
        for s in self.sample_names:
            res = self.results(sample_name=s)
            res = wald_pre_test(res)

            if s == "full_sample":
                return res

            else:
                _wald_pre_test[s] = res

        return _wald_pre_test

    def estimation_details(self, type_of_aggregation: str = None):
        """
        Estimation Details

        Group
        -----
            Results

        Parameters
        ----------
        type_of_aggregation

        Returns
        -------

        """

        details = {
            "estimation_method": self.est_method,
            "anticipation": self.anticipation,
            "base_period_type": self.base_period_type,
            "is_panel": self.is_panel,
        }

        if self.is_panel:
            details.update(
                {"is_balanced_panel": getattr(self, "is_balanced_panel", False)}
            )

        if self._result_dict is not None:
            details.update(
                {
                    "as_repeated_cross_section": self._as_rcs,
                    "control_group": self._control_group,
                    "model_matrix": f"{self._y_name} ~ {self._x_formula}",
                    "cluster_by_entity": self._cluster_by_entity,
                }
            )

            if not self._as_rcs and self.is_panel:
                details.update(
                    {
                        "base_delta": self._x_base_delta,
                    }
                )

        if type_of_aggregation is None:
            details.update({})

        return details

    def results(
            self,
            type_of_aggregation: Literal[
                                     "simple", "cohort", "event", "time"] | None = None,
            overall: bool = False,
            difference: bool = False,
            # sample_name: str = None,
            to_dataframe: bool = True,
            add_info: bool = False,
    ):
        """
        provides easy access to cached results.
        this method must be called after fit and/or aggregate depending
        on the parameters requested

        Group
        -----
            Results

        Parameters
        ----------
        type_of_aggregation: *str* | None, default: ``None``

            - ``"simple"``
                to return the weighted average of all cohort-time average treatment effects,
                with weights proportional to the cohort size.

            - ``"event"``
                to return the average effects in each relative period:
                periods relative to the treatment; as in an event study.

            - ``"cohort"``
                to return the average treatment effect in each cohort.

            - ``"time"``
                to return the average treatment effect in each time time.

        overall: *bool*, default: ``False``
            calculates the average effect within each type_of_aggregation.

            - if type_of_aggregation is set to ``"event"``
                to return the average effect of the treatment across positive relative periods

            - if type_of_aggregation is set to ``"cohort"``
                to return the average effect of the treatment across cohorts

            - if type_of_aggregation is set to ``"time"``
                to return the average effect of the treatment across time times

        difference: *bool*, default: ``False``
            to return the most recent estimated difference

        to_dataframe
            whether to return the result in a pd.DataFrame or a list of namedtuples

        Returns
        -------
        Either a pandas dataframe or a list of namedtuples
        """

        if self._result_dict is None:
            raise RuntimeError(".fit() must be called before accessing the results")

        if type_of_aggregation in self._agg_types:  # ['cohort', 'event', ...]

            if self._aggregate_cflag is None:
                raise RuntimeError(
                    ".aggregate() must be called before accessing the results"
                )

        if difference:
            if self._difference_ntl is None:
                raise RuntimeError(
                    ".aggregate(difference=) "
                    "must be called before accessing the results"
                )

            if to_dataframe:
                output = difference_ntl_to_dataframe(
                    ntl=self._difference_ntl, date_map=self._map_datetime
                )
                return output

            return self._difference_ntl

        # att gt / aggregation

        output = extract_dict_ntl(
            result_dict=self._result_dict,
            type_of_aggregation=type_of_aggregation,
            overall=overall,
            sample_names=self.sample_names,
        )

        if to_dataframe:
            output = output_dict_to_dataframe(
                output,
                stratum=bool(self._dosages),
                date_map=self._map_datetime,
                add_info=add_info,
            )
            return output

        return output
