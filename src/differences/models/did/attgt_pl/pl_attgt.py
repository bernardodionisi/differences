from __future__ import annotations

from typing import Callable, Literal

from formulaic import model_matrix
import polars as pl

from differences._OLD_NOW_PAGURO.shared.event.data_es import DataES
from differences.models.did.did.did_cal import did_cal_funcs, get_method_tuple

from differences.models.did.attgt.data_validator.panel_utility import (
    find_time_varying_covars,
    is_panel_balanced
)

from differences.models.did.attgt.data_validator import (
    preprocess_est_method,
    preprocess_fit_cluster_arg
)

from differences._OLD_NOW_PAGURO.shared import DataMatrix
from differences.models.did.attgt_pl.panel_to_cs import pre_process_base_delta, delta_col_to_create

from differences.models.did.attgt.results import ATTgtResult, insert_sample_mask_into_att_gt_dict

from differences.models.did.attgt.mboot import get_cluster_groups
from differences.models.did.attgt.attgt_cal import get_att_gt, get_standard_errors


class ATTgtPolars(DataES):
    def __init__(
            self,
            formula: str,
            data: pl.DataFrame,
            event_data: pl.DataFrame | None,
            anticipation: int = 0,
            *,
            entity_column: str,
            time_column: str,
            cohort_column: str,
            dosage_column: str = None,

            keep_columns: str | list[str] = None  # in case one doesn't want all the columns
    ):
        # --------------------------------------------------------------

        # moved formula up, and not in fit() anymore mainly so that if the panel is not
        # balanced because there are missing observations we'll know

        data_matrix = (
            DataMatrix(
                data=data,
                index_columns=[entity_column, time_column]
            )
            .build_model_matrix(formula=formula, include_intercept=True)
        )

        # --------------------------------------------------------------

        data_matrix.y_column
        data_matrix.x_columns

        super().__init__(
            data=data,
            event_data=event_data,
            entity_column=entity_column,
            time_column=time_column,
            cohort_column=cohort_column,
            dosage_column=dosage_column,
            anticipation=anticipation
        )

        # base_period: Literal["varying", "universal"] = "varying",

    def fit(
            self,
            est_method: Literal["reg", "dr", "dr-ipt", "std_ipw"] | Callable = "reg",
            control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",

            as_repeated_cross_section: bool = None,
            base_delta: str | list | dict = "base",
            *,
            sample_column: str = None,
            weights_column: str = None,
            cluster_column: str | list[str] = None,

            n_jobs: int = 1,
            backend: str = "loky",
            progress_bar: bool = True,
    ):
        pass


class ATTgtPolarsOLD:
    data = _ValiDIData()

    def __init__(
            self,
            data: PandasDF,
            cohort_column: str,
            dosage_column: str = None,  # extra treatment information
            base_period: Literal["varying", "universal"] = "varying",  # or 'universal'
            anticipation: int = 0,
    ):

        # from now on 'base_period' is called 'base_period_type'
        self.base_period_type = base_period

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

        self._data_matrix = PandasDF(self._data_matrix)

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

    # att gt
    def fit(
            self,
            formula: str,

            est_method: Literal["reg", "dr", "dr-ipt", "std_ipw"] | Callable = "reg",

            control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",

            as_repeated_cross_section: bool = None,
            base_delta: str | list | dict = "base",

            *,

            weights_column: str = None,
            sample_column: Callable | str | dict = None,
            cluster_column: list | str = None,

            alpha: float = 0.05,

            n_jobs: int = 1,
            backend: str = "loky",
            progress_bar: bool = True,

    ) -> ATTgtResult:

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
            cluster_var=cluster_column,
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

        add_cols = [self.cohort_column, self.dosage_column, weights_column, cluster_column]
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

        self._create_result_dict(sample_split_column=sample_column)

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
            if cluster_column:
                cluster_groups = get_cluster_groups(
                    data=(
                        self._data_matrix[cluster_column]
                        if s == "full_sample"
                        else self._data_matrix[cluster_column].loc[
                            self._result_dict[s]["sample_mask"]
                        ]
                    ),
                    cluster_var=cluster_column,
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

        return ATTgtResult(
            att_gt=self._result_dict,
            data_matrix=self._data_matrix,
            dosages=self._dosages,
            weights_column=self._weights_column,
            cluster_by_entity=self._cluster_by_entity
        )
