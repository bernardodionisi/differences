from __future__ import annotations

from functools import lru_cache

import pandas as pd
from pandas import DataFrame

from ..tools.feols.ols import FEols
from ..tools.panel_validation import _ValiDIData
from ..tools.relative_periods import (bin_relative_periods_dummies,
                                      cohort_info_table,
                                      get_relative_periods_dummies)
from ..tools.utility import (EventStudyResult, bin_start_end, process_formula)
from .utility import add_stack_fe, stack_did_data


class TWFE:
    """
    Two-way fixed effect regression

        - balanced panels, unbalanced panels
        - two or multiple periods
        - fixed or staggered treatment timing
        - binary treatment, with various intensities
        - one or multiple treatments per entity

    Parameters
    ----------
    data: DataFrame
        pandas DataFrame

        .. code-block:: python

            df = df.set_index(['entity', 'time'])

        where *df* is the dataframe to use, *'entity'* should be replaced with the
        name of the entity column and *'time'* should be replaced with
        the name of the time column.

    cohort_name: str
        cohort name

    cohort_data: DataFrame
        cohort data, in place of cohort name

    intensity_name: str
        name of the column with treatment intensities

    freq: *str* | None, default: ``None``
        the date frequency of the panel data. Required if the time index is datetime.
        For example, if the time column is a monthly datetime then freq='M'. Check
        `offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_,
        for a list of available frequencies.
    """

    data = _ValiDIData()

    def __init__(
            self,
            data: DataFrame,
            cohort_name: str,
            cohort_data: DataFrame = None,
            intensity_name: str = None,
            freq: str = None,  # frequency if cohort is datetime
            stacked: bool = False,
    ):

        self.cohort_name = cohort_name
        self.intensity_name = intensity_name

        self.freq = freq
        self.cohort_data = cohort_data

        self.copy_data = True

        self.data = data
        self._data_matrix = None

        self._relative_periods = None

        self._result = None
        self._formula = None
        self._cov_estimator = None

        self._stacked = stacked

        if self._stacked:
            self.stacked_data = stack_did_data(
                data=self.data, cohort_name=self.cohort_name
            )
        else:
            self.stacked_data = None

    @lru_cache(maxsize=2)
    def _get_relative_periods(
            self,
            start: int = None,
            end: int = None,
            intensity: bool = False,
            bin_start: bool = True,
            bin_end: bool = True,
            reference_period: int = None,
            _reindex: bool = True,
    ):

        cohort_table = cohort_info_table(
            cohort_data=self.cohort_data,
            pivot_values=self.cohort_name,
            data_index=self.data.index,
            fill_gaps=True,
        )

        if intensity:
            if self.intensity_name is None:
                raise ValueError(
                    "did not provide intensity_name when instantiating "
                    "the TWFE class"
                )

            intensity_table = cohort_info_table(
                cohort_data=self.cohort_data,
                pivot_values=self.intensity_name,
                data_index=self.data.index,
                fill_gaps=True,
            )
        else:
            intensity_table = None

        self._relative_periods, first_rp, last_rp = get_relative_periods_dummies(
            cohort_table=cohort_table,
            intensity_table=intensity_table,
            start=start,
            end=end,
        )

        if (
                (isinstance(start, int) and start != first_rp)
                or (isinstance(end, int) and end != last_rp)
        ) and (bin_start or bin_end):

            self._binned_relative_periods = bin_relative_periods_dummies(
                periods_dummies=self._relative_periods,
                bin_start=bin_start,
                bin_end=bin_end,
                copy_data=True,
            )

            # dropping the reference period
            if reference_period is not None:
                self._idx_reference_period = list(self._relative_periods).index(
                    reference_period
                )
                self._binned_relative_periods.drop(
                    columns=[reference_period], inplace=True
                )

            # if binned endpoints make the relative period names strings
            rp_names = [str(i) for i in list(self._binned_relative_periods)]

            if isinstance(start, int) and start != first_rp and bin_start:
                # careful if -n was dropped but was the endpoint
                if str(reference_period) != rp_names[0]:
                    rp_names[0] = f"≤ {rp_names[0]}"

            if isinstance(end, int) and end != last_rp and bin_end:
                # careful if n was dropped but was the endpoint
                if str(reference_period) != rp_names[0]:
                    rp_names[-1] = f"≥ {rp_names[-1]}"

            self._binned_relative_periods.columns = rp_names
            self._rt_names = list(self._binned_relative_periods)

            if _reindex:
                self._binned_relative_periods = self._binned_relative_periods.reindex(
                    self.data.index, fill_value=0
                )

            return self._binned_relative_periods

        if reference_period is not None:
            self._idx_reference_period = list(self._relative_periods).index(
                reference_period
            )
            self._relative_periods.drop(columns=[reference_period], inplace=True)

        self._rt_names = list(self._relative_periods)

        if _reindex:
            self._relative_periods = self._relative_periods.reindex(
                self.data.index, fill_value=0
            )

        return self._relative_periods

    def fit(
            self,
            formula: str,
            start: int = None,
            end: int = None,
            reference_period: int | None = -1,
            bin_endpoints: bool | str = True,
            weights_name: str = None,
            cluster_names: str | list = None,
            alpha: float = 0.05,
            use_intensity: bool = False,
            drop_singletons: bool = True,
            drop_absorbed: bool = False,
            dummies_names: list[str] = None,
            drop_names: list[str] = None,
    ):
        """
        fit two-way fixed effect

        Parameters
        ----------
        formula: str
            Wilkinson formula for the outcome variable and covariates

            If no covariates the formula must contain only the name of the outcome variable

            .. code-block:: python

                # example with covariates
                formula = 'y ~ a + b + a:b'

                # example without covariates
                formula = 'y'

            Formulas are implemented using
            `formulaic <https://matthewwardrop.github.io/formulaic/>`_,
            refer to its documentation for additional details.

        start
             first relative period
        end
             last relative period

        reference_period
            reference period

        bin_endpoints
                weights_name: *str* | None, default: ``None``

            The name of the column containing the sampling weights.
            If None, all observations have same weights.

        weights_name
        cluster_names
        alpha
        use_intensity
        drop_singletons
        drop_absorbed
        dummies_names
        drop_names

        Returns
        -------
        """

        self._formula, self._fe = process_formula(
            formula=formula,
            entity_name=self._entity_name,
            time_name=self._time_name,
            stacked=self._stacked,
            return_fe=True,
        )

        if self._stacked and self._fe:
            self.stacked_data = add_stack_fe(
                stacked_data=self.stacked_data, fixed_effects=self._fe
            )

        fe_ols = FEols(
            data=self.data if self.stacked_data is None else self.stacked_data,
            formula=self._formula,
            weights_name=weights_name,
            cluster_names=cluster_names,
            drop_singletons=drop_singletons,
            absorb_only_one_fe=False,
            copy_data=False,
        )

        bin_start, bin_end = bin_start_end(bin_endpoints)

        ols_result = fe_ols.result_table(
            extra_data_matrix=self._get_relative_periods(
                start=start,
                end=end,
                intensity=use_intensity,
                bin_start=bin_start,
                bin_end=bin_end,
                reference_period=reference_period,
            ),
            drop_absorbed=drop_absorbed,
            alpha=alpha,
            dummies_names=dummies_names,
            drop_names=drop_names,
        )

        self._cov_estimator = fe_ols._cov_estimator

        # save estimates by relative period
        event_study_est = ols_result[lambda x: x.index.isin(self._rt_names)]

        if reference_period is not None:
            if isinstance(self._rt_names[0], str):
                reference_period = str(reference_period)

            self._rt_names.insert(self._idx_reference_period, reference_period)

            reference_period_est = DataFrame(
                index=[reference_period], columns=list(event_study_est)
            ).fillna(0)

            event_study_est = pd.concat(
                [event_study_est, reference_period_est]
            ).reindex(self._rt_names)

        event_study_est.index.name = "relative_period"

        self._result = EventStudyResult(
            covariate_estimates=ols_result[lambda x: ~x.index.isin(self._rt_names)],
            event_study_est=event_study_est,
            formula=self._formula,
            weights_name=weights_name,
            cluster_names=cluster_names,
            use_intensity=use_intensity,
            bin_endpoints=bin_endpoints,
        )

        return ols_result

    @property
    def estimation_details(self):

        details = {
            "is_panel": self.is_panel,
        }

        if self.is_panel:
            details.update(
                {"is_balanced_panel": getattr(self, "is_balanced_panel", False)}
            )

        if self._result:
            details.update(
                {"formula": self._formula, "cov_estimator": self._cov_estimator}
            )

        return details
