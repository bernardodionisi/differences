from __future__ import annotations

import polars as pl

from typing import Literal

from differences._OLD_NOW_PAGURO.shared.event.data_es import DataES
from differences._OLD_NOW_PAGURO.shared import DataMatrix
from differences._OLD_NOW_PAGURO.shared import get_relative_periods_data

from differences.models.lm.panel_ols import panel_ols_fit
from differences.models.lm.panel_results import PanelEffectsResultsMultiple, PanelEffectsResults


class TWFE(DataES):
    """
    Two-Way Fixed Effects

    Group
    -----
        model

    """

    def __init__(
            self,
            formula: str,
            data: pl.DataFrame,
            event_data: pl.DataFrame = None,

            filter_entities: bool = False,
            treat_bounds: int | tuple[int | None, int | None] | None = 1,
            last_cohort_as_never_treated: bool = True,

            *,

            entity_column: str,
            time_column: str,
            cohort_column: str,
            dosage_column: str = None,  # should be continuous
            keep_columns: list[str] = None,
    ):
        data_matrix = (
            DataMatrix(data=data, index_columns=[entity_column, time_column])
            .build_model_matrix(formula=formula, include_intercept=True, other_columns=True)
        )

        super().__init__(
            data=data_matrix.data_matrix,

            event_data=event_data,

            cohort_column=cohort_column,
            dosage_column=dosage_column,

            entity_column=entity_column,
            time_column=time_column,

            treat_bounds=treat_bounds,
            last_cohort_as_never_treated=last_cohort_as_never_treated,
            filter_entities=filter_entities,

            anticipation=0
        )

        self._y_column = data_matrix.y_column
        self._x_columns = data_matrix.x_columns

    def fit(
            self,
            reference_period: int | None = -1,
            start: int = None,
            end: int = None,
            bin_endpoints: bool | Literal["start", "end", "both"] = True,
            **kwargs
    ) -> PanelEffectsResults | PanelEffectsResultsMultiple:
        """
        Fit Two-Way Fixed-Effects

        Group
        -----
            Estimation
        """
        # use_dosage: bool = False,

        data = self._get_data_with_relative_periods(
            start=start,
            end=end,
            bin_endpoints=bin_endpoints,
            reference_period=reference_period,
        )

        result = panel_ols_fit(
            data=data,
            entity_column=self._entity_column,
            time_column=self._time_column,

            y_column=self._y_column,
            x_columns=self._relative_periods_columns + self._x_columns,

            entity_fixed_effects=True,
            time_fixed_effects=True,

            **kwargs
        )

        return result

    # ------------------------------------------------------------------

    def _get_data_with_overall_columns(self):
        #     if not is_single_event:
        #         raise ValueError('data must be single event')
        #     # todo: check these columns are not in data already
        #
        #     data['time_to_treatment_'] = data.index.get_level_values(1) - data[cohort_column]
        #     data['post_'] = (data.index.get_level_values(1) > data[cohort_column]).astype(int)
        #     data['treated_'] = (data[cohort_column].notna()).astype(int)
        #
        #     if fillna_cohort is not None:
        #         data[cohort_column] = data[cohort_column].fillna(fillna_cohort).astype(int)
        #
        #     if fillna_rel_period is not None:
        #         data['time_to_treatment_'] = (
        #             data['time_to_treatment_'].fillna(fillna_rel_period).astype(int)
        #         )
        pass

    def _get_data_with_relative_periods(
            self,
            reference_period: int = None,
            start: int = None,
            end: int = None,
            bin_endpoints: Literal["start", "end", "both"] = "both",
    ):
        relative_periods = get_relative_periods_data(
            data=self._event_data,
            entity_column=self._entity_column,
            time_column_name=self._time_column,
            cohort_column=self._cohort_column,
            reference_period=reference_period,
            dosage_column=self._dosage_column,
            min_max_time_columns=("_min_panel_time", "_max_panel_time"),
            start=start,
            end=end,
            bin_endpoints=bin_endpoints,
        )

        self._relative_periods_columns = [
            i for i in relative_periods.columns
            if i not in [self._entity_column, self._time_column]
        ]

        data = (
            self._data
            .join(relative_periods, on=[self._entity_column, self._time_column], how="left")
            .with_columns([pl.col(i).fill_null(0) for i in self._relative_periods_columns])
        )

        return data
