from __future__ import annotations

import pandas as pd
from differences.models.attgt.data_validator.panel_utility import is_panel_balanced

from differences.models.attgt.data_validator.data_validator_utility import (
    invalid_columns,
    pre_process_treated_before,
    pre_process_treated_after,
    pre_process_no_never_treated,
    preprocess_cohort_data,
    replace_string_index_with_category_code,
    get_treatment_info_column,
    _convert_times
)


class _ValiDIData:
    def __set_name__(self, owner_class, prop_name):
        self.prop_name = prop_name

    def __set__(self, instance, data):

        # --------------- only pandas dataframes allowed ---------------

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{self.prop_name} must be a Pandas DataFrame")

        # ------------------ copy data ---------------------------------

        copy_data = instance.__dict__.get("copy_data", False)
        if copy_data:
            data = data.copy()

        # todo: change in place opeartions

        # operations on data will mostly be inplace given the class is
        # providing an option copy_data

        # convert entities to integer if they are strings
        data = replace_string_index_with_category_code(data)

        # -------------- make some columns names reserved --------------

        # data should not already contain columns with the following names,
        # these names will be used during computation

        # there are some extra, to be removed
        invalid_columns(
            data,
            ["_relative_time", "_cohort", "_treated", "_control", "_after_event", "_w"],
        )

        # ------------------- multi-index: too may ---------------------

        if data.index.nlevels != 2:
            raise ValueError(
                "DataFrame must have a MultiIndex with two levels: entity & time. The first index"
                "are the identifier for the entities and the second index the "
                "identifiers for calendar time. Time can be either an integer or a datetime."
            )

        # ----------------- multi-index: entity - time -----------------

        # data.index.nlevels == 2

        entity_name, time_name = data.index.names

        instance._entity_name = entity_name
        instance._time_name = time_name

        cohort_column = instance.cohort_column
        division_column = get_treatment_info_column(instance)

        # ---------------- is repeated cross section? --------------

        instance.is_panel = True

        if data.index.get_level_values(entity_name).nunique() == len(data):
            instance.is_repeated_cross_section = True
            instance.is_panel = False

        # ------------------- is balanced panel? -------------------

        instance._is_balanced_panel = None

        if instance.is_panel:
            instance.is_balanced_panel = is_panel_balanced(data)

        # --------- deal with cohort dates/periods -----------------

        cohort_data = getattr(instance, "cohort_data", None)

        # if cohort data preprocess, else create a cohort dataframe from data & cohort_column
        cohort_data = preprocess_cohort_data(
            data=data,
            cohort_data=cohort_data,
            cohort_column=cohort_column,
            treatment_info_column=division_column,
        )

        # to check (below) if 0s were passed instead of NAs
        cohort_data_has_nas = bool(
            len(cohort_data[lambda x: x[cohort_column].isna()])
        )

        if cohort_data_has_nas:
            cohort_data = cohort_data.dropna().copy()
        else:
            pass
            # warn('there are no never treated entity')

        # --------- deal with times and cohorts dates --------------
        # if time and cohorts are datetime: convert to int
        instance._map_datetime = None

        data, cohort_data = _convert_times(
            instance=instance, data=data, cohort_data=cohort_data
        )

        time_name, cohort_column = instance._time_name, instance.cohort_column
        # ----------------------------------------------------------

        # if there are no nas and there is a cohort of 0 but no 0 times
        if not cohort_data_has_nas and (
                min(data[time_name]) != min(cohort_data[cohort_column]) == 0
        ):
            polars_message = f"""
            df = df.with_columns(
                    pl.when(pl.col('{cohort_column}') == 0)
                    .then(None)
                    .otherwise(pl.col('{cohort_column}'))
                )
            """
            raise ValueError(
                f"{cohort_column} should not contain 0s for never treated groups, "
                f"\n replace the 0s with np.nan, for example by"
                f"\n >>> df['{cohort_column}'] = np.where(df['{cohort_column}'] == 0, "
                f"np.nan, df['{cohort_column}'])\n\nor in polars:\n {polars_message}"
            )

        cohort_info_list = (
            [instance.cohort_column, division_column]
            if division_column
            else [instance.cohort_column]
        )

        # for each entity-event_date,
        # create the start and end of the panel for the corresponding entity
        cohort_data = (
            cohort_data[cohort_info_list]
            .join(
                data
                .groupby(entity_name)[time_name]
                .agg(['min', 'max'])
                .set_axis(labels=['_min_time', '_max_time'], axis=1)
            ))

        # ----------- pre-process cohorts / data -------------------

        anticipation = getattr(instance, "anticipation", None)

        no_never_treated_flag = None

        # todo: look into non ATTgt classes how to handle this cases
        if instance.is_panel and anticipation is not None:
            cohort_data, data = pre_process_treated_before(
                cohort_data=cohort_data,
                cohort_column=cohort_column,
                data=data,
            )

            cohort_data = pre_process_treated_after(
                cohort_data=cohort_data,
                cohort_column=cohort_column,
                anticipation=anticipation,  # no effect for now
            )

            no_never_treated_flag, cohort_data, data = pre_process_no_never_treated(
                cohort_data=cohort_data,
                cohort_column=cohort_column,
                data=data,
                time_column=time_name,
                anticipation=anticipation,
            )

        instance._no_never_treated_flag = no_never_treated_flag

        # ------------------ multiple events -----------------------

        # if there is more than one event date per entity
        is_single_event = cohort_data.index.nunique() == len(cohort_data)

        instance._is_single_event = is_single_event

        # --------- replace cohort data in main data ---------------

        cohort_data[cohort_column] = cohort_data[cohort_column].astype(int)

        instance._cohort_data = cohort_data

        if is_single_event:
            # use the entity index to do the assignment
            data[cohort_column] = cohort_data[cohort_column]

            if division_column is not None:
                data[division_column] = cohort_data[division_column]

        # ----------------------------------------------------------

        data.set_index([time_name], append=True, inplace=True)

        # ----------------------------------------------------------

        data.sort_index(inplace=True)

        instance.__dict__[self.prop_name] = data

    def __get__(self, instance, owner_class):
        if instance is None:
            return self
        else:
            return instance.__dict__.get(self.prop_name, None)
