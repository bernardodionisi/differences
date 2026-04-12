from __future__ import annotations

from _warnings import warn

import numpy as np
import pandas as pd

from pandas.core.dtypes.common import (
    is_string_dtype,
    is_datetime64_dtype,
    is_integer_dtype,
    is_numeric_dtype
)

from differences.models.attgt.data_validator.panel_utility import map_dt_series_to_int


def invalid_columns(data: pd.DataFrame, cols: list):
    inv_names = [c for c in list(data) if c in cols]
    if inv_names:
        raise ValueError(f"columns {inv_names} already in dataframe")


def error_if_nan_in_vars(data: pd.DataFrame, var_list: list):
    lnotnull = data.notna().all().loc[var_list]
    if not np.all(lnotnull):
        raise ValueError(
            f"there should be no missing data among the variables provided: "
            f"{list(lnotnull[~lnotnull].index)} contain nas"
        )


def replace_string_index_with_category_code(data: pd.DataFrame):
    # print("converting to integer")

    entity_name, time_name = data.index.names

    if is_string_dtype(data.index.get_level_values(0)):
        # convert the first index level to category
        new_index = data.index.get_level_values(0).astype('category')

        # use the category codes as the new index, keep the second level as-is
        data.index = pd.MultiIndex.from_arrays([
            new_index.codes,
            data.index.get_level_values(1)
        ], names=[entity_name, time_name])
    return data


def get_treatment_info_column(
        instance  # _ValiDIData
):
    treatment_info_column = getattr(instance, "division_column", None)

    if treatment_info_column is None:  # in TWFE
        treatment_info_column = getattr(instance, "intensity_column", None)

    return treatment_info_column


# not used anymore
# def is_single_event(data: DataFrame, event_dummy_name: str):
#     entity_name = data.index.names[0]
#     return data.groupby(entity_name)[event_dummy_name].sum().max() == 1


def _convert_times(
        instance,  # _ValiDIData,
        data: pd.DataFrame,
        cohort_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_column = instance.cohort_column
    time_name = instance._time_name

    # ------------------------------------------------------------------

    times = data.index.get_level_values(time_name)
    instance._is_two_period = times.nunique() == 2

    # --------- deal with times and cohorts dates --------------
    # if time and cohorts are datetime: convert to int

    instance._map_datetime = None

    if is_datetime64_dtype(times) and is_datetime64_dtype(cohort_data[cohort_column]):

        # new names: '_' prior to old name
        new_time_name = f"_{time_name}"
        new_cohort_column = f"_{cohort_column}"

        freq = getattr(instance, "freq", None)

        if freq is None:
            raise TypeError(
                "a value for 'freq' of the datetime index " "must be provided"
            )

        map_datetime = map_dt_series_to_int(
            dates=times.append(pd.Index(cohort_data[cohort_column])), freq=freq
        )

        instance._map_datetime = {v: k for k, v in map_datetime.items()}

        # --------------------------------------------------------------

        data[new_time_name] = list(pd.Series(times).replace(map_datetime))

        cohort_data[new_cohort_column] = list(
            cohort_data[cohort_column].replace(map_datetime)
        )

        if not is_integer_dtype(data[new_time_name]):
            raise ValueError(
                f"unable to convert all Timestamps of {time_name} "
                f"into integers.  datetime_freq' may be incorrect"
            )

        if not is_integer_dtype(cohort_data[new_cohort_column]):
            raise ValueError(
                f"unable to convert all Timestamps of {cohort_column} "
                f"into integers.  datetime_freq' may be incorrect"
            )

        # --------------------------------------------------------------

        # new time and cohort name, do now use the datetime columns anymore

        data.reset_index(level=[1], drop=True, inplace=True)

        # cohort_column has changed
        instance._time_name = new_time_name
        instance.cohort_column = new_cohort_column

    # if time is int and cohort is either int or float (because of nas)
    elif not (is_integer_dtype(times) and is_numeric_dtype(cohort_data[cohort_column])):
        raise ValueError(
            f"{time_name} and {cohort_column} " f"must be both numeric or both datetime"
        )

    else:  # both numeric

        # drop time index:
        # in case it is datetime it will be replaced, if int it will be re-appended later
        data.reset_index(level=[1], drop=False, inplace=True)

    return data, cohort_data


# --------------------------- cohort data ------------------------------

def pre_process_treated_before(
        cohort_data: pd.DataFrame,
        cohort_column: str,
        data: pd.DataFrame,
):
    """drops always treated entities"""

    # entities whose event happened BEFORE the start of their time
    treated_before = cohort_data.loc[
        lambda x: x[cohort_column] <= x["_min_time"]
    ].index.unique()

    if len(treated_before):
        warn(
            f"{len(treated_before)} entities have been "
            f"dropped because always treated "
            f"(treated from before their first time)"
        )

        cohort_data = cohort_data.loc[lambda x: ~(x.index.isin(treated_before))].copy()

        # dropping the entities that are always treated
        data = data.loc[lambda x: ~(x.index.isin(treated_before))].copy()

    return cohort_data, data


def pre_process_treated_after(
        cohort_data: pd.DataFrame,
        cohort_column: str,
        anticipation: int = 0
):
    """drops event dates that come after the end of the panel (for the entity)"""

    # entities whose event happened AFTER the end of their time
    bool_after_end = cohort_data[cohort_column] > cohort_data["_max_time"]

    # shouldn't this be [cohort - anticipation] ?
    # line 85: https://github.com/bcallaway11/did/blob/master/R/pre_process_did.R
    # bool_after_end = (cohort_data[cohort_column] -
    #                   anticipation > cohort_data['amax'])

    n_entities_after_end = np.sum(bool_after_end)  # number of entities
    if n_entities_after_end:
        # remove those entities from the cohort_data (in data they will be nans)
        cohort_data = cohort_data.loc[~bool_after_end].copy()

        warn(
            f"{n_entities_after_end} entity-events ignored because the event (cohort) date"
            f"\n is after the end of the panel for the specific entity, "
            f"as if never treated"
        )

    return cohort_data


def pre_process_no_never_treated(

        cohort_data: pd.DataFrame,
        data: pd.DataFrame,

        cohort_column: str,
        time_column: str,

        anticipation: int,
):
    """makes the last cohort the comparison group, if no never treated"""

    n_treated_entities = cohort_data.index.nunique()
    n_all_entities = data.index.nunique()

    no_never_treated_flag = n_treated_entities == n_all_entities

    if no_never_treated_flag:
        warn(
            "No never treated entities. Using the last treated group as comparison group and "
            "keeping only the time periods before the last cohort date"
        )

        last_cohort = cohort_data[cohort_column].max()

        # dropping it from the cohorts will place it in never_treated:
        # keep track and only allow not_yet_treated option
        cohort_data = cohort_data.loc[lambda x: x[cohort_column] != last_cohort].copy()

        # restrict the sample to be before the treated periods for the last cohort
        data = data.loc[lambda x: x[time_column] < last_cohort - anticipation].copy()

    return no_never_treated_flag, cohort_data, data


def preprocess_cohort_data(
        data: pd.DataFrame,
        cohort_data: pd.DataFrame,
        cohort_column: str,
        treatment_info_column: str
):
    """generate valid cohort_data

    if cohort data preprocess, else create a cohort dataframe from data & cohort_column
    """

    entity_name = data.index.names[0]

    # cohort name + intensity name
    cohort_info_list = (
        [cohort_column, treatment_info_column] if treatment_info_column else [cohort_column]
    )

    if cohort_data is None:

        cohort_data = (
            data.reset_index(level=[0], drop=False)[[entity_name] + cohort_info_list]
            .drop_duplicates()
            .set_index(entity_name)
        )

    else:  # cohort data es separate input
        cohort_data = (
            cohort_data.reset_index()[[entity_name] + cohort_info_list]
            .drop_duplicates()
            .sort_values([entity_name, cohort_column])
            .reset_index(drop=True)
        )

        if len(cohort_data[lambda x: x.duplicated([entity_name, cohort_column])]):
            raise ValueError(
                "only one event per unit-time period allowed, consider averaging "
                "the event intensity within that unit-time period"
            )

        cohort_data.set_index([entity_name], inplace=True)

    return cohort_data
