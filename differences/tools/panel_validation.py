import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas.api.types import (is_numeric_dtype,
                              is_datetime64_dtype,
                              is_integer_dtype)

from warnings import warn

from ..tools.panel_utility import (is_panel_balanced,
                                   map_dt_series_to_int)

__all__ = ['_ValiDIData']


# ----------------------------------------------------------------------


class _ValiDIData:
    def __set_name__(self, owner_class, prop_name):
        self.prop_name = prop_name

    def __set__(self, instance, data):

        # --------------- only pandas dataframes allowed ---------------

        if not isinstance(data, DataFrame):
            raise ValueError(f'{self.prop_name} must be a Pandas DataFrame')

        # ------------------ copy data ---------------------------------

        copy_data = instance.__dict__.get('copy_data', False)
        if copy_data:
            data = data.copy()

        # operations on data will mostly be inplace given the class is
        # providing an option copy_data

        # -------------- make some columns names reserved --------------

        # data should not already contain columns with the following names,
        # these names will be used during computation

        # there are some extra, to be removed
        invalid_columns(data, ['_relative_time',
                               '_cohort',
                               '_treated',
                               '_control',
                               '_after_event',
                               '_w'
                               ]
                        )

        # ------------------- multi-index: too may ---------------------

        if data.index.nlevels != 2:
            raise ValueError("DataFrame must be MultiIndex with two levels: entity & time")

        # ----------------- multi-index: entity - time -----------------

        elif data.index.nlevels == 2:

            entity_name, time_name = data.index.names

            instance._entity_name = entity_name
            instance._time_name = time_name

            cohort_name = getattr(instance, 'cohort_name')
            division_name = get_division_name(instance)

            # ---------------- is repeated cross section? --------------

            instance.is_panel = True

            if data.index.get_level_values(entity_name).nunique() == len(data):
                instance.is_repeated_cross_section = True
                instance.is_panel = False

            # ------------------- is balanced panel? -------------------

            instance.is_balanced_panel = None

            if instance.is_panel:
                instance.is_balanced_panel = is_panel_balanced(data)

            # --------- deal with cohort dates/periods -----------------

            cohort_data = getattr(instance, 'cohort_data', None)

            # if cohort data preprocess, else create a cohort dataframe from data & cohort_name
            cohort_data = preprocess_cohort_data(
                data=data,
                cohort_data=cohort_data,
                cohort_name=cohort_name,
                intensity_name=division_name
            )

            # to check (below) if 0s were passed instead of NAs
            cohort_data_has_nas = bool(len(cohort_data[lambda x: x[cohort_name].isnull()]))

            if cohort_data_has_nas:
                cohort_data = cohort_data.dropna().copy()
            else:
                pass
                # warn('there are no never treated entity')

            # --------- deal with times and cohorts dates --------------
            # if time and cohorts are datetime: convert to int
            instance._map_datetime = None

            data, cohort_data = _convert_times(
                instance=instance, data=data, cohort_data=cohort_data)

            time_name, cohort_name = instance._time_name, instance.cohort_name
            # ----------------------------------------------------------

            # if there are no nas and there is a cohort of 0 but no 0 times
            if not cohort_data_has_nas and (min(data[time_name]) !=
                                            min(cohort_data[cohort_name]) == 0):
                raise ValueError(
                    f"{cohort_name} should not contain 0s for never treated groups, "
                    f"\n replace the 0s with np.nan, for example by"
                    f"\n >>> df['{cohort_name}'] = np.where(df['{cohort_name}'] == 0, "
                    f"np.nan, df['{cohort_name}'])")

            cohort_info_list = [instance.cohort_name,
                                division_name] if division_name else [instance.cohort_name]

            # for each entity-event_date,
            # create the start (amin) and end (amax) of the panel for the corresponding entity
            cohort_data = (
                cohort_data[cohort_info_list]
                .join(
                    data
                    .groupby(entity_name)[time_name]
                    .agg([np.min, np.max])
                )
            )

            # ----------- pre-process cohorts / data -------------------

            anticipation = getattr(instance, 'anticipation', None)

            no_never_treated_flag = None

            # todo: look into non ATTgt classes how to handle this cases
            if instance.is_panel and anticipation is not None:
                cohort_data, data = pre_process_treated_before(
                    cohort_data=cohort_data,
                    cohort_name=cohort_name,
                    data=data,
                    copy_data=copy_data
                )

                cohort_data = pre_process_treated_after(
                    cohort_data=cohort_data,
                    cohort_name=cohort_name,
                    anticipation=anticipation  # no effect for now
                )

                no_never_treated_flag, cohort_data, data = pre_process_no_never_treated(
                    cohort_data=cohort_data,
                    cohort_name=cohort_name,
                    data=data,
                    time_name=time_name,
                    anticipation=anticipation,
                    copy_data=copy_data
                )

            instance._no_never_treated_flag = no_never_treated_flag

            # ------------------ multiple events -----------------------

            # if there is more than one event date per entity
            is_single_event = cohort_data.index.nunique() == len(cohort_data)

            instance.is_single_event = is_single_event

            # --------- replace cohort data in main data ---------------

            cohort_data[cohort_name] = cohort_data[cohort_name].astype(int)

            instance.cohort_data = cohort_data

            if is_single_event:
                # use the entity index to do the assignment
                data[cohort_name] = cohort_data[cohort_name]

                if division_name is not None:
                    data[division_name] = cohort_data[division_name]

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


# ------------------------- helpers ------------------------------------

def get_division_name(instance):
    division_name = getattr(instance, 'division_name', None)

    if division_name is None:  # in TWFE
        division_name = getattr(instance, 'intensity_name', None)

    return division_name


def _convert_times(instance, data, cohort_data):
    cohort_name = getattr(instance, 'cohort_name')
    time_name = getattr(instance, '_time_name')

    # ------------------------------------------------------------------

    times = data.index.get_level_values(time_name)
    instance._is_two_period = times.nunique() == 2

    # --------- deal with times and cohorts dates --------------
    # if time and cohorts are datetime: convert to int

    instance._map_datetime = None

    if is_datetime64_dtype(times) and is_datetime64_dtype(cohort_data[cohort_name]):

        # new names: '_' prior to old name
        new_time_name = f'_{time_name}'
        new_cohort_name = f'_{cohort_name}'

        freq = getattr(instance, 'freq', None)

        if freq is None:
            raise TypeError("a value for 'freq' of the datetime index "
                            "must be provided")

        map_datetime = map_dt_series_to_int(
            dates=times.append(pd.Index(cohort_data[cohort_name])),
            freq=freq)

        instance._map_datetime = {v: k for k, v in map_datetime.items()}

        # --------------------------------------------------------------

        data[new_time_name] = list(pd.Series(times).replace(map_datetime))

        cohort_data[new_cohort_name] = list(cohort_data[cohort_name].replace(map_datetime))

        if not is_integer_dtype(data[new_time_name]):
            raise ValueError(f"unable to convert all Timestamps of {time_name} "
                             f"into integers.  datetime_freq' may be incorrect")

        if not is_integer_dtype(cohort_data[new_cohort_name]):
            raise ValueError(f"unable to convert all Timestamps of {cohort_name} "
                             f"into integers.  datetime_freq' may be incorrect")

        # --------------------------------------------------------------

        # new time and cohort name, do now use the datetime columns anymore

        data.reset_index(level=[1], drop=True, inplace=True)

        # cohort_name has changed
        instance._time_name = new_time_name
        instance.cohort_name = new_cohort_name

    # if time is int and cohort is either int or float (because of nas)
    elif not (is_integer_dtype(times) and is_numeric_dtype(cohort_data[cohort_name])):
        raise ValueError(f'{time_name} and {cohort_name} '
                         f'must be both numeric or both datetime')

    else:  # both numeric

        # drop time index:
        # in case it is datetime it will be replaced, if int it will be re-appended later
        data.reset_index(level=[1], drop=False, inplace=True)

    return data, cohort_data


def invalid_columns(data: DataFrame,
                    cols: list):
    inv_names = [c for c in list(data) if c in cols]
    if inv_names:
        raise ValueError(f'columns {inv_names} already in dataframe')


# extract relative time dummies, cohorts and event size variables

def error_if_nan_in_vars(data: DataFrame,
                         var_list: list):
    lnotnull = data.notnull().all().loc[var_list]
    if not np.all(lnotnull):
        raise ValueError(f'there should be no missing data among the variables provided: '
                         f'{list(lnotnull[~lnotnull].index)} contain nas')


# not used anymore
def is_single_event(data: DataFrame,
                    event_dummy_name: str):
    entity_name = data.index.names[0]
    return data.groupby(entity_name)[event_dummy_name].sum().max() == 1


def pre_process_treated_before(cohort_data: DataFrame,
                               cohort_name: str,
                               data: DataFrame,
                               copy_data: bool = True,  # just to issue a warning
                               ):
    """drops always treated entities"""

    # entities whose event happened BEFORE the start of their time
    treated_before = cohort_data.loc[
        lambda x: x[cohort_name] <= x['amin']].index.unique()

    if len(treated_before):
        warn(f'{len(treated_before)} entities have been '
             f'dropped because always treated '
             f'(treated from before their first time)')

        cohort_data = (cohort_data
                       .loc[lambda x: ~(x.index.isin(treated_before))].copy())

        # dropping the entities that are always treated
        data = data.loc[lambda x: ~(x.index.isin(treated_before))].copy()

        if not copy_data:
            warn(f'copied data')

    return cohort_data, data


def pre_process_treated_after(cohort_data: DataFrame,
                              cohort_name: str,
                              anticipation: int = 0
                              ):
    """drops event dates that come after the end of the panel (for the entity)"""

    # entities whose event happened AFTER the end of their time
    bool_after_end = (cohort_data[cohort_name] > cohort_data['amax'])

    # shouldn't this be [cohort - anticipation] ?
    # line 85: https://github.com/bcallaway11/did/blob/master/R/pre_process_did.R
    # bool_after_end = (cohort_data[cohort_name] -
    #                   anticipation > cohort_data['amax'])

    n_entities_after_end = np.sum(bool_after_end)  # number of entities
    if n_entities_after_end:
        # remove those entities from the cohort_data (in data they will be nans)
        cohort_data = cohort_data.loc[~bool_after_end].copy()

        warn(f'{n_entities_after_end} entity-events ignored because the event (cohort) date'
             f'\n is after the end of the panel for the specific entity, '
             f'as if never treated')

    return cohort_data


def pre_process_no_never_treated(cohort_data: DataFrame,
                                 cohort_name: str,
                                 data: DataFrame,
                                 time_name: str,
                                 anticipation: int,
                                 copy_data: bool = True,  # just to raise a warning
                                 ):
    """makes the last cohort the comparison group, if no never treated"""

    n_treated_entities = cohort_data.index.nunique()
    n_all_entities = data.index.nunique()

    no_never_treated_flag = n_treated_entities == n_all_entities

    if no_never_treated_flag:
        warn(f'No never treated entities. Using the last treated group as comparison group and '
             f'keeping only the time periods before the last cohort date')

        last_cohort = cohort_data[cohort_name].max()

        # dropping it from the cohorts will place it in never_treated:
        # keep track and only allow not_yet_treated option
        cohort_data = cohort_data.loc[lambda x: x[cohort_name] != last_cohort].copy()

        # restrict the sample to be before the treated periods for the last cohort
        data = data.loc[lambda x: x[time_name] < last_cohort - anticipation].copy()

        if not copy_data:
            warn(f'copied data')

    return no_never_treated_flag, cohort_data, data


# --------------------- cohort data ------------------------------------

def preprocess_cohort_data(data: DataFrame,
                           cohort_data: DataFrame,
                           cohort_name: str,
                           intensity_name: str):
    """generate valid cohort_data

    if cohort data preprocess, else create a cohort dataframe from data & cohort_name
    """

    entity_name = data.index.names[0]

    # cohort name + intensity name
    cohort_info_list = [cohort_name, intensity_name] if intensity_name else [cohort_name]

    if cohort_data is None:

        cohort_data = (
            data
            .reset_index(level=[0], drop=False)
            [[entity_name] + cohort_info_list]
            .drop_duplicates()
            .set_index(entity_name)
        )

    else:  # cohort data es separate input
        cohort_data = (
            cohort_data
            .reset_index()
            [[entity_name] + cohort_info_list]
            .drop_duplicates()
            .sort_values([entity_name, cohort_name])
            .reset_index(drop=True)
        )

        if len(cohort_data[lambda x: x.duplicated([entity_name, cohort_name])]):
            raise ValueError('only one event per unit-time period allowed, consider averaging '
                             'the event intensity within that unit-time period')

        cohort_data.set_index([entity_name], inplace=True)

    return cohort_data
