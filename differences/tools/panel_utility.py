import numpy as np
import pandas as pd

from pandas import DataFrame, MultiIndex

from typing import Union


# ------------------------- dates to int -------------------------------

def map_dt_to_int(start: str,
                  end: str,
                  freq: str = 'YS',
                  min_int_id: int = 1000) -> dict:
    """
    creates a dictionary mapping {datetime: int}

    in case the frequency of datetime is yearly,
    the integer will be the year

    todo: make sure the yearly flag is more comprehensive, ie A-DEC
    """

    date_map = pd.date_range(start=start, end=end, freq=freq)

    if freq in ['A', 'AS', 'Y', 'YS']:
        return dict(map(lambda x: (x, x.year), date_map))

    else:
        return {v: k for k, v in enumerate(date_map, start=min_int_id)}


def map_dt_series_to_int(dates,  # iterable
                         freq: str,
                         min_int_id: int = 1000) -> dict:
    """
    given a series of datetime values returns a dictionary mapping

    {datetime_value_1: int_1, dt_value_2: int_2, ...}

    helps convert dates to integers that preserve the time spacings
    between datetimes

    converting dates to integers makes it much easier to compute
    relative times. One can start with quarterly data or even minutes
    and the integers will make it easy  to calculate a difference in
    quarters, minutes, years...

    """
    start, end = dates.min(), dates.max()

    # # just to make sure not to miss the endpoints
    # start = str((start - np.timedelta64(1, 'Y')).date())
    # end = str((end + np.timedelta64(1, 'Y')).date())

    # get int map for the dates in the range between start and end
    return map_dt_to_int(start=start, end=end, freq=freq, min_int_id=min_int_id)


# -------------- gaps in the time var of the panel ---------------------


def panel_has_gaps(data: Union[DataFrame, MultiIndex],
                   return_gaps: bool = False
                   ) -> Union[bool, None, MultiIndex]:
    """
    checks whether there are gaps in the time var within entities

    Parameters
    ----------
    data
    return_gaps

    Returns
    -------
        - True/False indicating whether panel has gaps
        or
        - DataFrame with the number of gaps for each entity-time with the starting period
    """

    if isinstance(data, DataFrame):
        data = data.index

    entity_name, time_name = data.names
    entity_dtype, time_dtype = data.dtypes

    data = (
        DataFrame(index=data)
        .reset_index(level=[time_name], drop=False)
        .assign(
            shifted_time=lambda x: x.groupby(entity_name)[time_name].shift(1) + 1,
            time_gap=lambda x: x[time_name] - x['shifted_time'],
        )
    )

    has_gaps_bool = bool(data['time_gap'].max())

    if not return_gaps:  # only return True/False if there are gaps
        return has_gaps_bool

    elif not has_gaps_bool:  # if return_gaps, but no gaps
        return None

    gaps = data.loc[lambda x: x['time_gap'] > 0]

    ranges = map(lambda x: np.arange(*x), gaps[['shifted_time', time_name]].to_numpy())

    missing_times = (
        DataFrame(zip(gaps.index, ranges), columns=[entity_name, time_name])
        .explode([time_name])
        .astype({entity_name: entity_dtype, time_name: time_dtype})
    )
    return MultiIndex.from_frame(missing_times)


def reindex_gaps(data: Union[DataFrame, MultiIndex],
                 missing_index: MultiIndex = None,
                 fill_value: float = np.nan) -> DataFrame:
    """
    re-indexes a multiindex panel by filling the time gaps within entity

    currently pandas reindex on multiindex does not fill missing values
    and needs all levels to reindex
    issue: https://github.com/pandas-dev/pandas/issues/25460

    it is useful to temporarily fill the gaps in order to correctly
    create dummies in the current implementation of the relative dummies

    the gaps create issues mainly when there are multiple events per
    entity. binning of the endpoints would be off given the way the
    current implementation of binning works

    Parameters
    ----------
    data
    missing_index
    fill_value

    Returns
    -------

    """
    if missing_index is None:
        missing_index = panel_has_gaps(data=data, return_gaps=True)

    if isinstance(data, MultiIndex):
        data = DataFrame(index=data)

    return (
        data
        .reindex(
            data.index.append(missing_index),
            fill_value=fill_value)
        .sort_index()
    )


def reindex_times(data: DataFrame,
                  fill_value: float = np.nan,
                  start: int = None,
                  end: int = None) -> DataFrame:
    """
    re-indexes a multiindex panel by filling the time gaps within entity

    currently pandas reindex on multiindex does not fill missing values
    issue: https://github.com/pandas-dev/pandas/issues/25460

    it is useful to temporarily fill the gaps in order to correctly
    create dummies in the current implementation of the relative dummies

    the gaps create issues mainly when there are multiple events per
    entity. binning of the endpoints would be off given the way the
    current implementation of binning works

    Parameters
    ----------
    data
    fill_value
    start
    end

    Returns
    -------

    """
    entity_name, time_name = data.index.names
    entity_dtype, time_dtype = data.index.dtypes

    df_idx = (data
              .reset_index(level=time_name)
              .groupby(entity_name)[time_name]
              .agg(['min', 'max'])
              )

    if start is not None:
        df_idx['min'] = start

    if end is not None:
        df_idx['max'] = end

    df_idx['max'] += 1  # for range(x, x+1)

    ranges = map(lambda x: np.arange(*x),
                 df_idx[['min', 'max']].to_numpy())

    df_idx = (DataFrame(zip(df_idx.index, ranges),
                        columns=[entity_name, time_name])
              .explode([time_name]))

    # make sure it's an int. the time variable should always be int
    df_idx = df_idx.astype({entity_name: entity_dtype,
                            time_name: time_dtype})

    return data.reindex(df_idx
                        .sort_values([entity_name, time_name])
                        .to_records(index=False), fill_value=fill_value)


# ----------------------- make panel balanced --------------------------


def is_panel_balanced(data: DataFrame) -> bool:
    """checks if the panel is balanced: all units observed the same periods"""

    time_name = data.index.names[1]

    _, counts = np.unique(data.index.get_level_values(time_name),
                          return_counts=True)

    return len(set(counts)) == 1


# subset panel to create a balance panel with the same time periods

def n_entities_per_window(data: DataFrame,
                          min_time: int,
                          max_time: int,
                          min_n_periods: int) -> dict:
    """helper for: make_panel_balanced"""

    entity_name, time_name = data.index.names

    windows = [(t, t + min_n_periods - 1)
               for t in range(min_time, max_time)
               if t + min_n_periods - 1 <= max_time
               ]

    return {w: (data
                .loc[lambda x: x[time_name].between(*w)]
                .loc[lambda x: (x  # n_obs per entity
                                .groupby(entity_name)[time_name]
                                .transform('nunique') >= min_n_periods)]
                [entity_name]
                .nunique()  # n entities per time period span
                ) for w in windows}


def make_panel_balanced(data: DataFrame,
                        min_n_periods: int = None,
                        auto_increase_periods: bool = True,
                        outer_nested: bool = False) -> dict:
    if min_n_periods is not None and min_n_periods <= 1:
        raise ValueError("'min_n_periods' should be > 1")

    entity_name, time_name = data.index.names

    # maximum number of periods an entity is observed
    max_periods = data.groupby(entity_name)[time_name].nunique().max()

    # select outer_nested = True if one want to just to drop the entities
    # not observed for every time period. no min_n_periods specified
    if outer_nested and min_n_periods is None:
        return (data
                .loc[lambda x: (x
                                .groupby(entity_name)
                                [time_name]
                                .transform('nunique')
                                ) == max_periods]
                .reset_index(drop=True)
                )

    # minimum numbers of periods the user wants to observe an entity
    min_n_periods = min_n_periods or max_periods

    min_time = int(data[time_name].min())
    max_time = int(data[time_name].max())

    # time span = (start, end): n entities
    _options = n_entities_per_window(data=data,
                                     min_time=min_time,
                                     max_time=max_time,
                                     min_n_periods=min_n_periods)

    max_n_entities = max(_options.values())

    result = {k: v for k, v in _options.items()
              if v == max_n_entities}

    # if increase_periods == True: try to see if you can have the same
    # number of entities but observe them for more periods than the
    # chosen minimum number of periods.
    # It could be they end up being different entities
    if auto_increase_periods and max_n_entities:
        while True:

            min_n_periods += 1

            # time span = (start, end): n entities
            _options = n_entities_per_window(data=data,
                                             min_time=min_time,
                                             max_time=max_time,
                                             min_n_periods=min_n_periods)

            # break if no entities with that time period span
            if not _options:
                break
            else:
                new_max = max(_options.values())

            # break if less entities than
            # in the previous time period span
            if new_max < max_n_entities:
                break
            else:
                max_n_entities = new_max
                result = {k: v for k, v in _options.items()
                          if v == max_n_entities}
            print(f'increased to {min_n_periods} periods')

    # this return a dictionary which may have more than 1 entry:
    # one needs to select the entry
    return result


# ---------------------- panel 2 cross-section -------------------------


def find_time_varying_covars(data: DataFrame,
                             covariates: list,
                             rtol: float = None,  # 1e-05
                             atol: float = None  # 1e-08
                             ) -> list:
    """determines which columns vary within entity, over time"""
    # index must be set to entity-time

    entity_name = data.index.names[0]  # or 'time'

    if rtol is None and atol is None:
        varying = data.groupby([entity_name])[covariates].nunique().max(axis=0)
        return list(varying[varying > 1].index)

    # if one wants to set a tol for varying floats
    covars = data[covariates].sort_index()
    shift = covars.groupby([entity_name]).shift(1)

    varying = dict(zip(covariates,
                       np.sum(~np.isclose(covars[shift[covariates[0]].notnull()],
                                          shift[shift[covariates[0]].notnull()],
                                          rtol=1e-05,
                                          atol=1e-08),
                              axis=0)
                       )
                   )
    varying = [k for k, v in varying.items() if v]
    return varying


def delta_col_to_create(x_delta: list,
                        x_base: list,
                        return_idx: bool = False):
    """helper function for panel_2_cross_section_did

    identifies which columns to create with a prefix of 'delta_'
    """
    delta_to_create, create_idx, delta_to_replace, replace_idx = [], [], [], []

    for i, x in enumerate(x_delta):
        if x in x_base:
            delta_to_create.append(x)
            create_idx.append(i)
        else:
            delta_to_replace.append(x)
            replace_idx.append(i)
    if return_idx:
        return delta_to_create, create_idx, delta_to_replace, replace_idx
    else:
        return delta_to_create, delta_to_replace


def panel_2_cross_section_diff(data: DataFrame,
                               y_name: str,
                               x_base: list,
                               x_delta: list,
                               base_period: int = None,
                               time: int = None,
                               ):
    """
    Converts a balanced panel, with 2 periods, into a cross-section

    data: balanced panel with 2 periods. multilevel index must be set to _entity_, _time_
    """

    time_name = data.index.names[1]  # 'time'

    # maybe make this an argument instead of extracting it within the function
    times_values = data.index.get_level_values(time_name)

    if time is None or base_period is None:
        _times = times_values.unique()
        base_period, time = min(_times), max(_times)

    mask_time = times_values == time

    #  y[time] - y[base_period]
    y_delta = data[y_name].loc[mask_time].values - data[y_name].loc[~mask_time].values

    if x_delta:  # get delta Xs if requested: x[time] - x[base_period]
        # only time varying covariates
        tvc_delta = data[x_delta].loc[mask_time].values - data[x_delta].loc[~mask_time].values

    # keep vars in the earliest between time and base_period: x[time] or x[base_period]
    data = data.loc[mask_time].copy() if time < base_period else data.loc[~mask_time].copy()

    data[y_name] = y_delta

    if x_delta:
        delta_to_create, create_idx, delta_to_replace, replace_idx = delta_col_to_create(
            x_delta=x_delta,
            x_base=x_base,
            return_idx=True
        )

        if delta_to_create:  # if x is in both delta and base then add delta_ cols to data
            data[[f'delta_{c}' for c in delta_to_create]] = tvc_delta[:, create_idx]

        if delta_to_replace:  # replace base cols with delta, if these cols are not in x_base

            data[delta_to_replace] = tvc_delta[:, replace_idx]

        # warning: delta_to_replace cols will have the original name but they will be deltas
        # however the user has not access to this data

    return data
