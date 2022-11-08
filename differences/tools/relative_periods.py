import numpy as np
import pandas as pd

from typing import Optional, Union

from pandas import DataFrame, MultiIndex, Index

from ..tools.panel_utility import panel_has_gaps, reindex_gaps


def cohort_info_table(cohort_data: DataFrame,
                      pivot_values: str,
                      data_index: Optional[Union[MultiIndex, Index]] = None,
                      fill_gaps: bool = True,
                      ) -> DataFrame:
    """
    pivots cohort_data

    with rows=entities (or entities-times) & columns=event numbers
    values will be one of:
        - the times of treatment (cohort)
        - the treatment intensities

    Parameters
    ----------
    cohort_data
    pivot_values
    data_index:
        used t reindex and expand the cohort data to the panel level

        - full panel data index
        - index of the treated observations (pre&post)
    fill_gaps

    Returns
    -------

    """
    entity_name = cohort_data.index.name

    cohort_table = (
        cohort_data
        .sort_index()
        .assign(event_number=lambda x: x.groupby(entity_name).cumcount() + 1)
        .pivot_table(index=entity_name, values=pivot_values, columns='event_number')
    )

    if data_index is not None:  # expand the cohort_data to panel leve, just for treated
        entity_name, time_name = data_index.names

        cohort_table_rep = (  # only keep treated
            DataFrame(index=data_index)
            .loc[lambda x: x.index.get_level_values(entity_name).isin(cohort_table.index)]
        )

        if fill_gaps:  # fill the gaps in the panel
            gaps = panel_has_gaps(data=cohort_table_rep, return_gaps=True)
            if gaps is not None:  # fill time gaps within entity
                cohort_table_rep = reindex_gaps(data=cohort_table_rep, missing_index=gaps)

        cohort_table_rep.reset_index(level=[time_name], drop=False, inplace=True)
        cohort_table_rep[[i for i in list(cohort_table)]] = cohort_table

        cohort_table_rep.set_index([time_name], append=True, inplace=True)
        cohort_table_rep.sort_index(inplace=True)

        return cohort_table_rep

    return cohort_table


def get_relative_periods(cohort_table: DataFrame):
    return np.array(cohort_table.index.get_level_values(1))[:, None] - cohort_table


def get_relative_periods_dummies(cohort_table: DataFrame,
                                 intensity_table: DataFrame = None,
                                 start: int = None,
                                 end: int = None,
                                 ) -> tuple[DataFrame, int, int]:
    if start is not None and end is not None:
        if start * end > 0 and abs(start) > abs(end):
            raise ValueError(f'must be: start <= end')

    rp_table = get_relative_periods(cohort_table)

    first_rp = int(np.min(rp_table.fillna(0).values))
    last_rp = int(np.max(rp_table.fillna(0).values))

    start = first_rp if start is None else start
    end = last_rp if end is None else end

    window = np.arange(start, end + 1)

    # if there is only 1 column (named 1) in rp_table, only one event per entity
    is_single_event = len(list(rp_table)) == 1

    # include all relative times from the first to the last (should be faster with get_dummies)
    all_dummies = (start == first_rp) and (end == last_rp)

    if intensity_table is None:

        if is_single_event and all_dummies:
            # [1] because events are indexed starting from 1, in cohort_info_table - event_number
            events_cols = pd.get_dummies(rp_table[1])

        else:
            events_cols = (
                pd.concat([
                    DataFrame(np.sum(rp_table == t, axis=1), columns=[t])
                    for t in window], axis=1)
            )

    else:  # if intensity_table
        if is_single_event and all_dummies:
            events_cols = pd.get_dummies(rp_table[1]) * intensity_table[[1]].to_numpy()

        else:
            events_cols = (
                pd.concat([
                    DataFrame(np.sum(intensity_table[(rp_table == t)], axis=1), columns=[t])
                    for t in window], axis=1)
            )

    return events_cols, first_rp, last_rp


def bin_relative_periods_dummies(periods_dummies: DataFrame,
                                 bin_start: bool = True,
                                 bin_end: bool = True,
                                 copy_data: bool = True) -> DataFrame:
    """bins the relative times at the endpoints. if multiple events it's a sum"""
    if copy_data:
        periods_dummies = periods_dummies.copy()

    entity_name = periods_dummies.index.names[0]

    periods = list(periods_dummies)
    start, end = min(periods), max(periods)

    if start == end:
        if bin_start and bin_end:
            raise ValueError('when start=end, '
                             'bin endpoints should be either start or end, not both')

    if bin_start:
        periods_dummies[start] = (
            periods_dummies
            .sort_index(ascending=False)
            .groupby(entity_name)[start]
            .transform('cumsum')
        )

    if bin_end:
        periods_dummies[end] = (
            periods_dummies
            .sort_index()
            .groupby(entity_name)[end]
            .transform('cumsum')
        )

    if bin_start and not bin_end:
        periods_dummies.sort_index(inplace=True)

    return periods_dummies


def reindex_periods(periods_dummies: DataFrame,
                    reindex_index: MultiIndex) -> DataFrame:
    return periods_dummies.reindex(reindex_index, fill_value=0).astype(int)

# ----------------------------------------------------------------------
