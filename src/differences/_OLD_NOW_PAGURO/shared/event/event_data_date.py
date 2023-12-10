from __future__ import annotations

import warnings

import polars as pl
from polars.type_aliases import IntoExpr

from differences._OLD_NOW_PAGURO.shared.dates import date_as_int, PeriodFrequency


def event_data_to_period_frequency(
        event_data: pl.DataFrame,

        entity_column: str | None,
        cohort_column: str | None,

        period_frequency: PeriodFrequency,
        **named_aggs: IntoExpr,

) -> tuple[pl.DataFrame, str, str]:
    """preprocess event data: """

    # automatically identify entity_column and cohort_column
    entity_column, cohort_column = _discern_entity_cohort(
        event_data_columns=event_data.columns,
        entity_column=entity_column,
        cohort_column=cohort_column
    )

    # Deleting never-treated entities.
    event_data = event_data.filter(pl.col(cohort_column).is_not_null())

    # Convert date to integer.
    if event_data[cohort_column].dtype is pl.Date:
        event_data = (
            event_data
            .with_columns([
                date_as_int(cohort_column, period_frequency=period_frequency)
            ])
        )
    elif not event_data[cohort_column].is_integer():
        raise ValueError(
            f"'cohort_column': {cohort_column} must be either an integer or a pl.Date")

    # once the cohort column is converted, event_data may need to be aggregated
    event_data = _aggregate_event_data(
        event_data=event_data,
        entity_column=entity_column,
        cohort_column=cohort_column,
        **named_aggs,
    )

    return event_data, entity_column, cohort_column


def _aggregate_event_data(
        event_data: pl.DataFrame,
        entity_column: str,
        cohort_column: str,
        **named_aggs: IntoExpr,
):
    if named_aggs:
        event_data = (
            event_data
            .group_by([entity_column, cohort_column])
            .agg(**named_aggs)
        )
    else:
        if len(event_data) > len(event_data.unique(subset=[entity_column, cohort_column])):
            message = (
                '\n\tThere seems to be duplicate entity-time observations in the event_data \n '
                'these duplicates have now been deleted, you may want to consider the '
                '"named_aggs" parameter to keep the information in the duplicated rows: '
                'for example counts of a third variable\n'
            )
            warnings.warn(message)

        event_data = event_data.unique(subset=[entity_column, cohort_column])

    return event_data


def _discern_entity_cohort(
        event_data_columns: list,
        entity_column: str | None,
        cohort_column: str | None,
):
    """automatically identify entity and cohort column when needed/possible"""
    if entity_column is not None and cohort_column is not None:
        return entity_column, cohort_column

    # if 'entity_column' provided but no 'cohort_column'
    elif entity_column is not None and len(event_data_columns) == 2:

        # select the other column
        cohort_column = [i for i in event_data_columns if i != entity_column][0]

        return entity_column, cohort_column

    # if 'cohort_column' provided but no 'entity_column'
    elif cohort_column is not None and len(event_data_columns) == 2:

        # select the other column
        entity_column = [i for i in event_data_columns if i != cohort_column][0]

        return entity_column, cohort_column

    else:
        raise ValueError(
            f"\nPlease specify 'entity_column' and 'cohort_column' for 'event_data.'."
            f"\nChoose from 'event_data' columns: {event_data_columns}"
        )


# ----------------------------------------------------------------------

def pre_process_event_data_dates_and_entity_column(
        data: pl.DataFrame,
        event_data: pl.DataFrame,
        entity_column: str | None,
        cohort_column: str | None,

        panel_entity_column: str,
        period_frequency: PeriodFrequency,

        **named_aggs: IntoExpr
) -> pl.DataFrame:
    if entity_column is None and panel_entity_column in event_data.columns:
        entity_column = panel_entity_column

    # this function will raise errors if the columns are not automatically identified
    event_data, entity_column, cohort_column = event_data_to_period_frequency(
        event_data=event_data,
        entity_column=entity_column,
        cohort_column=cohort_column,
        period_frequency=period_frequency,
        **named_aggs
    )

    # if 'entity_column' in 'event_data' has
    # a different name then 'entity_column' in 'panel_data'
    if entity_column != panel_entity_column:
        event_data = event_data.rename({entity_column: panel_entity_column})

        message = (
            f"\n\tThe 'entity_column' in 'event_data' "
            f"has been renamed to '{panel_entity_column}'")
        warnings.warn(message)

    _panel_data_entity_dtype = data.schema[panel_entity_column]
    _event_data_entity_dtype = event_data[panel_entity_column].dtype

    if _event_data_entity_dtype != _panel_data_entity_dtype:

        # if they are both integers, but they are not the same integer, cast event_data
        if (
                _panel_data_entity_dtype in pl.INTEGER_DTYPES and
                event_data[panel_entity_column].is_integer()
        ):
            event_data = event_data.with_columns(pl.col(panel_entity_column).cast())

        # if they are not both integers or both strings
        elif not (_panel_data_entity_dtype is pl.Utf8 and _event_data_entity_dtype is pl.Utf8):
            raise ValueError(
                f"\n\t'entity_column', '{entity_column}', "
                f"in 'event_data' is {_event_data_entity_dtype} and "
                f"in 'panel_data' is {_panel_data_entity_dtype}.\n"
                f"'{entity_column}' should be the same dtype in both datasets."
            )

    return event_data
