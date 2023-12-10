from __future__ import annotations

import warnings
from typing import Literal

import polars as pl

from differences._OLD_NOW_PAGURO.shared.dates import int_as_date, PeriodFrequency


def pre_process_event_data_and_data(
        data: pl.DataFrame,  # 'time_column' must be integer
        event_data: pl.DataFrame,  # 'cohort_column' must be integer

        is_panel: bool,  # if False it is a repeated-cross-section

        *,
        entity_column: str,
        time_column: str,
        cohort_column: str,
        dosage_column: str,

        treat_bounds: int | None | tuple[int | None, int | None],
        last_cohort_as_never_treated: bool = True,
        anticipation: int = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series]:
    # separate 'event_data' from data
    data, event_data = select_event_data_from_data(
        data=data,
        event_data=event_data,
        entity_column=entity_column,
        cohort_column=cohort_column,
        dosage_column=dosage_column,
    )

    # ---> DROP never-treated entities from 'event_data' <---
    event_data = event_data.filter(pl.col(cohort_column).is_not_null())
    # todo: if there is a dosage column, it should not have nulls

    # are there never-treated entities (before cleaning up the treated_after
    has_never_treated = bool(len(
        get_never_treated_entities(data=data, event_data=event_data, entity_column=entity_column))
    )

    # todo: if repeated-cross-sections should filter entities?
    if is_panel:

        # add _min_panel_time, _max_panel_time to event_data
        event_data = _join_event_data_with_min_max_panel(
            event_data=event_data,
            data=data,
            cohort_column=cohort_column,
            entity_column=entity_column,
            time_column=time_column
        )

        # filter out:
        #   (1) if entities are in 'event_data' but not in 'data' (panel in this case)
        #   (2) entities based on 'treat_bounds'
        data, event_data = _filter_out_entities_from_data_and_event_data(
            data=data,
            event_data=event_data,
            entity_column=entity_column,
            cohort_column=cohort_column,
            treat_bounds=treat_bounds,
        )

        if last_cohort_as_never_treated and not has_never_treated:

            # has_never_treated may have changed in _filter_out_entities_from_data

            # the 'never_treated' group could be now the set of eventually treated
            # entities that are treated after the end of their time-series

            has_never_treated = bool(len(
                get_never_treated_entities(
                    data=data,
                    event_data=event_data,
                    entity_column=entity_column))
            )

            if not has_never_treated:
                # if still there are no "never-treated"
                data, event_data = _make_last_cohort_never_treated(
                    data=data,
                    event_data=event_data,
                    time_column=time_column,
                    cohort_column=cohort_column,
                    anticipation=anticipation,
                    has_never_treated=has_never_treated
                )

    # cast to integer just here, after entities have been filtered out (if needed)
    data, event_data, entity_category_codes = cast_entity_time_columns_dtype(
        data=data,
        event_data=event_data,
        entity_column=entity_column,
        time_column=time_column,
        cohort_column=cohort_column,
    )

    return data, event_data, entity_category_codes


def select_event_data_from_data(
        data: pl.DataFrame,
        event_data: pl.DataFrame | None,

        entity_column: str,
        cohort_column: str,
        dosage_column: str
):
    """
    Generate valid 'event_data'. No 'never-treated' (in 'event_data') and only relevant columns.
    """

    # cohort name + intensity name

    event_data_columns = [entity_column, cohort_column]
    if dosage_column:
        event_data_columns = [entity_column, cohort_column, dosage_column]

    if event_data is not None:

        # todo: identify the entity_column based on the values/ and the fact that often there are
        #  going to be only 2 columns entities and cohort.
        if entity_column not in event_data.columns:
            raise ValueError(
                f"\n'event_data' is missing the 'entity_column' named '{entity_column}'.\n"
                f"'entity_column' should contain entities identifiers overlapping with "
                f"the entities identifiers in 'data': 'event_data' should contain the event "
                f"information (cohort and dosage) for the entities "
                f"that undergo an event/treatment."
            )
        event_data = event_data.select(event_data_columns).unique()

    else:  # if no 'event_data' was passed, create it from the columns of 'data'
        event_data = data.select(event_data_columns).unique()

        # drop treatment columns from 'data' (will be joined back if is_single_event)
        data = data.drop(columns=[i for i in event_data_columns if i != entity_column])

    return data, event_data


# ----------------------------------------------------------------------

def cast_entity_time_columns_dtype(
        data: pl.DataFrame,
        event_data: pl.DataFrame | None,

        entity_column: str,
        time_column: str,
        cohort_column: str,
):
    if data[entity_column].is_integer():
        entity_category_codes = None

    elif data[entity_column].dtype is pl.Utf8:

        data, event_data, entity_category_codes = _cast_string_entities_to_integers(
            data=data,
            event_data=event_data,
            entity_column=entity_column
        )

    else:
        raise ValueError(
            f"'entity_column': '{entity_column}' must be either a string or an integer"
        )

    # make sure that entity and time columns are of the same type in 'event_data' and 'data'
    event_data = (
        event_data
        .with_columns([
            pl.col(entity_column).cast(data[entity_column].dtype),
            pl.col(cohort_column).cast(data[time_column].dtype),
        ])
    )

    return data, event_data, entity_category_codes


def _cast_string_entities_to_integers(data: pl.DataFrame, event_data: pl.DataFrame,
                                      entity_column: str):
    with pl.StringCache():
        data = data.with_columns(pl.col(entity_column).cast(pl.Categorical))
        event_data = event_data.with_columns(pl.col(entity_column).cast(pl.Categorical))

        entity_category_codes = data[entity_column]

        # this extension is redundant? this should be run once the entities that are in
        # 'event_event' data but not in 'data' have been dropped from 'event_data'
        entity_category_codes.extend(event_data[entity_column])

        entity_category_codes = entity_category_codes.unique()

        data = data.with_columns(pl.col(entity_column).to_physical())
        event_data = event_data.with_columns(pl.col(entity_column).to_physical())

    return data, event_data, entity_category_codes


# ----------------------------------------------------------------------


def _make_last_cohort_never_treated(
        data: pl.DataFrame,
        event_data: pl.DataFrame,

        time_column: str,
        cohort_column: str,

        anticipation: int,
        has_never_treated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if has_never_treated:
        return data, event_data

    last_cohort = event_data[cohort_column].max()

    # keep track and only allow not_yet_treated option

    message = (
        f"\n\tThere are no never-treated entities.\n"
        f"\t- Setting the last treated cohort (i.e. {last_cohort}) as the control cohort.\n"
        f"\t- Dropping all the time periods after {last_cohort} (the last cohort time).\n"
    )

    warnings.warn(message)

    # drop 'entities' treated in the last cohort from 'event_data'
    # these entities are still in the 'data', which makes them "never-treated"
    event_data = event_data.filter(pl.col(cohort_column).ne(last_cohort))

    # drop all calendar times before the last cohort in 'data'
    data = data.filter(pl.col(time_column).lt(last_cohort - anticipation))

    # TODO: look into possible issues if treat_after != 0
    #  (when allowing entities treated after the end of their time-series to be part
    #  of the treatment group)
    return data, event_data


def _join_event_data_with_min_max_panel(
        data: pl.DataFrame,  # pipeline
        event_data: pl.DataFrame,
        entity_column: str,
        time_column: str,
        cohort_column: str,
):
    event_data = (
        event_data
        .filter(pl.col(cohort_column).is_not_null())  # drop never-treated entities

        .join(  # keep entities that are in the panel dataset
            data
            .group_by(entity_column).agg([
                pl.col(time_column).min().alias('_min_panel_time'),  # this should be an int
                pl.col(time_column).max().alias('_max_panel_time'),  # this should be an int
            ]),
            on=entity_column  # inner join: dropping event_data entities not in pipeline
        )
    )  # this is the cohort_data:
    # only treated entities-cohorts, with the min-max period for each entity

    return event_data


# ----------------------- filter data/entities -------------------------


def _filter_out_entities_from_data_and_event_data(
        data: pl.DataFrame,  # pipeline
        event_data: pl.DataFrame,
        *,
        entity_column: str,
        cohort_column: str,

        treat_bounds: int | None | tuple[int | None, int | None],
) -> tuple[pl.DataFrame, pl.DataFrame]:  # data, event_data
    """
    Filters panel and event data based on a specified set of entities and conditions.

    This function excludes entities that have been treated 'treat_bounds' periods prior
    to the start of their time-series.

    Parameters
    ----------
    data : pl.DataFrame
        The panel data to be filtered.
    event_data : pl.DataFrame
        The event data to be filtered.
    entity_column : str
        Name of the column in `pipeline` and `event_data` that represents the entity.
    cohort_column : str
        Name of the column representing cohort groups.
    treat_bounds : int or tuple[int, int], optional
        Defines the bounds to exclude entities based on treatment.
        If a single int is provided, it represents the periods prior to
        the start of their time-series.
        If a tuple is provided, it represents (start_bound, end_bound). Default is 0.

    Returns
    -------
    pl.DataFrame or tuple[pl.DataFrame, pl.DataFrame]
        Depending on `data_to_return`, it returns:
        - a single pl.DataFrame if "event" or "panel" is selected.
        - a tuple of two pl.DataFrames if "both" is selected.
    """

    # treat_after: to identify entities treated N [treat_after] periods
    #               after end of time-series

    # treat_before: to identify entities treated N [treat_before] periods
    #               before start of time-series

    treat_before, treat_after = _process_treat_bounds_arg(treat_bounds=treat_bounds)

    # ------------------------ treat_after -----------------------------

    # these entities remain in 'data' they are eventually treated but outside the span of
    # their available time-series in the data. They are as if never-treated

    # todo: when treat_after = None, should this be + anticipation? Not
    _treat_after_ = treat_after if treat_after is not None else 0

    # using _treat_after_, since if treat_after is None we'll place the entities treated_after
    # into the never-treated group. If instead treat_after is not None, we'll drop them.

    entities_to_exclude: pl.Series = _get_filtered_entities(
        data=event_data,
        entity_column=entity_column,

        # entities for which the cohort time is after the max time-series
        filter_expr=pl.col(cohort_column).gt(pl.col("_max_panel_time") + _treat_after_),
        reason_to_exclude_entities="treat_after",

    )

    # todo: what to do if time_after != 0 ? and anticipation? when

    if len(entities_to_exclude):
        # always excluded from the event_data
        event_data = event_data.filter(~pl.col(entity_column).is_in(entities_to_exclude))

        if treat_after is not None:
            # exclude them also from 'data': do not place them in the never-treated group
            data = data.filter(~pl.col(entity_column).is_in(entities_to_exclude))

    # ------------------ treat_before + not in data --------------------

    # entities to exclude: those that have been treated
    # 'treat_bounds' periods prior to the start of their time-series
    entities_to_exclude: pl.Series = _get_entities_to_exclude(
        data=data,
        event_data=event_data,
        entity_column=entity_column,
        cohort_column=cohort_column,
        treat_before=treat_before
    )

    event_data = event_data.filter(~pl.col(entity_column).is_in(entities_to_exclude))
    data = data.filter(~pl.col(entity_column).is_in(entities_to_exclude))

    # ------------------------------------------------------------------

    # keep the entities that are not in the list of entities to exclude
    # both in the panel data and the cohort data

    return data, event_data  # event data here is cohort_data (because of this filtering)


def _get_entities_to_exclude(
        data: pl.DataFrame,  # pipeline
        event_data: pl.DataFrame,
        *,
        entity_column: str,
        cohort_column: str,

        treat_before: int | None
) -> pl.Series:
    """
    Find the entities for which the cohort >= treat_bounds + min panel time.

    Returns a Series of entities to exclude from the panel and the cohort data.

    The included ones are:
        - never treated (if they are included in event_data)
        - eventually treated (treated after the end of their time series)
        - treated 'treat_bounds' periods after cohort

    These are the entities that should be excluded to form the pipeline and cohort_data.
    """

    # drop entities in 'event_data' that are not in 'data'
    # likely redundant since the inner join below does the same thing
    entities_to_exclude = _get_filtered_entities(
        data=event_data,
        entity_column=entity_column,

        # entities in 'event_data' that are not in 'data'
        filter_expr=~pl.col(entity_column).is_in(data[entity_column]),
        reason_to_exclude_entities="entities_not_in_panel"
    )

    # entities not in cohort is fine, those are used as the never-treated
    # if the user didn't filter the entities

    # ------------------- join with min-max panel ----------------------

    if "_min_panel_time" not in event_data.columns or "_max_panel_time" not in event_data.columns:
        # need to _join_event_data_with_min_max_panel before
        raise ValueError("_min_panel_time, and _max_panel_time are not in event_data.columns")

    # entities to exclude from 'event_data':

    if treat_before is not None:
        entities_to_exclude.extend(
            _get_filtered_entities(
                data=event_data,
                entity_column=entity_column,

                # entities for which the cohort time is less then the min time-series
                filter_expr=pl.col(cohort_column).lt(pl.col("_min_panel_time") + treat_before),
                reason_to_exclude_entities="treat_before",
            )
        )

        # cohort = 2007 | _min_panel_time = 2006 | treat_before = 0 => 2007 < 2006 + 0
        # cohort = 2007 | _min_panel_time = 2007 | treat_before = 0 => 2007 < 2007 + 0
        # cohort = 2007 | _min_panel_time = 2008 | treat_before = 0 => 2007 < 2008 + 0    X

        # cohort = 2007 | _min_panel_time = 2006 | treat_before = 1 => 2007 < 2006 + 1
        # cohort = 2007 | _min_panel_time = 2007 | treat_before = 1 => 2007 < 2007 + 1    X
        # cohort = 2007 | _min_panel_time = 2008 | treat_before = 1 => 2007 < 2008 + 1    X

    # if the cohort is < time + treat_bounds
    # for example if treat_bounds = -1, all the entities that are
    # treated since 1 period prior to the start of their time series (or before)
    # will be excluded

    # If an entity is treated twice (has 2 cohorts), it will be
    # excluded if one of those cohorts is prior to: treat_bounds + min panel time

    return entities_to_exclude.unique()  # to exclude


def _get_filtered_entities(
        data: pl.DataFrame,
        entity_column: str,
        filter_expr: pl.Expr,
        reason_to_exclude_entities: Literal[
            "entities_not_in_panel", "treat_before", "treat_after"],
) -> pl.Series:
    entities = (
        data
        .filter(filter_expr)
        .select(entity_column)
        .unique()
        .to_series()
    )

    # issue warning for each reason
    _warning_entities_to_exclude(
        entities=entities,
        reason_to_exclude_entities=reason_to_exclude_entities,
        # <--- reason the entities were filtered out
    )

    return entities


def _warning_entities_to_exclude(
        entities: pl.Series,
        reason_to_exclude_entities: Literal[
            "entities_not_in_panel", "treat_before", "treat_after"],
) -> None:
    if not len(entities):
        return None

    if reason_to_exclude_entities == "entities_not_in_panel":
        if len(entities) == 1:
            message = f"\n\tThere is {len(entities)} entity that is"
        else:
            message = f"\n\tThere are {len(entities)} entities that are"

        message += (
            f" in 'event_data' but not in 'data'.\n"
            f"Dropping those entities from the 'event_data'.\n"
            f"Some examples are of dropped entities are {entities[:5].to_list()}"
        )

        warnings.warn(message, DroppedEntitiesWarning)

    elif reason_to_exclude_entities == "treat_before":
        if len(entities) == 1:
            message = f"\n\t{len(entities)} entity has been dropped; "
        else:
            message = f"\n\t{len(entities)} entities have been dropped; "

        message += f"Some dropped entities are: {entities[:3].to_list()}"

        # ete = f"+ {treat_before}" if treat_before >= 0 else f"{treat_before}"
        # f"Entities are dropped if their time of treatment/event (cohort) "
        # f"is ≤ start of their time-series {ete}.\n"
        # f"You can control this using 'treat_bounds'\n"

        warnings.warn(message, DroppedEntitiesWarning)

    elif reason_to_exclude_entities == "treat_after":

        n_entities = len(entities)

        if n_entities:
            message = f"\n\t{n_entities} "
            if n_entities == 1:
                message += "entity is treated after the end of its time-series."
            else:
                message += "entities have treated after the end of their time-series."
                # ete = f"+ {treat_before}" if treat_before >= 0 else f"{treat_before}"

            warnings.warn(message)


def _process_treat_bounds_arg(
        treat_bounds: int | None | tuple[int | None, int | None]
) -> tuple[int | None, int | None]:
    if isinstance(treat_bounds, tuple):
        if len(treat_bounds) != 2:
            raise ValueError(
                "If 'treat_bounds' is a tuple it should contain 2 elements:"
                "\t- the number of periods before the time-series of an entity starts "
                "in which the entity is allowed to be included in the 'event_data'\n"
                "\t- the number of periods, after the end of the time-series of an entity, "
                "in which the entity is allowed to included in the 'event_data'"
            )
        treat_before, treat_after = treat_bounds

    elif isinstance(treat_bounds, int):
        # if treat_bounds is an integer, then only treat_before is used:
        # for example if treat_before=-1 then an entity that has an yearly time-series starting
        # in 1999 will be included in the event_data if it belongs
        # to a treatment cohort starting from 1998 and after

        treat_before, treat_after = treat_bounds, None
    else:  # None
        treat_before, treat_after = None, None

    return treat_before, treat_after


# ----------------------------------------------------------------------

def get_never_treated_entities(
        data: pl.DataFrame,
        event_data: pl.DataFrame,

        entity_column: str,
) -> pl.Series:
    """Check if data has some never-treated entities"""

    # entities that are not in 'event_data' (after it has been filtered out of nulls)
    # are never-treated
    out = (
        data
        .select(entity_column)
        .filter(
            ~pl.col(entity_column).is_in(event_data[entity_column].unique())
        )
        .to_series()
        .unique(maintain_order=True)
    )

    return out


def is_single_event(event_data: pl.DataFrame, entity_column: str) -> bool:
    """
    Boolean indicating whether each entity undergoes an event/treatment only once.

    Not to be confused with is_single_cohort
    """
    # each entity has only one obs in 'event_data'
    return event_data[entity_column].n_unique() == len(event_data)


def is_single_cohort(event_data: pl.DataFrame, cohort_column: str) -> bool:
    """
    Boolean indicating whether all entities undergo an event/treatment at a common time

    Not to be confused with is_single_event
    """
    return event_data[cohort_column].n_unique() == 1


# ----------------------------------------------------------------------


def build_number_of_entities_per_time(
        panel_data: pl.DataFrame,
        event_data: pl.DataFrame,  # can be cohort_data
        entity_column: str,
        time_column: str,
        cohort_column: str,
        period_frequency: PeriodFrequency | None  # to convert back to date
) -> pl.DataFrame:
    """
    useful for the EDF plot
    be careful when event_data has 'entities' treated prior to the start of their time-series

    Parameters
    ----------
    panel_data
    event_data
    entity_column
    time_column
    cohort_column
    period_frequency

    Returns
    -------

    """
    # count how many entities there are in each time
    # (cumulative, an entity is counted in multiple times)
    data = (
        panel_data
        .group_by(time_column)

        # count the number of entities that are observed each time period
        .agg([pl.col(entity_column).n_unique().alias('num_entities_in_panel')])
    )

    total_n_entities = panel_data[entity_column].n_unique()

    # todo: add option for intensity/multivalued

    event_data = (
        event_data
        .filter(pl.col(cohort_column).is_not_null())  # if event_data is passed it may contain
        # the never treated
        .group_by(cohort_column)
        .agg([pl.col(entity_column).n_unique().alias('num_treated_entities')])
    )

    data = (
        data
        .join(
            event_data
            .with_columns(pl.col(cohort_column).cast(data[time_column].dtype))
            .rename({cohort_column: time_column}),
            on=time_column,
            how='left'  # keep the never-treated, they are not in the cohort_data,
            # but they are in the pipeline
        )
        .sort(time_column)
        .with_columns(pl.col('num_treated_entities').fill_null(0))
        .with_columns([
            (
                    pl.col('num_treated_entities').cumsum() / total_n_entities
            ).alias("cum_treated_density")
        ]
        )
    )

    # replace the 0s with None, in num_treated_entities, this is used for plotting
    # the text on top of the bars
    data = (
        data
        .with_columns(
            [
                pl
                .when(pl.col('num_treated_entities').eq(0))
                .then(None)  # before min time with a value
                .otherwise(pl.col('num_treated_entities'))
                .alias('num_treated_entities')
            ]
        )
    )

    if period_frequency is not None:
        data = (
            data
            .with_columns([
                int_as_date(time_column, period_frequency=period_frequency).alias(
                    f"{time_column}_date")
            ])
        )

    return data


class DroppedEntitiesWarning(Warning):
    pass
