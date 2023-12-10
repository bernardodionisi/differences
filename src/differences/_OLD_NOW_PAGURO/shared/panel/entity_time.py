from __future__ import annotations

import datetime
import warnings

import polars as pl

from differences._OLD_NOW_PAGURO.shared.dates import string_to_date


def to_entity_time(
        entities: list[str | int] | str | int | pl.Series,
        times: str | int | datetime.date | None | list[str | int | datetime.date | None],
        time_column: str = 'time',
        entity_column: str = "entity",
) -> pl.DataFrame:
    """
    Converts provided entities and times to a structured event data
    in the form of a `pl.DataFrame` object.

    Group
    -----
        general

    .. code-example::
        :collapsible:

        .. ipython:: python

            import differences

            data = differences.to_entity_time(
                entities=["e1", "e1", "e2", "e2"],
                times=["2011-02", "2011-05", 2012, "2013-04-30"],
            )

            data

    Parameters
    ----------
    entities : list | str | int
        A list, string or integer that represents the entities to be included in the event data.
        If a single str or int is passed, it is converted into a list.

    times : list | str | int | datetime.date
        A list, string, integer, or datetime object representing the times of events.
        Each time corresponds to an entity from the 'entities' argument.
        If fewer times than entities are provided, the last time is repeated.
        If a single str, int, or datetime.date is passed, it is duplicated for each entity.

    entity_column : str
        Name of the column in the resulting `pl.DataFrame` where the 'entities' data will be stored.

    time_column : str, optional, default is 'event_date'
        Name of the column in the resulting `pl.DataFrame` where the 'times' data will be stored.

    Returns
    -------
    pl.DataFrame
        A `pl.DataFrame` data frame containing the event data, with one row for each entity
        and columns for entity and time.

    Raises
    ------
    ValueError
        If there are fewer entities than event times.
    """

    # todo: add intensities + None in times (and rest) for never-treated
    # todo: fix entity_column string selection for pl.Series

    if not isinstance(entities, pl.Series):
        if entity_column is None:
            raise ValueError("Please provide a name for the entity_column")

    else:
        entity_column = entities.name
        if not entity_column:
            raise ValueError("Please provide a name for the entity_column")

        entities = entities.to_list()  # todo: make the function work for series

    if not isinstance(entities, list):
        entities = [entities]

    # ------------------------------------------------------------------

    if isinstance(times, int):
        times = str(times)  # int should just be used for year.

    if isinstance(times, str):
        times = string_to_date(times)

    # ------------------------------------------------------------------

    if isinstance(times, list):
        times = [string_to_date(str(t)) if isinstance(t, (int, str)) else t for t in times]

        missing_times = len(entities) - len(times)
        if missing_times > 0:
            times += [times[-1]] * missing_times
        elif missing_times < 0:
            raise ValueError("there can't be fewer entities than event times")

    else:  # it was not a list
        times = [times] * len(entities)

    return pl.DataFrame({entity_column: entities, time_column: times})


# ----------------------- find entity time -----------------------------

def extract_time_column(
        data: pl.DataFrame | pl.LazyFrame,
        time_column: str | None,
        none_if_no_dates: bool = False,  # if False raises error
        list_if_not_identifiable: bool = False,  # if False raises error
) -> str | None | list:
    # time_column should be the name of a pl.Date column,
    # used to construct the panel time
    possible_time_column = [k for k, v in data.schema.items() if v in [pl.Date]]

    if isinstance(time_column, str):  # time_column was provided
        if time_column not in data.columns:
            raise ValueError(f"'{time_column}' is not in the data.")

        elif time_column not in possible_time_column:
            raise ValueError(
                f"'{time_column}' is not a pl.Date, "
                f"the pl.Date columns in data are: {possible_time_column}"  # possibly empty
            )

        return time_column  # return the column that was passed

    elif time_column is None:
        if len(possible_time_column) == 0:
            if none_if_no_dates:
                return None
            else:
                raise ValueError(
                    "There is no pl.Date column in data to use as time_column"
                )

        elif len(possible_time_column) > 1:
            if not list_if_not_identifiable:
                raise ValueError(
                    f"There are too many pl.Date columns in the data "
                    f"to identify the time_column automatically.\n"
                    f"The pl.Date columns in the data are: {possible_time_column}\n"
                    f"Please select one using the 'time_column' parameter."
                )

            return possible_time_column  # return a list

        else:
            return possible_time_column[0]

    else:
        raise ValueError(
            "time_column should be the name of a pl.Date in the data to "
            "use to construct the panel time.")


def extract_entity_column(
        data: pl.LazyFrame,
        entity_column: str | None,
        list_if_not_identifiable: bool = False  # if False raises error
) -> str | list:
    # column candidates to be the entity_column. entity_column should not be a pl.Date

    possible_entity_columns = [k for k, v in data.schema.items() if v not in [pl.Date]]
    # possible_entity_columns = [c for c in data.columns if c != time_column]

    if isinstance(entity_column, str):  # and entity_column was provided
        if entity_column not in data.columns:
            raise ValueError(f"'{entity_column}' is not in the data.")

        elif entity_column not in possible_entity_columns:
            raise ValueError(f"'{entity_column}' should not be of dtype pl.Date")

        return entity_column  # return the entity that was passed

    elif entity_column is None:  # and entity_column was NOT provided

        if len(possible_entity_columns) == 0:
            raise ValueError(
                "All the columns provided are of dtype pl.Date. No column to use as the "
                "entity_column."
            )

        elif len(possible_entity_columns) == 1:
            return possible_entity_columns[0]  # return the only column

        elif len(possible_entity_columns) > 1:

            if not list_if_not_identifiable:
                raise ValueError(
                    f"entity_column can't automatically be selected. "
                    f"Please provide a value for the 'entity_column' parameter, "
                    f"possible values are: {possible_entity_columns}"
                )

            return possible_entity_columns  # returns a list

    else:
        raise ValueError(
            "entity_column should be the name of the column containing the entity identifiers"
            "use to construct the panel time."
        )


def find_entity_column(
        data: pl.LazyFrame,
        entities: pl.Series,
) -> str:
    # the entity column name used because it was the entity
    # in the event_data at initialization
    entity_column_in_event_data = entities.name

    # possible columns: the ones with the same dtype as the entities in the event_data
    possible_columns = [
        i for i, t in data.schema.items()
        if t != entities.dtype
    ]

    # f"Note that the data should contain a column of identifiers "
    # f"comparable to the identifiers in the event_data; "
    # f"the name of the entity column will be automatically changed "
    # f"to '{entity_column_in_event_data}' when possible. "

    if len(possible_columns) == 0:
        raise ValueError(
            f"The provided data does not have any column of of type {entities.dtype}.\n"
            f"The entities in the event_data are of type {entities.dtype}, "
            f"in order to build a panel one needs to provide data that has a column with "
            f"identifiers that overall with the entity identifiers in the event_data."
        )
    elif len(possible_columns) == 1:
        entity_column = select_column_if_is_in(
            data=data, series=entities, possible_columns=possible_columns[0]  # check only this col
        )

        if len(entity_column) == 1:
            entity_column = entity_column[0]
            return entity_column

        else:  # len(entity_column) == 0
            # there is a column named as entity_column_in_event_data but no overlapping ids
            raise ValueError(
                f"The data provided contains a column named '{possible_columns[0]}',"
                f" which is the only column that has the same dtype as the entity_column "
                f"in the event data. "
                f"However '{possible_columns[0]}' does not have any overlapping identifiers as "
                f"the column in event_data. Please make sure to provide a column that can be "
                f"used as the entity to construct the panel and that column contains identifiers "
                f"comparable to the identifiers in the entity_column in the event data. "
            )

    elif len(possible_columns) > 1:

        # if there is a column named as the entity_column in the event data
        if entity_column_in_event_data in possible_columns:

            # check if the column named as entity_column_in_event_data contains the
            # correct identifiers
            entity_column = select_column_if_is_in(
                data=data,
                series=entities,
                possible_columns=entity_column_in_event_data  # check only this col
            )

            if len(entity_column) == 1:
                return entity_column[0]

            else:  # len(entity_column) == 0
                # there is a column named as entity_column_in_event_data but no overlapping ids
                raise ValueError(
                    f"The data provided contains a column named '{entity_column_in_event_data}' "
                    f"(the same name as the entity_column in the event_data), "
                    f"but the column has no overlapping identifiers with the column in the "
                    f"event_data.\n"
                    f" Please make sure that if the data contains a column named "
                    f"{entity_column_in_event_data} that is the column to construct the panel."
                )
        else:
            entity_column = select_column_if_is_in(
                data=data,
                series=entities,
                possible_columns=possible_columns  # check all the columns of the same dtype
            )

            if len(entity_column) == 1:
                return entity_column[0]

            elif len(entity_column) == 0:
                raise ValueError(
                    "The data provided has no column that can be used as the entity_column: "
                    "the entity_column in the data must have the same dtype as the "
                    "entity_column in the event_data and it must contain identifiers that overlap "
                    "with the event_data. "
                )
            else:
                raise ValueError(
                    "The data provided contains multiple columns that could be used as "
                    "entity_column: they have the same dtype as the entity_column in the "
                    "event data and they contain identifiers that overlap with the event data.\n"
                    "Specify which column to use by setting the entity_column parameter."
                )


def rename_data_entity_column(
        data: pl.LazyFrame,
        entity_column: str,
        event_data_entity_column: str
) -> pl.LazyFrame:
    if entity_column != event_data_entity_column:

        if event_data_entity_column in data.columns:
            raise ValueError(
                f"'{event_data_entity_column}', already in data, however it was "
                f"not automatically selected to be the entity column. "
                f"The entity column seems to be '{entity_column}' which when possible is "
                f"renamed '{event_data_entity_column}'."
            )

        warnings.warn(f"Renaming {entity_column} to {event_data_entity_column}")
        data = data.rename({entity_column: event_data_entity_column})

    return data


def select_column_if_is_in(
        data: pl.LazyFrame,
        series: pl.Series,
        possible_columns: str | list = None
) -> list[str]:
    if possible_columns is None:
        # columns of the same dtype as the series
        possible_columns = [
            i for i, t in data.schema.items()
            if t != series.dtype
        ]

    if isinstance(possible_columns, str):
        possible_columns = [possible_columns]

    out = []
    for i in possible_columns:
        # if column i contains any of the entities in entity
        if series.is_in(data.select(pl.col(i).unique()).collect().to_series()).any():
            out.append(i)

    return out


def find_entity_column_and_rename_it(
        data: pl.LazyFrame,
        entities: pl.Series,
        entity_column: str,
) -> tuple[pl.LazyFrame, str]:  # data renamed + new entity_column

    # used to make sure that the event_data and data have the same entity_name
    entities_name = entities.name

    if not isinstance(entities_name, str):
        raise ValueError("'entities' must be a named pl.Series. Name seems to be missing.")

    entity_column = extract_entity_column(
        data=data,
        entity_column=entity_column,  # can be provided
        list_if_not_identifiable=True,  # if there are too many to identify return None
    )

    if isinstance(entity_column, str):

        if data.schema[entity_column] != entities.dtype:
            raise TypeError(
                f"The entity identifiers in the data provided to the panel are of different "
                f"type then the entity identifiers in the 'entities' "
                f"(likely coming from 'event_data')."
                f"\tThe entity_column ('{entity_column}') in the 'entities' is of type "
                f"{entities.dtype}\n"
                f"\tThe entity_column ('{entity_column}', renamed to "
                f"'{entities_name}') "
                f"in the data is of type {data.schema[entity_column]}\n."
            )

        selected_entity_column = entity_column  # just for the ValueError below

        # check there are overlapping identifiers
        entity_column = select_column_if_is_in(
            data=data, series=entities, possible_columns=entity_column
        )

        if len(entity_column) == 1:  # if there were overlapping identifiers

            entity_column = entity_column[0]

            data = rename_data_entity_column(
                data=data,
                entity_column=entity_column,
                event_data_entity_column=entities_name
            )

        else:  # len(entity_column) == 0
            raise ValueError(
                f"The column '{selected_entity_column}' does not contain any common entity "
                f"identifier with the event_data's column '{entities_name}'.\n"
                f"In order to build the panel either:\n"
                f"\t 1) select an entity_column with at least one common ID with "
                f"event_data's column '{entities_name}'\n"
                f"\t 2) set the argument entity_column to None, and the entity column will be "
                f"selected automatically, if possible.\n"
                f"1 is preferred to 2."
            )

    else:  # entity_column is a list of possible columns
        if not isinstance(entity_column, list):
            raise ValueError("Something went wrong, entity_column should be a list here")

        # find the column that has common elements with entities
        # (error will be raised if more than one)
        entity_column = find_entity_column(data=data, entities=entities)

        data = rename_data_entity_column(
            data=data,
            entity_column=entity_column,
            event_data_entity_column=entities_name
        )

    if entity_column != entities_name:
        if entity_column in data.columns or entities_name not in data.columns:
            raise ValueError(
                f"Something went wrong:\n "
                f"the data should have a column named '{entities_name}' and no column "
                f"named '{entity_column}'.\n"
                f"'{entity_column}' should have been (automatically) and carefully renamed to "
                f"'{entities_name}',after making sure that '{entity_column}' is"
                f" the correct column to use as the entity column:\n"
                f"\t- it has at least one overlapping entity identifiers "
                f"with the event data entity_column."
            )

    return data  # todo: fix return
