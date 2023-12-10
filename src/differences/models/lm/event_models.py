from __future__ import annotations

import polars as pl

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from differences._OLD_NOW_PAGURO.shared.event.data_es import DataES


class EventModels:
    """
    A factory class for event study models

    Group
    -----
        factory

    """

    def __init__(
            self,
            data: pl.DataFrame,
            event_data: pl.DataFrame,
            *,
            entity_column: str,
            time_column: str,
            cohort_column: str,
            dosage_column: str,
    ):
        # data is panel_data with no event info column

        self.data = data  # date column must be int
        self.event_data = event_data  # date column must be int

        # 'entity_column' in 'data' & 'event_data' should be the same
        self.entity_column = entity_column
        self.time_column = time_column

        self.cohort_column = cohort_column
        self.dosage_column = dosage_column

        # ---------------- store some details --------------------------

        self._event_info_columns = [cohort_column]

        if self.dosage_column is not None:
            self._event_info_columns.append(dosage_column)

        # --------------------------------------------------------------

        (
            self._n_entities_in_data,
            self._n_entities_in_event_data,
            self._n_entities_not_in_data,
            self._n_entities_not_in_event_data
        ) = count_entities(
            data=data,
            event_data=event_data,
            entity_column=entity_column
        )

    def __repr__(self):
        """
        Representation.
        """
        # this could also be delegated
        _spaces = ""

        data_repr = (
            self.data
            .__repr__()
            .replace('\n', f'\n{_spaces}  ')
        )

        event_data_repr = (
            self.event_data
            .select([self.entity_column] + self._event_info_columns)
            .__repr__()
            .replace('\n', f'\n{_spaces}  ')
        )

        # f"\n{self.__class__.__qualname__} [at {hex(id(self))}]\n"

        _info = {
            "entities_in_data": self._n_entities_in_data,
            "entities_in_event_data": self._n_entities_in_event_data,
            "entities_missing_in_data": self._n_entities_not_in_data,
            "entities_missing_in_event_data": self._n_entities_not_in_event_data
        }

        out = (
            f"{_spaces}data = {data_repr}\n"
            f"{_spaces}event_data = {event_data_repr}\n"
            f"{_spaces}entity_column = '{self.entity_column}'\n"
            f"{_spaces}time_column = '{self.time_column}'\n"
            f"{_spaces}cohort_column = '{self.cohort_column}'\n"
            f"{_spaces}dosage_column = '{self.dosage_column}'\n"
        )

        return out

    def __copy__(self):
        return type(self)(
            data=self.data,
            event_data=self.event_data,

            entity_column=self.entity_column,
            time_column=self.time_column,
            cohort_column=self.cohort_column,
            dosage_column=self.dosage_column,
        )

    def filter_entities(
            self,
            in_event_data: bool = False
    ) -> Self:

        # drops all the entities in data that are not in 'event_data'.
        # if 'event_data' only includes treated entities than all the never-treated entities will
        # be dropped.

        new_obj = self.__copy__()

        if in_event_data:
            new_obj.event_data = (
                new_obj.event_data
                .filter(
                    pl.col(new_obj.entity_column).is_in(new_obj.data[new_obj.entity_column])
                )
            )

        else:

            new_obj.data = (
                new_obj.data
                .filter(
                    pl.col(new_obj.entity_column).is_in(new_obj.event_data[new_obj.entity_column])
                )
            )

        # update counts after dropping the observations
        (
            new_obj._n_entities_in_data,
            new_obj._n_entities_in_event_data,
            new_obj._n_entities_not_in_data,
            new_obj._n_entities_not_in_event_data
        ) = count_entities(
            data=new_obj.data,
            event_data=new_obj.event_data,
            entity_column=new_obj.entity_column
        )

        return new_obj

    def to_data_es(
            self,
            treat_bounds: int | None | tuple[int | None, int | None] = 1,
            anticipation: int = 0,
    ) -> DataES:
        """
        Group
        -----
            Export

        Parameters
        ----------
        treat_bounds
        anticipation

        Returns
        -------

        """
        # all the other methods should use DataES
        return DataES(
            data=self.data,
            entity_column=self.entity_column,
            time_column=self.time_column,
            cohort_column=self.cohort_column,
            dosage_column=self.dosage_column,
            event_data=self.event_data,
            treat_bounds=treat_bounds,
            anticipation=anticipation,
        )

    # ------------------------- estimation -----------------------------

    def to_twfe(
            self,
            treat_bounds: int | None | tuple[int | None, int | None] = 1,
            **kwargs
    ):
        """
        Group
        -----
            Export

        Parameters
        ----------
        treat_bounds
        kwargs

        Returns
        -------

        """
        from differences.models.lm.twfe import TWFE

        twfe = TWFE(
            data=self.data,

            entity_column=self.entity_column,
            time_column=self.time_column,

            cohort_column=self.cohort_column,
            dosage_column=self.dosage_column,

            event_data=self.event_data,

            treat_bounds=treat_bounds,
            **kwargs
        )

        return twfe


# ----------------------------------------------------------------------

def count_entities(
        data: pl.DataFrame,
        event_data: pl.DataFrame,
        entity_column: str,
):
    entities_in_data = data[entity_column].unique()
    entities_in_event_data = event_data[entity_column].unique()

    n_entities_not_in_data = (
        entities_in_event_data
        .is_in(entities_in_data)
        .not_()
        .sum()
    )

    n_entities_not_in_event_data = (
        entities_in_data
        .is_in(entities_in_event_data)
        .not_()
        .sum()
    )

    n_entities_in_data = len(entities_in_data)
    n_entities_in_event_data = len(entities_in_event_data)

    return (
        n_entities_in_data,
        n_entities_in_event_data,
        n_entities_not_in_data,
        n_entities_not_in_event_data
    )
