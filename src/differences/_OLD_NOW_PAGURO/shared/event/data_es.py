from __future__ import annotations

import warnings

import polars as pl

from typing import Literal

from differences._OLD_NOW_PAGURO.info import Info
from differences._OLD_NOW_PAGURO.info import InfoList
from differences._OLD_NOW_PAGURO.shared.panel.panel_utils import (
    raise_reserved_column_names_error,
    raise_invalid_entity_time_data_error,

    determine_if_panel_or_rcs,
    get_time_varying_columns,

)

from differences._OLD_NOW_PAGURO.shared.event.event_data_int import (
    pre_process_event_data_and_data,
    is_single_event,
    get_never_treated_entities,
    build_number_of_entities_per_time
)

from differences._OLD_NOW_PAGURO.shared.event.group_time import (
    get_group_times_list_of_dicts,
    get_feasible_group_times,
    filter_group_time_list_of_dicts,
    add_expr_to_group_time_list_dicts
)


class DataES:
    """
    A class to set up the data for analysis of treatment effect

    Group
    -----
        base
    """

    def __init__(
            self,
            data: pl.DataFrame,
            event_data: pl.DataFrame | None,
            filter_entities: bool = False,
            *,
            treat_bounds: int | None | tuple[int | None, int | None] = 1,
            last_cohort_as_never_treated: bool = True,
            anticipation: int = 0,

            entity_column: str,
            time_column: str,
            cohort_column: str,
            dosage_column: str | None
    ) -> None:
        """

        Parameters
        ----------
        data
            Must be a panel or repeated-cross-section. The `time_column` must be an integer.
        event_data
        filter_entities
            Whether to filter out the entities that are in 'data' but not in 'event_data'.
            Note that, if not filtered, these entities. would be considered as never-treated.
            As described in 'event_data', 'event_data' may also contain never-treated entities.
        treat_bounds
        last_cohort_as_never_treated
        anticipation
        entity_column
        time_column
        cohort_column
        dosage_column
        """

        # -------------- is data passed valid? -------------------------

        super().__init__(data=data, name=None)

        raise_reserved_column_names_error(
            data=data,
            columns=[
                "_relative_time",
                "_cohort",
                "_treated",
                "_control",
                "_after_event",
                "_w"
            ],
        )

        raise_invalid_entity_time_data_error(
            data=data,
            entity_column=entity_column,
            time_column=time_column
        )

        # --------------------------------------------------------------

        if filter_entities:
            if event_data is None:
                raise ValueError("'filter_entities' can only be used if 'event_data' is not None")

            # drop all the entities from data that are not in 'event_data'
            # 'event_data' could contain never-treated now. later never-treated will only be in
            # 'data'
            _n_entities_before = data[entity_column].n_unique()

            data = data.filter(pl.col(entity_column).is_in(event_data[entity_column].unique()))

            _n_entities_after = data[entity_column].n_unique()

            warnings.warn(
                f"\n{_n_entities_before - _n_entities_after} entities have been"
                f" dropped from data because they are not in 'event_data'"
            )

        # ---------------- is repeated cross section? ------------------

        (
            self._is_panel,
            self._is_balanced_panel,
            self._is_repeated_cross_section
            # entity-time balanced. it could still be unbalanced depending on the variables
        ) = determine_if_panel_or_rcs(
            data=data,
            entity_column=entity_column,
            time_column=time_column
        )

        # -------------------- event data ------------------------------

        # missing_entities (entities in event_data but not in panel) are dropped here.
        data, event_data, entity_category_codes = pre_process_event_data_and_data(
            data=data,
            event_data=event_data,

            is_panel=self._is_panel,

            entity_column=entity_column,
            time_column=time_column,

            cohort_column=cohort_column,
            dosage_column=dosage_column,

            treat_bounds=treat_bounds,
            last_cohort_as_never_treated=last_cohort_as_never_treated,
            anticipation=anticipation,
        )

        # in case the entity identifiers are strings, they get converted to integers
        # but the conversion codes are stored here
        self._entity_category_codes: pl.Series = entity_category_codes

        # Boolean indicating whether each entity undergoes an event/treatment only once
        self._is_single_event = is_single_event(event_data=event_data, entity_column=entity_column)

        if self._is_single_event:
            data = (
                data
                .join(
                    event_data
                    .drop(columns=["_min_panel_time", "_max_panel_time"]),
                    on=entity_column,
                    how="left"  # # left to keep never-treated in data
                )
            )

        # Boolean indicating whether all entities undergo an event/treatment at a common time
        self._is_single_cohort = event_data[cohort_column].n_unique() == 1

        # --------------------------------------------------------------

        # replacing data which was already set
        self._data = data.sort([entity_column, time_column])
        self._event_data = event_data.sort([entity_column])

        self._entity_column = entity_column
        self._time_column = time_column

        self._cohort_column = cohort_column
        self._dosage_column = dosage_column

        self._anticipation = anticipation

        # ---------------- store some details --------------------------

        self._event_info_columns = [cohort_column]

        if self._dosage_column is not None:
            self._event_info_columns.append(dosage_column)

        # warning: this will also depend on missing values of outcome/controls
        # this would change for 'data_matrix'
        self._feasible_group_times = get_feasible_group_times(
            data=self._data,
            time_column=self._time_column,
            cohort_column=self._cohort_column,
            dosage_column=self._dosage_column,
        )

        # --------------------------------------------------------------

        self._info: InfoList = InfoList()

        self._info.append(

            # column names
            Info(name="_column_names")
            .update(
                _data_columns={
                    "entity_column": f'"{self._entity_column}"',
                    "time_column": f'"{self._time_column}"',
                },
                _event_data_columns={
                    "cohort_column": f'"{self._cohort_column}"',
                    "dosage_column": f'"{self._dosage_column}"'
                    if self._dosage_column else self._dosage_column
                }
            )
            .config_repr(same_line=2, n_columns=2),

            # stats
            Info(name="_stats")
            .update(
                Info={
                    "num_observations": self._data.shape[0],

                    "num_entities": len(self._entities()),
                    "num_times": len(self._times()),
                    "num_cohorts": len(self._cohorts()),

                    "num_treated_entities": len(self._entities(treated=True)),
                    "num_never_treated_entities": len(self._entities(treated=False)),
                }
            )
            .config_repr(same_line=2),
        )

        self._config: dict = {}  # unused

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return self._info._repr_string(
            data_for_repr={
                "_main_data": {
                    "data": self._data,
                    "event_data": self._event_data.drop(columns=[
                        "_min_panel_time", "_max_panel_time"
                    ])
                }
            }
        )

    # ------------------------------------------------------------------

    def _description(self) -> str:
        return data_es_description(
            times=self._times(),
            is_panel=self._is_panel,
            is_balanced_panel=self._is_balanced_panel,
            is_single_cohort=self._is_single_cohort,
            is_single_event=self._is_single_event,
            is_repeated_cross_section=self._is_repeated_cross_section
        )

    # ------------------------- plot -----------------------------------

    def _plot(
            self,
            # which: Literal["edf"] = "edf",
            **kwargs  # kwargs to be passed to the plot function, se the specific plot docs
    ):

        # for now its only cohort plot

        # data to plot
        data = build_number_of_entities_per_time(
            panel_data=self.panel_data,
            event_data=self.cohort_data,
            entity_column=self._entity_column,
            time_column=self._time_column,
            cohort_column=self._cohort_column,
            period_frequency=self._period_frequency  # creates a date if this is not None
        )

        # out = panel_event_edf_plot(
        #     data=data,
        #     time_column=f"{self.time_column}_date" if self._period_frequency else self.time_column,
        #
        #     step_column='cum_treated_density',  # cumulative density
        #     bar_column='num_entities_in_panel',  # frequency total entities in panel each time
        #
        #     text_column='num_treated_entities',  # num treated entities each time
        #     **kwargs
        # )
        # return out

    # ------------------------------------------------------------------

    def _entity_category_codes_map(self) -> dict | None:
        if self._entity_category_codes is not None:
            return dict(zip(
                self._entity_category_codes.to_physical(), self._entity_category_codes)
            )

    def _to_pandas(self, which_data: Literal["data", "event_data"] = "data"):
        if which_data == "data":
            return self._data.to_pandas().set_index([self._entity_column, self._time_column])
        else:
            return self._event_data.to_pandas().set_index([self._entity_column])

    # ------------------------------------------------------------------

    def _times(self) -> list[int]:
        return self._data[self._time_column].unique().sort().to_list()

    def _cohorts(self) -> list[int]:
        # example: first time = 1
        # anticipation
        #     0          x
        #     1          x     x
        #                1     2    3    4   <--- cohorts

        # cohorts[cohorts > self._times[0] + self.anticipation]

        # .drop_nulls(), there should be no never-treated in 'event_data', after __init__
        return self._event_data[self._cohort_column].unique().sort().to_list()

    def _dosages(self) -> list[int | str] | None:
        if self._dosage_column is None:
            return None
        # .drop_nulls(), there should be no never-treated in 'event_data'
        return self._event_data[self._dosage_column].drop_nulls().unique().sort().to_list()

    # ------------------------------------------------------------------

    def _entities(self, treated: bool = None, original_str: bool = False) -> pl.Series:
        """
        List `entities`

        Parameters
        ----------
        treated
        original_str
        """
        if treated is None:  # all 'entities'
            entities = self._data[self._entity_column].unique(maintain_order=True)

        elif treated:  # treated 'entities'
            entities = self._event_data[self._entity_column].unique(maintain_order=True)

        else:
            # if treated is False: # never-treated 'entities'
            entities = get_never_treated_entities(
                data=self._data, event_data=self._event_data, entity_column=self._entity_column)

        if original_str:
            entities = entities.map_dict(self._entity_category_codes_map())

        return entities

    def search_entity(
            self,
            value: str | int | list[str] | list[int],
            in_event_data: bool = False
    ) -> pl.DataFrame:

        if in_event_data:
            data = self._event_data
        else:
            data = self._data

        if isinstance(value, str) or isinstance(value[0], str):
            data = (
                data.with_columns(
                    pl.col(self._entity_column).map_dict(self._entity_category_codes_map())
                )
            )
        if isinstance(value, (str, int)):
            filter_expr = pl.col(self._entity_column).eq(value)
        else:
            filter_expr = pl.col(self._entity_column).is_in(value)

        return data.filter(filter_expr)

    # ------------------------------------------------------------------

    def _is_time_varying(
            self,
            column: str | list[str] = None,
    ) -> dict[str, bool]:
        """
        Time varying columns

        Dictionary with column names as keys and boolean values indicating whether the column is
        time varying (within entity).

        Parameters
        ----------
        column
            Select specific column(s)

        Returns
        -------
        dict
        filter_entities

        """

        # pipeline may not be filtered

        if self._is_panel:
            return get_time_varying_columns(
                data=self._data,
                entity_column=self._entity_column,
                time_column=self._time_column,
                column=column
            )
        raise ValueError("Time varying variables are only defined if 'data' is panel")

    # -------------------------- group times ---------------------------

    def _select_group_times(
            self,
            base_period: Literal["varying", "varying-multiple", "universal"],
            feasible: bool = False,
    ) -> list[dict]:
        group_times = get_group_times_list_of_dicts(
            cohorts=self._cohorts(),
            times=self._times(),
            anticipation=self._anticipation,
            base_period=base_period,
            dosages=self._dosages(),
            feasible_group_times=self._feasible_group_times if feasible else None
        )
        return group_times

    def _add_expr_to_group_times(self, group_times: list[dict]) -> list[dict]:
        return add_expr_to_group_time_list_dicts(
            group_times=group_times,
            time_column=self._time_column,
            cohort_column=self._cohort_column,
            anticipation=self._anticipation,
        )

    def _filter_group_times(
            self,
            base_period: Literal["varying", "varying-multiple", "universal"],
            feasible: bool = False,

            *,

            relative_period: int | list[int] = None,
            relative_period_start: int = None,
            relative_period_end: int = None,

            cohort: int | list[int] = None,
            cohort_start: int = None,
            cohort_end: int = None,

            time: int | list[int] = None,
            time_start: int = None,
            time_end: int = None,
    ):
        group_times = self._select_group_times(
            base_period=base_period,
            feasible=feasible,
        )

        group_times = filter_group_time_list_of_dicts(
            group_times=group_times,

            relative_period=relative_period,
            relative_period_start=relative_period_start,
            relative_period_end=relative_period_end,

            cohort=cohort,
            cohort_start=cohort_start,
            cohort_end=cohort_end,

            time=time,
            time_start=time_start,
            time_end=time_end,
        )

        group_times = self._add_expr_to_group_times(group_times=group_times)

        return group_times

    # ------------------------------------------------------------------
    def _get_data_split(
            self,
            sample_column: str = None,
            sample_value: str | int = None,
            columns: str | list[str] = None
    ) -> pl.DataFrame:

        # todo: sample_column should split the data without splitting each entity !!
        return (
            self._data
            .lazy()
            .filter(pl.col(sample_column).eq(sample_value))
            .select(columns)
            .collect()
        )

    def _get_unique_values(self, column: str) -> pl.Series:
        return self._data[column].unique()


def data_es_description(
        times: list[int],
        is_panel: bool,
        is_balanced_panel: bool,
        is_repeated_cross_section: bool,
        is_single_event: bool,
        is_single_cohort: bool
):
    out = ""

    if is_panel:

        first_time, last_time = min(times), max(times)

        if is_balanced_panel:
            out = (
                f"The data is a balanced panel, "
                f"between: [{first_time}, {last_time}]."
            )
        else:
            out = (
                f"The data is an unbalanced panel, "
                f"between: [{first_time}, {last_time}]."
            )

    elif is_repeated_cross_section:
        out = "The data is a repeated-cross-section."

    if is_single_event:
        # single cohort can only be for single_event data, since no repeated dates
        if is_single_cohort:
            out += " Each entity is treated at the same time."
        else:
            out += " Treatment is staggered."
    else:
        out += (
            " Treatment is staggered & "
            "some entities are treated multiple times."
        )
    return out
