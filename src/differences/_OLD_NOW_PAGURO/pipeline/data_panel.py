from __future__ import annotations

import sys
import copy
import datetime
import warnings

import polars as pl

from differences._OLD_NOW_PAGURO.info import InfoList

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from typing import Iterable, Literal, Callable, Any

from polars.series.series import ArrayLike
from polars.type_aliases import FrameInitTypes

from differences._OLD_NOW_PAGURO.shared.event.event_data_date import pre_process_event_data_dates_and_entity_column

from differences._OLD_NOW_PAGURO.shared import entity_time
from differences._OLD_NOW_PAGURO.info import Info
from differences._OLD_NOW_PAGURO.shared.dates import date_as_int, PeriodFrequency

from differences._OLD_NOW_PAGURO.shared.panel.panel_utils import (
    get_time_varying_columns,
    null_count_by_column,
    build_panel_data
)
from differences._OLD_NOW_PAGURO.info.info_repr._str_utility import (
    add_horizontal_line_to_list_of_strings,
    nested_dict_to_str
)

from differences._OLD_NOW_PAGURO.shared import utils_expr


class DataPanel:
    """
    A container for a Polars DataFrame/LazyFrame to facilitate working with panel data.

    DataPanel allows you to create a plan and construct a dataset structured as a panel dataset.

    Group
    -----
        data_prep
    """

    def __init__(
            self,
            data: pl.LazyFrame | pl.DataFrame | FrameInitTypes,
            entity_column: str = None,
            time_column: str = None,
            **kwargs
    ) -> None:
        """
        Initialize DataPanel

        Parameters
        ----------
        data : pl.LazyFrame
            Input dataset. This is the data that will be used (by PanelDataES)
            to create the panel (longitudinal) dataset. The data thus needs
            to have at least two columns: an entity column and a time column.

                - The entity column must contain entity identifiers, for example
                'employee_id', `customer_id`, `state_id` ...

                - The time column must be a date column indicating when the entity is observed.
                    The data type of the time column must be a pl.Date.

            Note that an entity can be observed at multiple times;
            an entity can also have multiple observations within the same time.
            When `data` will be passed to PanelDataES.build_data_from_dfp()
            the dates will be collapsed to the chosen period level
            (please refer PanelDataES for more details); so, if an entity is
            observed multiple times within the same chose period the columns
            in `data` will be aggregated using the `agg_data_by_entity_time` argument.
        time_column : str, optional
            Column in 'data' that represents the time variable, must be of dtype pl.Date.
            If not provided, it will be automatically identified.
        entity_column : str, optional
            Column in 'data' that contains the entity identifiers.
            If not provided, it will be automatically identified.
            this can be different from the name of the entity column used
            in the event_data, however the identifiers in the two columns:
            entity_column in event_data and entity_column in data must overlap.
        """

        # todo: allow to pass an integer time_column
        # todo: allow to pass DataPanel

        if isinstance(data, DataPanel):  # if it is already a DataPanel
            if entity_column is None:
                entity_column = data.entity_column
            if time_column is None:
                time_column = data.time_column

            data = data._data

        elif is_dataset(data):
            data = data._data

        elif not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            data = pl.LazyFrame(data, **kwargs)

        super().__init__(data=data, name=None)  # _DataInfo

        self._panel_data: pl.LazyFrame | pl.DataFrame | None = None

        self.entities_to_filter: ArrayLike | None = None

        # todo: in case no time_column or time_column is integer, this should be None
        self.period_frequency: PeriodFrequency = "year"  # default is year

        # try to automatically select the time_column; if not possible ask to provide one
        self.time_column = entity_time.extract_time_column(
            data=data,
            time_column=time_column,
            list_if_not_identifiable=False,
            none_if_no_dates=False,
        )

        # try to automatically select the entity_column; if not possible, ask to provide one
        self.entity_column = entity_time.extract_entity_column(
            data=data,
            entity_column=entity_column,
            list_if_not_identifiable=False,
        )

        # --------------------------------------------------------------

        self._agg: dict[str, dict[str, pl.Expr | Iterable[pl.Expr] | dict]] = {
            "entity": {"aggs": [], "named_aggs": {}},
            "entity-time": {"aggs": [], "named_aggs": {}}
        }

        #  store aggs for each the pivot columns (pivot column as key)
        self._pivot_agg: dict[str, dict[str, dict[str, pl.Expr | Iterable[pl.Expr] | dict]]] = {

            # "entity": {"pivot_column (name)": {"agg": ..., "named_aggs": ...}}
            "entity": {},

            # "entity-time": {"pivot_column (name)": {"agg": ..., "named_aggs": ...}}
            "entity-time": {}
        }

        # attributes to store the data after the group_by - agg
        self._other_data_panel: list[DataPanel] = []  # store other DataPanel(s) to be joined

        # ------------------------- info -------------------------------

        self._info: InfoList = InfoList()

        # set information of columns to display in string repr
        self._info.append(
            Info(name="_entity_time")
            .update(
                _entity_time={
                    "entity_column": f'"{self.entity_column}"',
                    "time_column": f'"{self.time_column}"'}
            )
            .config_repr(same_line=2)
        )

    def __str__(self) -> str:

        # append information indicating if aggregations have been set
        panel_data = self._panel_data
        if panel_data is None:
            panel_data = f"{panel_data}"
            if self._has_aggregations():
                panel_data += " [with aggregations]"

        if self._panel_data is None:
            operating_on = f"operating on data [{type(self._data).__name__}]"
        else:
            operating_on = f"operating on panel_data [{type(self._panel_data).__name__}]"

        return self._info._repr_string(
            bottom_name=operating_on,
            data_for_repr={
                "_main_data": {
                    "data": self._data,
                    "panel_data": panel_data
                }
            }
        )

    def __dir__(self) -> list:

        if self._panel_data is None:
            _dir = dir(self._data)  # may be DataFrame or LazyFrame
        else:
            _dir = dir(self._panel_data)  # may be DataFrame or LazyFrame

        return sorted(set(_dir + list(super().__dir__())))

    def __copy__(self) -> Self:

        cls = self.__class__
        new_obj = cls.__new__(cls)

        new_obj._panel_data = self._panel_data

        new_obj.entity_column = self.entity_column
        new_obj.time_column = self.time_column

        new_obj.entities_to_filter = self.entities_to_filter
        new_obj.period_frequency = self.period_frequency

        # deepcopy the dicts of parameters
        new_obj._agg = copy.deepcopy(self._agg)
        new_obj._pivot_agg = copy.deepcopy(self._pivot_agg)
        new_obj._info = self._info.__copy__()

        # recursively copy other DataPanel(s), used for joins
        new_obj._other_data_panel = [i.__copy__() for i in self._other_data_panel]

        return new_obj

    def __getattr__(self, attr) -> Self | Any:
        """
        Delegate to Polars DataFrame or LazyFrame

        if the return of the delegated attr call is a dataframe or
        lazyframe, set the current data we are operating on to the
        result. If It's not, return the output.
        """

        if attr.startswith("_repr_"):  # or attr.startswith("_ipython_")
            raise AttributeError(f"{type(self).__name__} object has no attribute {attr}")

        # select the data to operate on (either panel_data or data)
        if self._panel_data is not None:
            data = self._panel_data
            operating_on = 'panel_data'
        else:
            data = self._data
            operating_on = 'data'

        # where to get the methods from, depends if the data we are operating on is DF or LF
        if isinstance(data, pl.DataFrame):
            attr_from = pl.DataFrame
        elif isinstance(data, pl.LazyFrame):
            attr_from = pl.LazyFrame
        elif isinstance(data, pl.dataframe.group_by.GroupBy):
            attr_from = pl.dataframe.group_by.GroupBy
        elif isinstance(data, pl.lazyframe.group_by.LazyGroupBy):
            attr_from = pl.lazyframe.group_by.LazyGroupBy
        else:
            raise ValueError(
                f"'data' should be either a:"
                f"\n\t- Polars DataFrame"
                f"\n\t- Polars GroupBy"
                f"\n\t- Polars LazyFrame"
                f"\n\t- Polars LazyGroupBy"
                f"\nInstead a {self._data.__class__} was used."
            )

        if hasattr(attr_from, attr):
            attr_value = getattr(data, attr)

            new_obj = self.__copy__()  # new object (not self) so that operation is not inplace

            # print("data id local", id(data))
            # print("data id self", id(self._data))
            # print("data id new_obj", id(new_obj._data))

            if callable(attr_value):
                def wrapper(*args, **kwargs) -> Self | Any:
                    result = attr_value(*args, **kwargs)

                    if isinstance(result, (
                            pl.DataFrame,
                            pl.LazyFrame,
                            pl.dataframe.group_by.GroupBy,
                            pl.lazyframe.group_by.LazyGroupBy
                    )):
                        if new_obj._panel_data is not None:
                            new_obj._panel_data = result
                        else:
                            new_obj._data = result

                        return new_obj
                    else:
                        return result  # return attribute from Polars

                return wrapper
            else:
                return attr_value  # return attribute from Polars
        else:
            raise AttributeError(
                f"{type(self).__name__} ['{operating_on}' ({type(data).__name__})] "
                f"object has no attribute {attr}."
            )

    def __getitem__(self, key) -> pl.Series:
        if self._panel_data is not None:
            if isinstance(self._panel_data, pl.DataFrame):
                return self._panel_data[key]
            else:
                raise ValueError("'panel_data' must be a pl.DataFrame")

        if isinstance(self._data, pl.DataFrame):
            return self._data[key]

        raise ValueError("'data' must be a pl.DataFrame")

    # ------------------------------------------------------------------

    def _get_data(
            self,
            period_frequency: PeriodFrequency | None,
            filter_entities: bool = False,
    ) -> pl.LazyFrame:
        data = self._data.lazy()

        if period_frequency is not None:
            data = data.with_columns(
                [date_as_int(self.time_column, period_frequency=period_frequency)]
            )

        if filter_entities:
            if self.entities_to_filter is not None:  # just keep the requested entities
                data = data.filter(pl.col(self.entity_column).is_in(self.entities_to_filter))

            else:
                raise ValueError(
                    "You are trying to filter_entities but "
                    "no pl.Series of entities was provided.\n"
                    "\tUse .set_panel_info() to set the entities to use."
                )

        return data

    @property
    def _other_columns(self) -> list[str]:
        """Columns that are not entity or time"""
        return [
            i for i in self._data.columns if i not in [self.entity_column, self.time_column]
        ]

    def _raise_no_other_columns_error(self) -> None:
        """Raise error if there are no other columns. Useful when trying to get variables"""
        if not self._other_columns:
            raise ValueError(
                "You are trying to create variables by aggregating the data, "
                "however there are no columns to be used."
            )

    @staticmethod
    def _lazy_or_collect(
            data: pl.LazyFrame | pl.DataFrame,
            comparison: pl.LazyFrame | pl.DataFrame
    ) -> pl.LazyFrame | pl.DataFrame:

        if isinstance(data, pl.DataFrame) and isinstance(comparison, pl.LazyFrame):
            return data.lazy()

        elif isinstance(data, pl.LazyFrame) and isinstance(comparison, pl.DataFrame):
            return data.collect()

        return data

    def _rename_et_columns(
            self,
            entity_column: str = None,
            time_column: str = None
    ) -> DataPanel:
        """
        Rename entity and/or time column in 'data'. Useful for joining across DataPanel s
        """
        if entity_column is None and time_column is None:
            return self

        new_obj = self.__copy__()

        if entity_column is not None:
            if new_obj.entity_column != entity_column:
                new_obj._data = new_obj._data.rename({new_obj.entity_column: entity_column})
                new_obj.entity_column = entity_column

        if time_column is not None:

            # todo: make sure time was passed
            if new_obj.time_column != time_column:
                new_obj._data = new_obj._data.rename({new_obj.time_column: time_column})
                new_obj.time_column = time_column

        return new_obj

    # --------------------- configuration methods ----------------------

    def set_panel_info(
            self,
            period_frequency: PeriodFrequency = None,
            entities: ArrayLike = None,
    ) -> Self:
        """
        Panel Information

        Group
        -----
            Lazy panel configuration

        Parameters
        ----------
        period_frequency
        entities
        """
        if period_frequency is None and entities is None:
            return self

        new_obj = self.__copy__()

        if period_frequency is not None:
            new_obj.period_frequency = period_frequency
        if entities is not None:
            new_obj.entities_to_filter = pl.Series(entities)
            # todo: make sure the name is preserved and used?

        return new_obj

    # -------------------------- aggregations --------------------------

    def _get_aggregations_dict(
            self,
            *aggs: pl.Expr | Iterable[pl.Expr],
            **named_aggs: pl.Expr,
    ) -> dict[str, list[pl.Expr] | dict[str, Iterable[pl.Expr]]]:

        if aggs:
            if isinstance(aggs[0], list):
                if len(aggs) > 1:
                    raise ValueError("Only one element allowed if 'aggs' is a list.")
                aggs = aggs[0]
            else:  # aggs is a tuple
                aggs = list(aggs)

        output_names = utils_expr.expr_all_output_names(*aggs, **named_aggs)

        duplicates = []
        for i in output_names:
            if i in duplicates:
                raise ValueError(f"Duplicated column output name: '{i}'\n"
                                 f"Please use an alias.")

            duplicates.append(i)

        already_exist = set(output_names) & set([i for i in self._meta_output_names()])

        if already_exist:
            raise ValueError(
                f"The names in {already_exist} have already been used in aggregations."
                f"\n You can either reset the aggregations or give a different "
                f"alias to these columns."
            )

        return {
            "aggs": aggs,
            "named_aggs": named_aggs
        }

    def _add_to_aggregations_dict(
            self,
            aggregations_dict: dict[str, list[pl.Expr] | dict[str, Iterable[pl.Expr]]],
            level: Literal["entity", "entity-time"],
            pivot_column: str = None
    ) -> None:

        # adding aggregations in-place
        aggs = aggregations_dict["aggs"]
        named_aggs = aggregations_dict["named_aggs"]

        if pivot_column is None:

            if aggs:
                self._agg[level]["aggs"].extend(aggs)
            if named_aggs:
                self._agg[level]["named_aggs"].update(named_aggs)

        else:
            if not aggs and not named_aggs:
                raise ValueError(
                    "a 'pivot_column' was passed but no aggregations have been specified."
                )
            if aggs:
                self._pivot_agg[pivot_column][level]["aggs"].extend(aggs)
            if named_aggs:
                self._pivot_agg[pivot_column][level]["named_aggs"].update(named_aggs)

    def add_entity_agg(
            self,
            *aggs: pl.Expr | Iterable[pl.Expr],
            pivot_column: str = None,
            **named_aggs: pl.Expr,
    ) -> Self:
        """
        Entity level aggregations

        Group
        -----
            Lazy Panel Configuration

        Parameters
        ----------
        aggs
        pivot_column
        named_aggs
        """
        if pivot_column is None and not named_aggs and not aggs:
            return self

        # this dict: {"aggs": aggs, "named_aggs": named_aggs}
        aggregations_dict = self._get_aggregations_dict(*aggs, **named_aggs)

        new_obj = self.__copy__()

        new_obj._add_to_aggregations_dict(
            aggregations_dict=aggregations_dict,
            level="entity",
            pivot_column=pivot_column
        )

        return new_obj

    def add_entity_time_agg(
            self,
            *aggs: pl.Expr | Iterable[pl.Expr],
            pivot_column: str = None,
            **named_aggs: pl.Expr,
    ) -> Self:
        """
        Entity-Time level aggregations

        Group
        -----
            Lazy panel configuration

        Parameters
        ----------
        aggs
        pivot_column
        named_aggs
        """
        if pivot_column is None and not named_aggs and not aggs:
            return self

        # this dict: {"aggs": aggs, "named_aggs": named_aggs}
        aggregations_dict = self._get_aggregations_dict(*aggs, **named_aggs)

        new_obj = self.__copy__()

        new_obj._add_to_aggregations_dict(
            aggregations_dict=aggregations_dict,
            level="entity-time",
            pivot_column=pivot_column
        )

        return new_obj

    # ------------------------------ meta ------------------------------

    def _meta_entity_agg(
            self,
            named: bool = False
    ) -> list[pl.Expr] | dict[str, pl.Expr | Iterable[pl.Expr]]:

        if named:
            return self._agg["entity"]["named_aggs"]

        return self._agg["entity"]["aggs"]

    def _meta_entity_time_agg(
            self,
            named: bool = False

    ) -> list[pl.Expr] | dict[str, pl.Expr | Iterable[pl.Expr]]:

        if named:
            return self._agg["entity-time"]["named_aggs"]

        return self._agg["entity-time"]["aggs"]

    def _meta_agg(
            self,
            which: Literal["output_names", "all_names", "trees"]
    ) -> tuple[list[str] | dict, list[str] | dict]:
        """
        Pivots are not checked: we don't know what the column names will be

        Parameters
        ----------
        which

        Returns
        -------

        """

        # output_names for the entity aggregations

        if which == "trees":
            func = utils_expr.expr_all_trees
        elif which == "output_names":
            func = utils_expr.expr_all_output_names
        elif which == "all_names":
            func = utils_expr.expr_all_root_and_output_names
        else:
            raise ValueError

        out_entity = func(
            *self._meta_entity_agg(),
            **self._meta_entity_agg(named=True)
        )

        # output_names for the entity-time aggregations
        out_entity_time = func(
            *self._meta_entity_time_agg(),
            **self._meta_entity_time_agg(named=True)
        )

        return out_entity, out_entity_time

    def _meta_output_names(self) -> list[str]:
        """
        Returns the column names that are the output of the aggregations.

        Pivots are not checked: we don't know what the column names will be
        """
        out_entity, out_entity_time = self._meta_agg(which="output_names")
        out_entity.extend(out_entity_time)
        return out_entity

    def _meta_aggregations_columns(self) -> str:
        """
        Format a string to describe the origin column and the output name
        """
        # pivots are skipped add a note to that, we don't know the names of the columns for those
        out_entity, out_entity_time = self._meta_agg(which="all_names")

        out = {str(k): v for k, v in {**out_entity, **out_entity_time}.items()}

        return nested_dict_to_str(out)

    def _meta_aggregations_tree_format(self) -> str:
        """
        Format a string to describe the origin column and the output name. Tree format
        """
        # todo: PIVOT AGGREGATIONS

        if self.period_frequency is not None:
            agg_freq = f"({self.period_frequency}) [aggregations]"
        else:
            agg_freq = " [aggregations]"

        # --------------------------------------------------------------
        out = []

        out_entity, out_entity_time = self._meta_agg(which="trees")

        if out_entity:
            out_entity = add_horizontal_line_to_list_of_strings(
                strings=out_entity, title=f"entity [aggregations]"
            )

            out.extend(out_entity)

        if out_entity_time:
            out_entity_time = add_horizontal_line_to_list_of_strings(
                strings=out_entity_time,
                title=f"entity-time{agg_freq}"
            )
            out.extend(out_entity_time)

        out = "\n".join(out)

        return out

    def meta_agg(
            self,
            tree_format: bool = True,
            *,
            return_as_string: bool = False
    ) -> str | None:
        """
        Metadata on aggregations

        Group
        -----
            Metadata

        Parameters
        ----------
        tree_format
        return_as_string
        """

        if tree_format:
            out = self._meta_aggregations_tree_format()
        else:
            out = self._meta_aggregations_columns()

        if return_as_string:
            return out
        else:
            print(out)

    def _has_aggregations(
            self,
            by_entity_time: bool = None,
            is_pivot: bool = False
    ) -> bool:
        """

        Parameters
        ----------
        by_entity_time
            None is just used to check if it has aggregations at any level
        is_pivot
        """

        if by_entity_time is not None:
            dict_ = self._pivot_agg if is_pivot else self._agg

            key = "entity-time" if by_entity_time else "entity"

            return bool(dict_[key].get("aggs")) or bool(dict_[key].get("named_aggs"))

        else:  # check if it has any aggregation at any level
            for et in [True, False]:
                for p in [False, True]:  # pivot
                    has_aggregation = self._has_aggregations(et, p)
                    if has_aggregation:
                        return has_aggregation
            return False

    # ------------------------------------------------------------------

    def add_join_other(
            self,
            other: DataPanel | Iterable[DataPanel],
    ) -> Self:
        """
        Set join with other DataPanel

        Group
        -----
            Lazy panel configuration

        Parameters
        ----------
        other
        """

        new_obj = self.__copy__()

        if isinstance(other, DataPanel):
            new_obj._other_data_panel.append(other.__copy__())
        else:
            for o in other:
                new_obj._other_data_panel.append(o.__copy__())

        return new_obj

    # ------------------------------------------------------------------

    def reset(
            self,
            *args: Literal["panel_data", "entity-agg", "entity_time-agg", "join_other"],
            pivot_column: str = None
    ) -> Self:
        """
        Reset `set_entity_agg` and `set_entity_time_agg` to empty aggregations
        Reset `panel_data` to None

        Group
        -----
            Lazy panel configuration
        """
        new_obj = self.__copy__()

        if "entity-agg" in args:
            if pivot_column:
                new_obj._pivot_agg["entity"][pivot_column] = {
                }

            else:

                new_obj._agg["entity"] = {
                    "aggs": [], "named_aggs": {}
                }

        if "entity_time-agg" in args:
            if pivot_column:
                new_obj._pivot_agg["entity-time"][pivot_column] = {}
            else:

                new_obj._agg["entity-time"] = {
                    "aggs": [], "named_aggs": {}
                }

        if "join_other" in args:
            new_obj._other_data_panel = []  # store other DataPanel(s) to be joined

        if "panel_data" in args:
            new_obj._panel_data = None

        return new_obj

    # ------------------------------------------------------------------

    def _get_agg_variables(
            self,
            data: pl.LazyFrame | pl.DataFrame,
            by_entity_time: bool
    ) -> pl.LazyFrame | pl.DataFrame | None:
        """entity OR entity-time aggregations"""

        if not self._other_columns:
            return None  # in case there are no columns to aggregate

        if not self._has_aggregations(by_entity_time=by_entity_time, is_pivot=False):
            return None

        if by_entity_time:
            group_by_columns = [self.entity_column, self.time_column]
            aggregations = self._agg["entity-time"]

        else:
            group_by_columns = [self.entity_column]
            aggregations = self._agg["entity"]

        data = (
            data
            .group_by(group_by_columns)
            .agg(
                *aggregations["aggs"],
                **aggregations["named_aggs"]
            )
        )

        return data

    def _get_pivot_agg_variables(
            self,
            data: pl.LazyFrame | pl.DataFrame,
            by_entity_time: bool
    ) -> dict[str, pl.LazyFrame] | None:
        """entity OR entity-time PIVOT aggregations"""

        if not self._has_aggregations(by_entity_time=by_entity_time, is_pivot=True):
            return None

        if by_entity_time:
            pivot_aggs = self._pivot_agg["entity-time"]
        else:
            pivot_aggs = self._pivot_agg["entity"]

        out = []
        for pivot_column, aggregations in pivot_aggs.items():

            if by_entity_time:
                group_by_columns = [self.entity_column, self.time_column, pivot_column]
            else:
                group_by_columns = [self.entity_column, pivot_column]

            data_to_pivot = (
                data
                .group_by(group_by_columns)
                .agg(
                    *aggregations["aggs"],
                    **aggregations["named_aggs"]
                )
            )

            if isinstance(data_to_pivot, pl.LazyFrame):
                data_to_pivot = data_to_pivot.collect()  # no pivot for LazyFrame

            # index either entity or entity-time
            index_columns = [i for i in group_by_columns if i != pivot_column]

            # values: all the other columns created by the group_by
            value_columns = [i for i in data_to_pivot.columns if i not in group_by_columns]

            # add the dataframe to the dictionary
            out.append(
                data_to_pivot
                .pivot(
                    index=index_columns,
                    values=value_columns,

                    columns=pivot_column,  # always one column here, (not using built in option)

                    aggregate_function='first',  # there are no duplicates after group_by
                    maintain_order=True,
                    sort_columns=True,
                    separator="_"  # the separator is also used just below
                )
            )
            # columns: (we are iterating one by one, could also do multiple at a time)

            if len(value_columns) == 1:
                # add pivot_column as prefix to the columns created
                columns_created = [i for i in out.columns if i not in group_by_columns]
                out = out.rename({
                    k: f"{value_columns[0]}_{pivot_column}_{k}" for k in columns_created
                })
            # if len(value_columns) > 1:
            # Polars handles this case with the same pattern:
            # {column name in values}_{column name in columns (pivot)}_{entries in pivot}

        data = out[0]

        # join all frames that were created by different pivots
        for df in out[1:]:
            if by_entity_time:
                data = data.join(df, on=[self.entity_column, self.time_column], how="outer")
            else:
                data = data.join(df, on=[self.entity_column], how="outer")

        return data

    def _get_entity_or_entity_time_variables(
            self,
            by_entity_time: bool,
            *,
            filter_entities: bool = False,
            data: pl.LazyFrame | pl.DataFrame = None,
    ) -> tuple[pl.LazyFrame, bool]:

        if data is None:
            data = self._get_data(
                # period_frequency=None, so that if aggregations are performed on the time column
                # one can access the Date (not the integer)
                period_frequency=self.period_frequency if by_entity_time else None,
                filter_entities=filter_entities
            )

        else:  # data is not None

            if filter_entities:

                # if no entities were set no filtering is happening
                if self.entities_to_filter is not None:
                    data = data.filter(pl.col(self.entity_column).is_in(self.entities_to_filter))

        # entity_data OR entity_time_data
        data = [
            d for d in [
                # regular variables
                self._get_agg_variables(data=data, by_entity_time=by_entity_time),
                # pivot variables
                self._get_pivot_agg_variables(data=data, by_entity_time=by_entity_time)
            ]
            if d is not None  # d is None if there are no set_* aggregations
        ]

        # has_entity_data OR has_entity_time_data
        has_data = bool(len(data))  # data is a list now, if there is at least one element
        # there are aggregations

        # --------------------------------------------------------------

        if by_entity_time:
            join_with_pivot_columns = [self.entity_column, self.time_column]
        else:
            join_with_pivot_columns = [self.entity_column]

        if len(data) == 2:
            data = (
                data[0]  # regular variables
                .join(
                    data[1],  # pivot variables
                    on=join_with_pivot_columns,
                    how='left'  # entity_data/entity_time_data should contain all the observations
                )
            )
        elif len(data) == 1:
            data = data[0]

        return data, has_data

    def _get_all_variables(
            self,
            data: pl.DataFrame | pl.LazyFrame = None,
            filter_entities: bool = False,
    ) -> tuple[pl.LazyFrame | pl.DataFrame, bool] | tuple[None, None]:
        """
        Get aggregations + pivot aggregations
        """

        # todo: do not raise error if there are no columns
        if not self._other_columns:
            return None, None

        # ------------------- entity -----------------------------------

        entity_data, has_entity_data = self._get_entity_or_entity_time_variables(
            by_entity_time=False,
            filter_entities=filter_entities,
            data=data,
        )

        # ------------------- entity-time ------------------------------

        entity_time_data, has_entity_time_data = self._get_entity_or_entity_time_variables(
            by_entity_time=True,
            filter_entities=filter_entities,
            data=data,
        )

        # --------------------------------------------------------------

        # the bool return: 'has_entity_time_data' is useful to indicate whether the variables
        # should be joined with data/panel_data by entity-time or not.
        # It will be used as 'by_entity_time'
        if has_entity_time_data and has_entity_data:

            return (
                # outer because either of the two could just be the data from pivot
                entity_time_data.join(entity_data, on=self.entity_column, how="outer"),
                has_entity_time_data
            )

        elif has_entity_time_data:

            return entity_time_data, has_entity_time_data

        elif has_entity_data:
            # returns 'has_entity_time_data'=False if the join should be by entity
            return entity_data, has_entity_time_data

        else:
            return None, None

    def _join_data_or_panel_with_variables(
            self,
            data: pl.LazyFrame | pl.DataFrame,
            variables: pl.LazyFrame | pl.DataFrame | None,
            by_entity_time: bool
    ) -> pl.LazyFrame | pl.DataFrame | None:
        """
        Join aggregated variables with `panel_data`

        Only joins variable from focal class instance, if you want to join variables from
        other instanced of DataPanel use the public method `.join_other()`
        """
        # data should be panel_data if self, could be panel_data or data if other

        if variables is None:
            return data  # return input-data without the joined variables

        # variables must be same type of frame as data
        variables = self._lazy_or_collect(
            data=variables,
            comparison=data  # may be 'data' or 'panel_data'
        )

        if data.schema[self.entity_column] != variables.schema[self.entity_column]:
            # cast 'entity_column'
            variables = (
                variables
                .with_columns(
                    pl.col(self.entity_column).cast(data.schema[self.entity_column])
                )
            )

        if by_entity_time:
            if data.schema[self.time_column] != variables.schema[self.time_column]:
                # cast 'time_column'
                variables = (
                    variables
                    .with_columns(
                        pl.col(self.time_column).cast(data.schema[self.time_column])
                    )
                )

            data = (
                data
                .join(
                    variables,
                    on=[self.entity_column, self.time_column],
                    how="left"
                )
            )
        else:
            data = data.join(variables, on=[self.entity_column], how="left")

        return data

    # ----------------------- main methods -----------------------------

    def set_data_as_panel(
            self,
            validate: bool = False,
            *,
            with_date: bool = False,
            filter_entities: bool = False,
    ) -> Self:
        """
        Set `panel_data` equal to `data`

        Group
        -----
            Panel Data

        Parameters
        ----------
        validate
        with_date
        filter_entities
        """

        # todo: check that time_column is an integer

        # if data.schema[time_column] is pl.Date:
        #     # convert time_column from pl.Date to integer
        #     data = data.with_columns([
        #         date_as_int(time_column, period_frequency=period_frequency)
        #     ])

        # if with_date:  # add a pl.Date column corresponding to the chosen period frequency
        #     data = (
        #         data
        #         .with_columns([
        #             int_as_date(time_column, period_frequency=period_frequency)
        #             .alias(f"{time_column}_as_date")
        #         ])
        #     )

        if validate:
            if not isinstance(self._data, pl.DataFrame):
                raise ValueError(
                    "'validate' can only be used if data is a pl.DataFrame, "
                    "and not with a pl.LazyFrame."
                )

            # todo: check that the data has no duplicates at the entity-time level

        new_obj = self.__copy__()

        if self._has_aggregations():
            warnings.warn(
                "\nAggregations have been ignored. "
                "\n'panel_data' will contain the same columns as 'data'."
            )

        new_obj._panel_data = new_obj._data

        return new_obj

    def build_panel(
            self,
            start: int | str | datetime.date = None,
            end: int | str | datetime.date = None,

            relative_end: int = None,

            *,
            with_date: bool = False,
            filter_entities: bool = False,
    ) -> Self:
        """
        Convert the data to a panel dataset. This is the main manipulation method of DataPanel.

        Group
        -----
            Panel Data

        Parameters
        ----------
        start

            .. md-tab-set::
                .. md-tab-item:: int

                    Each entity's time-series will start from the specified integer.
                    For example if start is 2001, the first time for each entity will be 2001.

                    .. warning::

                        If you pass an integer when `period_frequency` is not "year", the number
                        must correspond to the datatime.date converted to integer using ...

                        To start from a specific time without the need to convert the date to
                        an integer you can pass a string or a datetime.date

                .. md-tab-item:: str

                    example is

                .. md-tab-item:: datetime.date

                    example is

                .. md-tab-item:: None

                    example is

        end
        relative_end
        with_date
        filter_entities
        """

        new_obj = self.__copy__()

        data = new_obj._get_data(
            period_frequency=new_obj.period_frequency,
            filter_entities=filter_entities
        )

        panel_data = build_panel_data(
            data=data,

            entity_column=new_obj.entity_column,
            time_column=new_obj.time_column,

            start=start,
            end=end,

            relative_end=relative_end,

            with_date=with_date,

            period_frequency=new_obj.period_frequency,
        )

        # ---------------------- self-variables ------------------------

        # data=None: to trigger ._get_data(period_frequency=...)
        variables, by_entity_time = new_obj._get_all_variables(data=None)

        if variables is not None:
            # join variables created through the aggregations with the panel
            # (no join_other variables here)
            panel_data = new_obj._join_data_or_panel_with_variables(
                data=panel_data,
                variables=variables,
                by_entity_time=by_entity_time  # temp None until allow for passing own vars
            )

        # ---------------------- other-variables -----------------------

        if new_obj._other_data_panel:
            panel_data = new_obj._join_other(
                new_obj=new_obj,
                data=panel_data,
                which_data="panel_data",
                filter_entities=filter_entities,
                other_data_panels=new_obj._other_data_panel
            )

        # --------------------------------------------------------------

        # if self._data is a DataFrame, collect panel_data as well
        new_obj._panel_data = new_obj._lazy_or_collect(
            data=panel_data,
            comparison=new_obj._data
        )

        return new_obj

    def join_other(
            self,
            other: DataPanel | list[DataPanel],
            filter_entities: bool = False,
    ) -> Self:
        if other is None and not self._other_data_panel:
            raise ValueError(
                "No 'other' DataPanel to join, either\n"
                "\t- provide a value for the argument named 'other'"
                "\t- use .set_join_other() to configure the join"
            )

        new_obj = self.__copy__()

        if isinstance(other, list):
            other = [o.__copy__() for o in other]
        else:
            other = [other.__copy__()]

        if new_obj._panel_data is not None:
            new_obj._panel_data = self._join_other(
                new_obj=new_obj,
                data=new_obj._panel_data,
                which_data="panel_data",
                filter_entities=filter_entities,
                other_data_panels=other
            )

        else:  # join other variables with 'data' if no panel_data has been created yet
            new_obj._data = self._join_other(
                new_obj=new_obj,
                data=new_obj._data,
                which_data="data",
                filter_entities=filter_entities,
                other_data_panels=other
            )

        return new_obj

    @staticmethod
    def _join_other(
            new_obj,
            data: pl.LazyFrame | pl.DataFrame,
            which_data: Literal["data", "panel_data"],
            other_data_panels: list[DataPanel],
            *,
            filter_entities: bool,
    ) -> pl.DataFrame | pl.LazyFrame:

        for other in other_data_panels:  # iter over all joined DataPanel(s)

            # if the entity_column have different names
            if other.entity_column != new_obj._entity_column:
                other = other._rename_et_columns(entity_column=new_obj._entity_column)

            # if the time_column have different names
            if isinstance(other.time_column, str) and isinstance(new_obj._time_column, str):
                if other.time_column != new_obj._time_column:
                    other = other._rename_et_columns(time_column=new_obj._time_column)

            # set other to have the same panel_info as self
            other = (
                other
                .set_panel_info(
                    period_frequency=new_obj.period_frequency,
                    entities=new_obj.entities_to_filter
                )
            )

            # get variables from other DataPanel:
            #   - either frame | bool, or
            #   - None | None
            other_variables, by_entity_time = other._get_all_variables(
                # data=None: to trigger ._get_data(period_frequency=...), in other
                data=None if which_data == "panel_data" else other._data,
                filter_entities=filter_entities
            )

            if other_variables is None:
                # there are no variables in other DataPanel
                # either no aggregations were set or no columns to aggregate
                continue

            # may be 'data' or 'panel_data'
            data = new_obj._join_data_or_panel_with_variables(
                data=data,  # may be 'data' or 'panel_data'
                variables=other_variables,
                by_entity_time=by_entity_time
            )

        return data

    # -------------------------- utilities -----------------------------

    def get_variables(
            self,
            filter_entities: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Get the `variables` that get created using the `set_*` methods.

        To insert variables into `panel_data`, when calling `.to_panel()`, either:
            - set the `variables` argument to True
            - pass a Polars DataFrame or LazyFrame to the `variables` argument.

        """
        data, _ = self._get_all_variables(data=None, filter_entities=filter_entities)
        return data

    def is_time_varying(
            self,
            columns: str | list[str] = None,
            filter_entities: bool = False
    ) -> dict[str, bool]:
        """
        Time varying columns

        Dictionary with column names as keys and boolean values indicating whether the column is
        time varying (within entity).

        Parameters
        ----------
        columns
            Select specific column(s)
        filter_entities

        Returns
        -------
        dict
        filter_entities
        """

        # panel_data may not be filtered
        data = self._get_data(
            period_frequency=self.period_frequency, filter_entities=filter_entities
        )

        return get_time_varying_columns(
            data=data,
            entity_column=self.entity_column,
            time_column=self.time_column,
            column=columns
        )

    def null_count_by(
            self,
            by: str | bool = None,
            share: bool = True,
            columns: str | list[str] = None,
    ) -> pl.DataFrame:
        """
        Find missing data: nulls by entity or time
        """
        return null_count_by_column(
            data=self._panel_data if self._panel_data is not None else self._data,
            by=by,
            columns=columns,
            share=share,
            time_column=self.time_column
        )

    def to_frame(self, filter_entities: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """
        Get the internally stored `panel_data` or `data` (with `time_column` as integer).

        This method returns the internal panel data as a DataFrame or LazyFrame.
        It provides an option to filter the data based on predefined entities.

        If 'data' is returned (in case `panel_data` is None) the time_column is coverted to
        an integer according to the predefined `period_frequency`.

        Parameters
        ----------
        filter_entities : bool, optional
            If True, filters the DataFrame/LazyFrame based on the entities specified in the class.

        Returns
        -------
            pl.DataFrame | pl.LazyFrame
        """
        if self._panel_data is not None:
            if filter_entities and self.entities_to_filter is not None:
                # entities may have already been filtered when calling .to_panel()
                return self._panel_data.filter(
                    pl.col(self.entity_column).is_in(self.entities_to_filter)
                )
            else:
                return self._panel_data

        # return the data with the pl.Date column converted into integer,
        # for the raw data get self._data
        data = self._get_data(
            period_frequency=self.period_frequency,
            filter_entities=filter_entities
        )

        return data

    def to_panel_models(self):
        """
        Instantiates PanelModels

        Returns
        -------
            PanelModels
        """
        from differences.models import PanelModels

        if self._panel_data is None:
            raise RuntimeError(
                f".to_panel() needs to be called in order to instantiate PanelModel"
            )

        panel_models = PanelModels(
            data=self._panel_data,
            entity_column=self.entity_column,
            time_column=self.time_column,
        )

        return panel_models

    def to_event_models(
            self,
            event_data: pl.DataFrame,
            *,
            cohort_column: str = None,
            dosage_column: str = None,
            entity_column: str = None,
            **named_aggs
    ):
        """
        Instantiates EventModels

        Returns
        -------
            EventModels
        """
        from differences.models.lm.event_models import EventModels

        if self._panel_data is None:
            raise RuntimeError(
                f".to_panel() needs to be called in order to instantiate PanelModel"
            )

        event_data = pre_process_event_data_dates_and_entity_column(
            data=self._data,
            event_data=event_data,
            entity_column=entity_column,
            cohort_column=cohort_column,
            panel_entity_column=self.entity_column,
            period_frequency=self.period_frequency,
            **named_aggs
        )

        models = EventModels(
            data=self._panel_data,
            event_data=event_data,
            entity_column=self.entity_column,
            time_column=self.time_column,
            cohort_column=cohort_column,
            dosage_column=dosage_column
        )

        return models


# ----------------------------------------------------------------------

def to_datapanel(entity_column: str = None, time_column: str = None) -> Callable:
    """Func for data.pipe(to_datapanel())"""

    def _to_datapanel(data: pl.DataFrame | pl.LazyFrame) -> DataPanel:
        return DataPanel(data=data, entity_column=entity_column, time_column=time_column)

    return _to_datapanel


def is_dataset(obj):
    from differences._OLD_NOW_PAGURO.pipeline.dataset import Dataset  # to avoid circular imports
    return isinstance(obj, Dataset)

# ----------------------------------------------------------------------

# def _period_frequency_warning(
#         self,
#         start: int | datetime.date | str | None,
#         end: int | datetime.date | str | None,
# ):
#     if self.period_frequency != "year":
#         # this warning is not needed but it's a good reminder if an integer is passed,
#         # the integer needs to be converted first.
#
#         if isinstance(start, int) or isinstance(end, int):
#             warnings.warn(
#                 "REMINDER: when you use an integer for the start and/or end arguments, "
#                 "the integer needs to correspond to the integer conversion of the date "
#                 "as returned by convert_date_to_int(date) "
#                 "in 'differences.panel_es.panel_data.dates'. Only if period_frequency='year' "
#                 "the integer is easily understandable "
#                 "(2010 will correspond to the year 2010).\n"
#                 "If you want the start/end to be from a specific date, but you do not want to "
#                 "convert that date to an integer, please use a string or a datetime.date, "
#                 "for example if you use end='2018-05' the time-series for each entity will "
#                 "end on May 2015."
#             )
