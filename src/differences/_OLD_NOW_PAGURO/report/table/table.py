from __future__ import annotations

import copy
from pathlib import Path

import polars as pl

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from typing import Literal, Any

from differences._OLD_NOW_PAGURO.info import Info
from differences._OLD_NOW_PAGURO.info import InfoList

from differences.report._switch_data import _DataInfoSwitch

from differences.report.table._utils_table import (
    add_reference_coefficient,
    _sort_as_numeric,
    identify_est_and_index_columns,
    from_pandas_remove_nested_index_from_dataframe,
    camel_to_snake
)


class Table:
    """
    Estimates/Coefficients Table

    Group
    -----
        table
    """

    def __init__(
            self,
            data: pl.DataFrame,
            name: str = None,
            *,
            id_column: str = None,
            estimate_column: str = None,
            lu_columns: tuple[str, str] | None = ("lower", "upper"),
            group_columns: str | list[str] = None,
            **group_literals
    ) -> None:
        """
        Coefficient Table.

        This class is designed to represent coefficient tables using the given parameters.

        Parameters
        ----------
        data : pl.DataFrame
            The data to be stored in the coefficient table.

        name : str, optional
            The name of the table. Default is None.

        id_column : str
            The column in `data` which represents the names/identifiers of the coefficients.

        estimate_column : str
            The column in `data` which represents the estimates of the coefficients.

        lu_columns : tuple[str, str], optional
            A tuple containing names of the columns that represent the
            lower and upper limits of the confidence interval respectively.
            Default is ("lower", "upper").

        group_columns : str or list[str], optional
            Column(s) in `data` by which coefficients can be grouped.
            Can be a single column name (string) or a list of column names.
            Default is None.

        **group_literals
            To create key (column name) value (literal) column name will be added to group_columns
            allows to create groups more easily.
            # used to be **info, now info should be added with method
        """

        # --------------------------------------------------------------

        super().__init__(data=data, name=name)

        id_column, estimate_column = identify_est_and_index_columns(
            data=data,
            id_column=id_column,
            estimate_column=estimate_column
        )

        # --------------------------------------------------------------

        columns_in_table = [id_column, estimate_column]

        if lu_columns is not None:
            columns_in_table += list(lu_columns)

        missing_columns = [i for i in columns_in_table if i not in data.columns]

        if missing_columns:
            raise ValueError(
                f"Missing: {missing_columns}. "
                f"Can't initialize Table using {columns_in_table}.\n"
                f"The available columns are {data.columns}"
            )

        # --------------------------------------------------------------

        for g in group_literals.keys():
            if g in self._data.columns:
                raise ValueError(
                    f"Invalid group name: {g}.\n"
                    f"{g} is already a column in 'data'."
                )

        self._data = self._data.with_columns(**{g: pl.lit(l) for g, l in group_literals.items()})

        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif group_columns is None:
            group_columns = []  # better to use a list for joining, _format_info

        self.group_columns = []
        for i in group_columns + list(group_literals.keys()):
            if i not in self.group_columns:
                self.group_columns.append(i)

        # --------------------------------------------------------------

        self.id_column = id_column
        self.estimate_column = estimate_column

        self._lu_columns = lu_columns

        self._lower_column, self._upper_column = None, None
        if lu_columns is not None:
            self._lower_column, self._upper_column = lu_columns

        # --------------------------------------------------------------

        # check that the data has no duplicates at the index-group level
        self._check_unique_rows()  # NO DUPLICATES ALLOWED

        # flag whether the table originated from concatenating tables
        self._is_concatenated_table = False

        # --------------------------------------------------------------

        self._plot_kwargs = None
        self._config: dict = {"PLOT_CONFIG": {}}  # unused

        self._info: InfoList = InfoList()

        self._info._config_repr(
            tbl_rows=100,
            tbl_formatting="UTF8_BORDERS_ONLY",  # check UTF8_NO_BORDERS
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        )

    @classmethod
    def _from_model_results(
            cls,
            *results
    ):
        # todo: linearmodels, statsmodels?
        ...

    @classmethod
    def from_pandas(
            cls,
            data,
            name: str = None,
            *,
            id_column: str = None,
            estimate_column: str = None,
            lu_columns: tuple[str, str] | None = ("lower", "upper"),
            group_columns: str | list[str] = None,
            **group_literals
    ) -> Self:
        """
        Create a Table from a pandas DataFrame

        Group
        -----
            Constructors

        Parameters
        ----------
        data: pandas.DataFrame
        name
        id_column
        estimate_column
        lu_columns
        group_columns
        group_literals

        Returns
        -------

        """
        data_pandas = data

        title, data = from_pandas_remove_nested_index_from_dataframe(
            data=data, with_title=True
        )

        if id_column is None:
            if (
                    len(data) == 1 and
                    title is not None and
                    data_pandas.index.names[0] is None
            ):
                id_column = "id_column"

                data.insert(
                    0,
                    id_column,
                    [camel_to_snake(title, separator=" ", capitalize=True)]
                )

        if title is not None and name is None:
            name = title

        data = pl.from_pandas(data)

        table = cls(
            data=data,
            name=name,
            id_column=id_column,
            estimate_column=estimate_column,
            lu_columns=lu_columns,
            group_columns=group_columns,
            **group_literals
        )

        table._data_pandas = data_pandas

        return table

    def __getitem__(self, key) -> pl.Series:
        if isinstance(self._data, pl.DataFrame):
            return self._data[key]
        else:
            raise ValueError("data must be a DataFrame")

    def __getattr__(self, attr) -> Self | Any:
        """
        Delegate to Polars DataFrame or LazyFrame
        """
        if attr.startswith("_repr_"):  # or attr.startswith("_ipython_")
            raise AttributeError(f"{type(self).__name__} object has no attribute {attr}")

        # print("ATTRIBUTE CALLED:", attr)

        if isinstance(self._data, pl.LazyFrame):
            attr_from = pl.LazyFrame
        else:
            attr_from = pl.DataFrame

        if hasattr(attr_from, attr):
            attr_value = getattr(self._data, attr)

            new_obj = self.__copy__()
            if callable(attr_value):
                def wrapper(*args, **kwargs) -> Table | Any:
                    result = attr_value(*args, **kwargs)

                    if isinstance(result, pl.DataFrame):
                        new_obj._data = result  # data is replaced only DataFrame, no LazyFrame
                        # todo: names should be dropped it not in table anymore

                        return new_obj  # return Table
                    else:
                        return result

                return wrapper
            else:
                return attr_value
        else:
            raise AttributeError(
                f"{type(self).__name__} ['data' ({type(self._data).__name__})] "
                f"object has no attribute {attr}."
            )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:

        top_name = f"{type(self).__qualname__}"
        if self._name is not None:
            top_name = f'{top_name.center(len(self._name))}\n{self._name}'

        return self._info._repr_string(
            top_name=top_name,
            data_for_repr=self._data
        )

    def _append_table_info(
            self,
            **mapping: dict[str, dict | float | int | str | list]
    ) -> Self:
        """
        Adds metadata to the self._info list

        This is a private method because each subclass may need to
        set up the information names differently.

        Parameters
        ----------
        **mapping : dict[str, dict | str | list]
            Information from dictionary: _info_add(statistics={"R squared": 0.7})
            The outer key will be automatically assigned to the table name

        Returns
        -------
        Self
            The instance with updated metadata.
        """

        if self._name is None:
            raise ValueError("Information can be added only "
                             "if Table has a name attribute assigned.")

        info = (
            Info(name=self._name)
            .update(**mapping)
            .config_repr(
                same_line=2,
                # n_columns: int = 1,
                # indent: int | tuple = (2, 2, 4, 2),
                # equal_symbol: str = ":",
                # bold_keys: bool = False,
                # top_lines: tuple[str] | str = ("─", "-"),
                # bottom_line: str = None,
                # center_frame: bool = True,
                # separator_when_join: str = "",
                # min_width_when_join: int = 0,
            )
        )

        self._info.append(info)

        return self

    # ------------------------------------------------------------------

    def _check_table_is_intact(self) -> None:

        for i in [
                     self.id_column,
                     self.estimate_column,
                     self._lower_column,
                     self._upper_column,
                 ] + self.group_columns:
            if i is None:
                continue

            if i not in self._data.columns:
                raise ValueError(
                    f"\nMissing column: '{i}'.\n"
                    f"The transformation has modified the name of "
                    f"the columns of {type(self).__name__}.\n"
                    f"If you want to rename the columns please use .rename_columns()."
                )

            # todo: check that the rows are unique at the id_column group level

    def _check_unique_rows(self) -> None:
        if self.group_columns is None:
            unique_rows = self.id_column

        else:
            unique_rows = [self.id_column] + self.group_columns

        if self._data.select(unique_rows).n_unique() != len(self._data):
            raise ValueError(
                f"\nThe values in {unique_rows} do not uniquely identify the rows of 'data'.\n"
                "Use 'group_columns' to specify, or add other column(s) which values "
                "uniquely identify the rows of 'data'."
            )

    def _add_group_columns(self, **group_literals) -> Self:

        for g, l in group_literals.items():
            if g in self._data.columns:
                raise ValueError(
                    f"Invalid group name: {g}.\n"
                    f"{g} is already a column in 'data'."
                )

            if l is None:
                ValueError(f"'{g}' should not be None")

            if g in self.group_columns:
                ValueError(f"'{g}' should not be in group_columns")

        new_obj = self.__copy__()

        for g in group_literals.keys():
            new_obj.group_columns.append(g)

        new_obj._data = new_obj._data.with_columns(
            **{g: pl.lit(l) for g, l in group_literals.items()}
        )

        return new_obj

    def _raname_columns(
            self,
            id_column: str = None,
            estimate_column: str = None,
            lu_columns: tuple[str, str] = None,
            **other_columns
    ) -> Self:
        new_obj = self.__copy__()

        lower_column, upper_column = None, None
        if lu_columns is not None:
            lower_column, upper_column = lu_columns

        new_names = {
            self.id_column: id_column,
            self.estimate_column: estimate_column,

            self._lower_column: lower_column,
            self._upper_column: upper_column,
        }

        new_names.update(other_columns)

        new_names = {k: v for k, v in new_names.items() if v is not None}
        new_obj._data = new_obj._data.rename(new_names)

        return new_obj

    # ------------------------------------------------------------------

    def concat_other(self, other: Self | list[Self]) -> Table:
        """
        Concatenate the focal Table with other Table(s) vertically

        Parameters
        ----------
        other
        """

        if not isinstance(other, list):
            other = [other]

        tables = [self] + other

        return concat_tables(tables=tables)

    def insert_zeros(
            self,
            id_values: int | str | list[int] | list[str] | bool = True,
    ) -> Table:  # todo: return Self
        """
        Insert zeros in est_column, and assign a value to the id_column based on id_values

        Parameters
        ----------
        id_values
        """
        new_obj = self.__copy__()

        if isinstance(id_values, bool):
            if not id_values:
                return new_obj
            else:  # if reference=True
                id_values = None  # if None, then it will try to
                # automatically identify a reference

        new_obj._data = add_reference_coefficient(
            data=new_obj._data,
            id_column=new_obj.id_column,
            estimate_column=new_obj.estimate_column,
            group_columns=new_obj.group_columns,
            reference_id=id_values
        )

        return new_obj

    def str_replace(
            self,
            pattern: str | pl.Expr,
            value: str | pl.Expr,
            *,
            literal: bool = False,
            n: int = 1,
            column: str = None
    ) -> Self:
        """
        Replace id values

        Shortcut to data.with_columns(pl.col(column).str.replace(...))

        Parameters
        ----------
        pattern
        value
        literal
        n
        column
        """
        new_obj = self.__copy__()

        if column is None:
            column = new_obj.id_column

        new_obj._data = (
            new_obj._data
            .with_columns(
                pl.col(column).str.replace(
                    pattern=pattern,
                    value=value,
                    literal=literal,
                    n=n
                )
            )
        )
        return new_obj

    def sort_by(
            self,
            column: str | list[str] = None,
            *,
            within_groups: bool = True,
            as_numeric: bool = False,
            **kwargs
    ) -> Self:
        """
        Sort by column

        Parameters
        ----------
        column
        within_groups
        as_numeric
        kwargs
        """
        new_obj = self.__copy__()

        if column is None:
            column = new_obj.id_column

        if not isinstance(column, list):
            column = [column]

        group_columns = [i for i in new_obj.group_columns if i not in column]
        if group_columns and within_groups:
            column = group_columns + column

        if as_numeric:
            new_obj._data = _sort_as_numeric(
                data=new_obj._data,
                id_column=new_obj.id_column,
                by=column,
                **kwargs
            )
        else:
            new_obj._data = new_obj._data.sort(by=column, **kwargs)

        return new_obj

    def _filter_by(
            self,
            column: str = None,
            **attr
    ) -> Self:
        """
        filter AND

        Parameters
        ----------
        column
        attr
        """
        # filter AND

        # todo: names should be dropped it not in table anymore

        if column is None:
            column = self.id_column

        new_obj = self.__copy__()

        if new_obj._data[column].dtype is pl.Utf8:
            attr_of = pl.col(column).str
        else:
            attr_of = pl.col(column)

        if attr:
            for k, v in attr.items():
                func = getattr(attr_of, k)

                if isinstance(v, dict):
                    func = func(**v)
                else:
                    func = func(v)

                new_obj._data = new_obj._data.filter(func)

        return new_obj

    # ------------------------ plot ------------------------------------

    def _set_plot_kwargs(
            self,
            backend: Literal["alt", "lp", "mpl"] = "lp",
            **kwargs
    ) -> Self:
        """
        Pre-set plot parameters

        Group
        -----
            Plot

        Parameters
        ----------
        backend
        kwargs
        """
        # todo: fix so one can update for various backends

        self._plot_kwargs = {
            "lp": {
                "x_column": self.id_column,
                "y_column": self.estimate_column,
                "ci_columns": self._lu_columns
            }
        }

        # self._plot_theme = {backend: theme}
        return self

    def plot(
            self,
            backend: Literal["alt", "lp", "mpl"] = "lp",
            **kwargs
    ):
        """
        Plot

        Group
        -----
            Plot

        Parameters
        ----------
        backend
        kwargs
        """
        from plotto import coefficient_plot

        if self._plot_kwargs is None:
            self._set_plot_kwargs(backend=backend, theme=None)

        theme = copy.deepcopy(self._plot_kwargs[backend])
        theme.update(kwargs)

        data = self._data
        if self.group_columns is not None:
            data = data.with_columns(pl.col(self.group_columns).cast(pl.Utf8))

        return coefficient_plot(data=data, backend=backend, **theme)

    # ------------------------------------------------------------------

    def to_summary(self, in_parenthesis: str = None) -> Summary:
        """
        Create a Summary of the Table

        Parameters
        ----------
        in_parenthesis
        """
        return Summary(table=self, in_parenthesis=in_parenthesis)


# ------------------------ Table helpers -------------------------------

def concat_tables(
        tables: list[Table]
) -> Table:
    _ = _all_table_names(tables)

    if len(tables) == 1:
        return tables[0]

    tables = [
        # add a column named 'name' in data containing all the table names.
        i._add_group_columns(name=i._name) if i._name is not None
        else i  # previously concatenated table (name=None)
        for i in tables
    ]

    # ------------------------------------------------------------------

    first_table = tables[0]

    id_column: str = first_table.id_column
    est_column: str = first_table.estimate_column
    lu_columns: tuple[str, str] | None = first_table._lu_columns

    if lu_columns is None:  # todo: relax this, take the first not None value
        raise ValueError("'ci_columns' can't be None when joining tables")

    other_tables = [
        t
        ._raname_columns(
            id_column=id_column,
            estimate_column=est_column,
            lu_columns=lu_columns,
        )
        for t in tables[1:]
    ]

    # ------------------------------------------------------------------

    table = Table(
        data=pl.concat([i._data for i in [first_table] + other_tables], how="diagonal_relaxed"),
        id_column=id_column,
        estimate_column=est_column,
        lu_columns=lu_columns,
        group_columns=list(set([j for i in tables for j in i.group_columns])),
        name=None,
    )

    table._info = []
    for i in tables:
        # names of the info should be unique because we checked before, on table names
        table._info.extend(i._info)

    table._is_concatenated_table = True

    return table


def _all_table_names(tables: list[Table]) -> list[str]:
    out = []

    for i in tables:

        if i._name is None:

            if 'name' in i._data.columns:

                if i._is_concatenated_table:

                    # table names from the previously concatenated tables
                    for name in i._data.select("name").to_series().unique().to_list():
                        if name in out:
                            raise ValueError(
                                f"\n'name'={name} is duplicated."
                                f"\nTo concatenate tables each name should be unique.")

                        out.append(name)

                else:
                    raise ValueError("'name' is a reserved column name.")
            else:
                raise ValueError(
                    "Missing 'name'; needed to concat tables."
                )

        else:  # if i.name is not None

            if i._name in out:
                raise ValueError(
                    f"\n'name'={i._name} is duplicated."
                    f"\nTo concatenate tables each name should be unique.")

            out.append(i._name)

    return out


# ----------------------------------------------------------------------


# TODO
class Summary(_DataInfoSwitch):
    """
    The summary of a :class:`Table`, to compare estimates across groups/models.

    Group
    -----
        report
    """

    def __init__(
            self,
            table: Table,
            in_parenthesis: str | list[str] = None
    ) -> None:
        # for i in range(3):
        #     setattr(self, f'footer_{i}', i)

        data_dict: dict[str, pl.DataFrame] = {}

        super().__init__(data_dict=data_dict)

    def _write_csv(self): ...

    def _write_text(self): ...

    def _write_latex(self): ...

    def write_html(
            self,
            file_path: str | Path = None,
            **kwargs
    ) -> str:
        """

        Parameters
        ----------
        file_path
        kwargs
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html

        Returns
        -------

        """
        return ""
