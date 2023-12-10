from __future__ import annotations

import sys

import polars as pl

from typing import Any, Sequence, Callable

from polars.type_aliases import FrameInitTypes, JoinStrategy, JoinValidation

from differences._OLD_NOW_PAGURO.info import InfoList

from differences._OLD_NOW_PAGURO.report.desc.skim_results import SkimResults

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from differences._OLD_NOW_PAGURO.pipeline.data_panel import DataPanel


class Dataset:
    """
    A container for a Polars DataFrame/LazyFrame which includes and displays metadata.


    See :doc:`Basic Usage </basic-usage/objects/dataset>`
    for an introduction and some examples.

    Group
    -----
        data_prep
    """

    def __init__(
            self,
            data: pl.DataFrame | pl.LazyFrame | FrameInitTypes,
            name: str = None,
            **kwargs
    ) -> None:

        if isinstance(data, Dataset):
            data = data._data
            name = data._name
        elif not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            data = pl.LazyFrame(data, **kwargs)

        self._data: pl.DataFrame | pl.LazyFrame = data
        self._name: str = name

        self._is_modified: bool = False

        self.info: InfoList = InfoList()

        self.info._config_repr(
            tbl_dataframe_shape_below=True
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:

        # need to know whether to show that info may refer to old data
        show_info = self.info._get_config("show_info", which_config="REPR_CONFIG")

        bottom_name = None
        if show_info and self._is_modified and self._info:
            bottom_name = "[info may refer to non-modified data]"

        top_name = f"{type(self).__qualname__}"
        if self._name is not None:
            top_name = f'{top_name.center(len(self._name))}\n{self._name}'

        if self._is_modified:
            top_name += "\n[modified]"

        return self.info._repr_string(
            top_name=top_name,
            bottom_name=bottom_name,
            data_for_repr=self._data
        )

    def __copy__(self) -> Dataset:
        cls = self.__class__
        new_obj = cls.__new__(cls)

        new_obj._data = self._data
        new_obj._name = self._name
        new_obj._is_modified = self._is_modified
        new_obj.info = self.info.__copy__()

        return new_obj

    def __dir__(self) -> list:
        return sorted(set(dir(self._data) + list(super().__dir__())))

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, attr) -> Self | Any:
        """
        Delegate to Polars DataFrame or LazyFrame
        """

        if attr.startswith("_repr_"):  # or attr.startswith("_ipython_")
            raise AttributeError(f"{type(self).__name__} object has no attribute {attr}")

        # where to get the methods from, depends if the data we are operating on is DF or LF
        # should a groupby abject still be a Dataset?

        if isinstance(self._data, pl.DataFrame):
            attr_from = pl.DataFrame
        elif isinstance(self._data, pl.LazyFrame):
            attr_from = pl.LazyFrame
        elif isinstance(self._data, pl.dataframe.group_by.GroupBy):
            attr_from = pl.dataframe.group_by.GroupBy
        elif isinstance(self._data, pl.lazyframe.group_by.LazyGroupBy):
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

            # print(f"type data when {attr} is called", self.data.__class__)
            attr_value = getattr(self._data, attr)
            # print(f"attr_value for {attr}", attr_value)

            if callable(attr_value):
                # print(f"attr_value is callable", attr_value)

                def wrapper(*args, **kwargs) -> Any:
                    result = attr_value(*args, **kwargs)

                    # print(type(result))
                    # print(result)

                    # only LazyFrame or DataFrame stay Dataset, if it's a
                    # group_by it will return a DataFrame

                    if isinstance(result, (
                            pl.DataFrame,
                            pl.LazyFrame,
                            pl.dataframe.group_by.GroupBy,
                            pl.lazyframe.group_by.LazyGroupBy
                    )):

                        # print("start copying the object")
                        new_obj = self.__copy__()  # not inplace
                        # print("finished copying the object")

                        # print("new_obj", new_obj)
                        new_obj._data = result

                        if attr not in ['lazy', 'collect'] and not new_obj._is_modified:
                            new_obj._is_modified = True

                        return new_obj
                    else:
                        return result  # non dataframe output

                return wrapper
            else:
                # print("attr_value", attr_value)
                return attr_value  # return attribute (not callable) from Polars
        else:
            raise AttributeError(
                f"{type(self).__name__} "
                # f"({type(self.data).__name__}) "
                f"has no attribute {attr}"
            )

    # ------------------- combine Dataset(s) ---------------------------

    def join(
            self,
            other: Dataset | pl.DataFrame,
            on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            how: JoinStrategy = 'inner',
            *,
            left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
            suffix: str = '_right',
            validate: JoinValidation = 'm:m',
    ) -> Self:
        # todo: return self

        # overwrite polars .join()

        if isinstance(other, Dataset):
            other_data: pl.DataFrame | pl.LazyFrame = other._data
        else:
            other_data: pl.DataFrame | pl.LazyFrame = other

        data = self._data.join(
            other=other_data,
            on=on,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
            validate=validate
        )

        # todo: join self.data with other.data
        # join self.info with other.info

        new_obj = self.__copy__()

        name = self._name

        # maybe don't write this?
        if name is not None and not name.endswith("[joined]"):
            name = f"{self._name} [joined]"

        new_obj._data = data
        new_obj.name = name

        new_obj._info = []  # todo: joined info
        new_obj._is_modified = True

        # info = self.info_get
        # self._is_modified = True

        # todo: return Self (left Self)

        # data = Dataset(data=data, name=name)

        return new_obj

    def skim(
            self,
            percentiles: Sequence[float] | float | None = (0.25, 0.75),
            *,
            by: str | list[str] = None,
            include_full_sample: bool = False,  # only relevant with by:
    ) -> SkimResults:
        ...

    # ---------------------- to other objects --------------------------

    def _to_dataframe(self) -> pl.DataFrame:
        if isinstance(self._data, pl.LazyFrame):
            return self._data.collect()
        return self._data

    def _to_lazyframe(self) -> pl.LazyFrame:
        return self._data.lazy()

    def to_datapanel(
            self,
            entity_column: str = None,
            time_column: str = None,
            **kwargs
    ) -> DataPanel:

        return DataPanel(
            data=self._data,
            entity_column=entity_column,
            time_column=time_column,
            **kwargs
        )


def to_dataset(name: str = None) -> Callable:
    """Func to data.pipe(to_dataset())"""

    def _to_dataset(data: pl.DataFrame | pl.LazyFrame) -> Dataset:
        return Dataset(data=data, name=name)

    return _to_dataset

#
# cls = self.__class__
# new_obj = cls.__new__(cls)
#
# new_super_obj = super().__copy__()
# new_obj.__dict__.update(new_super_obj.__dict__)
#
# new_obj._data = self._data
# new_obj._name = self._name
#
# return new_obj
