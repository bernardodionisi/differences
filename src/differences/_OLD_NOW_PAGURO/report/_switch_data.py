from __future__ import annotations

from typing import Any, Iterable

import polars as pl

from differences._OLD_NOW_PAGURO.info import Info

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class _DataInfoSwitch:
    """
    Switch between, display, and operate simultaneously on multiple Polars DataFrames (SMALL DFs).

    IMPLEMENTS a switching __getattr__

    DataPanel should NOT inherit from _DataInfoSwitch.
    """

    def __init__(self, data_dict: dict[str, pl.DataFrame]):
        super().__init__()

        self._data_dict = data_dict
        self._operating_on: str | None = None

    def __dir__(self) -> list:
        # no GroupBy
        return sorted(set(dir(pl.DataFrame) + list(super().__dir__())))

    def __copy__(self) -> Self:
        """
        Create a shallow copy of the DataInfo instance.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)

        new_obj._data_dict = {k: v for k, v in self._data_dict.items()}

        # # todo: copy info?
        new_obj.__dict__.update({
            k: v for k, v in self.__dict__.items() if k not in ["_data_dict"]}
        )

        return new_obj

    def __getattr__(self, attr) -> Self | Any:
        """
        Delegate to Polars DataFrame or LazyFrame
        """
        if attr.startswith("_repr_"):
            raise AttributeError(f"{type(self).__name__} object has no attribute {attr}")

        if hasattr(pl.DataFrame, attr):
            new_obj = self.__copy__()
            results = {}

            if new_obj._operating_on is None:
                operating_on_keys = list(new_obj._data_dict.keys())
            else:
                operating_on_keys = [new_obj._operating_on]

            def wrapper(*args, **kwargs) -> Self | Any:

                non_dataframe_results = {}

                for key in operating_on_keys:

                    data = new_obj._data_dict[key]

                    if not isinstance(data, pl.DataFrame):
                        # no GroupBy

                        raise ValueError(
                            f"{key} should be a Polars DataFrame, not {type(data).__name__}.")

                    method_or_property = getattr(data, attr)

                    if callable(method_or_property):
                        result = method_or_property(*args, **kwargs)

                        if isinstance(result, pl.DataFrame):
                            new_obj._data_dict[key] = result
                        else:
                            non_dataframe_results[key] = result
                    else:
                        non_dataframe_results[key] = method_or_property

                if non_dataframe_results:
                    keys_list = list(non_dataframe_results)
                    if len(keys_list) == 1:
                        return non_dataframe_results[keys_list[0]]
                    return non_dataframe_results
                else:
                    return new_obj

            # If the attribute is a method that will be called later,
            # return the wrapper function.
            # If the attribute is a property, the wrapper is not needed
            # and non-callable results are returned directly.
            if callable(getattr(pl.DataFrame, attr)):
                return wrapper
            else:
                # Directly return the property values from the DataFrames
                for key in operating_on_keys:
                    data = new_obj._data_dict[key]
                    results[key] = getattr(data, attr)

                if len(operating_on_keys) == 1:
                    return results[operating_on_keys[0]]
                else:
                    return results
        else:
            raise AttributeError(f"{type(self).__name__} object has no attribute {attr}")

    def __getitem__(self, key) -> pl.Series | dict[pl.Series]:
        """Get column (Series) from the dataframe we are currently operating on"""
        if self._operating_on is None:
            return {k: v[key] for k, v in self._data_dict.items()}
        else:
            return self._data_dict[self._operating_on][key]

    def __str__(self) -> str:
        operating_on = self._operating_on
        if operating_on is not None:
            operating_on = f"operating on '{operating_on}' [pl.DataFrame]"

        return self._repr_string(bottom_name=operating_on)

    def _repr_data(
            self,
            width_chars: int,
            # additional way to set polars config?
    ) -> Info:
        """
         Helper function to create a formatted string representation of the data.

         Parameters
         ----------
         width_chars : int
             The width of the representation in characters.

         Returns
         -------
         Info
         """

        info = Info(name="_DATA_INFO", width_chars=width_chars)

        info.update(**self._data_dict)  # sub-dict has key _data.
        # info.update_info(_data={"data": self.data}) # to display as an "index"/"dict"

        info.config_polars_repr(
            **self._config.get("set_polars_config", {})  # careful with the config strings
        )

        return info

    def _operate_on(self, key: str | None) -> Self:

        if key is not None and key not in self._data_dict.keys():
            raise ValueError(f"'on' must be one of {self._data_dict.keys()}")
        new_obj = self.__copy__()
        new_obj._operating_on = key
        return new_obj

    def _to_dataframe(self, concatenate: bool | Iterable[str] = False) -> pl.DataFrame:

        # todo: add a separator row --- + add a row with the name of the
        if concatenate or self._operating_on is None:
            if isinstance(concatenate, Iterable):
                return pl.concat(
                    [v for k, v in self._data_dict.items() if k in concatenate],
                    how="diagonal_relaxed"
                )
            return pl.concat(self._data_dict.values(), how="diagonal_relaxed")

        return self._data_dict[self._operating_on]

    def _add_row(self, values: Iterable, index: int = 0):

        data = self._data_dict[self._operating_on]

        if index < 0:
            index = range(len(data))[index]
        pass
