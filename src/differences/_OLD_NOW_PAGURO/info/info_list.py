from __future__ import annotations

import sys
from copy import deepcopy
from typing import Literal, Mapping, Iterable, Union

import polars as pl

from differences._OLD_NOW_PAGURO.info.info import Info
from differences._OLD_NOW_PAGURO.info.info_repr._str_boxed import boxed_str
from differences._OLD_NOW_PAGURO.info.utils import preprocess_set_config

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

InfoMappingTypes = dict[str, Union[dict, float, int, str, list, tuple]]
InfoTypes = Union[Info, Iterable[Info], InfoMappingTypes]

DataReprTypes = Union[
    pl.DataFrame, dict[str, pl.DataFrame], dict[str, dict[str, pl.DataFrame]]
]


class InfoList:
    """
    A container for a list with multiple Info + string representation utility methods
    """

    def __init__(self, info: InfoTypes = None):

        if info is None:
            self.info: list[Info] = list()
        elif isinstance(info, Info):
            self.info: list[Info] = [info.__copy__()]
        elif isinstance(info, Iterable):
            if not all(isinstance(i, Info) for i in info):
                raise TypeError("'info' must contain only Info")
            self.info: list[Info] = [i.__copy__() for i in info]
        else:
            TypeError("'info' must be either Info or Iterable[Info]")

        self._config: dict[str, dict] = {
            "REPR_CONFIG": dict()
        }

        self._config_repr()  # populate _config["REPR_CONFIG"]

    def __repr__(self) -> str:
        return self.info.__repr__()

    def __str__(self) -> str:
        # need to overwrite this method in a sub-class if you want to display main data
        return self._repr_string()

    def __copy__(self) -> Self:
        # problem with:

        # type(self)(data=self.data, name=self._name)
        # new_obj.__dict__.update(**self._copy_dict_no_data())

        # if data is not a dataframe (grouby obj) this gets stuck.

        cls = self.__class__
        new_obj = cls.__new__(cls)

        new_obj.info = [i.__copy__() for i in self.info]
        new_obj._config = deepcopy(self._config)

        return new_obj

    def __iter__(self):
        return iter(self.info)

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, key: int | str) -> Info:
        if isinstance(key, str):
            key = [i.name for i in self.info].index(key)

        return self.info[key]

    def __setitem__(self, key: int, value: Info) -> None:
        if not isinstance(value, Info):
            raise TypeError("Assigned value must be an Info object")

        self.info[key] = value

    def __contains__(self, name: str):
        return name in [i.name for i in self.info]

    def __delitem__(self, key: int | str):
        del self.info[key]

    def __add__(self, other):
        if isinstance(other, InfoList):
            return InfoList(self.info + other.info)
        elif isinstance(other, Iterable) and all(isinstance(item, Info) for item in other):
            return InfoList(self.info + list(other))
        else:
            raise TypeError("Can only concatenate with InfoList or iterable of Info objects")

    def __iadd__(self, other):
        if isinstance(other, InfoList):
            self.info += other.info
        elif isinstance(other, Iterable) and all(isinstance(item, Info) for item in other):
            self.info += list(other)
        else:
            raise TypeError("Can only concatenate with InfoList or iterable of Info objects")
        return self

    def __eq__(self, other):
        # TODO: check that is works as intended
        if isinstance(other, InfoList):
            return self.info == other.info
        return False

    # ------------------------------------------------------------------

    def _get_config(
            self,
            key: str,
            which_config: Literal["REPR_CONFIG"] = "REPR_CONFIG"
    ):
        return self._config[which_config].get(key, {})

    def _set_config(
            self,
            which_config: Literal["REPR_CONFIG"] = "REPR_CONFIG",
            **kwargs
    ) -> None:  # inplace operation
        self._config[which_config] = preprocess_set_config(
            obj_config=self._config, **kwargs
        )

    def _config_repr(
            self,
            width_chars: Literal["auto"] | int = 80,
            show_info: str | list[str] | bool = True,
            top_info: str | list[str] = None,
            **polars_tbl_config
    ) -> None:
        """
        Configuration of how to display the metadata/info representation.

        Parameters
        ----------
        show_info : Union[str, List[str], bool], default True
            Specify which metadata to show. If True, show all. If False, show none.
            If str or List[str], show only specified metadata.
        width_chars : int, default 80
            The width of the representation in characters.
        top_info
            The names (outer names) of the information to be displayed at the top
        **polars_tbl_config
            Additional configuration settings for Polars.
        """
        # todo: add set_column_info to store the outer and inner key that point to col info
        # not here: in _info_set_config

        # inplace operation
        self._set_config(
            which_config="REPR_CONFIG",
            show_info=show_info,
            width_chars=width_chars,
            top_info=top_info,

            polars_tbl_config=polars_tbl_config
        )

    # ------------------------------------------------------------------

    def _repr_string(
            self,
            top_name: str = None,
            bottom_name: str = None,
            data_for_repr: DataReprTypes = None,
    ) -> str:
        """
        Helper function to create a formatted string representation.

        Parameters
        ----------
        top_name : str, optional
            Name to be displayed at the top of the representation.
        bottom_name : str, optional
            Name to be displayed at the bottom of the representation.
        data_for_repr
        """

        obj_repr = self._repr_info(
            top_name=top_name,
            bottom_name=bottom_name,
            data_for_repr=data_for_repr
        )

        obj_repr = f"{self.__class__.__qualname__} [at {hex(id(self))}]\n{obj_repr}"

        return obj_repr

    def _repr_data(
            self,
            data_for_repr: DataReprTypes | None,
            width_chars: int,
    ) -> Info | None:
        """
        Helper function to create a formatted string representation of the data.

        Parameters
        ----------
        data_for_repr
        width_chars : int
            The width of the representation in characters.
        """
        # additional way to set polars config?

        if data_for_repr is None:
            return None

        info = Info(name="_DATA_INFO")

        polars_tbl_config = self._get_config(
            key="polars_tbl_config", which_config="REPR_CONFIG"
        )

        # pass the configuration down to Info
        info.config_repr(width_chars=width_chars, **polars_tbl_config)

        if isinstance(data_for_repr, (pl.DataFrame, pl.LazyFrame)):
            info.update(_data=data_for_repr)  # sub-dict has key _data.
            # info.update_info(_data={"data": self.data}) # to display as an "index"/"dict"

        elif isinstance(data_for_repr, dict):  # could be _data={"data": self.data}
            info.update(**data_for_repr)  # sub-dict has key _data.

        else:
            raise TypeError("'data_for_repr' must be either a DataFrame, a LazyFrame or a dict")

        return info

    def _repr_info(
            self,
            top_name: str = None,
            bottom_name: str = None,
            data_for_repr: DataReprTypes = None,
    ) -> str:
        """
        Helper function to create a formatted string representation of the metadata.

        Parameters
        ----------
        top_name : str, optional
            Name to be displayed at the top of the representation.
        bottom_name : str, optional
            Name to be displayed at the bottom of the representation.
        data_for_repr

        Returns
        -------
        str
            A formatted string representation of the metadata.
        """
        if top_name is None:
            top_name = f"{type(self).__qualname__}"  # always use the name of the class as top name
            name = getattr(self, "_name", None)
            if name is not None:
                if isinstance(name, str):
                    # center to create space around the class name if needed
                    top_name = f'{top_name.center(len(name))}\n{name}'

        # --------------------------------------------------------------

        width_chars = self._get_config("width_chars", which_config="REPR_CONFIG")

        # if width_chars == "auto":
        #     width_chars = max(len(i) for i in self.data.__repr__().split("\n")) + 10

        top_info = self._get_config("top_info", which_config="REPR_CONFIG")

        margin = 2

        # --------------------------------------------------------------

        data_for_repr: Info | None = self._repr_data(
            data_for_repr=data_for_repr,
            width_chars=width_chars - margin
        )

        out = concat_info_to_str(
            data_for_repr,
            *self._filter_show_info(),
            width_chars=width_chars,
            top_name=top_name,
            bottom_name=bottom_name,
            top_info=top_info,
            margin=margin
        )

        return out

    # ------------------------------------------------------------------

    def _filter_show_info(self) -> list[Info]:
        show_info = self._get_config("show_info", which_config="REPR_CONFIG")

        if show_info:
            if isinstance(show_info, bool):
                return [i for i in self.info]  # don't return self.info so can modify list

            if isinstance(show_info, str):
                show_info = [show_info]

            return [i for i in self.info if i.name in show_info]
        return []

    def insert(self, index: int, item: Info):
        if not isinstance(item, Info):
            raise TypeError("Inserted item must be an Info object")
        self.info.insert(index, item)

    def extend(self, iterable: Iterable[Info]):
        if not all(isinstance(item, Info) for item in iterable):
            raise TypeError("All items in the iterable must be Info objects")
        self.info.extend(iterable)

    def clear(self):
        self.info.clear()

    def copy(self):
        return self.__copy__()

    def append(
            self,
            *info: Info,
            **mapping: InfoMappingTypes
    ) -> Self:
        """
        Adds metadata to the self._info list

        This is a private method because each subclass may need to
        set up the information names differently.

        For example Table enforces the outer name of the information
        to be the table-name because concatenating multiple Tables keeps
        track of the information for each table. On the other hand the
        information for a Dataset is fixed to the original Dataset,
        thus the outer name can be set by the user at their preference.

        Each subclass may (or may not) need an info_add method.

        Parameters
        ----------
        *info : Info
            Variable length Info objects.
        **mapping : dict[str, dict | str | list]
            Additional metadata in dictionary form.
        """
        self._append_from_info(*info)
        self._append_from_mapping(**mapping)
        return self

    def _append_from_info(self, *info):
        for i in info:
            if not isinstance(i, Info):
                raise TypeError("info must be of type Info")

            if i.name in self:
                raise ValueError(f"'{i.name}' has already been used as a name for an Info")

            self.info.append(i.__copy__())

    def _append_from_mapping(self, **mapping: dict[str, dict | float | int | str | list]):
        self.extend(mapping_to_info(**mapping))

    def get(self, name: str) -> Info:
        return self[name]

    def filter(self, name: str | Iterable[str]) -> Self:
        if isinstance(name, str):
            name = [name]

        cls = self.__class__
        new_obj = cls.__new__(cls)

        new_obj.info = [i.__copy__() for i in self.info if i.name in name]
        new_obj._config = deepcopy(self._config)

        return new_obj

    # ------------------------------------------------------------------


# ------------------------- helpers ------------------------------------

def mapping_to_info(info_list: InfoList, **mapping) -> list[Info]:
    out = []
    for k, v in mapping.items():
        if not isinstance(v, Mapping):
            raise TypeError(
                f"argument, when adding info using kwargs, must be a "
                f"Mapping and not a {type(v).__name__}. "
                f"Alternatively use an Info object\n\n"
                f"for example, info_add({k}={{'info': '{v}'}})"
            )
        if k in info_list:
            raise ValueError(f"'{k}' has already been used as a name for an Info")

        info = Info(name=k).update(**v)
        out.append(info)

    return out


# def concat_info_to_frames(*info: Info):
#     """
#     Creates a dictionary in which he key is the inner_name of and the value
#     is a joined frame where the columns correspond to the information displayed
#     (for each Info). So for example if the information is "statistics" with a dictionary
#     of
#
#     this function is mainly relevant for Table
#     in which the information contained in Table may contain info
#     for multiple Tables (when Tables are concatenated)
#
#     This is ok only if a table correspond to a single "model",
#     sometimes a single table may contain estimates for multiple "models"
#     for example if we created a table that has group_columns.
#
#     In that case the group_columns need to become part of the information and
#     the Table needs to be split up in multiple sub tables and then concatenated again
#     into a single Table (but created from the concatenation)
#
#     Returns
#     -------
#
#     """
#
#     pass
def concat_info(
        *info: Info,
        width_chars: int,
        top_info: str | list[str] | None  # info to be placed at the top
) -> list[Info]:
    if isinstance(top_info, str):
        top_info = [top_info]

    top = []
    bottom = []

    for i in info:

        if not isinstance(i, Info):  # in case None or something else
            continue

        i.config_repr(width_chars=width_chars)

        if top_info:
            if i.name in top_info:
                top.append(i)

        bottom.append(i)

    return top + bottom


def concat_info_to_str(
        *info: Info,
        width_chars: int,
        top_name: str | None,
        bottom_name: str | None,
        top_info: str | list[str] | None,  # info to be placed at the top
        margin: int = 0
) -> str:
    width_chars_margin = width_chars - margin

    out = "\n".join(
        [
            i.to_str()
            for i in concat_info(
            *info, width_chars=width_chars_margin, top_info=top_info)
        ]
    )

    out = boxed_str(
        input_str=out,
        top_name=top_name,
        bottom_name=bottom_name,

        horizontal_line="thin",
        vertical_line="thin",
        corner="thin-curved",

        include_top_line=True,
        include_bottom_line=True,
        fixed_width=width_chars,
        # increment_auto_width=2,
    )

    return out
