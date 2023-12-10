# todo: metadata columns:
#   - self.info_fmt, add an option to save the name and the inner
#   name where the columns are stored
#   - self.info_update should work without having to specify name and
#   inner name for the metadata
#       storing the columns information, using the stored location in self.info_fmt
#   - self._auto_update_columns [NO NEED just use info_update],
#       calls self.info_update to auto update columns:
#       - if rename is called
#       - at the end of __getattr__ call autoupdate. If the columns are
#       not there anymore delete
#           from metadata
#       - if join is called? overwrite join?
#       - are there other methods?

# enforce_columns is used only if info is a dictionary containing column information
# where the keys of the dictionary are the column names. The user may want to
# update column information. If enforce_columns is set to False then all the columns
# that are in info but are not in data.columns will be dropped. If enforce_columns
# is set to True the info

# new_obj = self.__copy__()

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Literal

import polars as pl

from differences._OLD_NOW_PAGURO.info.info_repr._str_repr import (
    line_with_centered_caption
)

from differences._OLD_NOW_PAGURO.info.utils import (
    info_dict_to_str_multiple,
    flip_info_dict_for_frames,
    is_json_serializable,
    preprocess_set_config, rename_dict_keys, info_dict_to_str_single
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


# Info contains a single set of info: name must always be provided.
# self.indo
class Info:
    """
    Container for information or metadata, providing a structured way to store, update,
    and represent data in a readable format.

    Group
    -----
        info
    """

    def __init__(
            self,
            name: str,
    ):
        """
        Initializes the Info class.

        Parameters
        ----------
        name : str
            The top-level name for the information.
            - Example: If the information contains descriptions (column, data, etc.), the name could be "Descriptions".
            - Note: If the name is preceded by an underscore ("_"), it will not be used in the string representation,
              but it will remain a key to access the information.
        width_chars : int, optional, default=80
            The total width of the final string representation.
        """

        self.info: dict = {}
        self.name = name  # if the name starts with _ it will not be displayed as a title

        self._is_serializable: bool = True
        self._is_column_info: bool = False

        self._config: dict[str, dict] = {
            "REPR_CONFIG": dict(),
            "INFO_CONFIG": dict(),
            "POLARS_CONFIG": dict()
        }

        self.config_repr()  # populate self._config["REPR_CONFIG"]

        # self._overall_left_padding = 0  # left padding for the entire info block
        # self._overall_right_padding = 0  # right padding for the entire info block

    def _get_config(
            self,
            key: str,
            which_config: Literal["REPR_CONFIG", "INFO_CONFIG", "POLARS_CONFIG"]
    ):
        return self._config[which_config].get(key, {})

    def _set_config(
            self,
            which_config: Literal["REPR_CONFIG", "INFO_CONFIG", "POLARS_CONFIG"],
            **kwargs
    ) -> None:  # inplace operation
        self._config[which_config] = preprocess_set_config(
            obj_config=self._config[which_config], **kwargs
        )

    def config_info(
            self,
            allow_non_serializable: bool = False,
            data_column_info: bool | str = False,
    ) -> Self:
        self._set_config(
            which_config="REPR_CONFIG",
            allow_non_serializable=allow_non_serializable,
        )
        return self

    def config_repr(
            self,
            width_chars: int = 80,

            n_levels: Literal[1, 2] = 1,

            same_line: bool = False,
            n_columns: int = 1,

            indent: int | tuple = (2, 2, 4, 2),

            equal_symbol: str = ":",

            top_lines: tuple[str] | str = ("─", "-"),
            bottom_line: str = None,

            center_frame: bool = True,

            separator_when_join: str = "",
            min_width_when_join: int = 0,

            **polars_tbl_config,
    ) -> Self:
        """
        Sets configuration settings for the string representation.

        Parameters
        ----------
        width_chars : int, default=80
        n_levels : Literal[1, 2]
        same_line : bool, optional, default=False
            Determines the format type.
        n_columns : int, optional, default=1
            Determines the number of columns in the string representation.
            for now all in the same line
        indent : int or tuple, optional, default=(2, 2, 4, 2)
            Defines the padding as (base left, key left, value left, right).
            This will define the padding for each sub-block
            All sub-blocks will have the same padding
        equal_symbol : str, optional, default=":"
            The symbol used to separate keys from values.
        top_lines : tuple of str or str, optional, default=("─", "-")
            Defines the symbols used for top lines in the string representation.
        bottom_line : str, optional
            Defines the symbol used for the bottom line in the string representation.
        center_frame : bool, optional, default=True
            Determines if the frame should be centered (only if not in dict format).
        separator_when_join : str, optional, default=""
            The separator used when joining.
        min_width_when_join : int, optional, default=0
            The minimum width when joining.
        """

        self._set_config(
            which_config="REPR_CONFIG",

            width_chars=width_chars,

            n_levels=n_levels,
            same_line=same_line,

            n_columns=n_columns,
            indent=indent,
            equal_symbol=equal_symbol,
            bold_keys=False,
            top_lines=top_lines,
            bottom_line=bottom_line,
            center_frame=center_frame,
            separator_when_join=separator_when_join,
            min_width_when_join=min_width_when_join,

        )

        self._set_config(
            which_config="POLARS_CONFIG",
            **polars_tbl_config
        )

        return self

    def __repr__(self) -> str:
        # Returns the string representation of the object.
        return f"<{self.__class__.__qualname__}(name='{self.name}') at {hex(id(self))}>"

    def __str__(self) -> str:
        # Returns the string representation of the object.
        return self.to_str()

    def __deepcopy__(self) -> Self:
        """
        Polars DataFrames are cheap to deepcopy

        polars clone: https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.clone.html#polars.DataFrame.clone
        polars deepcopy: https://github.com/pola-rs/polars/blob/main/py-polars/polars/dataframe/frame.py#L1795
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.__dict__ = deepcopy(self.__dict__)
        return new_obj

    def __copy__(self) -> Self:
        return self.__deepcopy__()

    def __getitem__(self, key: str):
        return self.info[key]

    def update(self, **info) -> Self:
        # if the name/title starts with _ it will not be displayed as a title

        allow_non_serializable = self._config.get('allow_non_serializable', False)

        for key, value in info.items():
            if not is_json_serializable(value):

                if not allow_non_serializable:
                    raise ValueError(f"Value for key '{key}' is not JSON serializable")
                else:
                    self._is_serializable = False

            self.info[key] = value

        return self

    def rename(self, name: str) -> Self:
        # inplace, maybe should not be?
        self.name = name
        return self

    def rename_keys(self, mapping: dict[str, str]) -> Self:
        # inplace, maybe should not be?
        self.info = rename_dict_keys(original_dict=self.info, mapping=mapping)
        return self

    # -------------------------- data columns --------------------------

    def _set_data_columns(self, columns: list[str]) -> Self:
        self._is_column_info = True
        self._columns = columns
        # todo: check that the keys are columns, drop missing columns
        # todo: check that it is a serializable mapping

        # self.info: dict = {}
        return self

    def _update_data_columns(self, columns: list[str], validate: bool = False) -> Self:

        self._columns = columns
        # todo

        # self.info: dict = {}

        return self

    # ---------------------- export ------------------------------------
    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the information.

        Returns
        -------
        dict
            A dictionary with a single key-value pair,
            where the key is the `name` attribute and the value is the `info` attribute.
        """
        return {self.name: self.info}

    def to_frames(self) -> dict[str, pl.DataFrame]:
        # useful for table. First column is the inner name
        # (which in most cases should be the fine categorization of information)
        # for example: statistics, fixed_effects, ...
        flipped: dict = flip_info_dict_for_frames(name_info_dict=self.to_dict())

        out = {}
        for k, v in flipped.items():
            out[k] = pl.DataFrame(v)

        return out

    def to_str(self) -> str:
        """
        Converts the information stored in the `info` attribute
        to a formatted string representation.

        Returns
        -------
        str
            The formatted string representation of the information.
        """

        top_lines: tuple[str, str] | str = self._config["REPR_CONFIG"].get("top_lines", "")
        n_levels: int = self._config["REPR_CONFIG"].get("n_levels", 1)

        if not isinstance(top_lines, tuple):
            top_lines = (top_lines, top_lines)

        title_line, sub_title_line = top_lines

        if n_levels == 1:
            # string representation without subtitle

            config_repr = {
                k: v
                for k, v in self._config["REPR_CONFIG"].items()
                if k in [
                    "width_chars",
                    "same_line",
                    "indent",
                    "equal_symbol",
                    "bold_keys",
                    "center_frame"
                ]
            }

            out: str = info_dict_to_str_single(
                info=self.info,
                **config_repr,  # there are extra args here that end up in the polars
                **self._config["POLARS_CONFIG"]
            )

        else:  # n_levels = 2

            # string representation with as a subtitle: ------- key ---------
            # if the key starts with _ the sub-title will not be displayed

            config_repr = {
                k: v
                for k, v in self._config["REPR_CONFIG"].items()
                if k not in ["top_lines", "n_levels"]
            }

            out: str = info_dict_to_str_multiple(
                info=self.info,
                top_line=sub_title_line,

                **config_repr,

                **self._config["POLARS_CONFIG"]
            )

        if not self.name.startswith("_"):
            # outer title

            name_str = line_with_centered_caption(
                width_chars=self._config["REPR_CONFIG"].get("width_chars"),
                left_padding=1,
                right_padding=1,
                title=self.name,
                symbol=title_line
            )
            out = f"{name_str}\n{out}"  # append line on top

        return out
