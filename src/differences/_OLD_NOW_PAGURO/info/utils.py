from __future__ import annotations

import json
from typing import Literal, Callable

import polars as pl

from differences._OLD_NOW_PAGURO.info.info_repr._str_repr import (
    _determine_indent,
    line_with_centered_caption,
    join_strings,
    split_text_with_padding,
    format_frames,
    format_dicts_with_frames,
    determine_fmt_func
)
from differences._OLD_NOW_PAGURO.utils.utility import update_nested_dict


def preprocess_set_config(obj_config: dict, **kwargs):
    new_config = {k: v for k, v in kwargs.items() if v is not None}

    for k, v in kwargs.items():
        if k is not None and v != obj_config.get(k):
            new_config[k] = v

    return update_nested_dict(template=obj_config, user_in=new_config)


def info_dict_to_str_single(
        info: dict,
        width_chars: int = 80,  # total width of the final repr

        same_line: bool = False,
        *,

        indent: int | tuple = (0, 2, 4, 0),  # this will define the padding for each sub-block
        # all sub-blocks will have the same padding

        equal_symbol: str = ":",
        bold_keys: bool = False,

        center_frame: bool = True,  # only if not in dict

        **polars_tbl_config
) -> str:
    """
    for each key value,

    - if the value is a string process the string to fit the width_char
    - if the value is a list process the list,

    Parameters
    ----------
    info
    width_chars
    same_line
    n_columns
    indent
    equal_symbol
    bold_keys
    top_line
    bottom_line
    center_frame
    separator_when_join
    min_width_when_join

    Returns
    -------

    """
    fmt_type: Literal[1, 2] = 2 if same_line else 1

    out: str = format_single_value_of_info_dict(
        value=info,
        width_chars=width_chars,
        fmt_type=fmt_type,
        indent=indent,
        equal_symbol=equal_symbol,
        bold_keys=bold_keys,
        center_frame=center_frame,
        **polars_tbl_config,
    )

    return out


def info_dict_to_str_multiple(
        info: dict,
        width_chars: int = 80,  # total width of the final repr

        same_line: bool = False,
        n_columns: int = 1,  # for now all in the same line

        *,

        indent: int | tuple = (0, 2, 4, 0),  # this will define the padding for each sub-block
        # all sub-blocks will have the same padding

        equal_symbol: str = ":",
        bold_keys: bool = False,

        top_line: str = "-",
        bottom_line: str = None,

        center_frame: bool = True,  # only if not in dict

        separator_when_join: str = "",
        min_width_when_join: int = 0,

        **set_polars_config
) -> str:
    """
    for each key value,

    - if the value is a string process the string to fit the width_char
    - if the value is a list process the list,

    Parameters
    ----------
    info
    width_chars
    same_line
    n_columns
    indent
    equal_symbol
    bold_keys
    top_line
    bottom_line
    center_frame
    separator_when_join
    min_width_when_join

    Returns
    -------

    """
    fmt_type: Literal[1, 2] = 2 if same_line else 1

    (
        base_indent,
        ki,
        vi,
        right_padding
    ) = _determine_indent(indent=indent)

    # todo: adjust width_chars if separator_when_join
    width_chars_single = width_chars // n_columns

    out = []
    for k, v in info.items():
        single: str = format_single_value_of_info_dict(
            value=v,
            width_chars=width_chars_single,
            fmt_type=fmt_type,
            indent=indent,
            equal_symbol=equal_symbol,
            bold_keys=bold_keys,
            center_frame=center_frame,
            **set_polars_config,
        )

        if not k.startswith("_"):
            top_line_str = line_with_centered_caption(
                width_chars=width_chars_single,
                left_padding=base_indent,
                right_padding=right_padding,
                title=k,
                symbol=top_line
            )

            single = f"{top_line_str}\n{single}"

        if bottom_line:
            pass

        out.append(single)

    if n_columns > 1:
        # join the strings horizontally if more than one column is requested
        out = join_strings(
            *out,
            separator=separator_when_join,
            min_width=min_width_when_join
        )

    return "\n".join(out)


def format_single_value_of_info_dict(
        value: str | list | dict,
        width_chars: int,  # width of the single, can be total width or divided up

        fmt_type: Literal[1, 2] | Callable = 1,

        *,

        indent: int | tuple = (0, 2, 4, 0),
        # this will define the padding for each sub-block
        # all sub-blocks will have the same padding

        equal_symbol: str = ":",
        bold_keys: bool = False,

        center_frame: bool = True,  # only if not in dict
        **set_polars_config
):
    """

    Parameters
    ----------
    value
        information value
    width_chars
        total with of the table
    fmt_type
    indent
    equal_symbol
    bold_keys
    center_frame
    set_polars_config

    Returns
    -------

    """
    (
        base_indent,
        key_indent,
        value_indent,
        right_padding
    ) = _determine_indent(indent=indent)

    # value could be a string or a list
    if isinstance(value, str):
        out: list = split_text_with_padding(
            text=value,
            width_chars=width_chars,
            left_padding=base_indent,
            right_padding=right_padding
        )

        return "\n".join(out)

    if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
        if center_frame:
            indent = 0, 0, 0, 0

        return format_frames(
            data=value,
            width_chars=width_chars,
            indent=indent,
            center=center_frame,
            **set_polars_config

        )

    elif isinstance(value, dict):
        value = format_dicts_with_frames(
            data=value,
            width_chars=width_chars,
            indent=indent,
            **set_polars_config
        )

        func = determine_fmt_func(fmt_type=fmt_type, to_string=True)
        return func(
            value,
            indent=indent,
            bold_keys=bold_keys,
            equal_symbol=equal_symbol,
            width_chars=width_chars
        )

    elif isinstance(value, list):
        out = [
            format_single_value_of_info_dict(
                value=sub_item,
                width_chars=width_chars,
                fmt_type=fmt_type,

                indent=indent,

                equal_symbol=equal_symbol,
                bold_keys=bold_keys,

                center_frame=center_frame,
                **set_polars_config

            ) for sub_item in value
        ]

        # todo: raise error if list of lists
        return "\n".join(out)


class ValidationError(Exception):
    pass


def validate_input(value):
    if isinstance(value, str):
        return True

    elif isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValidationError("All elements in the list must be strings.")
        return True

    elif isinstance(value, dict):
        for key, val in value.items():

            if not isinstance(key, str):
                raise ValidationError(f"Dictionary key '{key}' is not a string.")

            if isinstance(val, (str, int, float)):
                continue

            elif isinstance(val, pl.DataFrame):
                continue

            elif isinstance(val, dict):
                # For nested dicts, values should not be dataframes
                if any(isinstance(inner_val, pl.DataFrame) for inner_val in val.values()):
                    raise ValidationError(
                        "DataFrames are only allowed at the first level of nestedness."
                    )
                validate_input(val)  # Recursive validation for deeper nested dictionaries

            else:

                raise ValidationError(f"Unsupported value type '{type(val)}'.")
        return True
    else:
        raise ValidationError(f"Unsupported input type '{type(value)}'.")


def flip_info_dict_for_frames(
        name_info_dict: dict[str, dict[str, dict[str, int | float | str]]]
) -> dict[str, dict[str, list[int | float | str]]]:
    """
    Convert a nested dictionary structure into a specific format.

    The function transforms a dictionary with nested dictionaries into
    a format where the keys of the innermost dictionaries become
    keys in the output dictionary with their values stored in lists.
    The order of values in the lists corresponds to the order of
    the keys in the input dictionaries. With extra outer keys.

    Parameters
    ----------
    name_info_dict : dict[str, dict[str, dict[str, int | float | str]]]
        The input dictionary with a nested structure.
        Example: {"table-0": {"statistics": {"N Observations": 100, "R Squared": 0.2}}}

    Returns
    -------
    dict[str, dict[str, list[int | float | str]]]
        A dictionary where the keys of the innermost dictionaries are
        now keys in the output dictionary, and their values are stored in lists.
        Example: {'statistics': {'statistics': ['N Observations', 'R Squared'], 'table-0': [100, 0.2]}}

    Raises
    ------
    TypeError
        If the input is not a dictionary, or if any of the
        nested structures are not dictionaries.

    Examples
    --------
    # >>> flip_info_dict({"table-0": {"statistics": {"N Observations": 100, "R Squared": 0.2}}})
    # {'statistics': {'statistics': ['N Observations', 'R Squared'], 'table-0': [100, 0.2]}}
    """

    # EXAMPLE
    # flip_info_dict_for_frames_new(
    #     {"table-0": {"statistics": {"N Observations": 100, "R Squared": 0.2},
    #                  "fixed_effects": {"a": "yes"}}})
    #

    if not isinstance(name_info_dict, dict):
        raise TypeError("Input must be a dictionary.")

    out = {}
    for name, info_dict in name_info_dict.items():

        if not isinstance(info_dict, dict):
            raise TypeError(f"Value associated with {name} must be a dictionary.")

        for inner_name, value in info_dict.items():

            if inner_name not in out:
                out[inner_name] = {inner_name: [], name: []}

            if isinstance(value, dict):

                for stat_name, stat_value in value.items():
                    out[inner_name][inner_name].append(stat_name)
                    out[inner_name][name].append(stat_value)

            else:
                out[inner_name][inner_name].append(inner_name)

                # str value? if not int/float/str?
                if isinstance(value, (int, float, str)):
                    value = str(value)

                out[inner_name][name].append(value)

    return out


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def rename_dict_keys(original_dict: dict, mapping: dict[str, str]):
    # check for conflicts:
    # if a new key is in the original keys and not meant to be renamed
    for key, new_key in mapping.items():
        if new_key in original_dict and key != new_key and new_key not in mapping:
            raise ValueError(
                f"trying to rename the dictionary key '{key}' to '{new_key}'\n"
                f"but '{new_key}' is already a key in the dictionary "
                f"(and is not being renamed)"
            )

    # create a temporary dictionary to store the new key-value pairs
    out: dict = {}

    for key, value in original_dict.items():
        if key in mapping:
            # rename key if it's in the key_mapping
            new_key = mapping[key]
            out[new_key] = value
        elif key not in mapping.values():
            # keep the original key-value pair if the key is not supposed to be renamed
            out[key] = value

    return out
