from __future__ import annotations

from typing import Literal, Callable, Any
import textwrap

import polars as pl


def determine_fmt_func(
        fmt_type: Literal[1, 2] | Callable = 1,
        to_string: bool = True
) -> Callable:
    if fmt_type == 1:
        func = format_dict_v1_to_str if to_string else _format_dict_v1_to_list
    elif fmt_type == 2:
        func = format_dict_v2_to_str if to_string else _format_dict_v2_to_list
    elif isinstance(fmt_type, Callable):
        func = fmt_type
    else:
        raise ValueError("no such version")

    return func


# ---------------------------- version 1 -------------------------------

def format_dict_v1_to_str(
        data: dict,
        width_chars: int = 80,
        *,
        indent: int | tuple = (0, 2, 4),
        bold_keys: bool = False,
        equal_symbol: str = ":",
        **set_polars_config
):
    out = _format_dict_v1_to_list(
        data=data,
        width_chars=width_chars,
        bold_keys=bold_keys,
        indent=indent,
        equal_symbol=equal_symbol,
        **set_polars_config
    )
    return "\n".join(out)


def _format_dict_v1_to_list(
        data: dict,
        width_chars: int,
        *,
        bold_keys: bool,
        indent: int | tuple,
        equal_symbol: str,
        _current_level: int = 0,
        **set_polars_config
) -> list[str]:
    (
        base_indent,
        key_indent,
        value_indent,
        right_padding
    ) = _determine_indent(indent=indent)

    # ------------------------------------------------------------------

    data = format_dicts_with_frames(
        data=data,
        width_chars=width_chars,
        indent=(
            base_indent,
            key_indent,
            value_indent,
            right_padding  # no right padding for frames?
        ),
        **set_polars_config
    )

    # ------------------------------------------------------------------

    bold_start = "\033[1m"
    bold_end = "\033[0m"

    left_padding_str = ' ' * base_indent + ' ' * (key_indent * _current_level)

    lines = []
    for key, value in data.items():
        key_str = (
            f"{bold_start if bold_keys else ''}"
            f"{left_padding_str}{key}{equal_symbol}{bold_end if bold_keys else ''}"
        )
        if isinstance(value, dict):
            lines.append(key_str)

            sub_lines = _format_dict_v1_to_list(
                data=value,
                width_chars=width_chars,
                bold_keys=bold_keys,
                indent=indent,
                equal_symbol=equal_symbol,
                _current_level=_current_level + 1
            )

            lines.extend(sub_lines)

        else:
            lines.append(key_str)

            value_str = str(value)

            value_lines = format_list_of_strings(
                data=value_str,
                width_chars=width_chars,
                left_padding=len(left_padding_str) + value_indent,
                right_padding=right_padding
            )
            lines.extend(value_lines)

    return lines


# --------------------------- version 2 --------------------------------

def format_dict_v2_to_str(
        data: dict,
        width_chars: int = 80,
        *,
        indent: int | tuple = (0, 2, 4),
        bold_keys: bool = False,
        equal_symbol: str = ":",

        **set_polars_config
) -> str:
    out = _format_dict_v2_to_list(
        data=data,
        width_chars=width_chars,
        bold_keys=bold_keys,
        indent=indent,
        equal_symbol=equal_symbol,
        **set_polars_config
    )
    return "\n".join(out)


def _format_dict_v2_to_list(
        data: dict,
        width_chars: int,
        *,
        indent: int | tuple,
        equal_symbol: str,
        bold_keys: bool,
        _current_level: int = 0,
        **set_polars_config
) -> list[str]:
    (
        base_indent,
        key_indent,
        value_indent,
        right_padding
    ) = _determine_indent(indent=indent)

    # ------------------------------------------------------------------

    data = format_dicts_with_frames(
        data=data,
        width_chars=width_chars,
        indent=(
            base_indent * 2,
            key_indent,
            value_indent,
            right_padding  # no right padding for frames?
        ),
        **set_polars_config
    )

    # ------------------------------------------------------------------

    left_padding_str = ' ' * base_indent + ' ' * (key_indent * _current_level)

    bold_start = "\033[1m"
    bold_end = "\033[0m"

    lines = []
    for key, value in data.items():
        key_str = (
            f"{bold_start if bold_keys else ''}"
            f"{key}{equal_symbol}{bold_end if bold_keys else ''}"
        )

        if isinstance(value, dict):

            lines.append(left_padding_str + key_str)

            sub_lines = _format_dict_v2_to_list(
                data=value,
                width_chars=width_chars,
                bold_keys=bold_keys,
                indent=indent,
                equal_symbol=equal_symbol,
                _current_level=_current_level + 1
            )

            lines.extend(sub_lines)
        else:
            value_str = str(value)

            total_padding = (
                    width_chars -
                    len(left_padding_str) -
                    len(key_str) -
                    len(value_str) -
                    right_padding
            )

            line = (
                    f"{left_padding_str}{key_str}"
                    f"{' ' * total_padding}{value_str}"
                    + " " * right_padding
            )

            if len(line) <= width_chars:
                lines.append(line)

            else:
                # If it doesn't fit, then fall back to putting value on new line

                lines.append(left_padding_str + key_str)

                value_lines = format_list_of_strings(
                    data=value_str,
                    width_chars=width_chars,
                    left_padding=len(left_padding_str) + value_indent,
                    right_padding=right_padding
                )

                lines.extend(value_lines)

    return lines


# ----------------------------------------------------------------------

def format_list_of_strings(
        data: str | list[str],
        width_chars: int,
        left_padding: int,
        right_padding: int = 0,
        to_string: bool = False,
):
    """
    Formats a string or list of strings with specified padding on each line.

    The function processes each line in the input string or list of strings,
    adjusts its width according to the specified character count, and applies
    the desired left and right padding. The result can either be returned as
    a list of strings or a single concatenated string.

    Parameters
    ----------
    data : str or list[str]
        The string or list of strings to be formatted.

    width_chars : int
        The total width of each resulting line, in characters.

    left_padding : int
        The number of space characters to be added to the left side of each line.

    right_padding : int, optional
        The number of space characters to be added to the right side of each line.
        Default is 0.

    to_string : bool, optional
        If True, the result will be a single concatenated string.
        If False, the result will be a list of strings. Default is False.

    Returns
    -------
    str or list[str]
        A formatted string or list of strings with the desired width and padding.

    Notes
    -----
    The function uses the `split_text_with_padding` utility internally to handle
    the splitting and padding of each line. If the input `data` is a list of strings,
    it first concatenates the list with newline characters to process the data uniformly.
    """
    if isinstance(data, list):
        # joining because the strings in the list may also have new lines
        data = "\n".join(data)

    out = []
    for val_line in data.split('\n'):
        value_lines = split_text_with_padding(
            text=val_line,
            width_chars=width_chars,
            left_padding=left_padding,
            right_padding=right_padding
        )
        out.extend(value_lines)

    if to_string:
        return "\n".join(out)

    return out


# -------------------------- format dataframe --------------------------

def format_dicts_with_frames(
        data: dict[str, pl.DataFrame | pl.LazyFrame | Any],
        width_chars: int,
        *,
        indent: int | tuple,
        **set_polars_config
) -> dict:
    base_indent, _, value_indent, right_padding = _determine_indent(indent=indent)

    # data should just be at first level of nestedness.

    width_dataframe = width_chars - (base_indent + value_indent + right_padding)

    out = {}
    for key, value in data.items():
        if isinstance(value, pl.DataFrame):
            out[key] = _format_dataframe(
                data=value,
                tbl_width_chars=width_dataframe,
                **set_polars_config
            )
        elif isinstance(value, pl.LazyFrame):
            out[key] = value.__repr__()
        else:
            out[key] = value

    return out


def format_frames(  # not dict of frames
        data: list[pl.DataFrame, pl.LazyFrame] | pl.DataFrame | pl.LazyFrame,
        width_chars: int,
        *,
        indent: int | tuple,
        center: bool = True,
        **set_polars_config
) -> str:
    # todo: indent here is used only to find the width, it does not actually indent
    base_indent, _, value_indent, right_padding = _determine_indent(indent=indent)

    # data should just be at first level of nestedness.

    width_dataframe = width_chars - (base_indent + value_indent + right_padding)

    if not isinstance(data, list):
        data = [data]  # careful here !!

    out = []
    for d in data:
        if isinstance(d, pl.DataFrame):

            data_repr_ = _format_dataframe(
                data=d,
                tbl_width_chars=width_dataframe,
                **set_polars_config
            )

        elif isinstance(d, pl.LazyFrame):
            data_repr_ = d.__repr__()

            data_repr_ = format_list_of_strings(
                data=data_repr_,
                width_chars=width_chars,
                left_padding=base_indent,
                right_padding=right_padding,
                to_string=True
            )

            data_repr_ = f"\n{data_repr_}"

        else:
            data_repr_ = d.__repr__()  # careful??

        data_repr_ = f"{data_repr_}\n"

        if center:
            data_repr_ = "\n".join([
                i.strip().center(width_chars) for i in data_repr_.split("\n")
            ])

        out.append(data_repr_)

    # -------------------------- pad frame -----------------------------

    lines = []

    for i in out:
        line = format_list_of_strings(
            data=i,
            width_chars=width_chars,
            left_padding=base_indent,
            right_padding=right_padding
        )
        lines.extend(line)

    return "\n".join(lines)


def _format_dataframe(
        data: pl.DataFrame,
        tbl_width_chars: int = None,
        **set_polars_config
) -> str:
    with pl.Config(
            **set_polars_config
    ):

        if tbl_width_chars is None:  # if no pre-specified width
            return data.__repr__()

        # if the width has been specified, but there are many displayed columns (from tbl_cols):
        # polars will still display all the columns (maybe this will change)

        with pl.Config(
                tbl_width_chars=tbl_width_chars,
                ascii_tables=True  # using ascii_tables to find out how many columns are displayed
        ):
            data_repr_ = data.__repr__()

            # this is very hacky, but the user defined format may not include a top line
            # this approach may break, oh well, if it breaks, the repr may not look as desired

            # when 3 columns the top bar should be a string as +---+---+---+

            rows = [line for line in data_repr_.split("\n") if line.startswith("+")]

            if rows:
                top_line = rows[0]  # 0 because here we only kept the +
                n_displayed_columns = top_line.count("+") - 1

        # --------------------------------------------------------------

        if not rows or len(top_line) <= tbl_width_chars:
            with pl.Config(tbl_width_chars=tbl_width_chars):
                # out of ascii_tables=True (unless it was requested)
                return data.__repr__()

        # --------------------------------------------------------------

        while len(top_line) > tbl_width_chars:
            # reduce the number of displayed columns if the table is too wide
            n_displayed_columns -= 1

            with pl.Config(
                    tbl_width_chars=tbl_width_chars,
                    tbl_cols=n_displayed_columns
            ):
                data_repr_ = data.__repr__()

            # actual_frame_repr_width = max(len(i) for i in data_repr_.split("\n"))
            top_line = data_repr_.split("\n")[1]

        return data_repr_


# -------------------------- helpers -----------------------------------


def split_text_with_padding(
        text: str,
        width_chars: int,
        left_padding: int,
        right_padding: int = 0,
) -> list[str]:
    """
    Splits a text based on width_chars and appends hyphens for word breaks.
    """
    width_chars_with_padding = width_chars - left_padding - right_padding

    if width_chars_with_padding <= 1:
        raise ValueError("width_chars_with_padding must be > 1")
    # Use textwrap to wrap the text
    wrapped_lines = textwrap.wrap(
        text,
        width=width_chars_with_padding,
        break_long_words=False,
        replace_whitespace=False
    )

    # Manually handle long words that couldn't be wrapped by textwrap
    split_lines = []
    for line in wrapped_lines:
        while len(line) > width_chars_with_padding:
            break_point = width_chars_with_padding - 1  # Account for the hyphen
            split_lines.append(line[:break_point] + "-")
            line = line[break_point:]

        split_lines.append(line)

    # Apply padding
    padded_lines = [left_padding * ' ' + line + right_padding * ' ' for line in split_lines]

    # Ensure each line is of width `width_chars`
    padded_lines = [line.ljust(width_chars) for line in padded_lines]

    return padded_lines


def line_with_centered_caption(
        width_chars: int,
        left_padding: int = 0,
        right_padding: int = 0,
        title: str = None,
        symbol: str = "-"
) -> str:
    """
    Creates a line of specified width with an optional centered title.

    The function generates a string of a given width, primarily composed of a specified
    symbol. Optionally, a title can be centered within this line. Additional padding can
    be added to the left or right side of the line separately.

    Parameters
    ----------
    width_chars : int
        The total width of the resulting line, in characters.

    left_padding : int, optional
        The number of space characters to be added to the left side of the line.
        Default is 0.

    right_padding : int, optional
        The number of space characters to be added to the right side of the line.
        Default is 0.

    title : str, optional
        The title to be centered within the line. If not provided, the line will
        consist only of the specified symbol. Default is None.

    symbol : str, optional
        The character used to create the line. Default is '-'.

    Returns
    -------
    str
        A string of specified width, with the given symbol, and an optional centered
        title.

    Notes
    -----
    If the total width of the line (including the title, symbols, and padding) exceeds
    `width_chars`, the line will be truncated to `width_chars`.

    If `title` is provided and its length combined with symbols exceeds `width_chars`,
    the title may be surrounded by fewer symbols than expected or may be partially
    truncated.
    """
    # Calculate the total width for the dashes considering the padding on both sides
    dash_length = width_chars - left_padding - right_padding

    if title:
        title = ' ' + title + ' '

    # If title is provided, adjust the dash length to fit the title in the center
    if title:
        title_length = len(title)
        dash_side_length = (dash_length - title_length) // 2

        line = (
                symbol * dash_side_length +
                title +
                symbol * (dash_length - dash_side_length - title_length)
        )
    else:
        line = symbol * dash_length

    # Add padding
    line = ' ' * left_padding + line + ' ' * right_padding

    # Ensure the line doesn't exceed the max_length
    return line[:width_chars]


def join_strings(
        *strings,
        separator: str = '|',
        min_width: int = 0
) -> list[str]:
    """
    Join multiple strings vertically, side by side, separated by a given separator.

    The function takes care to pad strings with spaces such that the output is neat
    and the strings are vertically aligned. Strings can have multiple lines.

    Parameters
    ----------
    *strings : str
        Any number of input strings that need to be joined. Multiple lines in a string
        should be separated by newline characters.

    separator : str, optional
        The separator string used to join the input strings side by side.
        Default is '|'.

    min_width : int, optional
        Minimum width for each string. If the actual string width is smaller than this,
        it will be padded with spaces on the right. Default is 0.

    Returns
    -------
    list
        A list of strings with the input strings joined side by side, separated by the specified
        separator. Each line of the input strings is joined with the corresponding line
        of other strings. If a string has fewer lines than others, it will be padded
        with empty lines.

    Notes
    -----
    If the input strings have different number of lines, the output will be aligned
    based on the string with the maximum number of lines.

    The function ensures that each string block in the output has a width equal to the
    maximum width of its lines or the specified `min_width`, whichever is larger.
    """

    # Convert each string representation into lists of lines
    lists_of_lines = [s.split('\n') for s in strings]

    # Get the maximum width for each string representation, considering the minimum width
    max_widths = [max(len(line) for line in lines) for lines in lists_of_lines]
    max_widths = [max(w, min_width) for w in max_widths]

    # Ensure each line in each string representation is of the correct width (by padding)
    for i, lines in enumerate(lists_of_lines):
        for j, line in enumerate(lines):
            lists_of_lines[i][j] = line.ljust(max_widths[i])

    # Get the maximum number of lines among all string representations
    max_lines = max(len(lines) for lines in lists_of_lines)

    # Ensure all lists of lines are of equal length (fill with empty lines if necessary)
    for lines in lists_of_lines:
        while len(lines) < max_lines:
            lines.append(' ' * max_widths[lists_of_lines.index(lines)])

    # Join lines side by side
    joined_lines = []
    for i in range(max_lines):
        line = separator.join(lines[i] for lines in lists_of_lines)
        joined_lines.append(line)

    return joined_lines


def _determine_indent(
        indent: int | tuple,
) -> tuple[int, int, int, int]:
    base_indent, key_indent, value_indent, right_padding = 0, 0, 0, 0

    if isinstance(indent, int):
        base_indent = indent

    elif isinstance(indent, tuple):

        if len(indent) == 1:
            base_indent = indent

        elif len(indent) == 2:
            base_indent, key_indent = indent

        elif len(indent) == 3:
            base_indent, key_indent, value_indent = indent

        elif len(indent) == 4:
            base_indent, key_indent, value_indent, right_padding = indent

        else:
            raise ValueError("len of indent should max 4")

    return base_indent, key_indent, value_indent, right_padding


# ----------------------------------------------------------------------

def line_str_with_title_old(
        width_chars: int,
        symmetric_padding: int = 0,
        title: str = None,
        symbol: str = "-"
):
    """Creates a line with an optional centered title."""
    # Calculate the total width for the dashes considering the padding on both sides
    dash_length = width_chars - (2 * symmetric_padding)

    if title:
        title = ' ' + title + ' '

    # If title is provided, adjust the dash length to fit the title in the center
    if title:
        title_length = len(title)
        dash_side_length = (dash_length - title_length) // 2

        line = (
                symbol * dash_side_length +
                title +
                symbol * (dash_length - dash_side_length - title_length)
        )
    else:
        line = symbol * dash_length

    # Add padding
    line = ' ' * symmetric_padding + line + ' ' * symmetric_padding

    # Ensure the line doesn't exceed the max_length
    return line[:width_chars]


def is_dict_of_dicts(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    for key, value in data.items():
        if not isinstance(value, dict):
            return False
    return True


def split_text_with_padding_old_old(
        text: str,
        width_chars: int,
        padding: str
) -> list[str]:
    """
    Splits a text based on max_line_length and appends hyphens for word breaks.
    """
    lines = []

    counter = 0
    _original_text = text

    while text:
        if len(text) <= width_chars:
            lines.append(padding + text)
            break
        else:
            # If there's a space to break the line at
            break_point = text.rfind(' ', 0, width_chars)
            if break_point != -1:
                lines.append(padding + text[:break_point])
                text = text[break_point + 1:]

            else:  # If no space, use hyphen and continue
                lines.append(padding + text[:width_chars - 1] + "-")
                text = text[width_chars - 1:]

            counter += 1

            if counter > 10:
                raise ValueError(f"string is too long: '{_original_text}'")

    return lines


def split_text_with_padding_old(
        text: str,
        width_chars: int,
        left_padding: int,
        right_padding: int = 0,
) -> list[str]:
    """
    Splits a text based on max_line_length and appends hyphens for word breaks.
    """
    lines = []
    left_padding_str = ' ' * left_padding
    right_padding_str = ' ' * right_padding

    width_chars_with_padding = width_chars - left_padding - right_padding

    counter = 0

    _original_text = text

    while text:
        if len(text) <= width_chars_with_padding:

            lin = left_padding_str + text + right_padding_str
            # lin = lin.ljust(width_chars, ' ')

            lines.append(lin)
            break
        else:
            # If there's a space to break the line at
            break_point = text.rfind(' ', 0, width_chars_with_padding)

            if break_point != -1:

                lin = left_padding_str + text[:break_point] + right_padding_str

                # lin = lin.ljust(width_chars, ' ')

                lines.append(lin)

                text = text[break_point + 1:]

            else:  # If no space, use hyphen and continue

                lin = (
                        left_padding_str +
                        text[:width_chars_with_padding - 1] +
                        "-" +
                        right_padding_str
                )

                # lin = lin.ljust(width_chars, ' ')

                lines.append(lin)

                text = text[width_chars_with_padding - 1:]

            counter += 1

            if counter > 10:
                raise ValueError(f"string is too long: '{_original_text}'")

    return lines
