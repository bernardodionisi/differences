from __future__ import annotations

from typing import Literal

# check out: https://rich.readthedocs.io/en/stable/panel.html

HLINES = {"double": "═", "thin": "─", "bold": "━"}
VLINES = {"thin": "│", "bold": "┃"}

# https://www.compart.com/en/unicode/U+251B
CORNER_LIENS = {
    "thin": {'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘'},
    "bold": {'tl': '┏', 'tr': '┓', 'bl': '┗', 'br': '┛'},
    "thin-curved": {'tl': '╭', 'tr': '╮', 'bl': '╰', 'br': '╯'}
}


def boxed_str(
        input_str: str,

        top_name: str = None,
        bottom_name: str = None,

        horizontal_line: Literal["double", "thin", "bold"] = "bold",
        vertical_line: Literal["thin", "bold"] = "thin",
        corner: Literal["thin", "bold", "thin-curved"] = "thin",

        include_top_line: bool = True,
        include_bottom_line: bool = True,

        fixed_width: int = None,
        increment_auto_width: int = 2,
) -> str:
    out_str = box_around_string(
        input_str=input_str,

        top_name=top_name,
        bottom_name=bottom_name,

        horizontal_line=horizontal_line,
        vertical_line=vertical_line,
        corner=corner,

        include_top_line=include_top_line,
        include_bottom_line=include_bottom_line,

        increment_auto_width=increment_auto_width,
        fixed_width=fixed_width,
    )

    out_str = name_in_line(
        input_str=out_str,
        top_name=top_name,
        bottom_name=bottom_name
    )

    return out_str


def box_width(
        lines: list,
        top_name: str = None,
        bottom_name: str = None,

        fixed_width: int = None,
        increment_auto_width: int = 2,
) -> int:
    if fixed_width is not None:
        width = fixed_width
        if top_name:
            if max(len(i) for i in top_name.split("\n")) >= width:
                raise ValueError("top_name too long")

        if bottom_name:
            if max(len(i) for i in bottom_name.split("\n")) >= width:
                raise ValueError("bottom_name too long")

    else:
        # Determine the width based on the longest line in the content
        width = max(len(line) for line in lines)

        if top_name:
            width = max(width, len(top_name))

        if bottom_name:
            width = max(width, len(bottom_name))

        width += increment_auto_width

    width -= 2  # the corners

    return width


def box_around_string(
        input_str: str,

        horizontal_line: Literal["double", "thin", "bold"],
        vertical_line: Literal["thin", "bold"],
        corner: Literal["thin", "bold", "thin-curved"],

        fixed_width: int | None,
        increment_auto_width: int,  # right padding

        top_name: str | None,  # just for len
        bottom_name: str | None,  # just for len
        include_top_line: bool | True,
        include_bottom_line: bool | True
) -> str:
    # Split the string into lines
    lines = input_str.split('\n')

    width = box_width(
        lines=lines,
        fixed_width=fixed_width,
        increment_auto_width=increment_auto_width,
        top_name=top_name,
        bottom_name=bottom_name,
    )
    # ------------------------------------------------------------------

    # Create the top and bottom borders
    top_border = (
            CORNER_LIENS[corner]["tl"] +
            HLINES[horizontal_line] * width +
            CORNER_LIENS[corner]["tr"]
    )

    bottom_border = (
            CORNER_LIENS[corner]["bl"] +
            HLINES[horizontal_line] * width +
            CORNER_LIENS[corner]["br"]
    )

    # Add vertical borders to the sides of each line
    bordered_lines = [
        f'{VLINES[vertical_line]}' + line.ljust(width) + f'{VLINES[vertical_line]}'
        for line in lines
    ]

    # Combine everything

    final_output = ""
    if include_top_line:
        final_output += top_border

    final_output += "\n" + '\n'.join(bordered_lines) + '\n'

    if include_bottom_line:
        final_output += bottom_border

    return final_output


def name_in_line(
        input_str: str,
        top_name: str | None,
        bottom_name: str | None
) -> str:
    input_str = input_str.split("\n")

    if top_name:
        top_name_line = replace_center_line_with_name_multiple(
            line=input_str[0],
            name=top_name,
            middle=False
        )
        # input_str[0] = top_name_line
        input_str = (
            # to make sure to add the borders back to the box
                modify_strings_after_index(
                    strings=top_name_line.split("\n"),
                    symbol=input_str[1][0]) + input_str[1:]
        )

    if bottom_name:
        input_str[-1] = replace_center_line_with_name_multiple(
            line=input_str[-1],
            name=bottom_name,
            middle=False
        )

    return "\n".join(input_str)


def modify_strings_after_index(strings: list, symbol: str) -> list:
    # Find the index of the first string with a non-space starting character
    index = next((i for i, s in enumerate(strings) if s and not s[0].isspace()), -1)

    # If no such string found, return the original list
    if index == -1:
        return strings

    # Modify the strings after the found index
    modified_strings = strings[:index + 1]
    for s in strings[index + 1:]:
        if s.startswith(' ') and s.endswith(' '):
            s = symbol + s[1:-1] + symbol
        modified_strings.append(s)

    return modified_strings


def replace_center_line_with_name(line: str, name: str):
    name = f" {name} "
    # If the replacement is longer than the original, return replacement \n original
    if len(name) > len(line):
        return name + '\n' + line

    # Calculate the start and end positions for the replacement in the original string
    start_pos = (len(line) - len(name)) // 2
    end_pos = start_pos + len(name)

    # Replace the center of the original with the replacement string
    return f"{line[:start_pos]}{name}{line[end_pos:]}"


def replace_center_line_with_name_multiple(
        line: str, name: str, middle: bool = False) -> str:
    if middle:
        return _replace_center_line_with_name_multiple_middle(line=line, name=name)
    return _replace_center_line_with_name_multiple_top(line=line, name=name)


def _replace_center_line_with_name_multiple_top(line: str, name: str) -> str:
    names = name.split("\n")
    if len(names) == 1:
        return replace_center_line_with_name(line=line, name=name)

    first_name, other_names = names[0], names[1:]
    if first_name[0] == line[0]:
        raise ValueError("the name should not start with the same characters as the line")

    first_line = replace_center_line_with_name(line=line, name=first_name)

    other_str = "\n".join([i.center(len(first_line)) for i in other_names])

    return f"{first_line}\n" + other_str


def _replace_center_line_with_name_multiple_middle(line: str, name: str) -> str:
    names = name.split("\n")

    if len(names) == 1:
        return replace_center_line_with_name(line=line, name=name)

    before, middle, after = _split_list_middle(names)

    # warning: if the string starts with the same characters as a line
    if middle[0] == line[0]:
        raise ValueError("the name should not start with the same characters as the line")

    middle_line = replace_center_line_with_name(line=line, name=middle)

    before_str = "\n".join([i.center(len(middle_line)) for i in before])
    after_str = "\n".join([i.center(len(middle_line)) for i in after])

    if before and after:
        out = before_str + f"\n{middle_line}\n" + after_str
    elif before:
        out = before_str + f"\n{middle_line}"
    else:  # elif after:
        out = f"{middle_line}\n" + after_str

    return out


def _split_list_middle(lst) -> tuple[list, str, list]:
    n = len(lst)

    # If the list is empty
    if n == 0:
        return [], "", []

    middle_index = n // 2

    # If the list length is odd
    if n % 2 == 1:
        before = lst[:middle_index]
        middle = lst[middle_index]
        after = lst[middle_index + 1:]

    # If the list length is even
    else:
        before = lst[:middle_index - 1]  # - 1 if want to move down
        middle = lst[middle_index - 1]  # - 1 if want to move down
        after = lst[middle_index:]  # remove +1 if want to move down

    return before, middle, after
