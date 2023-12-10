from __future__ import annotations


# ---------------------- additional information ------------------------

def format_info_single_dictionary(
        info: dict,
        n_columns: int,
        sort: bool,
        align_equals: bool
) -> str:
    # # Example usage:
    # d = {
    #     "info_1": 5,
    #     "info_2": 3,
    #     "info_3": 7,
    #     "info_4": 6,
    #     "info_5": 8
    # }
    #
    # print(format_info_dictionary(d, 2))

    if not info:
        return ""

    info = info.items()

    if sort:
        # Sort the dictionary by its keys
        info = sorted(info)

    info = list(info)

    # Chunk the list into the specified number of columns
    chunked_items = [info[i:i + n_columns] for i in range(0, len(info), n_columns)]

    # Find the maximum key width for each column
    max_key_widths = [
        max((len(chunk[i][0]) for chunk in chunked_items if i < len(chunk)), default=0)
        for i in range(n_columns)
    ]

    formatted_chunks = []

    for chunk in chunked_items:
        formatted_chunk = []
        for i, (key, value) in enumerate(chunk):
            if align_equals:
                # Pad the key based on the max key width of its respective column
                formatted_key = key.ljust(max_key_widths[i])
                formatted_chunk.append(f"{formatted_key} = {value}")
            else:
                formatted_chunk.append(f"{key} = {value}")
        formatted_chunks.append(formatted_chunk)

    # Adjust each string in the chunked list according to the max width of its respective column
    max_widths = [
        max((len(chunk[i]) for chunk in formatted_chunks if i < len(chunk)), default=0)
        for i in range(n_columns)
    ]

    for i in range(len(formatted_chunks)):
        for j in range(len(formatted_chunks[i])):
            formatted_chunks[i][j] = formatted_chunks[i][j].ljust(max_widths[j])

    # Join the inner lists with '|', prepend and append an additional '|' to each chunk,
    # and then join the outer list with newlines
    return '\n'.join(['| ' + ' | '.join(chunk) + ' |' for chunk in formatted_chunks])


def format_info_multiple_dictionaries(
        info: dict[str, dict],
        n_columns: int,
        sort: bool,
        align_equals: bool
) -> str:
    if not info:
        return ""

    info_str = []
    for k, v in info.items():
        info_str_single = format_info_single_dictionary(
            info=v, n_columns=n_columns, sort=sort, align_equals=align_equals
        )

        if info_str_single:
            info_str_single = f"{k}\n" + info_str_single
        else:
            info_str_single = f"{k}"
        info_str.append(info_str_single)

    return "\n".join(info_str)


def join_dictionaries(dicts: list[dict]):
    """
    Joins multiple dictionaries into a single dictionary.
    Raises an error if there are overlapping keys.

    Args:
    * dicts: The dictionaries to join.

    Returns:
    dict: The joined dictionary.
    """
    result = {}
    for dictionary in dicts:
        for key in dictionary:
            if key in result:
                raise ValueError(f"Duplicate key detected: {key}")
            result[key] = dictionary[key]
    return result


# ----------------------------------------------------------------------

def nested_dict_to_str(d: dict, indent: int = 0, n_words: int = 10) -> str:
    lines = _nested_dict_to_list(d=d, indent=indent, n_words=n_words)
    return "\n".join(lines)


def _nested_dict_to_list(d: dict, indent: int, n_words: int):
    lines = []
    items = list(d.items())

    def adjust_indent(multiline_str, prefix):
        """Apply the same indentation to every line of a multiline string."""
        return '\n'.join([prefix + line for line in multiline_str.split('\n')])

    for i, (key, value) in enumerate(items):
        prefix = ' ' * indent

        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")

            lines.extend(
                _nested_dict_to_list(
                    d=value,
                    indent=indent + 3,
                    n_words=n_words
                )
            )
        else:

            if isinstance(value, str):
                value = _limit_words_in_line(
                    text=value, n_words=n_words
                )

            value_str = str(value)  # Convert value to string

            if '\n' in value_str:
                lines.append(f"{prefix}{key}:")
                lines.append(adjust_indent(value_str, prefix + ' ' * 2))
            else:
                lines.append(f"{prefix}{key}: {value_str}")

    return lines


def _limit_words_in_line(text: str, n_words: int):
    words = text.split()
    reconstructed = []
    current = []
    for word in words:
        if len(current) == n_words:
            reconstructed.append(' '.join(current))
            current = []
        current.append(word)
    if current:
        reconstructed.append(' '.join(current))
    return '\n'.join(reconstructed)


# ----------------------------------------------------------------------

def add_horizontal_line_to_list_of_strings(strings: list[str], title: str = None):
    # this function is just used for expressions trees for now

    # Find the widest line
    max_width = max(len(line) for s in strings for line in s.split('\n'))

    # Adjust max_width to accommodate the title if it's longer
    if title and len(title) > max_width:
        max_width = len(title)

    result = []

    # Add title if provided
    if title:
        result.append('=' * max_width)
        result.append(title.center(max_width))
        result.append('=' * max_width)

    # Add horizontal line after each string
    for s in strings:
        result.append(s)
        result.append('-' * max_width)

    return result
