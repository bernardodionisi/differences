from __future__ import annotations

import copy
from typing import Literal
import polars as pl


def get_group_times_list_of_dicts(
        cohorts: list[int],
        times: list[int],
        anticipation: int,
        base_period: Literal["varying", "varying-multiple", "universal"],
        dosages: list[int | str] | None,
        feasible_group_times: list[dict] | None,
) -> list[dict]:
    if anticipation < 0:
        raise ValueError("'anticipation' must be >= 0")

    cohorts = sorted(cohorts)
    times = sorted(times)

    if base_period == "varying":
        func = _varying_base_period_single

    elif base_period == "varying-multiple":
        func = _varying_base_period_multiple

    elif base_period == "universal":
        func = _universal_base_period_single

    else:
        raise ValueError(
            "'base_period' must be set to either:\n"
            "'varying', "
            "'varying-multiple' or "
            "'universal'"
        )

    out = func(
        cohorts=cohorts,
        times=times,
        anticipation=anticipation,
    )

    if dosages is not None:
        out = [{**i, "dosage": d} for d in dosages for i in out]

    if feasible_group_times is not None:
        # keep only group-time that are in 'feasible_group_times': list[dict]
        # one can get the 'feasible_group_times' using the 'get_feasible_group_times' func
        out = _overlap_group_times_with_feasible(
            gt_source=out,
            gt_reference=feasible_group_times
        )

    return out


def get_feasible_group_times(
        data: pl.DataFrame,
        time_column: str,
        cohort_column: str,
        dosage_column: str,
) -> list[dict]:
    """
    data = pl.DataFrame(
    {"time": [1, 2, 3, 4], "cohort": [2, 3, None, 2], "dosage": ["a", None, None, None]})

    get_feasible_group_times(data, "time", "cohort", "dosage")
    """

    keep_info = [time_column, cohort_column]
    rename_columns = {time_column: "time", cohort_column: "cohort"}

    if dosage_column is not None:
        keep_info = [time_column, cohort_column, dosage_column]
        rename_columns.update({dosage_column: "dosage"})

    # returns something like {'time': 1, 'cohort': 2, 'dosage': 'a'}, {...}]
    # time is the second level of the index
    return (
        data
        .select(keep_info)
        .unique()
        .drop_nulls()  # drop never-treated
        .rename(rename_columns)  # rename columns so the keys match the group_time dict[list]
        .to_dicts()
    )


# -------------------------- expressions -------------------------------

def add_expr_to_group_time_list_dicts(
        group_times: list[dict],

        time_column: str,
        cohort_column: str,
        anticipation: int,
) -> list[dict]:
    group_times = copy.deepcopy(group_times)

    for gt in group_times:
        gt.update(_get_group_time_expr(
            time_column=time_column,
            cohort_column=cohort_column,

            anticipation=anticipation,

            time=gt["time"],
            base_period=gt["base_period"],
            cohort=gt["cohort"]
        ))

    return group_times


def _get_group_time_expr(
        time_column: str,
        cohort_column: str,

        anticipation: int,

        time: int,
        base_period: int | list[int],
        cohort: int,

) -> dict[str, dict[str, pl.Expr]]:
    pre_post = _expr_pre_post(time_column=time_column, time=time, base_period=base_period)

    treated = _expr_treated(cohort_column=cohort_column, cohort=cohort)

    never_treated = _expr_never_treated(cohort_column=cohort_column)

    not_yet_treated = _expr_not_yet_treated(
        cohort_column=cohort_column,
        time=time,
        base_period=base_period,
        cohort=cohort,
        anticipation=anticipation
    )

    out = {
        "expr": {
            "never_treated": pre_post & (treated | never_treated),
            "not_yet_treated": pre_post & (treated | never_treated | not_yet_treated)
        }
    }

    return out


def _expr_pre_post(
        time_column: str,
        time: int,
        base_period: int | list[int]

) -> pl.Expr:
    if isinstance(base_period, int):
        base_period = pl.col(time_column).eq(base_period)
    else:  # list
        base_period = pl.col(time_column).is_in(base_period)

    return base_period | pl.col(time_column).eq(time)


def _expr_treated(
        cohort_column: str,
        cohort: int,
) -> pl.Expr:
    return pl.col(cohort_column).eq(cohort)


def _expr_never_treated(
        cohort_column: str,
) -> pl.Expr:
    return pl.col(cohort_column).is_null()


def _expr_not_yet_treated(
        cohort_column: str,
        time: int,
        base_period: int | list[int],
        cohort: int,
        anticipation: int
) -> pl.Expr:
    if isinstance(base_period, int):
        not_yet_expr = (pl.col(cohort_column).gt(max(time, base_period) + anticipation))
    else:  # list
        # todo: check that this is ok
        not_yet_expr = pl.any_horizontal([
            pl.col(cohort_column).gt(max(time, bp) + anticipation)
            for bp in base_period
        ])

    not_yet_expr = pl.col(cohort_column).ne(cohort) & not_yet_expr

    return not_yet_expr


# ------------------------ base_period ---------------------------------

def _universal_base_period_single(  # new function to be tested
        cohorts: list,
        times: list,
        anticipation: int,
):
    cbt = []

    for c in cohorts:
        # Find the latest time less than (cohort - anticipation)
        base_period = None
        for t in reversed(times):
            if t < (c - anticipation):
                base_period = t
                break

        if base_period is None:
            continue

        for current_time in times:
            cbt.append({
                "cohort": c,
                "time": current_time,
                "base_period": base_period,
            })

    return cbt


def _varying_base_period_single(  # new function to be tested
        cohorts: list,
        times: list,
        anticipation: int,
):
    cbt = []

    for c in cohorts:
        for t_idx, current_time in enumerate(times[:-1]):

            next_time = times[t_idx + 1]

            # Default: pre-treatment base period is current_time
            base_period = current_time  # next_time < cohort

            if next_time >= c:  # check if post-treatment
                # Try to find the latest time less than (cohort - anticipation)
                for previous_time in reversed(times[:t_idx + 1]):
                    if previous_time < (c - anticipation):
                        base_period = previous_time
                        break
                else:  # No suitable base_period found
                    continue

            cbt.append({
                "cohort": c,
                "time": next_time,
                "base_period": base_period,
            })
    return cbt


def _varying_base_period_multiple(cohorts: list, times: list, anticipation: int):
    cbt = []

    for c in cohorts:
        for t_idx, current_time in enumerate(times[:-1]):

            next_time = times[t_idx + 1]

            # Default: pre-treatment base periods are all times up to current_time
            base_periods = times[:t_idx + 1] if next_time < c else []

            # If post-treatment, gather all prior times before (c - anticipation)
            if next_time >= c:
                base_periods = [t for t in times[:t_idx + 1] if t < (c - anticipation)]

            # If no base_periods found, skip to the next iteration
            if not base_periods:
                continue

            cbt.append({
                "cohort": c,
                "time": next_time,
                "base_period": base_periods,
            })
    return cbt


def _overlap_group_times_with_feasible(
        gt_source: list[dict],
        gt_reference: list[dict]
) -> list[dict]:
    # # Example
    # source = [
    #     {"a": 1, "b": 2},
    #     {"a": 1, "b": 3},
    #     {"c": 4}
    # ]
    #
    # reference = [
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 4},
    #     {"c": 4}
    # ]
    #
    # print(overlap_group_times_with_feasible(source, reference))

    if not all(isinstance(item, dict) for item in gt_source) or not all(
            isinstance(item, dict) for item in gt_reference):
        return []

    def matches_reference_dict(source_dict: dict) -> bool:
        for ref_dict in gt_reference:
            if all(
                    key
                    in ref_dict and
                    source_dict[key] == ref_dict[key] for key in source_dict
            ):
                return True
        return False

    out = []
    for s_dict in gt_source:
        if "base_period" in s_dict and "time" in s_dict:
            bp = s_dict.get("base_period") == s_dict.get("time")
        else:
            bp = False
        if matches_reference_dict(s_dict) or bp:
            out.append(s_dict)
    return out


# ---------------------- filter group-times ----------------------------
def filter_group_time_list_of_dicts(
        group_times: list[dict],

        *,

        relative_period: int | list[int] = None,
        relative_period_start: int = None,
        relative_period_end: int = None,

        cohort: int | list[int] = None,
        cohort_start: int = None,
        cohort_end: int = None,

        time: int | list[int] = None,
        time_start: int = None,
        time_end: int = None,

) -> list[dict]:
    """
    Filter a list of dictionaries based on cohort, time, and event criteria.

    Parameters
    ----------
    relative_period : int, list[int], optional
        Specific event values (time - cohort) to include.
    relative_period_start : int, optional
        Minimum event value (time - cohort) to include.
    relative_period_end : int, optional
        Maximum event value (time - cohort) to include.
    group_times : list[dict]
        list of dictionaries containing cohort and time information.
    cohort : int, list[int], optional
        Specific cohort values to include.
    time : int, list[int], optional
        Specific time values to include.
    cohort_start : int, optional
        Minimum cohort value to include.
    cohort_end : int, optional
        Maximum cohort value to include.
    time_start : int, optional
        Minimum time value to include.
    time_end : int, optional
        Maximum time value to include.

    Returns
    -------
    list[dict]
        Filtered list of dictionaries.
    """
    # Filtering by cohort and time
    for attribute, value, start_value, end_value in [
        ('cohort', cohort, cohort_start, cohort_end),
        ('time', time, time_start, time_end)
    ]:
        group_times = _filter_by_attribute(group_times, attribute, value, start_value, end_value)

    # Filtering by event (difference between time and cohort)
    group_times = _filter_by_event(
        group_times, relative_period, relative_period_start, relative_period_end
    )

    return group_times


def _filter_by_attribute(
        group_times: list[dict],
        attribute: str,
        value: int | list[int] = None,
        start_value: int = None,
        end_value: int = None
) -> list[dict]:
    """Filter a list of dictionaries based on an attribute's value, start, and end.

    Parameters
    ----------
    group_times : list[dict]
        list of dictionaries containing cohort, time information.
    attribute : str
        The key to filter on, e.g., 'cohort' or 'time'.
    value : int, list[int], optional
        Specific values to include.
    start_value : int, optional
        The minimum value for the attribute.
    end_value : int, optional
        The maximum value for the attribute.

    Returns
    -------
    list[dict]
        Filtered list of dictionaries.
    """
    if value is not None:
        if isinstance(value, int):
            value = [value]
        group_times = [d for d in group_times if d[attribute] in value]

    if start_value is not None:
        group_times = [d for d in group_times if d[attribute] >= start_value]

    if end_value is not None:
        group_times = [d for d in group_times if d[attribute] <= end_value]

    return group_times


def _filter_by_event(
        group_times: list[dict],
        event_values: int | list[int] = None,
        event_start: int = None,
        event_end: int = None
) -> list[dict]:
    """Filter a list based on event, defined as the difference between time and cohort.

    Parameters
    ----------
    group_times : list[dict]
        list of dictionaries containing cohort, time information.
    event_values : int, list[int], optional
        Specific event values to include.
    event_start : int, optional
        The minimum event value.
    event_end : int, optional
        The maximum event value.

    Returns
    -------
    list[dict]
        Filtered list of dictionaries.
    """
    if event_values is not None:
        if isinstance(event_values, int):
            event_values = [event_values]
        group_times = [d for d in group_times if d["time"] - d["cohort"] in event_values]

    if event_start is not None:
        group_times = [d for d in group_times if d["time"] - d["cohort"] >= event_start]

    if event_end is not None:
        group_times = [d for d in group_times if d["time"] - d["cohort"] <= event_end]

    return group_times


# ----------------------------------------------------------------------


# todo: delete this once sure everything works fine
def filter_gt_dict_old(group_time: list[dict], filter_gt: dict) -> list[dict]:
    """
    filters the cohorts-times-strata to run the estimation by

    this filter is applied before calculating the ATT, it is called in .fit()

    Parameters
    ----------
    group_time:
        dictionary of cohorts-times-strata
    filter_gt:
        user input filter as a dictionary

    Returns
    -------

    """

    for i in ["cohort", "time"]:
        gt = filter_gt.get(i)
        if gt is not None:
            if isinstance(gt, int):
                gt = [gt]
            group_time = [d for d in group_time if d[i] in gt]

        ct_start = filter_gt.get(f"{i}_start")
        if ct_start is not None:
            group_time = [d for d in group_time if d[i] >= ct_start]

        ct_end = filter_gt.get(f"{i}_end")
        if ct_end is not None:
            group_time = [d for d in group_time if d[i] <= ct_end]

    event = filter_gt.get("")
    if event is not None:
        if isinstance(event, int):
            event = [event]

        group_time = [d for d in group_time if d["time"] - d["cohort"] in event]

    event_start = filter_gt.get("event_start")
    if event_start is not None:
        group_time = [d for d in group_time if d["time"] - d["cohort"] >= event_start]

    event_end = filter_gt.get("event_end")
    if event_end is not None:
        group_time = [d for d in group_time if d["time"] - d["cohort"] <= event_end]

    return group_time
