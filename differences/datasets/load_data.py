from __future__ import annotations

from os.path import abspath, join, split

import numpy as np
from pandas import qcut, read_csv, to_datetime

__all__ = ["rc_two_periods", "rc_multi_period", "mpdta", "simulated_data"]


def rc_two_periods(py_format: bool = True, seed: int = 4321):
    """two-period repeated cross section, sim_rc santanna zhao"""
    data = _load_data(__file__, "rcs-two_periods.csv")

    data["cohort"] = np.where(data["d"], 2000, np.nan)
    data["time"] = np.where(data["post"], 1999, 2000)

    np.random.seed(seed)
    data["random_weights"] = np.random.uniform(low=0, high=1, size=len(data))

    if py_format:
        data.set_index(["id", "time"], inplace=True)
    return {
        "data": data,
        "entity_name": "id",
        "post_dummy": "post",
        "time_name": "year",
        "treat_dummy": "d",
        "cohort_name": "cohort",
        "y_name": "y",
    }


def rc_multi_period(py_format: bool = True, seed: int = 4321):
    """multi-period repeated cross section, sim_rc santanna zhao"""
    data = _load_data(__file__, "rcs-multi_period.csv")

    np.random.seed(seed)
    data["random_weights"] = np.random.uniform(low=0, high=1, size=len(data))

    if py_format:
        data["G"] = np.where(data["G"] == 0, np.nan, data["G"])
        data.set_index(["id", "period"], inplace=True)

    return {
        "data": data,
        "entity_name": "id",
        "time_name": "period",
        "cohort_name": "G",
        "y_name": "Y",
        "panel": False,
        "allow_unbalanced_panel": False,
        "weights_name": "random_weights",
    }


def mpdta(py_format: bool = True, seed: int = 4321):
    """multi-period balanced panel"""

    data = _load_data(__file__, "mpdta.csv")

    np.random.seed(seed)
    data["random_weights"] = np.random.uniform(low=0, high=1, size=len(data))

    if py_format:
        data["first_treat"] = np.where(
            data["first_treat"] == 0, np.nan, data["first_treat"]
        )
        data.set_index(["countyreal", "year"], inplace=True)

    return {
        "data": data,
        "entity_name": "countyreal",
        "time_name": "year",
        "cohort_name": "first_treat",
        "y_name": "lemp",
        "panel": True,
        "allow_unbalanced_panel": False,
    }


# ----------------------------------------------------------------------


def simulated_data(
    py_format: bool = True,
    single_event: bool = True,
    separate_cohort_data: bool = False,
    seed: int = 4321,
):
    if single_event:
        data_name = "se"
    else:
        data_name = "me"
    data = _load_data(__file__, f"{data_name}.csv")

    np.random.seed(seed)
    data["random_weights"] = np.random.uniform(low=0, high=1, size=len(data))

    data = data.assign(
        T=lambda x: to_datetime(x["T"]),
        cohort=lambda x: to_datetime(x["cohort"]),
    )

    cohorts = (
        data[["cohort", "event_size"]].dropna().reset_index(drop=True).drop_duplicates()
    )
    cohorts["qevent_size"] = qcut(cohorts["event_size"], 3, labels=False)

    data = data.merge(cohorts[["cohort", "qevent_size"]], on=["cohort"], how="left")

    if py_format:
        data.set_index(["I", "T"], inplace=True)

    data_dict = {
        "data": data,
        "entity_name": "I",
        "time_name": "T",
        "cohort_name": "cohort",
        "y_name": "y",
    }

    if separate_cohort_data:
        cohort_data = (
            data.reset_index()[["I", "cohort", "event_size", "qevent_size"]]
            .dropna()
            .reset_index(drop=True)
        )
        data_dict.update({"cohort_data": cohort_data})

        del data["cohort"], data["event_size"]

    return data_dict


def _get_path(f: str):
    return split(abspath(f))[0]


def _load_data(module: str, file_name: str):
    return read_csv(join(_get_path(module), file_name))
