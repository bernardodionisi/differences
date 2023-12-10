from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

import numpy as np

from pandas import DataFrame as PandasDF

from differences.datasets.utility import _load_data, Dataset

__all___ = [
    "mpdta",
    "fix_did",
    "fix_stagg",
    "rcs_two_periods",
    "rcs_multi_period",
    "baker",
    "cs_simulation"
]


# ----------------------------------------------------------------------


def mpdta(
        x_formula: Literal["~ 1", "~ lpop"] = "~ 1",
        weights_column: Literal["random_weights_varying", "random_weights_by_entity"] = None
) -> Dataset:
    """
    multi-period balanced panel


    Group
    -----
        dataset
    """

    data = _load_data(__file__, "mpdta.csv")

    # "panel": True,
    # "allow_unbalanced_panel": False,

    data = Dataset(
        data=data.set_index(["countyreal", "year"]),
        data_name="mpdta",

        entity_column="countyreal",
        time_column="year",
        cohort_column="first_treat",
        y_column="lemp",
        x_formula=x_formula,  # example model_matrix for this data
        formula=f"lemp {x_formula}",
        weights_column=weights_column,
        is_panel=True,
        post_column=None,
        treat_column=None,
    )

    return data


def rcs_multi_period(
        x_formula: Literal["~ 1", "~ X"] = "~ 1",
        weights_column: Literal["random_weights"] = None
) -> Dataset:
    """multi-period repeated cross section, sim_rc santanna zhao"""
    data = _load_data(__file__, "rcs-multi_period.csv")

    data = Dataset(
        data=data.set_index(["id", "period"]),
        data_name="rcs_multi_period",

        entity_column="id",
        time_column="period",
        cohort_column="G",
        y_column="Y",
        x_formula=x_formula,
        formula=f"Y {x_formula}",
        weights_column=weights_column,
        is_panel=False,
        post_column=None,
        treat_column=None,
    )

    # "panel": False,
    # "allow_unbalanced_panel": False,

    return data


# ----------------------------------------------------------------------


def fix_did() -> Dataset:
    """
    Group
    -----
        dataset
    """
    data = _load_data(__file__, "fix-did.csv", Path("data"))

    data = Dataset(
        data=data,
        data_name="fix_did",

        entity_column="id",
        time_column="period",
        cohort_column="cohort",
        y_column="y",
        post_column="post",
        treat_column="treat",
    )

    return data


def _process_fix_did(data: PandasDF):
    data = data.sort_values(['id', 'period']).reset_index(drop=True)

    data['cohort'] = np.where(data["treat"], 5, np.nan)
    data = data.set_index(['id', 'period'])

    return data


def fix_stagg() -> Dataset:
    """
    Group
    -----
        dataset
    """
    data = _load_data(__file__, "fix-stagg.csv", Path("data"))

    data = Dataset(
        data=data,
        data_name="fix_stagg",

        entity_column="id",
        time_column="year",
        cohort_column="cohort",
        y_column="y",
        post_column=None,
        treat_column=None,
        true_effect=_true_effect(data, "cohort", "treatment_effect_true"),
    )

    return data


def _process_fix_stagg(data: PandasDF):
    data = data.sort_values(['id', 'year']).reset_index(drop=True)

    data['cohort'] = np.where(data['year_treated'] == 10000, np.nan, data['year_treated'])
    data["relative_period"] = data["year"] - data["cohort"]

    data = (
        data
        .set_index(['id', 'year'])
        [["y", "x1", "cohort", "relative_period", "treatment_effect_true"]]
    )

    return data


# --------------------- repeated cross-section -------------------------


def rcs_two_periods(weights_seed: int = 4321) -> Dataset:
    """two-period repeated cross section, sim_rc santanna zhao"""
    data = _load_data(__file__, "rcs-two_periods.csv", Path("data"))

    data = _process_rcs_two_periods(data=data, weights_seed=weights_seed)

    data = Dataset(
        data=data,
        data_name="rcs_two_periods",

        entity_column="id",
        time_column="time",
        cohort_column="cohort",
        y_column="y",
        post_column="post",
        treat_column="d",
    )

    return data


def _process_rcs_two_periods(data: PandasDF, weights_seed: int):
    data["cohort"] = np.where(data["d"], 2000, np.nan)
    data["time"] = np.where(data["post"], 1999, 2000)

    np.random.seed(weights_seed)
    data["random_weights"] = np.random.uniform(low=0, high=1, size=len(data))

    return data.set_index(["id", "time"])


# --------------------- data from sc substack --------------------------

def baker():
    # source:
    # data = pd.read_stata(
    #     "https://github.com/scunning1975/codechella/blob/main/programs/baker.dta?raw=true",
    # )

    data = _load_data(__file__, "baker.parquet", Path("data", "post"))

    data, cohort_data = _process_baker(data=data)

    data = Dataset(
        data=data,
        data_name="baker",

        entity_column="id",
        time_column="year",
        cohort_column="treat_date",
        y_column="y",
        cohort_data=cohort_data,
        post_column=None,
        treat_column=None,
        true_effect=None
    )

    return data


def _process_baker(data: PandasDF):
    data = (
        pl.from_pandas(data)
        .with_columns(
            relative_period=pl.col("year") - pl.col("treat_date"),
            treatment_effect_y2=(pl.col("treat") * pl.col("te"))
        )
        .with_columns(
            treatment_effect_y=pl.col("treatment_effect_y2") *
                               (pl.col("year") - pl.col("treat_date") + 1)
        )
    )

    cohort_data = (
        data
        .select(["id", "treat_date"])
        .unique()
        .sort(["id", "treat_date"])
        .to_pandas()
    )

    # checl = (
    #     data
    #     .groupby(["treat_date", "relative_period"])
    #     .agg((pl.col("treatement_effect")).mean())
    #     .sort(["treat_date", "relative_period"])
    #     .to_pandas()
    # )

    data = data.to_pandas().set_index(["id", "year"])

    return data, cohort_data


# ------------------ data from ps/did website --------------------------


def cs_simulation(heterogeneous: bool = True, never_treated: bool = True) -> Dataset:
    """Homogeneous treatment effect w/ no-never-treated"""

    if not heterogeneous and not never_treated:
        # Homogeneous treatment effect w/ no-never-treated
        data = _process_cs_simulation("cs-panel-homogeneous-nnt.csv")
    elif not heterogeneous and never_treated:
        # Homogeneous treatment effect w/ no-never-treated
        data = _process_cs_simulation("cs-panel-homogeneous-ynt.csv")
    elif heterogeneous and not never_treated:
        # Heterogeneous treatment effect w/ never-treated
        data = _process_cs_simulation("cs-panel-heterogeneous-ynt.csv")
    else:
        raise ValueError

    data = Dataset(
        data=data,
        data_name="cs_simulation",
        entity_column="unit",
        time_column="year",
        cohort_column="cohort_year",
        y_column="dep_var",
        post_column=None,
        treat_column=None,
        true_effect=_true_effect(data, "cohort_year", "tau_cum")
    )

    return data


def _process_cs_simulation(data_name: str):
    data = _load_data(__file__, data_name, Path("data", "post"))

    data["cohort_year"] = np.where(data["cohort_year"] == np.Inf, np.nan, data["cohort_year"])
    data["relative_period"] = data["year"] - data["cohort_year"]

    data = data.set_index(['unit', 'year'])

    data = data.loc[lambda x: x["cohort_year"] != x["cohort_year"].max()]

    return data


def _true_effect(data: PandasDF, cohort_column: str, effect_column: str):
    true_effect = (
        data
        .reset_index(drop=True)
        .loc[lambda x: x['relative_period'].notnull()]
        [[cohort_column, "relative_period", effect_column]]
        .drop_duplicates()
        .assign(
            **{cohort_column: lambda x: x[cohort_column].astype(int)},
            relative_period=lambda x: x['relative_period'].astype(int),
            min_period=lambda x: x.groupby(cohort_column)['relative_period'].transform('min')
        )
        .sort_values(["min_period", cohort_column, "relative_period"])
        .assign(
            relative_period=lambda x: x['relative_period'].astype(str),
        )
        .drop(columns=['min_period'])
        .reset_index(drop=True)
    )
    return true_effect
