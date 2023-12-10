from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl
import numpy as np

from random import randint

from pandas import DataFrame as PandasDF


def generate_panel(
        n_entities=1_000,
        start_year=1950,
        end_year=1980,

        panel: Literal["balanced", "unbalanced-start", "unbalanced-end"] = "balanced"

):
    if panel == "balanced":
        times = [np.arange(start_year, end_year + 1) for i in range(n_entities)]

    elif panel == "unbalanced-start":
        times = [
            np.arange(start_year, randint(start_year + 1, end_year + 1))
            for i in range(n_entities)
        ]
    elif panel == "unbalanced-end":
        times = [
            np.arange(randint(start_year, end_year - 1), end_year + 1)
            for i in range(n_entities)
        ]
    else:
        raise ValueError

    rng = np.random.RandomState(
        [
            0xA14E2429,
            0x448D2E51,
            0x91B558E7,
            0x6A3F5CD2,
            0x22B43ABB,
            0xE746C92D,
            0xCE691A7D,
            0x66746EE7,
        ]
    )

    entity_fe = [rng.normal(loc=randint(0, 100_000), size=len(t)) for t in times]

    data = pl.DataFrame({
        "entity": np.arange(n_entities),
        "time": times,
        "entity_fe": entity_fe
    }).explode(["time", "entity_fe"])

    data = (
        data
        .with_columns([
            pl.col("time").map_dict({
                y: rng.normal(loc=randint(-200, 200), size=1)[0]
                for y in data["time"].unique()}
            ).alias("time_fe"),
            pl.lit(rng.standard_normal(size=len(data))).alias("error"),
            pl.lit(rng.normal(loc=5, size=len(data))).alias("x")
        ])
        .with_columns(
            y=11 * pl.col("x") + pl.col("entity_fe") + pl.col("time_fe") + pl.col("error")
        )
    )

    return data.to_pandas().set_index(["entity", "time"])


# ----------------------------------------------------------------------


def _load_data(module: str, file_name: str, sub_path: str | Path = "data"):
    if isinstance(file_name, str):
        file_name = Path(file_name)

    if isinstance(module, str):
        module = Path(module)

    file_path = module.parent / sub_path / file_name

    if file_name.suffix.endswith("csv"):
        return pd.read_csv(file_path)
    elif file_name.suffix.endswith("parquet"):
        return pd.read_parquet(file_path)


class Dataset:
    """
    A class to represent a dataset containing various columns.

    Group
    -----
        dataset

    """

    def __init__(
            self,
            data: PandasDF,
            data_name: str,
            entity_column: str,
            time_column: str,
            y_column: str,
            cohort_column: str | None,
            cohort_data: PandasDF = None,
            x_formula: str | None = None,
            formula: str | None = None,
            post_column: str | None = None,
            treat_column: str | None = None,
            weights_column: str | None = None,
            is_panel: bool | None = None,
            true_effect: PandasDF | None = None,
    ):
        """

        Properties
        ----------
        data : pd.DataFrame
            The main dataset containing the entire data.
        entity_column : str
            The column name representing the entity.
        time_column : str
            The column name representing the time.
        y_column : str
            The column name representing the dependent variable.
        cohort_column : str, optional
            The column name representing the cohort. If not relevant, set to None. Default is None.
        cohort_data : pd.DataFrame, optional
            Additional dataset containing cohort data. Default is None.
        x_formula
            Possible controls
        post_column : str, optional
            The column name representing the post treatment period. Default is None.
        treat_column : str, optional
            The column name representing the treatment variable. Default is None.
        weights_column : str, optional
            The column name representing the weights for each observation. Default is None.
        true_effect : pd.DataFrame, optional
            The dataset containing the true effect of the treatment. Default is None.
        """

        self._data = data
        self.data_name = data_name

        self.entity_column = entity_column
        self.time_column = time_column
        self.y_column = y_column
        self.x_formula = x_formula
        self.formula = formula

        self.cohort_column = cohort_column
        self._cohort_data = cohort_data
        self.post_column = post_column
        self.treat_column = treat_column
        self.weights_column = weights_column
        self.is_panel = is_panel

        self.true_effect = true_effect

    def data(
            self,
            fill_cohort_column: int = None,
            reset_index: bool = False,
            to_polars: bool = False,
    ):
        data = self._data.copy()

        if fill_cohort_column is not None:
            data = data.assign(
                **{
                    self.cohort_column:
                        lambda x: x[self.cohort_column].fillna(fill_cohort_column).astype(int)}
            )

        if to_polars:
            return pl.from_pandas(data.reset_index())

        if reset_index:
            data = data.reset_index()

        return data

    def cohort_data(self, to_polars: bool = False):
        if to_polars:
            return pl.from_pandas(self._cohort_data)
        return self._cohort_data

# ----------------------------------------------------------------------

#
# def simulated_data(
#         py_format: bool = True,
#         single_event: bool = True,
#         separate_cohort_data: bool = False,
#         seed: int = 4321,
# ):
#     if single_event:
#         data_name = "se"
#     else:
#         data_name = "me"
#     data = _load_data(__file__, f"{data_name}.csv")
#
#     np.random.seed(seed)
#     data["random_weights"] = np.random.uniform(low=0, high=1, size=len(data))
#
#     data = data.assign(
#         T=lambda x: to_datetime(x["T"]),
#         cohort=lambda x: to_datetime(x["cohort"]),
#     )
#
#     cohorts = (
#         data[["cohort", "event_size"]].dropna().reset_index(drop=True).drop_duplicates()
#     )
#     cohorts["qevent_size"] = qcut(cohorts["event_size"], 3, labels=False)
#
#     data = data.merge(cohorts[["cohort", "qevent_size"]], on=["cohort"], how="left")
#
#     if py_format:
#         data.set_index(["I", "T"], inplace=True)
#
#     data_dict = {
#         "data": data,
#         "entity_name": "I",
#         "time_name": "T",
#         "cohort_name": "cohort",
#         "y_name": "y",
#     }
#
#     if separate_cohort_data:
#         cohort_data = (
#             data.reset_index()[["I", "cohort", "event_size", "qevent_size"]]
#             .dropna()
#             .reset_index(drop=True)
#         )
#         data_dict.update({"cohort_data": cohort_data})
#
#         del data["cohort"], data["event_size"]
#
#     return data_dict
#
#
# def multiple_events():
#     data = _load_data(__file__, f"sim-me.csv")
#
#     data = Dataset(
#         data=data,
#         entity_column='I',
#         time_column='T',
#         y_column='y',
#         cohort_column=None,
#         cohort_data=None
#     )
#     return data
#
#
# def single_event():
#     data = _load_data(__file__, f"sim-se.csv")
#
#     data = Dataset(
#         data=data,
#         entity_column='I',
#         time_column='T',
#         y_column='y',
#         cohort_column=None,
#         cohort_data=None
#     )
#     return data


# ----------------------------------------------------------------------
