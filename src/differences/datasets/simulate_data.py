from __future__ import annotations

import random
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd
from linearmodels.panel import generate_panel_data
from pandas import DataFrame as PandasDF

from differences.models.tools.relative_periods import cohort_info_table, get_relative_periods
from differences.models.did.attgt.attgt import replace_string_index_with_category_code
from differences.datasets.utility import generate_panel


def simulate_data(
        n_entities: int = 1000,
        n_times: int = 8,
        n_exog: int = 1,
        n_cohorts: int = 3,

        *,

        effect_size: float = 10,
        share_treated: float = 0.8,
        cohort_shares: int | list[float] = None,

        const: bool = False,
        missing: float = 0,
        n_cats: int | list[int] = 3,
        other_effects: int = 2,

        tau: float = -1,
        alpha: float = 1,
        reverse_effect: bool = False,
        bound_effect: bool = False,

        unbalance: bool = False,
        intensity_by: str | int | tuple[str, int] = "cohort",
        low: float | int = 0.5,
        high: float | int = 10,
        random_func=None,
        samples: int = 0,
        datetime: bool = False,
        eid_as_int: bool = True,
        lm: bool = True
):
    sim_data = SimulateData(
        n_entities=n_entities,
        n_times=n_times,
        n_exog=n_exog,
        const=const,
        missing=missing,
        other_effects=other_effects,
        n_cats=n_cats,
        share_treated=share_treated,
        n_cohorts=n_cohorts,
        cohort_shares=cohort_shares,
        intensity_by=intensity_by,
        low=low,
        high=high,
        random_func=random_func,
        samples=samples,
        lm=lm
    )

    sim_data.dgp(
        effect_size=effect_size,
        tau=tau,
        alpha=alpha,
        reverse_effect=reverse_effect,
        bound_effect=bound_effect,
    )

    if unbalance:
        return sim_data.panel_data.sample(frac=0.8)

    out_df = sim_data.panel_data

    if datetime:
        out_df = make_datetime(out_df)

    if eid_as_int:
        out_df = replace_string_index_with_category_code(out_df)

    return out_df


# ----------------------------------------------------------------------

def data_generating_process(
        relative_times,
        tau: float,
        alpha: float,
        reverse_effect: bool,
        bound_effect: bool,
):
    """
    Generate data following a specific process based on the input parameters.

    This function uses relative times and various parameters to generate an output
    according to a defined data generating process. Depending on the inputs,
    different mechanisms can be applied to obtain the desired effect.

    Parameters
    ----------
    relative_times : np.ndarray
        Array representing the relative times for the process.
    tau : float, optional
        Parameter for scaling the effect, default is 1.
    alpha : float, optional
        Parameter to adjust the effect after time tau, default is 0.
    reverse_effect : bool, optional
    bound_effect : bool, optional
        If True, the effect will be bounded to 1, otherwise unbounded. Default is True.

    Returns
    -------
    np.ndarray
        The resulting array after applying the data generating process.

    Example
    -------
    >>> data_generating_process(np.arange(-5, 5))
    """

    reverse_effect = -1 if reverse_effect else 1

    dgp_0 = relative_times > 0

    if not tau:
        return dgp_0 * reverse_effect

    dgp_1 = relative_times / tau

    if bound_effect:
        dgp_1[dgp_1 > 1] = 1

    dgp_1 = np.multiply(dgp_0, dgp_1)

    if not alpha:
        return dgp_1 * reverse_effect

    dgp_2 = (relative_times - tau) / tau

    dgp_2[dgp_2 > 1] = 1

    dgp_2 = dgp_1 - np.multiply((relative_times > tau), dgp_2) * (1 / alpha)

    return dgp_2 * reverse_effect


def group_random(ser: pd.Series, low=0.0, high=1.0, random_func=np.random.uniform):
    values = ser.unique()

    if random_func is None:
        numbers = np.linspace(start=low, stop=high, num=len(values))
    else:
        numbers = random_func(low=low, high=high, size=len(values))

    vrn = dict(zip(values, numbers))

    # print(vrn)
    return ser.map(vrn)


# ----------------------------------------------------------------------


def single_event_per_entity(
        entities: list,
        times: list,
        share_treated: float,
        n_cohorts: int,
        cohort_shares: float | list[float] = None,
) -> PandasDF:
    """creates the cohort_data: a dataframe with cohort dates for each datafreme

    if cohort_shares is a list then n cohorts = len(cohort_shares)

    """

    tentities = random.sample(list(entities), int(share_treated * len(entities)))

    if cohort_shares is None:
        cohort_shares = list(np.ones(n_cohorts) / n_cohorts)
    else:
        if not isinstance(cohort_shares, list):
            cohort_shares = [cohort_shares]
        n_cohorts = len(cohort_shares)

    if sum(cohort_shares) > 1:
        raise ValueError("cohort_shares can not sum more than 1")

    sizes = len(tentities) * np.array(cohort_shares)

    te_to_sample = tentities.copy()
    te_splits = []
    for i, s in enumerate(sizes):

        if i == len(sizes) - 1:
            te_splits.append(te_to_sample)
        else:

            sample = np.random.choice(te_to_sample, size=int(s), replace=False)
            te_splits.append(list(sample))

            te_to_sample = [e for e in te_to_sample if e not in sample]

    cohorts = random.sample(list(times), n_cohorts)
    cohorts = np.repeat(cohorts, repeats=list(map(len, te_splits)))

    cohort_data = PandasDF(
        zip([e for l in te_splits for e in l], cohorts), columns=["entity", "cohort"]
    ).set_index(["entity"])

    return cohort_data


def event_intensities(
        cohort_data: PandasDF,
        intensity_by: str | int | tuple = None,
        low: float | int = 0,
        high: float | int = 2,
        random_func=np.random.uniform,
        samples: int = 0,
):
    """create the event intensity col in cohort_data"""

    intensity = cohort_data.copy()

    groups = []

    if samples:
        intensity["samples"] = np.random.randint(samples, size=len(intensity))
        groups.append("samples")

    if intensity_by == "entity":
        intensity["intensity"] = np.random.uniform(
            low=low, high=high, size=len(intensity)
        )

    elif intensity_by == "cohort":
        groups.append("cohort")

    elif isinstance(intensity_by, int):
        intensity["strata"] = np.random.randint(intensity_by, size=len(intensity))
        groups.append("strata")

    elif isinstance(intensity_by, tuple):

        multiple_strata = [i for i in intensity_by if i != "cohort"]

        for idx, ms in enumerate(multiple_strata):
            if idx == 0:
                idx = ""

            intensity[f"strata{idx}"] = np.random.randint(ms, size=len(intensity))

        groups.extend([i for i in list(intensity) if "strata" in i])

        if "cohort" in intensity_by:
            groups.append("cohort")

    else:
        if intensity_by is not None:
            raise ValueError("wrong input for intensity_by")

    intensity["intensity"] = group_random(
        ser=intensity.groupby(groups).ngroup(),
        low=low,
        high=high,
        random_func=random_func,
    )

    return intensity


def event_intensities_old(
        cohort_data: PandasDF, intensity_by: str | int | tuple = None
):
    """create the event intensity col in cohort_data"""

    if intensity_by == "cohort":

        intensity = (
            cohort_data.drop_duplicates()
            .reset_index(drop=True)
            .assign(intensity=lambda x: np.random.uniform(low=0.5, high=1, size=len(x)))
            .set_index(["cohort"])
        )

        intensity = cohort_data.replace(intensity._parameters_for_panel_es()["intensity"]).rename(
            columns={"cohort": "intensity"}
        )

    elif intensity_by == "entity":
        cohort_data["intensity"] = np.random.uniform(
            low=0.5, high=1, size=len(cohort_data)
        )
        intensity = cohort_data.drop(columns=["cohort"])

    else:
        if not isinstance(intensity_by, int):
            raise ValueError(
                "'intensity_by' can be either 'cohort', '' or an integer "
                "representing the number of groups that partition the entities "
                "in multiple treatments"
            )

        cohort_data = cohort_data.assign(
            strata=lambda x: np.random.randint(intensity_by, size=len(x))
        ).drop(columns=["cohort"])

        intensity = (
            cohort_data.drop_duplicates()
            .reset_index(drop=True)
            .assign(intensity=lambda x: np.random.uniform(low=0.5, high=1, size=len(x)))
            .set_index(["strata"])
        )

        intensity = cohort_data.replace(intensity._parameters_for_panel_es()["intensity"]).rename(
            columns={"strata": "intensity"}
        )

    return intensity


# ----------------------------------------------------------------------


def never_treated_to_samples(data: PandasDF, samples: int):
    """inplace operations on panel data to replace NaNs with sample numbers"""

    if not samples:
        raise ValueError("samples must be > 0")

    # never treated entities
    nte = data[lambda x: x["cohort"].isnull()].index.get_level_values("entity").unique()
    numbers = PandasDF(
        np.random.randint(samples, size=len(nte)), index=nte, columns=["temp"]
    )

    data.reset_index(["time"], inplace=True)

    data["temp"] = numbers

    data["samples"] = np.where(data["samples"].isnull(), data["temp"], data["samples"])
    data["samples"] = data["samples"].astype(int)

    del data["temp"]

    data.set_index(["time"], append=True, inplace=True)


def make_datetime(data):
    data = data.reset_index()

    c = {i: str(int(i)) for i in data["cohort"].dropna().unique()}
    data["cohort"] = data["cohort"].map(c)

    data = data.assign(
        time=lambda x: pd.to_datetime(x["time"].astype(str)),
        cohort=lambda x: pd.to_datetime(data["cohort"]),
    )

    data = data.set_index(["entity", "time"])
    return data


# ----------------------------------------------------------------------


class SimulateData:
    def __init__(
            self,
            n_entities: int = 971,
            n_times: int = 7,
            n_exog: int = 2,
            n_cats: int | list[int] = 4,
            n_cohorts: int = 3,

            const: bool = False,
            missing: float = 0,
            other_effects: int = 2,
            share_treated: float = 0.6,
            cohort_shares: int | list[float] = None,
            intensity_by: str | int | tuple[str, int] = None,
            low=0,
            high=1,
            random_func=np.random.uniform,
            samples: int = 0,
            lm: bool = False
    ):

        self.nentity = n_entities
        self.ntime = n_times
        self.nexog = n_exog
        self.const = const
        self.missing = missing
        self.other_effects = other_effects
        self.ncats = n_cats
        self.share_treated = share_treated
        self.n_cohorts = n_cohorts
        self.cohort_shares = cohort_shares

        # rng: np.random.RandomState | None = None

        self.intensity_by = intensity_by
        self.low = low
        self.high = high
        self.random_func = random_func
        self.samples = samples

        self.lm = lm

    @cached_property
    def panel_data(self):

        if self.lm:
            data = self.lm_panel_data()
        else:
            data = self.not_lm_panel_data()

        return data

    def lm_panel_data(self):

        data = generate_panel_data(
            nentity=self.nentity,
            ntime=self.ntime,
            nexog=self.nexog,
            const=self.const,
            missing=self.missing,
            other_effects=self.other_effects,
            ncats=self.ncats,
        )

        data = pd.concat([data.data, data.weights, data.other_effects], axis=1)

        data.index.names = ["entity", "time"]

        data.reset_index(inplace=True)

        data["time"] = data["time"].dt.year
        data["entity"] = data["entity"].str.replace("firm", "e")

        return data.set_index(["entity", "time"])

    def not_lm_panel_data(self):
        return generate_panel()

    @cached_property
    def entities(self):
        return list(self.panel_data.index.get_level_values(0).unique())

    @cached_property
    def times(self):
        return list(self.panel_data.index.get_level_values(1).unique())

    @cached_property
    def cohort_data(self):
        # todo: add multiple events per entity

        return single_event_per_entity(
            entities=self.entities,
            times=self.times,
            share_treated=self.share_treated,
            n_cohorts=self.n_cohorts,
            cohort_shares=self.cohort_shares,
        )

    @cached_property
    def cohort_table(self):
        return cohort_info_table(
            cohort_data=self.cohort_data,
            pivot_values="cohort",
            data_index=self.panel_data.index,
        )

    def intensity_table(self):

        return cohort_info_table(
            cohort_data=self.event_intensities(),
            pivot_values="intensity",
            data_index=self.panel_data.index,
        )

    def _add_info_to_panel(self):
        info = self.event_intensities()

        for i in list(info):
            col = cohort_info_table(
                cohort_data=info, pivot_values=i, data_index=self.panel_data.index
            )

            if len(list(col)) == 1:
                col = col.rename(columns={1: i})
                self.panel_data[i] = col

    @lru_cache(maxsize=3)
    def event_intensities(self):

        return event_intensities(
            cohort_data=self.cohort_data,
            intensity_by=self.intensity_by,
            low=self.low,
            high=self.high,
            random_func=self.random_func,
            samples=self.samples,
        )

    @lru_cache(maxsize=1)
    def dgp(
            self,
            effect_size: float,
            tau: float,
            alpha: float,
            reverse_effect: bool,
            bound_effect: bool,
    ):

        dgp = data_generating_process(
            relative_times=get_relative_periods(self.cohort_table),
            tau=tau,
            alpha=alpha,
            reverse_effect=reverse_effect,
            bound_effect=bound_effect,
        )

        effect = dgp.values

        if self.intensity_by is not None:
            event_intensity = self.intensity_table()
            effect = effect * event_intensity.fillna(1).values

        effect = effect_size * np.sum(effect, axis=1)

        effect = PandasDF(effect, index=dgp.index).reindex(
            index=self.panel_data.index, fill_value=0
        )

        self.panel_data["effect"] = effect

        self.panel_data["y"] = (
                self.panel_data["y"]
                + self.panel_data["effect"]
                + np.random.normal(0, effect_size / 1, size=len(self.panel_data))
        )

        self._add_info_to_panel()

        if self.samples:
            never_treated_to_samples(self.panel_data, self.samples)
