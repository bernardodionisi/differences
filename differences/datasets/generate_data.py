from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series
import pandas as pd

import random

from functools import cached_property, lru_cache

from linearmodels.panel import generate_panel_data

from differences.tools.relative_periods import cohort_info_table, get_relative_periods


def group_random(ser: Series, low=0.0, high=1.0, random_func=np.random.uniform):
    values = ser.unique()

    if random_func is None:
        numbers = np.linspace(start=low, stop=high, num=len(values))
    else:
        numbers = random_func(low=low, high=high, size=len(values))

    vrn = dict(zip(values, numbers))

    # print(vrn)
    return ser.map(vrn)


def data_generating_process(relative_times,
                            tau: float = 1,
                            alpha: float = 0,
                            positive_effect: bool = True,
                            bound_effect: bool = True):
    """data_generating_process(np.arange(-5, 5))"""

    positive_effect = int(positive_effect)
    bound_effect = int(bound_effect)

    dgp_0 = (relative_times > 0)
    if not tau:
        return dgp_0 * positive_effect

    dgp_1 = (relative_times / tau)
    if bound_effect == 1:
        dgp_1[dgp_1 > 1] = 1

    dgp_1 = np.multiply(dgp_0, dgp_1)

    if not alpha:
        return dgp_1 * positive_effect

    dgp_2 = ((relative_times - tau) / tau)
    dgp_2[dgp_2 > 1] = 1

    dgp_2 = dgp_1 - np.multiply((relative_times > tau), dgp_2) * (1 / alpha)

    return dgp_2 * positive_effect


# ----------------------------------------------------------------------

def single_event_per_entity(entities: list,
                            times: list,
                            share_treated: float,
                            n_cohorts: int,
                            cohort_shares: float | list[float] = None,
                            ) -> DataFrame:
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
        raise ValueError('cohort_shares can not sum more than 1')

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

    cohort_data = pd.DataFrame(zip([e for l in te_splits for e in l], cohorts),
                               columns=['entity', 'cohort']).set_index(['entity'])

    return cohort_data


def event_intensities(cohort_data: DataFrame,
                      intensity_by: str | int | tuple = None,
                      low: float | int = 0,
                      high: float | int = 2,
                      random_func=np.random.uniform,
                      samples: int = 0):
    """create the event intensity col in cohort_data"""

    intensity = cohort_data.copy()

    groups = []

    if samples:
        intensity[f'samples'] = np.random.randint(samples, size=len(intensity))
        groups.append('samples')

    if intensity_by == 'entity':
        intensity['intensity'] = np.random.uniform(low=low, high=high, size=len(intensity))

    elif intensity_by == 'cohort':
        groups.append('cohort')

    elif isinstance(intensity_by, int):
        intensity['strata'] = np.random.randint(intensity_by, size=len(intensity))
        groups.append('strata')

    elif isinstance(intensity_by, tuple):

        multiple_strata = [i for i in intensity_by if i != 'cohort']

        for idx, ms in enumerate(multiple_strata):
            if idx == 0:
                idx = ''

            intensity[f'strata{idx}'] = np.random.randint(ms, size=len(intensity))

        groups.extend([i for i in list(intensity) if 'strata' in i])

        if 'cohort' in intensity_by:
            groups.append('cohort')

    else:
        if intensity_by is not None:
            raise ValueError('wrong input for intensity_by')

    intensity['intensity'] = group_random(
        ser=intensity.groupby(groups).ngroup(), low=low, high=high, random_func=random_func)

    return intensity


def event_intensities_old(cohort_data: DataFrame,
                          intensity_by: str | int | tuple = None):
    """create the event intensity col in cohort_data"""

    if intensity_by == 'cohort':

        intensity = (
            cohort_data
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(intensity=lambda x: np.random.uniform(low=0.5, high=1, size=len(x)))
            .set_index(['cohort'])
        )

        intensity = (
            cohort_data
            .replace(intensity.to_dict()['intensity'])
            .rename(columns={'cohort': 'intensity'})
        )

    elif intensity_by == 'entity':
        cohort_data['intensity'] = np.random.uniform(low=0.5, high=1, size=len(cohort_data))
        intensity = cohort_data.drop(columns=['cohort'])


    else:
        if not isinstance(intensity_by, int):
            raise ValueError("'intensity_by' can be either 'cohort', '' or an integer "
                             "representing the number of groups that partition the entities "
                             "in multiple treatments")

        cohort_data = (
            cohort_data
            .assign(strata=lambda x: np.random.randint(intensity_by, size=len(x)))
            .drop(columns=['cohort'])
        )

        intensity = (
            cohort_data
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(intensity=lambda x: np.random.uniform(low=0.5, high=1, size=len(x)))
            .set_index(['strata'])
        )

        intensity = (
            cohort_data
            .replace(intensity.to_dict()['intensity'])
            .rename(columns={'strata': 'intensity'})
        )

    return intensity


# ----------------------------------------------------------------------

def never_treated_to_samples(data: DataFrame, samples: int):
    """inplace operations on panel data to replace NaNs with sample numbers"""

    if not samples:
        raise ValueError('samples must be > 0')

    # never treated entities
    nte = data[lambda x: x['cohort'].isnull()].index.get_level_values('entity').unique()
    numbers = pd.DataFrame(np.random.randint(samples, size=len(nte)), index=nte, columns=['temp'])

    data.reset_index(['time'], inplace=True)

    data['temp'] = numbers

    data['samples'] = np.where(data['samples'].isnull(), data['temp'], data['samples'])
    data['samples'] = data['samples'].astype(int)

    del data['temp']

    data.set_index(['time'], append=True, inplace=True)


def make_datetime(data):
    data = data.reset_index()

    c = {i: str(int(i)) for i in data['cohort'].dropna().unique()}
    data['cohort'] = data['cohort'].map(c)

    data = data.assign(
        time=lambda x: pd.to_datetime(x['time'].astype(str)),
        cohort=lambda x: pd.to_datetime(data['cohort'])
    )

    data = data.set_index(['entity', 'time'])
    return data


# ----------------------------------------------------------------------


class SimulateData:
    def __init__(self,
                 nentity: int = 971,
                 ntime: int = 7,
                 nexog: int = 2,
                 const: bool = False,
                 missing: float = 0,
                 other_effects: int = 2,
                 ncats: int | list[int] = 4,
                 share_treated: float = 0.6,
                 n_cohorts: int = 3,
                 cohort_shares: int | list[float] = None,
                 intensity_by: str | int | tuple[str, int] = None,
                 low=0,
                 high=1,
                 random_func=np.random.uniform,
                 samples: int = 0
                 ):

        self.nentity = nentity
        self.ntime = ntime
        self.nexog = nexog
        self.const = const
        self.missing = missing
        self.other_effects = other_effects
        self.ncats = ncats
        self.share_treated = share_treated
        self.n_cohorts = n_cohorts
        self.cohort_shares = cohort_shares

        # rng: np.random.RandomState | None = None

        self.intensity_by = intensity_by
        self.low = low
        self.high = high
        self.random_func = random_func
        self.samples = samples

    @cached_property
    def panel_data(self):
        panel_data = generate_panel_data(
            nentity=self.nentity,
            ntime=self.ntime,
            nexog=self.nexog,
            const=self.const,
            missing=self.missing,
            other_effects=self.other_effects,
            ncats=self.ncats
        )

        panel_data = pd.concat([panel_data.data,
                                panel_data.weights,
                                panel_data.other_effects], axis=1)

        panel_data.index.names = ['entity', 'time']

        panel_data.reset_index(inplace=True)

        panel_data['time'] = panel_data['time'].dt.year
        panel_data['entity'] = panel_data['entity'].str.replace('firm', 'e')

        panel_data.set_index(['entity', 'time'], inplace=True)

        return panel_data

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
            cohort_shares=self.cohort_shares
        )

    @cached_property
    def cohort_table(self):
        return cohort_info_table(
            cohort_data=self.cohort_data,
            pivot_values='cohort',
            data_index=self.panel_data.index
        )

    def intensity_table(self):

        return cohort_info_table(
            cohort_data=self.event_intensities(),
            pivot_values='intensity',
            data_index=self.panel_data.index)

    def _add_info_to_panel(self):
        info = self.event_intensities()

        for i in list(info):
            col = cohort_info_table(
                cohort_data=info,
                pivot_values=i,
                data_index=self.panel_data.index
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
            samples=self.samples
        )

    @lru_cache(maxsize=1)
    def dgp(self,
            effect_size: float = 1,
            tau: float = 1,
            alpha: float = 0,
            positive_effect: bool = True,
            bound_effect: bool = True,
            ):

        dgp = data_generating_process(
            relative_times=get_relative_periods(self.cohort_table),
            tau=tau,
            alpha=alpha,
            positive_effect=positive_effect,
            bound_effect=bound_effect
        )

        effect = dgp.values

        if self.intensity_by is not None:
            event_intensity = self.intensity_table()
            effect = effect * event_intensity.fillna(1).values

        effect = effect_size * np.sum(effect, axis=1)

        effect = (
            DataFrame(effect, index=dgp.index)
            .reindex(index=self.panel_data.index, fill_value=0)
        )

        self.panel_data['effect'] = effect

        self.panel_data['y'] = (
                self.panel_data['y']
                + self.panel_data['effect']
                + np.random.normal(0, effect_size / 1, size=len(self.panel_data))
        )

        self._add_info_to_panel()

        if self.samples:
            never_treated_to_samples(self.panel_data, self.samples)


def simulate_data(nentity: int = 1000,
                  ntime: int = 8,
                  nexog: int = 1,
                  const: bool = False,
                  missing: float = 0,
                  other_effects: int = 2,
                  ncats: int | list[int] = 1,
                  share_treated: float = 0.8,
                  n_cohorts: int = 3,
                  cohort_shares: int | list[float] = None,
                  effect_size: float = 10,
                  tau: float = 10,
                  alpha: float = 0,
                  positive_effect: bool = True,
                  bound_effect: bool = False,
                  unbalance: bool = False,
                  intensity_by: str | int | tuple[str, int] = 'cohort',
                  low: float | int = 0.5,
                  high: float | int = 10,
                  random_func=None,
                  samples: int = 0,
                  datetime: bool = False
                  ):
    sim_data = SimulateData(
        nentity=nentity,
        ntime=ntime,
        nexog=nexog,
        const=const,
        missing=missing,
        other_effects=other_effects,
        ncats=ncats,
        share_treated=share_treated,
        n_cohorts=n_cohorts,
        cohort_shares=cohort_shares,
        intensity_by=intensity_by,
        low=low,
        high=high,
        random_func=random_func,
        samples=samples
    )

    sim_data.dgp(
        effect_size=effect_size,
        tau=tau,
        alpha=alpha,
        positive_effect=positive_effect,
        bound_effect=bound_effect
    )

    if unbalance:
        return sim_data.panel_data.sample(frac=0.8)

    out_df = sim_data.panel_data

    if datetime:
        out_df = make_datetime(out_df)

    return out_df
