import pandas as pd
import numpy as np

from pandas import DataFrame


def stack_did_data(data: DataFrame, cohort_name: str):
    cohorts = get_cohorts(data=data, cohort_name=cohort_name)

    stacked_masks = get_stacked_masks(
        data=data,
        cohorts=cohorts,
        cohort_name=cohort_name
    )

    stack_data = []
    for idx, m in enumerate(stacked_masks):
        cdata = data[m].copy()
        cdata['stack'] = idx

        stack_data.append(cdata)

    return pd.concat(stack_data).sort_index()


def cohort_mask(data: DataFrame, cohort_name: str, cohort: int):
    return (data[cohort_name] == cohort).to_numpy()


def never_treated_mask(data: DataFrame, cohort_name: str):
    return data[cohort_name].isnull().to_numpy()


def get_stacked_masks(data: DataFrame, cohorts: list, cohort_name: str):
    ntm = never_treated_mask(data=data, cohort_name=cohort_name)

    stacked_masks = []
    for c in cohorts:
        mask = cohort_mask(data=data, cohort_name=cohort_name, cohort=c) | ntm
        stacked_masks.append(mask)

    return stacked_masks


def get_cohorts(data: DataFrame, cohort_name: str):
    return [i for i in data[cohort_name].unique() if not np.isnan(i)]


def add_stack_fe(stacked_data: DataFrame, fixed_effects: list):
    for fe in fixed_effects:
        stacked_data[f'{fe}_stack'] = stacked_data.groupby([fe, 'stack']).ngroup()

    return stacked_data
