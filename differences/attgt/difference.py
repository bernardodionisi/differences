from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Index

import inspect
from itertools import chain

from collections import namedtuple
from typing import Callable

from scipy.sparse import issparse

from ..attgt.attgt_cal import get_standard_errors
from ..attgt.utility import (get_agg_attr_name,
                             list_fs_ntl,
                             fix_std_error_cols,
                             out_df_index,
                             replace_dates)


class _Difference:
    def __init__(self,
                 alpha: float = 0.05,
                 boot_iterations: int = 0,
                 random_state: int = None,

                 n_jobs_boot: int = 1,
                 backend_boot: str = 'loky',
                 ):

        # if isinstance(diff_ntls, list):
        #     if len(diff_ntls) != 2:
        #         raise ValueError('difference must be between 2 distinct elements, '
        #                          'identified by stratum, sample split')
        # else:
        #     for v in diff_ntls.values():
        #         if len(v) != 2:
        #             raise ValueError('difference must be between 2 distinct elements, '
        #                              'identified by stratum, sample split')

        self.alpha = alpha
        self.boot_iterations = boot_iterations
        self.random_state = random_state
        self.n_jobs_boot = n_jobs_boot
        self.backend_boot = backend_boot

    def _get_standard_errors(self,
                             ntl: list[namedtuple],
                             cluster_groups: np.ndarray = None,
                             ) -> list[namedtuple]:
        return get_standard_errors(
            ntl=ntl,
            alpha=self.alpha,
            cluster_groups=cluster_groups,
            boot_iterations=self.boot_iterations,
            random_state=self.random_state,
            backend_boot=self.backend_boot,
            n_jobs_boot=self.n_jobs_boot
        )

    def get_difference(self,
                       diff_pairs_ntls: dict[str, list] | list,
                       type_of_aggregation: str | None,
                       iterating_samples: bool,  # 'sample' 'stratum'
                       overall: bool = False,
                       sample_masks: list = None,
                       cluster_groups: np.ndarray | dict[str, np.ndarray] = None):

        # res = getattr(self, attr_name, None)
        # if res is None:

        # difference between: "A - B", string with the names
        difference_between = name_difference_between(diff_pairs_ntls=diff_pairs_ntls)

        names = None  # if there are multiple samples/strata
        if isinstance(diff_pairs_ntls, dict):
            # either sample or stratum names
            names = list(diff_pairs_ntls.keys())

        all_pairs_ntls = list_fs_ntl(x=diff_pairs_ntls)

        if names:
            if len(names) != len(all_pairs_ntls):
                raise ValueError('something went wrong in the stratum/samples difference, '
                                 'there are more names than pairs')

        sample_name, strata_name = None, None
        clusters = cluster_groups  # this may be a dict

        res = []
        for n, ntls in enumerate(all_pairs_ntls):

            if names:
                # current iteration name
                if iterating_samples:
                    sample_name = names[n]

                    if cluster_groups is not None:
                        # cluster_groups: should be a dict
                        clusters = cluster_groups[sample_name]

                else:  # iterating strata
                    strata_name = names[n]

            first_ntl, second_ntl = ntls

            res_ntl = get_difference_ntl(
                first_ntl=first_ntl,
                second_ntl=second_ntl,
                sample_masks=sample_masks,
                type_of_aggregation=type_of_aggregation,
                overall=overall,
                difference_between=difference_between,
                sample_name=sample_name,
                strata_name=strata_name
            )

            res_ntl = self._get_standard_errors(
                ntl=res_ntl,
                cluster_groups=clusters
            )

            res.extend(res_ntl)

            # setattr(self, attr_name, res)

        return res


def name_difference_between(diff_pairs_ntls):
    """get names of the two elements to subtract out of list of tuples"""

    if isinstance(diff_pairs_ntls, list):
        first_name, second_name = map(lambda x: x[0], diff_pairs_ntls)

    else:  # dict
        first_name, second_name = map(lambda x: x[0], list(diff_pairs_ntls.values())[0])

    # if isinstance(first_name, tuple):
    #     first_name = ' & '.join(map(str, first_name))
    #     second_name = ' & '.join(map(str, first_name))

    return ' - '.join((str(first_name), str(second_name)))


# ------------------------- differences --------------------------------

def get_difference_ntl(first_ntl: list[namedtuple],
                       second_ntl: list[namedtuple],
                       difference_between: str,
                       sample_name: str,
                       strata_name: str,
                       sample_masks: list[np.ndarray] = None,  # list of two sample masks
                       type_of_aggregation: str = None,
                       overall: bool = False,
                       ) -> list[namedtuple]:
    """set up for difference between two estimates"""

    attr_name = get_agg_attr_name(
        type_of_aggregation=type_of_aggregation,
        overall=overall
    )

    # convert type_of_aggregation to id_fields to be used to find the comparisons
    id_fields = {
        'att_ctg': ['cohort', 'base_period', 'time'],
        'cohort': ['cohort'],
        'event': ['relative_period'],
        'time': ['time'],
        'simple': [],
        'cohort_overall': [],
        'event_overall': [],
        'time_overall': [],
    }

    difference_ntl = get_differences_between(
        first_ntl=first_ntl,
        second_ntl=second_ntl,
        sample_masks=sample_masks,
        id_fields=id_fields[attr_name],
        difference_between=difference_between,
        sample_name=sample_name,
        strata_name=strata_name
    )

    return difference_ntl


def get_differences_between(first_ntl: list[namedtuple],
                            second_ntl: list[namedtuple],
                            difference_between: str,
                            sample_name: str,
                            strata_name: str,
                            id_fields: list[str] = None,
                            sample_masks: list[np.ndarray] = None,
                            ) -> list[namedtuple]:
    """
    finds the namedtuple(s) to subtract & subtracts them

    how does it find what nt to subtract?

        - given the id_fields, for example ['cohort']
        - tuple(getattr(nt, i) for i in id_fields): (1999,)
        - if (1999,) of the first namedtuple == the tuple from the second
        - it's the ATT t o subtract

    Parameters
    ----------
    first_ntl: list[namedtuple]

    second_ntl: list[namedtuple]

    difference_between
    sample_name
    strata_name

    id_fields: list[str] | None, default: ``None``
        - difference in ATT by cohort aggregates: ['cohort']
        - difference in ATT by time aggregates: ['time']
        - difference in ATT by event aggregates: ['relative_period']
        - difference in ATTctg: ['cohort', 'time', 'base_period']

        if id_fields is None for simple & overall aggregates

    sample_masks: list[np.ndarray] | None

    Returns
    -------

    list[namedtuple]

    """
    ntl_output = []

    if id_fields:
        for first_nt in first_ntl:
            if first_nt.ATT is None or np.isnan(first_nt.ATT):
                continue

            first_nt_id = tuple(getattr(first_nt, i) for i in id_fields)

            for second_nt in second_ntl:
                if second_nt.ATT is None or np.isnan(second_nt.ATT):
                    continue

                second_nt_id = tuple(getattr(second_nt, i) for i in id_fields)

                if first_nt_id == second_nt_id:
                    nt = diff_att_if(
                        first_nt=first_nt,
                        second_nt=second_nt,
                        id_fields=id_fields,
                        id_fields_values=first_nt_id,
                        sample_masks=sample_masks,
                        difference_between=difference_between,
                        sample_name=sample_name,
                        strata_name=strata_name
                    )

                    ntl_output.append(nt)

    else:  # simple aggregation + overall aggregations
        nt = diff_att_if(
            first_nt=first_ntl,
            second_nt=second_ntl,
            id_fields=id_fields,
            id_fields_values=(),
            sample_masks=sample_masks,
            difference_between=difference_between,
            sample_name=sample_name,
            strata_name=strata_name
        )

        ntl_output.append(nt)

    return ntl_output


def diff_att_if(first_nt: namedtuple | list,
                second_nt: namedtuple | list,
                difference_between: str,
                sample_name: str,
                strata_name: str,
                id_fields: list,
                id_fields_values: tuple,
                sample_masks: list = None,
                ) -> namedtuple:
    """calculates the difference in estimates and returns namedtuple"""

    # sample idxs to get the right indx to insert the influence function values
    # need to insert in the right spot to cluster

    if isinstance(first_nt, list):
        if len(first_nt) != 1:
            raise ValueError('len of first_nt, when list, must be equal to 1')
        # simple aggregate, overall fr example
        first_nt = first_nt[0]

    if isinstance(second_nt, list):
        if len(second_nt) != 1:
            raise ValueError('len of second_nt, when list, must be equal to 1')
        # simple aggregate, overall fr example
        second_nt = second_nt[0]

    # att
    diff_att = first_nt.ATT - second_nt.ATT

    # influence function

    if sample_masks is not None:  # difference between sample splits
        n = len(sample_masks[0])

        diff_influence_func = np.zeros(shape=(n,))
        first_idxs, second_idxs = get_sample_idxs(sample_masks)

        if issparse(first_nt.influence_func):
            diff_influence_func[first_idxs] = first_nt.influence_func.todense().flatten()
            diff_influence_func[second_idxs] = -1 * second_nt.influence_func.todense().flatten()
        else:
            diff_influence_func[first_idxs] = first_nt.influence_func
            diff_influence_func[second_idxs] = -1 * second_nt.influence_func

        # # todo: check that the above gives the same result as the commented out code
        # if issparse(first_nt.influence_func):
        #     influence_func = vstack([first_nt.influence_func, -1 * second_nt.influence_func])
        # else:
        #     influence_func = np.r_[first_nt.influence_func, -1 * second_nt.influence_func]

    else:  # difference between strata
        if issparse(first_nt.influence_func):
            diff_influence_func = (
                    first_nt.influence_func.todense().flatten() -
                    second_nt.influence_func.todense().flatten()
            )
        else:
            diff_influence_func = (
                    first_nt.influence_func - second_nt.influence_func
            )

    nt = output_difference_namedtuple(
        difference_between,
        *id_fields_values,
        diff_att,
        diff_influence_func,
        insert_fields=id_fields,
        nt_name=type(first_nt).__name__,
        sample_name=sample_name,
        strata_name=strata_name
    )

    return nt


# ------------------------ output --------------------------------------


def output_difference_namedtuple(*args,
                                 nt_name: str,
                                 insert_fields: list,
                                 sample_name: str,
                                 strata_name: str,
                                 ):
    """creates the namedtuple containing the difference in estimates results"""

    # namedtuple needs to be consistent with the other nt,
    # these namedtuple approach will likely be changed, should not affect the API

    if strata_name is not None:
        insert_name, insert_value = ['stratum'], strata_name
    elif sample_name is not None:
        insert_name, insert_value = ['sample_name'], sample_name
    else:
        insert_name, insert_value = [], None

    fields = list(
        chain(
            insert_name,
            ['difference_between'],
            insert_fields,
            ['ATT',
             'influence_func',

             'std_error',
             'lower',
             'upper',
             'boot_iterations'
             ])
    )

    # std_error, lower, upper, boot_iterations
    missing = len(fields) - len(args)

    # fill the last entries with NAs if missing
    if missing:
        missing = 4  # std_error, lower, upper, boot_iterations
        args = *args, *[np.NaN] * missing

    nt = namedtuple(f'Difference{nt_name}', fields)

    if insert_value is not None:
        return nt(insert_value, *args)
    else:
        return nt(*args)


def difference_ntl_to_dataframe(ntl: list,
                                date_map: dict = None) -> DataFrame:
    """difference results to a dataframe"""

    name_table = type(ntl[0]).__name__
    boot_iterations = getattr(ntl[0], 'boot_iterations')

    if boot_iterations:
        se_info, conf_info = 'bootstrap', 'simult. conf. band'
    else:
        se_info, conf_info = 'analytic', 'pointwise conf. band'

    exclude_fields = [
        'influence_func',
        'boot_iterations'
    ]

    include_fields = [f for f in ntl[0]._fields
                      if f not in exclude_fields]

    values = map(lambda x: (getattr(x, v) for v in include_fields), ntl)
    out_df = DataFrame(values, columns=include_fields)

    out_df = fix_std_error_cols(out_df=out_df)

    output = replace_dates(out_df=out_df, date_map=date_map)

    out_df = out_df_index(
        out_df=out_df,
        name_table=name_table,
        se_info=se_info,
        conf_info=conf_info
    )
    return out_df


# ------------------ pre-processing helpers ----------------------------


def parse_split_sample(data: DataFrame,
                       split_sample_by: Callable | str = None,
                       ):
    """finds the masks for the data given split_sample_by

    returns a dict {'name of sample': data mask for sample}"""

    if split_sample_by is None:
        return None

    elif isinstance(split_sample_by, str):
        return {f"{split_sample_by} = {i}": {'sample_mask': np.array(data[split_sample_by] == i)}
                for i in data[split_sample_by].dropna().unique()
                }

    elif isinstance(split_sample_by, Callable):

        source_code = inspect.getsource(split_sample_by)

        # make the string look nicer if lambda is passed
        if 'lambda' in source_code:
            source_code = source_code.split('split_sample_by=')[-1]
            if source_code.count(':') == 1:
                source_code = source_code.split(':')[-1]

        source_code = source_code.strip()
        if source_code.endswith(','):
            source_code = source_code[:-1]

        mask = np.array(split_sample_by(data))

        return {source_code: {'sample_mask': mask}, f'NOT-{source_code}': {'sample_mask': ~mask}}

    else:
        raise ValueError("invalid 'split_sample_by'")


def preprocess_difference(difference: bool | list,
                          sample_names: list,
                          strata: list | None
                          ) -> dict[str, list] | tuple:  # strata, samples
    difference_between = {}

    if sample_names is None or sample_names == ['full_sample']:
        sample_names = []

    if strata is None:
        strata = []

    if not sample_names and not strata:
        raise ValueError('in order to calculate the difference the estimation must be run on a '
                         'split sample and/or for different treatment strata')

    if isinstance(difference, bool):
        if min(len(sample_names), len(strata)) != 2:
            raise ValueError('difference must be between two elements, provide the names '
                             'of the two samples or strata to subtract in a list')
        elif len(sample_names) == 2 and len(strata) == 2:
            raise ValueError('unable to determine the dimension to subtract '
                             'between samples and strata, please dictionary')

        # if the min len is 2 then subtract that for each of the other

        if len(strata) == 2:

            difference_between = {
                'difference_strata': strata,
                'iterate_samples': sample_names
            }

        elif len(sample_names) == 2:

            difference_between = {
                'difference_samples': sample_names,
                'iterate_strata': strata
            }

        return difference_between

    elif isinstance(difference, list):
        if len(difference) != 2:
            raise ValueError('difference must be between two elements')

        sample_names_diff = [s for s in difference if s in sample_names]
        strata_diff = [d for d in difference if d in strata]

        if strata_diff and sample_names_diff:
            raise ValueError("not able to distinguish between sample names and strata names, "
                             "provide "
                             "the list of sample names and/or stratum names in a dictionary as "
                             "{'sample_names': [sample names], strata: [stratum names]}")

        elif strata_diff and not sample_names_diff:

            if len(strata_diff) == 2:
                difference_between = {
                    'difference_strata': strata_diff,
                    'iterate_samples': sample_names
                }
            else:
                raise ValueError('difference must be between two elements, provide the names '
                                 'of the two strata to subtract in a list')

            return difference_between

        elif not strata_diff and sample_names_diff:

            if len(sample_names_diff) == 2:
                difference_between = {
                    'difference_samples': sample_names_diff,
                    'iterate_strata': strata
                }
            else:
                raise ValueError('difference must be between two elements, provide the names '
                                 'of the two samples to subtract in a list')

            return difference_between

    elif isinstance(difference, dict):

        strata_diff, sample_names_diff = difference.get('strata'), difference.get(
            'sample_names')

        if not strata_diff and not sample_names_diff:
            raise ValueError("dictionary must be of the format: "
                             "{'sample_names': [sample names]} or "
                             "{'strata': [stratum names]} or"
                             "{'sample_names': [sample names], 'strata': [stratum names]}")

        elif strata_diff and not sample_names_diff:
            strata_diff = [d for d in strata_diff if d in strata]

            if len(strata_diff) == 2:
                difference_between = {
                    'difference_strata': strata_diff,
                    'iterate_samples': sample_names
                }
            else:
                raise ValueError('difference must be between two elements, provide the names '
                                 'of the two strata to subtract in a list')

            return difference_between

        elif not strata_diff and sample_names_diff:
            sample_names_diff = [s for s in sample_names_diff if s in sample_names]

            if len(sample_names_diff) == 2:
                difference_between = {'difference_samples': sample_names_diff,
                                      'iterate_strata': strata}
            else:
                raise ValueError('difference must be between two elements, provide the names '
                                 'of the two samples to subtract in a list')

            return difference_between

        else:
            if len(strata_diff) != 2 or len(sample_names_diff) != 2:
                raise ValueError("if a dictionary with two dictionary must be of the format: "
                                 "{'sample_names': [2 sample names], strata: [2 stratum names]}"
                                 "and the difference will be taken between "
                                 "(sample 0, stratum 0) - (sample 1, stratum 1)")

            # (sample 0, stratum 0), (sample 1, stratum 1)
            return tuple(zip(sample_names_diff, strata_diff))

    else:
        raise ValueError("invalid 'difference'")


# ----------------------- helpers --------------------------------------

def resize_sample_masks(sample_masks: list
                        ) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """helper function to resize masks when the data split > 2

    in case the two samples do not make up the full sample

    this function is used to find the correct location to insert
    the values of the influence function in order to do clustering

    the data_mask will help filter the clusters from the full dataset

    first_mask, second_mask will give the
    correct locations for each sample
    """

    if len(sample_masks) != 2:
        raise ValueError('only 2 masks allowed')

    first_mask, second_mask = sample_masks

    data_mask = first_mask | second_mask

    # whether the 2 samples make up the full sample
    if not np.alltrue(data_mask):
        second_mask_int = second_mask.astype(int)
        second_mask_int[second_mask] = 2

        samples = first_mask + second_mask_int
        resized_array = samples[samples != 0]

        first_mask, second_mask = resized_array == 1, resized_array == 2

        return data_mask, [first_mask, second_mask]

    return None, sample_masks


def get_ds_masks(result_dict: dict,
                 difference: list,
                 entity_index: Index,
                 cluster_by_entity: bool
                 ) -> tuple[None | np.ndarray, list]:
    """
    get data mask + sample masks [2 masks only]

    Parameters
    ----------
    result_dict
    difference
    entity_index
    cluster_by_entity

    Returns
    -------

    """
    # boolean masks for each (of the 2) samples. same order as difference list
    # data mask is None if the 2 samples make up the full data
    data_mask, sample_masks = resize_sample_masks(
        sample_masks=[result_dict[n]['sample_mask'] for n in difference]
    )

    if data_mask is not None:
        entity_index = entity_index[data_mask]

    # entity level sample masks
    sample_masks = get_masks_for_difference(
        entity_index=entity_index,
        entity_name=entity_index.name,
        first_mask=sample_masks[0],
        entity_level=cluster_by_entity,
    )

    return data_mask, sample_masks


def get_masks_for_difference(entity_index: pd.Index,
                             entity_name: str,
                             first_mask: np.ndarray,
                             entity_level: bool = False
                             ) -> list[np.ndarray, np.ndarray]:
    if not entity_level:
        return [first_mask, ~first_mask]

    entity_mask = (
        DataFrame(first_mask, index=entity_index)
        .groupby(entity_name)
        .first()
        .to_numpy()
        .flatten()
    )

    return [entity_mask, ~entity_mask]


def get_sample_idxs(sample_masks: list
                    ) -> list[np.ndarray, np.ndarray]:
    """finds the indexes where the masks are True"""
    if len(sample_masks) != 2:
        raise ValueError('only 2 masks allowed')

    return [np.where(sample_masks[0])[0], np.where(sample_masks[1])[0]]
