from __future__ import annotations

import numpy as np
import joblib

from pandas import DataFrame
from typing import NamedTuple

from scipy.sparse import issparse
from contextlib import contextmanager


# ------------------------- formulaic ----------------------------------

# useful issues:
# https://github.com/matthewwardrop/formulaic/issues/32
# https://github.com/matthewwardrop/formulaic/issues/60

def parse_fe_from_formula(formula) -> tuple[str, list]:
    """
    basic extraction of fixed effects from formula

    formula: 'y ~ x + z | fe1 + fe2'
    """
    try:
        formula, fe = formula.split('|')
    except ValueError:
        return formula, []

    fe = [i.strip() for i in fe.split('+') if i.strip()]

    return formula, fe


def process_formula(formula: str,
                    entity_name: str,
                    time_name: str,
                    stacked: bool = False,
                    return_fe: bool = False):
    if '~' not in formula:
        formula = f'{formula} ~ 1'

    try:
        spec, fe = formula.split('|')

        if not fe.strip():  # not ''
            fe = []
            formula = spec.strip()
        else:
            fe = [i.strip() for i in fe.split('+') if i.strip()]

    except ValueError:  # if no | then use two-way fe

        fe = [entity_name, time_name]

        if stacked:
            formula = f'{formula} | {entity_name}_stack + {time_name}_stack'
        else:
            formula = f'{formula} | {entity_name} + {time_name}'

        # to use fe, then just specify a formula with |

    if return_fe:
        return formula, fe

    return formula


def parse_fe_from_formula(formula: str) -> tuple[str, list]:
    """
    basic extraction of fixed effects from formula

    formula: 'y ~ x + z | fe1 + fe2'
    """
    try:
        formula, fe = formula.split('|')
    except ValueError:
        return formula, []

    fe = [i.strip() for i in fe.split('+') if i.strip()]

    return formula, fe


# --------------------------- groupby ----------------------------------

def group_sums(x, codes):
    """sourced from: https://github.com/iamlemec/fastreg/blob/master/fastreg/tools.py"""
    if issparse(x):
        _, K = x.shape
        x = x.tocsc()
        C = max(codes) + 1
        idx = [(x.indptr[i], x.indptr[i + 1]) for i in range(K)]
        return np.vstack([
            np.bincount(
                codes[x.indices[i:j]], weights=x.data[i:j], minlength=C
            ) for i, j in idx
        ]).T
    if np.ndim(x) == 1:
        return np.bincount(codes, weights=x)
    else:
        _, K = x.shape
        return np.vstack([
            np.bincount(codes, weights=x[:, j]) for j in range(K)
        ]).T


# sparsity handled by group_sums
def group_means(x, codes):
    """sourced from: https://github.com/iamlemec/fastreg/blob/master/fastreg/tools.py"""
    if np.ndim(x) == 1:
        return group_sums(x, codes) / np.bincount(codes)
    else:
        return group_sums(x, codes) / np.bincount(codes)[:, None]


# ----------------------------------------------------------------------


@contextmanager
def tqdm_joblib(tqdm_object):
    """context manager to patch joblib to report into tqdm progress bar given as argument

    sourced from: https://stackoverflow.com/a/58936697/5133167
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# --------------------------- plot helpers -----------------------------

def get_title(df):
    title = [c[0] for c in list(df)][0]
    title = ''.join([c if c.islower() else f' {c}' for c in title]).strip()
    title = title.replace('A T T', 'ATT')

    return title


def single_idx(df: DataFrame, return_idx_names: bool = True):
    cols = list(df)

    if isinstance(cols[0], tuple):
        df.columns = [c[-1] for c in cols]

    if return_idx_names:
        return df.index.names, df.reset_index()
    else:
        df.reset_index()


def capitalize_details(estimation_details: dict):
    estimation_details = {
        str(k).replace('_', ' ').capitalize(): str(v)
        for k, v in estimation_details.items()
    } if estimation_details else None

    return estimation_details


# --------------------- twfe + stacked helpers -------------------------


def bin_start_end(bin_endpoints: bool):
    bin_start, bin_end = False, False
    if bin_endpoints:
        if bin_endpoints == 'start':
            bin_start = True
        elif bin_endpoints == 'end':
            bin_end = True
        else:
            bin_start, bin_end = True, True

    return bin_start, bin_end


class EventStudyResult(NamedTuple):
    event_study_est: DataFrame
    covariate_estimates: DataFrame
    formula: str
    weights_name: str | None
    cluster_names: list | str | None
    use_intensity: bool
    bin_endpoints: bool | str
    # r2: float


# -------------------------- general -----------------------------------

def flatten_cols(data):
    cols = ['_'.join(map(str, vals))
            for vals in data.columns.to_flat_index()]
    data.columns = cols
    return data
