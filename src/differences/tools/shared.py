from __future__ import annotations

from contextlib import contextmanager

import joblib
import numpy as np
from pandas import DataFrame as PandasDF
from scipy.sparse import issparse


# --------------------------- group_by ---------------------------------


def group_sums(x, codes):
    """sourced from: https://github.com/iamlemec/fastreg/blob/master/fastreg/tools.py"""
    if issparse(x):
        _, K = x.shape
        x = x.tocsc()
        C = max(codes) + 1
        idx = [(x.indptr[i], x.indptr[i + 1]) for i in range(K)]
        return np.vstack(
            [
                np.bincount(codes[x.indices[i:j]], weights=x._data[i:j], minlength=C)
                for i, j in idx
            ]
        ).T
    if np.ndim(x) == 1:
        return np.bincount(codes, weights=x)
    else:
        _, K = x.shape
        return np.vstack([np.bincount(codes, weights=x[:, j]) for j in range(K)]).T


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


def parse_att_gt_titles(data: PandasDF | str):
    if isinstance(data, PandasDF):
        # upper column index is the title in the resulting dataframes of the ATTgt class
        title = [c[0] for c in list(data)][0]
    else:
        title = data

    # split title on capital letters
    title = "".join([c if c.islower() else f" {c}" for c in title]).strip()
    title = title.replace("A T T", "ATT")

    return title


def single_idx(df: PandasDF, return_idx_names: bool = True):
    cols = list(df)

    if isinstance(cols[0], tuple):
        df.columns = [c[-1] for c in cols]

    if return_idx_names:
        return df.index.names, df.reset_index()
    else:
        df.reset_index()


def capitalize_details(estimation_details: dict):
    estimation_details = (
        {
            str(k).replace("_", " ").capitalize(): str(v)
            for k, v in estimation_details.items()
        }
        if estimation_details
        else None
    )

    return estimation_details


# -------------------------- general -----------------------------------


def flatten_cols(data):
    cols = ["_".join(map(str, vals)) for vals in data.columns.to_flat_index()]
    data.columns = cols
    return data
