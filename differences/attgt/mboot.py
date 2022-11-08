from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series

from scipy.stats import norm
from scipy.sparse import issparse, csr_matrix

from tqdm import tqdm

from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import get_reusable_executor

from ..tools.panel_utility import find_time_varying_covars
from ..tools.utility import group_means


def mboot(inf_funcs: np.ndarray,
          cluster_groups: np.ndarray = None,
          alpha: float = 0.05,
          boot_iterations: int = 1000,
          random_state: int = None,
          boot_backend: str = 'loky',
          boot_n_jobs: int = -1,

          progress_bar: bool = True,
          sample_name: str = None,
          release_workers: bool = True
          ):
    # n: only entity cluster, overwritten if other cluster var present
    n_clusters = inf_funcs.shape[0]

    if cluster_groups is not None:  # if there is a cluster var other than entity
        inf_funcs = mean_inf_func_by_cluster(
            cluster_groups=cluster_groups,
            inf_funcs=inf_funcs)

        n_clusters = np.max(cluster_groups) + 1  # nunique clusters

    mb_res = np.sqrt(n_clusters) * multiplier_bootstrap_joblib(
        inf_funcs=inf_funcs,
        iterations=boot_iterations,
        random_state=random_state,
        backend=boot_backend,
        n_jobs=boot_n_jobs,
        progress_bar=progress_bar,
        sample_name=sample_name,
        release_workers=release_workers
    )

    crit_val, b_sigma = boot_crit_val(mb_res=mb_res, alpha=alpha)

    # bootstrap variance matrix
    # (this matrix can be defective because of degenerate cases)
    var_bres = np.cov(mb_res, rowvar=False)

    return {'mb_res': mb_res,
            'se': b_sigma / np.sqrt(n_clusters),
            'var_bres': var_bres,
            'crit_val': crit_val}


# ---------------------- multiplier bootstrap --------------------------

def multiplier_bootstrap_joblib(inf_funcs: np.ndarray | csr_matrix,
                                iterations: int = 1000,
                                random_state: int = None,
                                backend: str = 'loky',
                                n_jobs: int = -1,

                                progress_bar: bool = True,
                                sample_name: str = None,
                                release_workers: bool = True
                                ):
    jobs = cpu_count() + 1 - n_jobs if n_jobs < 0 else n_jobs

    if random_state:
        rng = np.random.default_rng(random_state)
        random_states = rng.integers(0, np.iinfo(np.int32).max, size=jobs)
    else:
        random_states = np.random.randint(0, np.iinfo(np.int32).max, size=jobs)

    iterations_per_job = np.repeat(iterations // jobs, jobs)
    iterations_per_job[:iterations % jobs] += 1

    # look into leaked resources joblib
    # https://stackoverflow.com/questions/72590664/leaked-folders-by-randomizedsearchcv
    # https://github.com/joblib/joblib/issues/1076

    if progress_bar:
        positions_progress = np.arange(jobs) + 1
        mask_max_iterations = list(positions_progress != positions_progress.max())
    else:
        mask_max_iterations = np.ones(np.arange(jobs) + 1, dtype=bool)

    output = Parallel(n_jobs=jobs,
                      backend=backend,
                      temp_folder=None,
                      # verbose=9
                      # max_nbytes=None,
                      # mmap_mode=None
                      )(delayed(multiplier_bootstrap
                                )(inf_funcs, *it)
                        for it in zip(iterations_per_job,
                                      random_states,
                                      mask_max_iterations
                                      )
                        )

    if release_workers:
        get_reusable_executor().shutdown(wait=True)

    return np.vstack(output)


def multiplier_bootstrap(inf_funcs: np.ndarray | csr_matrix,
                         iterations: int = 1000,
                         random_state: int = None,
                         disable_tqdm: bool = False):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.default_rng(random_state)

    n, k = inf_funcs.shape

    out_mat = np.zeros(shape=(iterations, k))

    for bit in tqdm(range(iterations),
                    desc=f'Bootstrap',
                    disable=disable_tqdm,
                    bar_format='{desc:<30}{percentage:3.0f}%|{bar:20}| [{elapsed}<{remaining}, '
                               '{rate_fmt}{postfix}]'
                    ):

        ub = rng.choice([1, -1], size=(n, 1))
        if issparse(inf_funcs):  # slower to multiply/broadcast with sparse matrix
            out_mat[bit] = np.mean(inf_funcs.multiply(ub), axis=0)
        else:
            out_mat[bit] = np.mean(inf_funcs * ub, axis=0)

    return out_mat  # iterations x k


def boot_crit_val(mb_res: np.ndarray,  # n x k
                  alpha: float = 0.05):
    b_sigma = (
            (np.quantile(a=mb_res, q=.75, axis=0, method='inverted_cdf')
             - np.quantile(a=mb_res, q=.25, axis=0, method='inverted_cdf'))
            / (norm.ppf(.75) - norm.ppf(.25))
    )

    # critical value for uniform confidence band
    b_t = np.max(np.abs(mb_res / b_sigma), axis=1)

    crit_val = np.quantile(b_t, 1 - alpha, method='inverted_cdf')

    return crit_val, b_sigma


# -------------------------- clusters ----------------------------------

def mean_inf_func_by_cluster(cluster_groups: np.ndarray,
                             inf_funcs: np.ndarray,  # n x k
                             ):
    return group_means(x=inf_funcs, codes=cluster_groups)


def get_cluster_groups(data: DataFrame,
                       cluster_var: str | list | None):
    if cluster_var is None:
        cluster_var = []
    elif isinstance(cluster_var, str):  # entity cluster is automatic, exclude from list
        cluster_var = [c for c in [cluster_var] if c != data.index.names[0]]

    if cluster_var:  # if there is a cluster var other than entity
        if len(cluster_var) > 1:  # one is always entity
            raise ValueError("can't have more than 2 cluster variables")

        if find_time_varying_covars(data=data, covariates=cluster_var):
            raise ValueError("can't have time-varying cluster variables")

        cluster_groups = _get_cluster_groups_from_series(cluster=data[cluster_var[0]])

    else:
        cluster_groups = None

    return cluster_groups


# todo: fix if I estimate panel std_errors as repeated cross-section this will be grouped by entity
#   this should raise an error because n_obs of influence function greater than n entities
def _get_cluster_groups_from_series(cluster: Series):
    """returns a 0 'indexed' array with the clusters to be used"""
    entity_name = cluster.index.names[0]
    cluster_name = cluster.name

    # (just keep n obs) cluster should not be nested
    # (check above not time varying)
    # len(cluster_n) must be = len(inf_funcs)
    cluster_n = (
        cluster
        .reset_index()
        [[entity_name, cluster_name]]
        .drop_duplicates()
    )

    cluster_groups = cluster_n.groupby(cluster_name).ngroup().to_numpy()

    return cluster_groups
