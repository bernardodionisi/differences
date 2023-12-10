from __future__ import annotations

import inspect
from typing import Callable, Literal

import numpy as np
from pandas import DataFrame as PandasDF

from differences.models.did.attgt.aggregate import get_weights, _AggregateGT
from differences.models.did.attgt.difference import (
    preprocess_difference,
    get_ds_masks, _Difference, difference_ntl_to_dataframe)
from differences.models.did.attgt.mboot import get_cluster_groups

from differences.models.did.attgt.data_validator import (
    output_dict_to_dataframe, extract_dict_ntl,
    extract_dict_ntl_for_difference)


class ATTgtResult:

    def __init__(
            self,
            att_gt: dict,
            data_matrix: PandasDF,
            dosages: list,
            weights_column: str,
            cluster_by_entity: bool
    ):
        self._att_gt = att_gt  # _result_dict
        self._dosages = dosages

        self.data_matrix = data_matrix

        self._weights_column = weights_column
        self._cluster_by_entity = cluster_by_entity

        self._sample_names = None
        self._aggregate_locals = None
        self._boot_iterations_difference = None
        self._difference_inst = None
        self._difference_ntl = None

    def to_pandas(self):
        return output_dict_to_dataframe(
            extract_dict_ntl(self._att_gt),
            stratum=bool(self._dosages),
        )

    def __repr__(self):

        att_gt = self.to_pandas()
        return str(att_gt)

    def aggregate(
            self,
            type_of_aggregation: Literal["simple", "cohort", "event", "time"] | None = "simple",
            overall: bool = False,
            difference: bool | list | dict[str, list] = False,
            *,
            alpha: float = 0.05,
            cluster_var: list | str = None,
            boot_iterations: int = 0,
            random_state: int = None,
            n_jobs: int = 1,
            backend: str = "loky",
    ) -> PandasDF:
        """
        Aggregate the ATTgt

        Group
        -----
            Estimation

        Parameters
        ----------
        type_of_aggregation: *str* | None, default: ``None``

            - ``"simple"``
                to calculate the weighted average of all cohort-time average treatment effects,
                with weights proportional to the cohort size.

            - ``"event"``
                to calculate the average effects in each relative period:
                periods relative to the treatment; as in an event study.

            - ``"cohort"``
                to calculate the average treatment effect in each cohort.

            - ``"time"``
                to calculate the average treatment effect in each time.

        overall: *bool*, default: ``False``
            calculates the average effect within each type_of_aggregation.

            - if type_of_aggregation is set to ``"event"``
                to calculate the average effect of the treatment across positive relative periods

            - if type_of_aggregation is set to ``"cohort"``
                to calculate the average effect of the treatment across cohorts

            - if type_of_aggregation is set to ``"time"``
                to calculate the average effect of the treatment across time times

        difference: *bool* | *list* | *dict*, default: ``False``
            take the difference of the estimates

            Available options are:

            - ``True``
                to calculate the difference between 2 samples or 2 strata of treatments

            .. note::
                - Samples difference: if the estimation is run on 2 samples and more than
                  2 strata,
                  the estimates for the two samples will be subtracted, as long as there are no
                  strata that have the same names as the samples, in that case use
                  a dictionary as indicated below

                - strata difference: if the estimation is run on 2 strata and more than
                  2 samples,
                  the estimates for the two strata will be subtracted, as long as there are no
                  samples that have the same names as the strata, in that case use
                  a dictionary as indicated below

            - ``[sample-0, sample-1]`` or ``[stratum-A, stratum-B]``
                to calculate the difference between 2 samples listed in the argument or
                the 2 strata of treatments listed in the argument

            .. note::
                - Samples difference: if there are strata with the same name as the two samples
                  listed, use a dictionary as indicated below

                - strata difference: if there are samples with the same name as the two strata
                  listed, use a dictionary as indicated below

            - ``{'strata': [stratum-A, stratum-B]}`` or
              ``{'sample_names': [sample-0, sample-1]}``

        alpha: *float*, default: ``0.05``

            The significance level.

        cluster_var: *str* | *list* | *None*, default: ``None``
            cluster variables

        boot_iterations: *int*, default: ``0``
            bootstrap iterations

        random_state: *int* | None, default: ``None``
            seed for bootstrap

        n_jobs: *int*, default: ``1``
            The maximum number of concurrently running jobs. If -1 all CPUs are used.

            If ≠ 1, concurrent jobs will be run for:

            - computing the bootstrap; the influence function is split into n_jobs parts and the
              boostrap is computed concurrently for each part

            Parallelization is implemented using
            `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_,
            refer to its documentation for additional details on n_jobs.

        backend: *int*, default: ``"loky"``
            Parallelization backend implementation.

            Parallelization is implemented using
            `joblib <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_,
            refer to its documentation for additional details on backend.

        Returns
        -------
        A PandasDF with the requested aggregation

        """

        # todo: MISTAKE HERE, go back to what it was, agg can be None if difference !!!
        if type_of_aggregation not in ["simple", "cohort", "event", "time"]:
            raise ValueError(
                "'type_of_aggregation' must be one of: 'simple', 'cohort', 'event', 'time'"
            )
        if isinstance(
                cluster_var, str
        ):  # entity cluster is automatic, exclude from list
            cluster_var = [
                c for c in [cluster_var] if c != self.data_matrix.index.names[0]
            ]

        if isinstance(difference, dict):
            correct = {"d": "strata", "s": "sample_names"}
            for k, v in [(k[0], k) for k in difference.keys()]:
                difference[correct[k]] = difference.pop(v)

        # -------------------------- cache -----------------------------

        # decide when to un-cache the aggregation class
        _locals_aggregate = tuple(
            (k, v)
            for k, v in locals().items()
            if k not in ["self", "type_of_aggregation", "overall"]
        )

        if _locals_aggregate != self._aggregate_locals:
            self._aggregate_cflag = None
            self._aggregate_locals = _locals_aggregate

        # if difference, no need to bootstrap the standard errors for the aggregation
        self._boot_iterations_difference = boot_iterations
        if difference:
            boot_iterations = 0

        # ---------- instances of _Aggregategt for each sample --------

        if type_of_aggregation is not None:

            if self._aggregate_cflag is None:  # set to None in .fit()
                self._aggregate_cflag = True

                for s in self.sample_names:

                    if self._att_gt[s].get("weights") is None:
                        self._att_gt[s]["weights"] = get_weights(
                            data=(
                                self.data_matrix[[self._weights_column]]
                                if s == "full_sample"
                                else self.data_matrix[[self._weights_column]].loc[
                                    self._att_gt[s]["sample_mask"]
                                ]
                            ),
                            weights_name=self._weights_column,
                            entity_level=self._cluster_by_entity,
                        )

                    cluster_groups = None
                    if cluster_var:
                        cluster_groups = get_cluster_groups(
                            data=(
                                self.data_matrix[cluster_var]
                                if s == "full_sample"
                                else self.data_matrix[cluster_var].loc[
                                    self._att_gt[s]["sample_mask"]
                                ]
                            ),
                            cluster_var=cluster_var,
                        )

                    # save the instance of the aggregation class in the result dict
                    self._att_gt[s]["aggregate_inst"] = _AggregateGT(
                        ntl=self._att_gt[s]["ATTgt_ntl"],
                        weights=self._att_gt[s]["weights"],
                        strata=self._dosages,
                        cluster_groups=cluster_groups,
                        alpha=alpha,
                        boot_iterations=boot_iterations,
                        random_state=random_state,
                        backend_boot=backend,
                        n_jobs_boot=n_jobs,
                    )

            # ------ perform aggregation within _Aggregategt ----------

            for s in self.sample_names:
                self._att_gt[s]["aggregate_inst"].aggregate(
                    type_of_aggregation=type_of_aggregation,
                    overall=overall,
                )

        # ----------------- differences between ATTs -------------------

        # differences between
        #   - strata
        #   - sample splits
        #   - sample splits & strata

        if difference:

            # pre-process difference input: strata, samples
            difference_between = preprocess_difference(
                difference=difference,
                sample_names=self._sample_names,
                strata=self._dosages,
            )

            # if isinstance(difference_between, tuple):
            #     pass

            # else:  # dict
            difference_strata, iterate_samples = [
                difference_between.get(key)
                for key in ["difference_strata", "iterate_samples"]
            ]

            difference_samples, iterate_strata = [
                difference_between.get(key)
                for key in ["difference_samples", "iterate_strata"]
            ]

            data_mask, sample_masks = None, None
            if difference_samples:  # if the difference is between samples, get masks

                # data_mask: numpy array masking the full data to filter the two samples
                # sample_masks: list of np arrays with mask for each sample (resized)
                data_mask, sample_masks = get_ds_masks(
                    result_dict=self._att_gt,
                    difference=difference_samples,
                    entity_index=self.data_matrix.index.get_level_values(0),
                    cluster_by_entity=self._cluster_by_entity,
                )

            cluster_groups = self._get_clusters_for_difference(
                cluster_var=cluster_var,
                difference_samples=difference_samples,
                data_mask=data_mask,
                iterate_samples=iterate_samples,
            )

            # set up data for difference
            diff_pairs_ntls = extract_dict_ntl_for_difference(
                result_dict=self._att_gt,
                type_of_aggregation=type_of_aggregation,
                overall=overall,
                difference_samples=difference_samples,
                iterate_strata=iterate_strata,
                difference_strata=difference_strata,
                iterate_samples=iterate_samples,
            )

            self._difference_inst = _Difference(
                alpha=alpha,
                boot_iterations=self._boot_iterations_difference,
                random_state=random_state,
                n_jobs_boot=n_jobs,
                backend_boot=backend,
            )

            # get difference between estimates
            self._difference_ntl = self._difference_inst.get_difference(
                diff_pairs_ntls=diff_pairs_ntls,
                sample_masks=sample_masks,  # need it when subtracting two samples
                cluster_groups=cluster_groups,
                type_of_aggregation=type_of_aggregation,
                overall=overall,
                iterating_samples=bool(iterate_samples)
                # false if iter strata or only one sample
            )

            output = difference_ntl_to_dataframe(
                ntl=self._difference_ntl,
            )

            return output

        # aggregation if no difference requested

        output = extract_dict_ntl(
            result_dict=self._att_gt,
            type_of_aggregation=type_of_aggregation,
            overall=overall,
            sample_names=self.sample_names,
        )

        output = output_dict_to_dataframe(
            output,
            stratum=bool(self._dosages),
        )

        return output

    @property
    def sample_names(self):
        """
        Names of the sample splits

        Group
        -----
            Info

        Returns
        -------

        """
        try:
            self._sample_names = list(self._att_gt.keys())
        except AttributeError:
            return self._sample_names

        return self._sample_names

    def plot(
            self,
            type_of_aggregation: Literal["simple", "cohort", "event", "time"] | None = None,
            overall: bool = False,
            difference: bool = False,  # I need this mainly to retrieve the correct
            **kwargs,
    ):
        """
        Plot the estimates

        Group
        -----
            Results

        Parameters
        ----------
        type_of_aggregation: *str* | None, default: ``None``

            .. md-tab-set::
                .. md-tab-item:: "simple"

                    to plot the weighted average of all cohort-time average treatment effects,
                    with weights proportional to the cohort size.

                .. md-tab-item:: "event"

                    to plot the average effects in each relative period:
                    periods relative to the treatment; as in an event study.

                .. md-tab-item:: "cohort"

                    to plot the average treatment effect in each cohort.

                .. md-tab-item:: "time"

                     to plot the average treatment effect in each time time.

        overall: *bool*, default: ``False``
            to plot the average effect within each type_of_aggregation.

        difference: *bool*, default: ``False``
            take the difference of the estimates

            Available options are:

            - ``True``
                to plot the difference between 2 samples or 2 strata of treatments

        estimation_details: *bool* | *list* | *str*, default: ``True``
            include the estimation details in the plot. One can modify the format
            through plotting_parameters

        estimate_in_x_axis: *bool*, default: ``False``
            whether to display the ATT estimates in the x-axis

        plotting_parameters
            a set of parameters to customize the plot. Please refer to the separate documentation
            for the plotting functionalities built in the library

        Returns
        -------
        An interactive plot for the requested estimates
        """

        data = self.results(
            type_of_aggregation=type_of_aggregation,
            overall=overall,
            difference=difference,
            # sample_name=sample_name,
            to_dataframe=True,
            add_info=not bool(type_of_aggregation),
        )

    # -------------------------- helpers -------------------------------

    def _get_clusters_for_difference(
            self,
            cluster_var: list | str | None,
            difference_samples: list,
            data_mask: np.ndarray,
            iterate_samples: list,
    ) -> np.ndarray | dict:

        # clusters
        cluster_groups = None
        if cluster_var:

            if difference_samples:
                cluster_groups = get_cluster_groups(
                    data=self.data_matrix[cluster_var].loc[data_mask],
                    cluster_var=cluster_var,
                )

            elif iterate_samples:
                cluster_groups = {
                    s: get_cluster_groups(
                        data=self.data_matrix[cluster_var].loc[
                            self._att_gt[s]["sample_mask"]
                        ],
                        cluster_var=cluster_var,
                    )
                    for s in self.sample_names
                }

            else:  # main case, no samples splits or treat strata
                cluster_groups = get_cluster_groups(
                    data=self.data_matrix[cluster_var], cluster_var=cluster_var
                )

        return cluster_groups


# ----------------------------------------------------------------------


def insert_sample_mask_into_att_gt_dict(
        data: PandasDF,
        split_sample_by: Callable | str = None,
):
    """finds the masks for the data given split_sample_by

    returns a dict {'name of sample': data mask for sample}"""

    if split_sample_by is None:
        return None

    elif isinstance(split_sample_by, str):
        return {
            f"{split_sample_by} = {i}": {
                "sample_mask": np.array(data[split_sample_by] == i)
            }
            for i in data[split_sample_by].dropna().unique()  # for each value in the sample col
        }

    # ------------------------------------------------------------------

    # TODO: REMOVE CALLABLE?
    elif isinstance(split_sample_by, Callable):

        source_code = inspect.getsource(split_sample_by)

        # make the string look nicer if lambda is passed
        if "lambda" in source_code:
            source_code = source_code.split("split_sample_by=")[-1]
            if source_code.count(":") == 1:
                source_code = source_code.split(":")[-1]

        source_code = source_code.strip()
        if source_code.endswith(","):
            source_code = source_code[:-1]

        mask = np.array(split_sample_by(data))

        return {
            source_code: {"sample_mask": mask},
            f"NOT-{source_code}": {"sample_mask": ~mask},
        }

    else:
        raise ValueError("invalid 'split_sample_by'")
