from __future__ import annotations

import numpy as np
import pandas as pd

from pandas import DataFrame, Series

import warnings

from pyhdfe import create
from linearmodels.iv import AbsorbingLS

from ..feols.utility import (find_singletons,
                             get_categories,
                             find_nested_fe,
                             estimate_absorbed_fixed_effects,
                             get_t_stats,
                             get_p_value,
                             get_conf_int,
                             one_way_clustered_vcv,
                             build_model_matrix)

from ..utility import parse_fe_from_formula


class FEols:
    """
    wrapper for linearmodels.AbsorbingLS

    https://bashtage.github.io/linearmodels/iv/absorbing/linearmodels.iv.absorbing.AbsorbingLS.html

    used by the TWFE class
    """

    def __init__(self,
                 data: DataFrame,
                 formula: str,  # includes fe as: 'y ~ x1 | fe1 + fe2 + fe3'
                 weights_name: str = None,
                 cluster_names: str | list = None,

                 drop_singletons: bool = False,
                 absorb_only_one_fe: bool = False,

                 copy_data: bool = False
                 ):

        if copy_data:
            self.data = data.copy()
        else:
            self.data = data

        self._major_idx, self._minor_idx = self.data.index.names

        self.drop_singletons = drop_singletons

        self.weights_name = weights_name

        self._y_data_matrix = None
        self.formula, self.fe = parse_fe_from_formula(formula)

        self.absorb_only_one = absorb_only_one_fe

        # --------------------------------------------------------------

        if isinstance(cluster_names, str):
            cluster_names = [cluster_names]

        self.cluster_names = cluster_names if cluster_names is not None else []

        # fixed effects + cluster_names that are stored in the data multi-index
        self._categories = list(set(self.fe + self.cluster_names))

        self._idx_categories = [i for i in self._categories
                                if i in list(self.data.index.names)]

        # fixed effects + cluster_names that are stored in the data columns
        self._cols_categories = [i for i in self._categories
                                 if i not in self._idx_categories]

        # --------------------------------------------------------------
        # drop singletons
        self._categorical_dataframe = None

        if self.drop_singletons and self.fe:
            self.singletons = find_singletons(
                categorical_data=self.categorical_dataframe[self.fe], fe=self.fe)

            self.n_singletons = np.sum(self.singletons)

            if self.n_singletons:
                self._categorical_dataframe = None  # un-cache self._categorical_dataframe

                print(f'dropping {self.n_singletons} singletons observations')
                self.data = self.data[~self.singletons].copy()

        self._nested_fe = None  # list of fe nested within cluster_names
        self._x_data_matrix = None  # covars matrix + fe dummies if requested
        self._x_data_matrix_cols = None  # names of cols in _x_data_matrix

        # --------------------------------------------------------------
        self.dummy_fe = []  # list of fe to make dummies
        self.absorb_fe = []  # list of fe to absorb

        # create fe/cluster_names cats + determine which fe to absorb
        if self.fe:
            if self.absorb_only_one:
                # when this func is called it also creates the categorical cols
                self.determine_fixed_effects_to_absorb()
            else:
                self.absorb_fe = self.fe

        # --------------------------------------------------------------
        self._cov_estimator = None

        # containers for AbsorbingLS objects

        self.mod = None  # AbsorbingLS instance
        self.res = None  # result of AbsorbingLS
        self.estimated_fe = None

        self.dropped_cols = None  # cols in _x_data_matrix_cols dropped because collinear

        self._dof_m = None  # degrees of freedom for the model
        self._dof_a = None  # absorbed degrees of freedom

        # --------------------------------------------------------------

    @property
    def categorical_dataframe(self) -> DataFrame:
        """categorical dataframe with fixed effects & cluster_names"""

        if self._categorical_dataframe is None:
            self._categorical_dataframe = get_categories(
                data=self.data,
                cols_categories=self._cols_categories,
                idx_categories=self._idx_categories
            )

        return self._categorical_dataframe

    @property
    def fe_categories_dict(self) -> dict:
        """n categories for each specified fe var

        Returns
        -------
        dict
            {'fe_name': 'number of categories', ...} descending by n of cats

        """
        fe_cats = {c: len(self.categorical_dataframe[c].cat.categories)
                   for c in self.fe}

        fe_cats = dict(sorted(fe_cats.items(), key=lambda x: -x[1]))

        return fe_cats

    @property
    def clusters_categories_dict(self) -> dict:
        """n categories for each specified cluster var"""
        clusters_cats = {c: len(self.categorical_dataframe[c].cat.categories)
                         for c in self.cluster_names}

        clusters_cats = dict(sorted(clusters_cats.items(), key=lambda x: -x[1]))

        return clusters_cats

    @property
    def nested_fe(self) -> list:
        """finds list of fe nested in cluster vars"""

        if self._nested_fe is None:
            self._nested_fe = find_nested_fe(
                categorical_data=self.categorical_dataframe,
                fe=self.fe,
                cluster_names=self.cluster_names
            )

        return self._nested_fe

    def determine_fixed_effects_to_absorb(self) -> None:
        """generates two lists: dummy_fe, absorb_fe

        dummy_fe: the fixed effects to be made dummies
        absorb_fe: the fixed effects to be asborbed

        """

        # which fe to convert to dummies
        cats_keys = list(self.fe_categories_dict.keys())

        # absorb the first and generate dummies for the rest
        self.dummy_fe = [f for f in cats_keys[1:]]
        self.absorb_fe = [f for f in self.fe if f not in self.dummy_fe]

    # todo: check that the size of all is the same even if formulaic drops nas
    def _data_matrix(self,
                     extra_data_matrix: DataFrame = None
                     ) -> None:
        """covars matrix + dummy fe if 'absorb_only_one' is True"""

        self._y_data_matrix, self._x_data_matrix = build_model_matrix(
            formula=self.formula,
            data=self.data,
            categories_data=self.categorical_dataframe,  # created without singletons if dropped
            dummy_fe=self.dummy_fe
        )

        self._x_data_matrix_cols = list(self._x_data_matrix)

        if extra_data_matrix is not None:
            self._x_data_matrix = self._x_data_matrix.join(extra_data_matrix)

            if len(self._y_data_matrix) != len(self._x_data_matrix):
                pass  # todo: drop some obs of self._y_data_matrix

    @property
    def n_obs(self) -> int:
        return len(self.data)

    def fit(self,
            drop_absorbed: bool = True,
            extra_data_matrix: DataFrame = None,
            dummies_names: list[str] = None,  # formulaic drops the NaNs when getting dummies
            drop_names: list[str] = None,
            ):
        """fit linearmodels AbsorbingLS"""

        self._data_matrix(extra_data_matrix=extra_data_matrix)

        if dummies_names is not None:
            dummies = [pd.get_dummies(self.data[d]) for d in dummies_names]
            self._x_data_matrix = pd.concat([self._x_data_matrix] + dummies, axis=1)

        if drop_names:
            self._x_data_matrix = self._x_data_matrix.drop(columns=drop_names)

        self.mod = AbsorbingLS(
            dependent=self._y_data_matrix,
            exog=self._x_data_matrix,
            absorb=self.categorical_dataframe[self.absorb_fe] if self.absorb_fe else None,
            weights=self.data[self.weights_name] if self.weights_name else None,
            drop_absorbed=drop_absorbed
        )

        with warnings.catch_warnings():
            # AbsorbingEffectWarning
            warnings.simplefilter('ignore')

            self.res = self.mod.fit()

            if extra_data_matrix is not None:
                all_cols = self._x_data_matrix_cols + list(extra_data_matrix)
            else:
                all_cols = self._x_data_matrix_cols

            self.dropped_cols = [i for i in all_cols
                                 if i not in self.res.params.index]

            if self.dropped_cols:
                print(f'{self.dropped_cols} dropped due to multicollinearity')

        self.estimated_fe = estimate_absorbed_fixed_effects(
            mod=self.mod, res=self.res)

        return self.res

    @property
    def params(self) -> Series:
        """betas"""
        if self.res is None:
            raise RuntimeError('call fit')

        return self.res.params

    @property
    def std_errors(self) -> Series:
        """std errors"""

        vcv = self.vcv
        return pd.Series(np.sqrt(np.diag(vcv)), index=vcv.index, name='std_error')

    def result_table(self,
                     drop_absorbed: bool = True,
                     extra_data_matrix: DataFrame = None,
                     alpha: float = 0.05,
                     dummies_names: list[str] = None,  # formulaic drops the NaNs
                     drop_names: list[str] = None,
                     ) -> DataFrame:

        if self.res is None:  # run AbsorbingLS
            self.fit(
                drop_absorbed=drop_absorbed,
                extra_data_matrix=extra_data_matrix,
                dummies_names=dummies_names,
                drop_names=drop_names
            )

        params = self.params
        std_errors = self.std_errors

        t_stats = get_t_stats(params, std_errors)

        pv = get_p_value(
            params=params,
            std_errors=std_errors,
            t_stats=t_stats,
            resid=None,  # todo: resid, with debiased
        )
        ci = get_conf_int(
            params=params,
            std_errors=std_errors,
            resid=None,  # todo: resid, with debiased
            alpha=alpha
        )

        return pd.concat([params, std_errors, t_stats, pv, ci], axis=1)

    @property
    def vcv(self) -> DataFrame:

        if self.res is None:
            raise RuntimeError('call fit')

        if self.cluster_names:
            if len(self.cluster_names) > 1:
                # todo: two-way clustering + other types if needed
                raise ValueError('only one way clustering allowed')

            self._cov_estimator = f'clustered, with {self.cluster_names} clusters'

            return one_way_clustered_vcv(
                y=self.absorbed_dependent.to_numpy().flatten(),
                X=self.absorbed_exog.to_numpy(),
                params=self.params,
                clusters=self.categorical_dataframe[self.cluster_names[0]].cat.codes,
                dof_a=self.dof_a,
                dof_m=self.dof_m
            )
        else:
            self._cov_estimator = self.res.cov_estimator
            return self.res.cov

    @property
    def absorbed_dependent(self):
        if self.mod is not None:
            return self.mod.absorbed_dependent

    @property
    def absorbed_exog(self):
        if self.mod is not None:
            return self.mod.absorbed_exog

    @property
    def resids(self) -> Series:

        if self.res is None:
            raise RuntimeError('call fit')

        return self.res.resids

    @property
    def wresids(self) -> Series:

        if self.res is None:
            raise RuntimeError('call fit')

        return self.res.wresids

    @property
    def dof_r(self) -> int:
        """degrees of freedom residual"""
        return self.n_obs - self.dof_m - self.dof_a

    @property
    def dof_m(self) -> int:
        """degrees of freedom model"""

        if self.res is None:
            raise RuntimeError('call fit')

        self._dof_m = (len(self.params) if not self.fe else
                       len(self.params[lambda x: ~x.index.isin(['Intercept'])]))

        return self._dof_m

    @property
    def dof_a(self) -> int:
        """degrees of freedom absorbed"""

        if self._dof_a is None:

            if self.cluster_names:
                not_nested_fe = [i for i in self.fe if i not in self.nested_fe]

                # sorted by largest number of categories
                not_nested_fe = [k for k in self.fe_categories_dict if k in not_nested_fe]

            else:
                not_nested_fe = self.fe

            # if all the fe are nested within a cluster
            if not not_nested_fe:
                self._dof_a = 0

            else:  # if there are fe not nested within a cluster

                if len(not_nested_fe) == 1:
                    # return the number of categories for that fe
                    self._dof_a = self.fe_categories_dict[not_nested_fe[0]]

                else:  # if there is more than 1 non nested fe

                    # this can be slow for hdfe >= 3

                    algo = create(
                        self.categorical_dataframe[not_nested_fe],
                        drop_singletons=False,
                        compute_degrees=True
                    )

                    if not self.drop_singletons:
                        singletons = algo.singletons if algo.singletons is not None else 0

                        # be careful, just using this for checks in the imputation code
                        return algo.degrees + singletons

                    self._dof_a = algo.degrees

        return self._dof_a
