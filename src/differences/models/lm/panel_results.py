from __future__ import annotations

import inspect
import textwrap
import polars as pl
import linearmodels
import pandas as pd
from pandas import DataFrame as PandasDF

from differences.report.table.table import Table


class PanelEffectsResults:
    def __init__(
            self,
            instance: linearmodels.panel.results.PanelEffectsResults = None
    ):
        # it would be ideal to just inherit from 'linearmodels.panel.results.PanelEffectsResult'
        # but we have no access to the fit's 'res' which is the argument to __init__

        self.instance = instance

        # self.update_docstring()  # Dynamically generate docstring.

    def __getattr__(self, attr):
        attr_value = getattr(self.instance, attr)

        if callable(attr_value):
            def method(*args, **kwargs):
                return attr_value(*args, **kwargs)

            method.__signature__ = inspect.signature(attr_value)
            return method
        else:
            return attr_value

    def __dir__(self):
        return sorted(set(dir(self.instance) + list(super().__dir__())))

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}\n"
            f"{self.instance.__repr__()}"
        )

    def _update_docstring(self):
        instance_class = type(self.instance)
        docs = [f"Methods from {instance_class.__name__}:"]

        for name, method in inspect.getmembers(instance_class, predicate=inspect.isfunction):
            sig = str(inspect.signature(method))
            doc = textwrap.indent(method.__doc__ or '', ' ' * 4)
            docs.append(f"{name}{sig}\n{doc}")

        self.__doc__ = "\n".join(docs)

    def _coefficient_table_data(
            self,
            level: float = .95,
    ) -> PandasDF:
        out = pd.concat([
            self.params,
            self.std_errors,
            self.pvalues,
            self.conf_int(level=level),
        ], axis=1)

        out.insert(0, 'coefficient', out.index)
        return out.reset_index(drop=True)

    def coefficient_table(
            self,
            level: float = .95,
    ) -> Table:
        data = pl.from_pandas(self._coefficient_table_data(level=level))

        data = Table(
            data=data,
            id_column="coefficient",
            estimate_column="parameter",
            lu_columns=("lower", "upper"),
        )

        return data


class PanelEffectsResultsMultiple:
    def __init__(
            self,
            instances: dict[str, PanelEffectsResults] = None
    ):
        self.instances = instances if instances is not None else {}

        self._update_docstring()  # Dynamically generate docstring.

    def _update_docstring(self):
        if self.instances:
            docs = [
                f"Methods from {type(next(iter(self.instances.values()))).__name__}:"
            ]

            for name, method in inspect.getmembers(
                    type(next(iter(self.instances.values()))), predicate=inspect.isfunction):
                sig = str(inspect.signature(method))

                doc = textwrap.indent(method.__doc__ or '', ' ' * 4)

                docs.append(f"{name}{sig}\n{doc}")

            self.__doc__ = "\n".join(docs)

    def __getattr__(self, attr):

        attr_values = {
            name: getattr(instance, attr)
            for name, instance in self.instances.items()
        }

        if all(callable(val) for val in attr_values.values()):
            def method(*args, **kwargs):
                return {
                    name: callable_attr(*args, **kwargs)
                    for name, callable_attr in attr_values.items()
                }

            method.__signature__ = inspect.signature(next(iter(attr_values.values())))

            return method
        else:
            return attr_values

    def __dir__(self):
        methods_and_attrs = set()
        for instance in self.instances.values():
            methods_and_attrs.update(dir(instance))
        return sorted(methods_and_attrs)  # sorted for easier reading

    def __repr__(self):
        reprs = [f"{name}: {repr(instance)}" for name, instance in self.instances.items()]
        separator = '-' * 80
        separator = '\n\n' + separator + '\n\n'

        return (
            f"\n{self.__class__.__qualname__}\n"
            f"{separator.join(reprs)}"
        )

    def coefficient_table(
            self,
            level: float = .95,
    ) -> Table:

        data = {}
        for name, inst in self.instances:
            # get pandas dataframe from coefficient table
            temp = inst._coefficient_table_data(level=level)
            temp.insert(0, 'sample', name)  # insert a column with the name of the sample
            data[name] = temp

        data = pd.concat(data.values()).reset_index(drop=True)

        data = Table(
            data=data,
            id_column="coefficient",
            estimate_column="parameter",
            lu_columns=("lower", "upper"),
            group_columns='sample'
        )

        return data


# ----------------------------------------------------------------------


from linearmodels.compat.statsmodels import Summary

import numpy as np
from pandas import Series, concat
from statsmodels.iolib.summary import SimpleTable

from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
from linearmodels.shared.base import _ModelComparison
from linearmodels.shared.io import _str, add_star, pval_format
from linearmodels.panel.results import (
    PanelResults, RandomEffectsResults, FamaMacBethResults, PanelModelResults
)


# _ModelComparison: https://github.com/bashtage/linearmodels/blob/49c93db58abd9ddd2c05ca8210e8f4754efc8e42/linearmodels/shared/base.py#L12

class PanelModelComparison(_ModelComparison):
    """
    Comparison of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {"tstats","std_errors", "std-errors", "pvalues"}
        Estimator precision estimator to include in the comparison output.
        Default is "tstats".
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
    """

    _supported = (
        PanelEffectsResults,
        PanelResults,
        RandomEffectsResults,
        FamaMacBethResults,
    )

    def __init__(
            self,
            results: list[PanelModelResults] | dict[str, PanelModelResults],
            *,
            precision: str = "tstats",
            stars: bool = False,
    ) -> None:
        super().__init__(results, precision=precision, stars=stars)

    @property
    def rsquared_between(self) -> Series:
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_between")

    @property
    def rsquared_within(self) -> Series:
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_within")

    @property
    def rsquared_overall(self) -> Series:
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_overall")

    @property
    def estimator_method(self) -> Series:
        """Estimation methods"""
        return self._get_property("name")

    @property
    def cov_estimator(self) -> Series:
        """Covariance estimator descriptions"""
        return self._get_property("_cov_type")

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

        Notes
        -----
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """

        smry = Summary()
        models = list(self._results.keys())
        title = "Model Comparison"
        stubs = [
            "Dep. Variable",
            "Estimator",
            "No. Observations",
            "Cov. Est.",
            "R-squared",
            "R-Squared (Within)",
            "R-Squared (Between)",
            "R-Squared (Overall)",
            "F-statistic",
            "P-value (F-stat)",
        ]
        dep_name = {}
        for key in self._results:
            dep_name[key] = self._results[key].model.dependent.vars[0]

        vals = concat(
            [
                Series(dep_name),
                self.estimator_method,
                self.nobs,
                self.cov_estimator,
                self.rsquared,
                self.rsquared_within,
                self.rsquared_between,
                self.rsquared_overall,
                self.f_statistic,
            ],
            axis=1,
        )
        vals_lst = [[i for i in v] for v in vals.T.values]
        vals_lst[2] = [str(v) for v in vals_lst[2]]
        for i in range(4, len(vals_lst)):
            f = _str
            if i == 9:
                f = pval_format
            vals_lst[i] = [f(v) for v in vals_lst[i]]

        params = self.params
        precision = getattr(self, self._precision)
        pvalues = np.asarray(self.pvalues)
        params_fmt = []
        params_stub: list[str] = []
        for i in range(len(params)):
            formatted_and_starred = []
            for v, pv in zip(params.values[i], pvalues[i]):
                formatted_and_starred.append(add_star(_str(v), pv, self._stars))
            params_fmt.append(formatted_and_starred)

            precision_fmt = []
            for v in precision.values[i]:
                v_str = _str(v)
                v_str = f"({v_str})" if v_str.strip() else v_str
                precision_fmt.append(v_str)
            params_fmt.append(precision_fmt)
            params_stub.append(str(params.index[i]))
            params_stub.append(" ")

        vals_lst = table_concat((vals_lst, params_fmt))
        stubs = stub_concat((stubs, params_stub))

        all_effects = []
        for key in self._results:
            res = self._results[key]
            effects = getattr(res, "included_effects", [])
            all_effects.append(effects)

        neffect = max(map(len, all_effects))
        effects = []
        effects_stub = ["Effects"]
        for i in range(neffect):
            if i > 0:
                effects_stub.append("")
            row = []
            for j in range(len(self._results)):
                effect = all_effects[j]
                if len(effect) > i:
                    row.append(effect[i])
                else:
                    row.append("")
            effects.append(row)
        if effects:
            vals_lst = table_concat((vals_lst, effects))
            stubs = stub_concat((stubs, effects_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "r"
        txt_fmt["header_align"] = "r"
        table = SimpleTable(
            vals_lst, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)
        prec_type = self._PRECISION_TYPES[self._precision]
        smry.add_extra_txt([f"{prec_type} reported in parentheses"])
        return smry


def compare(
        results: list[PanelModelResults] | dict[str, PanelModelResults],
        *,
        precision: str = "tstats",
        stars: bool = False,
) -> PanelModelComparison:
    """
    Compare the results of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {"tstats","std_errors", "std-errors", "pvalues"}
        Estimator precision estimator to include in the comparison output.
        Default is "tstats".
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.

    Returns
    -------
    PanelModelComparison
        The model comparison object.
    """
    return PanelModelComparison(results, precision=precision, stars=stars)
