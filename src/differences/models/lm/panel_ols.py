from __future__ import annotations

from typing import Literal

from linearmodels.panel.model import PanelOLS
from pandas import DataFrame as PandasDF
from polars import DataFrame as PolarsDF

from warnings import warn
from functools import partial

from differences.models.lm.panel_results import PanelEffectsResultsMultiple, PanelEffectsResults


def panel_ols_fit(
        data: PolarsDF,
        entity_column: str,
        time_column: str,

        y_column: str,
        x_columns: list[str],
        sample_column: str = None,
        weights_column: str = None,
        other_effects_column: str = None,

        *,

        vcov: Literal["unadjusted", "robust", "kernel", "clustered"] | str | list[str] = "robust",

        entity_fixed_effects: bool = False,
        time_fixed_effects: bool = False,

        drop_singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,

        **fit_kws
) -> PanelEffectsResults | PanelEffectsResultsMultiple:
    if sample_column is None:
        func = _panel_ols_fit_single
    else:
        func = partial(_panel_ols_fit_multiple, sample_column=sample_column)

    out = func(
        data=data,
        entity_column=entity_column,
        time_column=time_column,
        y_column=y_column,
        x_columns=x_columns,
        # sample_column=sample_column,
        weights_column=weights_column,
        other_effects_column=other_effects_column,
        vcov=vcov,

        entity_fixed_effects=entity_fixed_effects,
        time_fixed_effects=time_fixed_effects,

        drop_singletons=drop_singletons,
        drop_absorbed=drop_absorbed,
        check_rank=check_rank,

        **fit_kws
    )

    return out


def _panel_ols_fit_multiple(
        data: PolarsDF,
        entity_column: str,
        time_column: str,

        y_column: str,
        x_columns: list[str],
        sample_column: str,
        weights_column: str = None,
        other_effects_column: str = None,

        *,

        vcov: Literal["unadjusted", "robust", "kernel", "clustered"] | str | list[str] = "robust",

        entity_fixed_effects: bool = False,
        time_fixed_effects: bool = False,

        drop_singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,

        **fit_kws
) -> PanelEffectsResultsMultiple:
    """get PanelOLS est for sample splits"""
    out = {}

    for s, d in data.groupby(sample_column):
        out[s] = _panel_ols_fit_single(
            entity_column=entity_column,
            time_column=time_column,
            y_column=y_column,
            x_columns=x_columns,
            weights_column=weights_column,
            other_effects_column=other_effects_column,
            vcov=vcov,
            entity_fixed_effects=entity_fixed_effects,
            time_fixed_effects=time_fixed_effects,
            drop_singletons=drop_singletons,
            drop_absorbed=drop_absorbed,
            check_rank=check_rank,
            **fit_kws
        )

    return PanelEffectsResultsMultiple(out)


def _panel_ols_fit_single(
        data: PolarsDF,
        entity_column: str,
        time_column: str,

        y_column: str,
        x_columns: list[str],
        weights_column: str = None,
        other_effects_column: str = None,

        *,

        # vcov can take one or multiple column names instead of "clustered"
        vcov: Literal["unadjusted", "robust", "kernel", "clustered"] | str | list[str] = "robust",

        entity_fixed_effects: bool = False,
        time_fixed_effects: bool = False,

        drop_singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,

        **fit_kws
) -> PanelEffectsResults:
    model = _panel_ols_model(
        data=data,
        entity_column=entity_column,
        time_column=time_column,

        y_column=y_column,
        x_columns=x_columns,
        weights_column=weights_column,
        other_effects_column=other_effects_column,

        entity_fixed_effects=entity_fixed_effects,
        time_fixed_effects=time_fixed_effects,

        drop_singletons=drop_singletons,
        drop_absorbed=drop_absorbed,
        check_rank=check_rank,
    )

    fit_kws, cluster_columns = _vcov_dispatcher(
        vcov=vcov,
        entity_column=entity_column,
        time_column=time_column,
        data_columns=data.columns,
        cov_config=fit_kws,  # the dictionary to update
    )

    if cluster_columns is not None:
        fit_kws.update(
            clusters=
            data
            .select([[entity_column, time_column] + cluster_columns])
            .to_pandas()
            .set_index([entity_column, time_column])
        )

    result = model.fit(**fit_kws)

    # from linearmodels.PanelEffectsResults to differences.PanelEffectsResults
    result = PanelEffectsResults(result)

    patch_model = False
    if patch_model:
        result._panel_ols_single_est = model

    return result


def _panel_ols_model(
        data: PolarsDF,
        entity_column: str,
        time_column: str,

        y_column: str,
        x_columns: list[str],
        weights_column: str = None,
        other_effects_column: str = None,

        *,

        entity_fixed_effects: bool = False,
        time_fixed_effects: bool = False,

        drop_singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,
):
    panel_columns = _PanelDataColumns(
        data=data,
        entity_column=entity_column,
        time_column=time_column
    )

    model = PanelOLS(
        dependent=panel_columns.get_data_columns(y_column),
        exog=panel_columns.get_data_columns(x_columns),
        weights=panel_columns.get_data_columns(weights_column),
        other_effects=panel_columns.get_data_columns(other_effects_column),

        entity_effects=entity_fixed_effects,
        time_effects=time_fixed_effects,
        singletons=drop_singletons,
        drop_absorbed=drop_absorbed,
        check_rank=check_rank,
    )

    return model


class _PanelDataColumns:
    def __init__(
            self,
            data: PolarsDF,
            entity_column: str,
            time_column: str,
            to_pandas: bool = True
    ):

        self.to_pandas = to_pandas

        if to_pandas:
            self.data = data.to_pandas().set_index([entity_column, time_column])
        else:
            self.data = data

        self.entity_column = entity_column
        self.time_column = time_column

    def get_data_columns(
            self,
            columns: str | list[str] | None,
    ) -> PolarsDF | PandasDF | None:
        if columns is None:
            return None

        if isinstance(columns, str):
            columns = [columns]

        if self.to_pandas:
            return self.data[columns]
        else:
            return self.data.select([self.entity_column, self.time_column] + columns)


def _vcov_dispatcher(
        vcov: str | list,
        entity_column: str,
        time_column: str,
        data_columns: list[str],
        cov_config: dict
) -> tuple[dict, list[str] | None]:
    cov_type = None
    cluster_columns = None

    print("vcov", vcov)
    if isinstance(vcov, str):
        if vcov in [
            "unadjusted",
            "homoskedastic",
            "robust",
            "heteroskedastic",
            "clustered",
            "kernel"
        ]:

            cov_type = vcov

            if vcov in data_columns:
                warn(
                    f'{vcov} is both a cov_type as well as a column in the dataframe'
                )

        else:  # if it is a column name then make into a list
            vcov = [vcov]

    if isinstance(vcov, list):
        cov_type = "clustered"

        for i in vcov:
            if i not in data_columns:
                raise ValueError(f'{i} not in data')

            if i == entity_column:
                cov_config.update(cluster_entity=True)

            if i == time_column:
                cov_config.update(cluster_time=True)

        cluster_columns = [
            c for c in vcov
            if (c in data_columns) and
               (c not in [entity_column, time_column])
        ]
        if not cluster_columns:
            cluster_columns = None

        # cov_config.update(clusters=data[cluster_columns])

    if cov_type is None:
        raise ValueError('unable to discern vcov')

    cov_config["cov_type"] = cov_type

    return cov_config, cluster_columns
