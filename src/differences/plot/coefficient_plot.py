from __future__ import annotations

import sys
from typing import Literal

import pandas as pd
from differences.models.tools.shared import parse_att_gt_titles

# --------------------------- att_gt ----------------------------------

# plot_att_gt:
# base: one plot
# sample splits: facet on sample splits vertically
# groups: facet on groups splits vertically
# groups + sample splits: facet on sample splits vertically & groups horizontally


CoefficientPlotSources = Literal[
    "twfe_panel_ols",

    "att_gt",
    "att_gt_event_aggregation",
    "att_gt_cohort_aggregation",
    "att_gt_time_aggregation",
    "att_gt_overall_aggregation"
]


def coefficient_plot(
        source: CoefficientPlotSources,
        backend: Literal["mpl", "lp", "alt"] = "lp",
        **kwargs
):
    try:
        import plotto
    except ImportError:
        raise ImportError(
            "In order to plot the coefficients "
            "you need to install the python package named 'plotto'\n\n"
            "pip install plotto\n"
        )

    params_func = getattr(sys.modules[__name__], f"_{source}_plot_params")

    kwargs = params_func(**kwargs)  # get some predefined parameters

    if not kwargs:
        raise ValueError(f"{source} is not available for backend={backend}")

    return plotto.coefficient_plot(backend=backend, **kwargs)


def coefficient_plot_att_gt(
        data: pd.DataFrame,
        **kwargs
):
    title, data = data.no_index()
    title = parse_att_gt_titles(title)

    # type_of_aggregation: Literal["simple", "cohort", "event", "time"] | None = None,
    # overall: bool = False,
    # difference: bool = False,  # I need this mainly to retrieve the correct

    type_of_aggregation = title.split()[0].lower()

    if type_of_aggregation != "att":
        source = f"att_gt_{type_of_aggregation}_aggregation"
    else:
        source = "att_gt"

    return coefficient_plot(
        source=source, backend="lp", data=data, title=title, **kwargs
    )

    # if type_of_aggregation is None:
    #     return attgt_plot.plot_att_gt(
    #         df=df,
    #         plotting_parameters=plotting_parameters,
    #         estimation_details=estimation_details,
    #     )
    #
    # elif type_of_aggregation == "simple" or overall:
    #     return attgt_plot.plot_overall_agg(
    #         df=df,
    #         plotting_parameters=plotting_parameters,
    #         estimation_details=estimation_details,
    #     )
    #
    # elif not overall:
    #
    #     plot_func = getattr(attgt_plot, f"plot_{type_of_aggregation}_agg")
    #     return plot_func(
    #         df=df,
    #         plotting_parameters=plotting_parameters,
    #         estimation_details=estimation_details,
    #     )


# ---------------------------- TWFE ------------------------------------

def _twfe_panel_ols_plot_params(
        backend: Literal["mpl", "lp", "alt"] = "lp",
        **kwargs
) -> dict:
    if backend == "alt":
        out = {}
    elif backend == "mpl":
        out = {}
    elif backend == "lp":
        out = dict(
            x_column="coefficient",
            y_column="parameter",
        )
    else:
        raise ValueError("'backend' must be one of: 'mpl', 'lp', 'alt'")

    out.update(kwargs)

    return out


# ------------------------- ATTgt --------------------------------------

def _att_gt_event_aggregation_plot_params(
        backend: Literal["mpl", "lp", "alt"] = "lp",
        **kwargs
) -> dict:
    if backend == "alt":
        out = {}
    elif backend == "mpl":
        out = {}
    elif backend == "lp":
        out = dict(
            x_column="relative_period",
            y_column="ATT",
        )
    else:
        raise ValueError("'backend' must be one of: 'mpl', 'lp', 'alt'")

    out.update(kwargs)

    return out


def _att_gt_plot_params(
        data: pd.DataFrame,
        backend: Literal["mpl", "lp", "alt"] = "mpl",
) -> dict:
    if backend == "alt":

        out = {

            # "data": data,
            # "title": title,
            # "table_note": caption_table,

            "y": "ATT",
            "x": "time",

            "points": True,
            "lines": True,
            "eticks": True,
            "ebars": True,

            "shape_by": "cohort",
            "color_by": "post",
            "select_by": "hc",
            "configure_legend_shape_by": {"title": "Cohort"},

            "zero_in_xscale": False,

        }

    elif backend == "mpl":

        out = {

        }

    elif backend == "lp":

        out = {

        }

    else:
        raise ValueError("'backend' must be one of: 'mpl', 'lp', 'alt'")

    return out
