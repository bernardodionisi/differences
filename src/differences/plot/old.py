
def plot_att_gt(
        df: PandasDF,
        plotting_parameters: dict,
        estimation_details: dict = None,
        save_fname: str = None,
):
    from plotto import mark_plot

    title = get_title(df)

    _, df = single_idx(df)

    if "sample_name" in list(df) or "stratum" in list(df):
        print("Plot not implemented yet")
        return None

    df["cohort"] = df["cohort"].astype(str)
    # df['time'] = df['time'].astype(str)

    plot_att_gt_params.update(plotting_parameters)

    return mark_plot(**plot_att_gt_params)


# --------------------------- event ----------------------------------


def plot_event_agg(
        df: PandasDF, plotting_parameters: dict, estimation_details: list = None
):
    from plotto import mark_plot

    params = determine_params(df=df)

    params["data"] = params["data"].assign(
        relative_period=lambda x: x["relative_period"].astype(int)
    )

    agg_plot_params = {
        **params,
        "y": "ATT",
        "x": "relative_period",
        "points": True,
        "lines": True,
        "ebands": True,
        "table_note": estimation_details,
        "tooltip": "ATT",
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# --------------------------- cohort -----------------------------------


def plot_cohort_agg(
        df: PandasDF, plotting_parameters: dict, estimation_details: list = None
):
    from plotto import mark_plot

    params = determine_params(df=df)

    params["data"] = params["data"].assign(cohort=lambda x: x["cohort"].astype(str))

    agg_plot_params = {
        **params,
        "y": "ATT",
        "x": "cohort",
        "points": True,
        "lines": True,
        "eticks": True,
        "ebars": True,
        "table_note": estimation_details,
        "tooltip": "ATT",
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# --------------------------- time ---------------------------------


def plot_time_agg(
        df: PandasDF, plotting_parameters: dict, estimation_details: list = None
):
    from plotto import mark_plot

    params = determine_params(df=df)

    # params['data'] = params['data'].assign(time=lambda x: x['time'].astype(str))

    agg_plot_params = {
        **params,
        "y": "ATT",
        "x": "time",
        "points": True,
        "lines": True,
        "eticks": True,
        "ebars": True,
        "table_note": estimation_details,
        "tooltip": "ATT",
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# --------------------------- overall ----------------------------------


# plot_overall_agg:
# base:
# sample splits:
# groups:
# groups + sample splits:


def plot_overall_agg(
        df: PandasDF, plotting_parameters: dict, estimation_details: list = None
):
    from plotto import mark_plot

    title = get_title(df)

    idx_names, df = single_idx(df)

    if len(df) == 1:
        print("Plot not implemented")
        return None

    if "sample_name" in list(df) and "stratum" in list(df):
        print("Plot not implemented yet")
        return None

    if "sample_name" in list(df):
        y = "sample_name"
        df["sample_name"] = df["sample_name"].astype(str)
    elif "stratum" in list(df):
        y = "stratum"
        df["stratum"] = df["stratum"].astype(str)

    else:
        print("Plot not implemented")
        return None

    plot_event_agg_params = {
        "data": df,
        "x": "ATT",
        "y": y,
        "vertical": False,
        "points": True,
        "lines": False,
        "ebars": True,
        "eticks": True,
        # 'dash_lines_by': None,
        # 'shape_by': grouping_name,
        "title": title,
        "table_note": estimation_details,
        "tooltip": "ATT",
    }

    plot_event_agg_params.update(plotting_parameters)

    return mark_plot(**plot_event_agg_params)


# --------------------------- overall ----------------------------------


def xcoeff_plot(
        df: PandasDF, plotting_parameters: dict, estimation_details: list = None
):
    from plotto import mark_plot

    title = get_title(df)

    idx_names, df = single_idx(df)

    plot_event_agg_params = {
        "data": df,
        "x": "ATT",
        "vertical": False,
        "points": True,
        "lines": False,
        "ebands": False,
        "ebars": True,
        "eticks": True,
        # 'dash_lines_by': None,
        # 'shape_by': grouping_name,
        "title": title,
        "table_note": estimation_details,
        "tooltip": "ATT",
    }

    plot_event_agg_params.update(plotting_parameters)

    return mark_plot(**plot_event_agg_params)


# ------------------------- helpers ------------------------------------


# ---------

from pandas import DataFrame as PandasDF

from differences.models.tools.shared import single_idx


def plot_event_study(
        df: PandasDF, plotting_parameters: dict, estimation_details: dict = None
):
    _, df = single_idx(df)

    agg_plot_params = {
        "data": df,
        "y": "parameter",
        "x": "relative_period",
        "points": True,
        "lines": True,
        "ebands": True,
        "table_note": estimation_details,
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# ----------------------------------------------------------------------


def get_template_es_plot_params():
    template_es_params = dict(
        spines='b',

        errorbar_hkws={
            'markerfacecolor': 'white',
            'markeredgecolor': 'black',
            'markersize': 6
        },

        axhline_kws={
            'y': 0,
            'dashes': (3, 3),
            "alpha": 0.4
        },

        axvline_kws={
            'x': '0',
            'dashes': (3, 3),
            "alpha": 0.4

        },
        # remove the y axis ticks
        tick_params_kws={
            'axis': 'y',
            'length': 0
        },

        grid_kws={
            "axis": "y",
            "alpha": 0.4
        },

        set_xlabel_kws={
            'xlabel': 'Relative Periods',
            'labelpad': 10
        },

        set_ylabel_kws={
            'ylabel': 'Beta',
            'labelpad': 10
        },

        set_title_kws={
            'label': 'Event Study Plot',
            "pad": 20
        }
    )

    return template_es_params


