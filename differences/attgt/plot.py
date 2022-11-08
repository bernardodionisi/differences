from pandas import DataFrame
from plotto import mark_plot

from ..tools.utility import get_title, single_idx


# --------------------------- att_gt ----------------------------------

# plot_att_gt:
# base: one plot
# sample splits: facet on sample splits vertically
# groups: facet on groups splits vertically
# groups + sample splits: facet on sample splits vertically & groups horizontally

def plot_att_gt(df: DataFrame,
                plotting_parameters: dict,
                estimation_details: dict = None,
                save_fname: str = None,
                ):
    title = get_title(df)

    _, df = single_idx(df)

    if 'sample_name' in list(df) or 'stratum' in list(df):
        print('Plot not implemented yet')
        return None

    df['cohort'] = df['cohort'].astype(str)
    # df['time'] = df['time'].astype(str)

    plot_att_gt_params = {
        'data': df,

        'y': 'ATT',
        'x': 'time',

        'points': True,
        'lines': True,
        'eticks': True,
        'ebars': True,

        'shape_by': 'cohort',
        'color_by': 'post',
        'select_by': 'hc',

        'title': title,
        'configure_legend_shape_by': {'title': 'Cohort'},

        'table_note': estimation_details,
        'save_fname': save_fname,
        'zero_in_xscale': False,

    }

    plot_att_gt_params.update(plotting_parameters)

    return mark_plot(**plot_att_gt_params)


# --------------------------- event ----------------------------------


def plot_event_agg(df: DataFrame,
                   plotting_parameters: dict,
                   estimation_details: list = None
                   ):
    params = determine_params(df=df)

    params['data'] = params[
        'data'].assign(relative_period=lambda x: x['relative_period'].astype(int))

    agg_plot_params = {

        **params,

        'y': 'ATT',
        'x': 'relative_period',

        'points': True,
        'lines': True,
        'ebands': True,

        'table_note': estimation_details,
        'tooltip': 'ATT'
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# --------------------------- cohort -----------------------------------


def plot_cohort_agg(df: DataFrame,
                    plotting_parameters: dict,
                    estimation_details: list = None
                    ):
    params = determine_params(df=df)

    params['data'] = params['data'].assign(cohort=lambda x: x['cohort'].astype(str))

    agg_plot_params = {
        **params,

        'y': 'ATT',
        'x': 'cohort',

        'points': True,
        'lines': True,
        'eticks': True,
        'ebars': True,

        'table_note': estimation_details,
        'tooltip': 'ATT'
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# --------------------------- time ---------------------------------

def plot_time_agg(df: DataFrame,
                  plotting_parameters: dict,
                  estimation_details: list = None
                  ):
    params = determine_params(df=df)

    # params['data'] = params['data'].assign(time=lambda x: x['time'].astype(str))

    agg_plot_params = {
        **params,

        'y': 'ATT',
        'x': 'time',

        'points': True,
        'lines': True,
        'eticks': True,
        'ebars': True,

        'table_note': estimation_details,
        'tooltip': 'ATT'
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)


# --------------------------- overall ----------------------------------


# plot_overall_agg:
# base:
# sample splits:
# groups:
# groups + sample splits:

def plot_overall_agg(df: DataFrame,
                     plotting_parameters: dict,
                     estimation_details: list = None
                     ):
    title = get_title(df)

    idx_names, df = single_idx(df)

    if len(df) == 1:
        print('Plot not implemented')
        return None

    if 'sample_name' in list(df) and 'stratum' in list(df):
        print('Plot not implemented yet')
        return None

    if 'sample_name' in list(df):
        y = 'sample_name'
        df['sample_name'] = df['sample_name'].astype(str)
    elif 'stratum' in list(df):
        y = 'stratum'
        df['stratum'] = df['stratum'].astype(str)

    else:
        print('Plot not implemented')
        return None

    plot_event_agg_params = {

        'data': df,

        'x': 'ATT',
        'y': y,

        'vertical': False,

        'points': True,
        'lines': False,
        'ebars': True,
        'eticks': True,

        # 'dash_lines_by': None,
        # 'shape_by': grouping_name,
        'title': title,
        'table_note': estimation_details,

        'tooltip': 'ATT'
    }

    plot_event_agg_params.update(plotting_parameters)

    return mark_plot(**plot_event_agg_params)


# --------------------------- overall ----------------------------------


def xcoeff_plot(df: DataFrame,
                plotting_parameters: dict,
                estimation_details: list = None
                ):
    title = get_title(df)

    idx_names, df = single_idx(df)

    plot_event_agg_params = {
        'data': df,

        'x': 'ATT',
        'vertical': False,

        'points': True,
        'lines': False,
        'ebands': False,
        'ebars': True,
        'eticks': True,

        # 'dash_lines_by': None,
        # 'shape_by': grouping_name,
        'title': title,
        'table_note': estimation_details,
        'tooltip': 'ATT'
    }

    plot_event_agg_params.update(plotting_parameters)

    return mark_plot(**plot_event_agg_params)


# ------------------------- helpers ------------------------------------


def grouping_facet_difference(idx_names: list,
                              grouping_name: str = None,
                              facet_name: str = None,
                              difference: bool = False,
                              diff_name: str = None):
    """
    in case of difference, there is a 'difference_between' variable,
    which can indicate the difference between samples or strata.
    However, in case the result dataframe includes a 'stratum' column
    then the difference is taken between samples and if it includes a
    'sample_name' column then the difference is taken between 'strata'

    grouping_name: determines the color/shapes within the sample plot
    facet_name: splits the plot into multiple facets
    diff_name is just for the name
    """

    if 'sample_name' in idx_names and 'stratum' in idx_names:
        grouping_name, facet_name = 'sample_name', 'stratum'

    elif 'sample_name' in idx_names:
        grouping_name = 'sample_name'

    elif 'stratum' in idx_names:
        grouping_name = 'stratum'

    if difference:
        if 'stratum' in idx_names:
            grouping_name = 'stratum'
            diff_name = 'samples'

        if 'sample_name' in idx_names:
            grouping_name = 'sample_name'
            diff_name = 'strata'

    return grouping_name, facet_name, diff_name


def determine_params(df: DataFrame,
                     grouping_name: str = None,
                     facet_name: str = None,
                     diff_name: str = None):
    title = get_title(df)  # plot title: from the nt

    idx_names, df = single_idx(df)  # names of the table indexes: cohort-time...

    if 'stratum' in list(df):
        df['stratum'] = df['stratum'].astype(str)

    difference = 'difference_between' in idx_names

    grouping_name, facet_name, diff_name = grouping_facet_difference(
        idx_names=idx_names,
        grouping_name=grouping_name,
        facet_name=facet_name,
        diff_name=diff_name,
        difference=difference)

    if difference:
        diff_name = diff_name if diff_name is not None else ''
        difference = f'difference between {diff_name}: ' + df['difference_between'].unique()[0]

    params = {
        'data': df,
        'title': title,
        # 'subtitle': difference,

        'shape_by': grouping_name,
        'facet_group': facet_name
    }

    return params
