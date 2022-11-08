from pandas import DataFrame
from plotto import mark_plot

from ..tools.utility import single_idx


def plot_event_study(df: DataFrame,
                     plotting_parameters: dict,
                     estimation_details: dict = None
                     ):
    _, df = single_idx(df)

    agg_plot_params = {
        'data': df,
        'y': 'parameter',
        'x': 'relative_period',

        'points': True,
        'lines': True,
        'ebands': True,

        'table_note': estimation_details
    }

    agg_plot_params.update(plotting_parameters)

    return mark_plot(**agg_plot_params)
