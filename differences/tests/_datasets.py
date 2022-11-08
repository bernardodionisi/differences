import os
import numpy as np
import pandas as pd

from differences.datasets import load_data

# ------------------------ load data -----------------------------------

# pd.set_option('display.max_columns', 20)

py_data = {'mpdta': {**load_data.mpdta()},
           'rc_multi_period': {**load_data.rc_multi_period()}
           }

R_data = {'mpdta': {**load_data.mpdta(py_format=False),
                    'panel': True,
                    'allow_unbalanced_panel': False},

          'rc_multi_period': {**load_data.rc_multi_period(py_format=False),
                              'panel': False,
                              'allow_unbalanced_panel': False},
          }


# ----------------------------------------------------------------------

data_name = 'mpdta'

weights = np.random.uniform(
    low=0, high=1,
    size=len(py_data[data_name]['data']))

py_data[data_name]['data']['random_weights'] = weights
R_data[data_name]['data']['random_weights'] = weights


# ----------------------------------------------------------------------

data_name = 'rc_multi_period'

weights = np.random.uniform(
    low=0, high=1,
    size=len(py_data[data_name]['data']))

py_data[data_name]['data']['random_weights'] = weights
R_data[data_name]['data']['random_weights'] = weights
