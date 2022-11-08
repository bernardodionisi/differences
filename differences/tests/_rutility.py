import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import rpy2.robjects.conversion as R_converter
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter, Formula


# ------------------------ R related resources -------------------------

# os.environ['PATH'] = "/Library/Frameworks/R.framework/Resources:$PATH"
# os.environ['PATH'].split(':')


def _none2null(none_obj):
    return ro.r("NULL")


none_to_R = R_converter.Converter("None converter")
none_to_R.py2rpy.register(type(None), _none2null)

# R_utils = importr('utils')
# R_base = importr('base')

R_did = importr('did')


# R_did.__version__


# ------------------------- select -------------------------------------

def r_attgt(data: pd.DataFrame,
            entity_name: str,
            time_name: str,
            cohort_name: str,
            y_name: str,
            panel: bool,
            allow_unbalanced_panel: bool,
            control_group: str,
            anticipation: int,
            base_period: str,
            est_method: str,
            weights_name: str = None,
            xformla: str = None
            ):
    with R_converter.localconverter(default_converter + none_to_R):
        r_attgt_res = R_did.att_gt(
            idname=entity_name,
            tname=time_name,
            gname=cohort_name,
            yname=y_name,

            panel=panel,
            allow_unbalanced_panel=allow_unbalanced_panel,
            data=pandas_2_R(data),
            weightsname=weights_name,

            xformla=Formula(xformla) if xformla is not None else None,

            control_group=control_group,
            anticipation=anticipation,
            base_period=base_period,
            est_method=est_method,

            cband=False,
            bstrap=False,
        )

    return r_attgt_res


# ------------------------- select -------------------------------------

def r_agg(r_attgt_res, agg_type: str):
    r_agg_res = R_did.aggte(
        r_attgt_res,
        type=agg_type,
        na_rm=True
    )

    return r_agg_res


def r_agg_overall(r_agg_res):
    # R overall aggregation
    r_agg_overall_res = R_att_overall_to_pandas(r_agg_res)

    r_agg_overall_res = r_agg_overall_res.rename(
        columns={'overall.att': 'att',
                 'overall.se': 'se'}
    )

    return r_agg_overall_res


def r_agg_not_overall(r_agg_res):
    # R aggregation
    r_agg_not_overall_res = R_att_aggregate_to_pandas(r_agg_res)
    r_agg_not_overall_res = r_agg_not_overall_res.rename(
        columns={'att.egt': 'att',
                 'se.egt': 'se'}
    )
    return r_agg_not_overall_res


# ------------------------ rpy2 helpers --------------------------------

def R_2_pandas(r_data):
    """converts R dataframe to pandas dataframe
    https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html"""
    with R_converter.localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.rpy2py(r_data)
    return df


def pandas_2_R(pd_df):
    """converts pandas dataframe to R dataframe
    https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html"""
    with R_converter.localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.py2rpy(pd_df)
    return data


def R_list_to_python_dict(R_result):
    output = {}
    for i in range(len(R_result)):
        try:
            output.update({R_result.names[i]: list(R_result[i])})
        except Exception as e:
            # print(f'exception for {R_result.names[i]}')
            continue
    return output


def R_attgt_to_pandas(R_result):
    return pd.DataFrame({k: v for k, v in R_list_to_python_dict(R_result).items()
                         if k in ['group', 't', 'att', 'se']})


def R_att_overall_to_pandas(R_result):
    return pd.DataFrame({k: v for k, v in R_list_to_python_dict(R_result).items()
                         if k in ['overall.att', 'overall.se']})


def R_att_aggregate_to_pandas(R_result):
    return pd.DataFrame({k: v for k, v in R_list_to_python_dict(R_result).items()
                         if k in ['egt', 'att.egt', 'se.egt']})

# # before installing rpy2,
# if you want link to main R version: export PATH="/Library/Frameworks/R.framework/Resources:$PATH"

# import os
# os.environ['PATH'] = "/Library/Frameworks/R.framework/Resources:$PATH"

# # import rpy2
# # print(rpy2.__version__)
#

#
# # R --version
# print(f'{ro.r("R.version").rx2("major")[0]}.'
#       f'{ro.r("R.version").rx2("minor")[0]}')
#
# # from rpy2.robjects.packages import importr
# # r_utils = importr('utils')
# # # select CRAN mirror for installing packages. but do not install from here
# # # utils.chooseCRANmirror(ind=1) # select the first mirror in the list
