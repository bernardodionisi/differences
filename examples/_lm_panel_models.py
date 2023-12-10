import pandas as pd
from linearmodels.datasets import wage_panel
from linearmodels.panel.utility import AbsorbingEffectWarning

data = wage_panel.load()
year = pd.Categorical(data.year)
data = data.set_index(["nr", "year"])
data["year"] = year
print(wage_panel.DESCR)
print(data.head())

import statsmodels.api as sm
from linearmodels.panel import PooledOLS

exog_vars = ["black", "hisp", "exper", "expersq", "married", "educ", "union", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PooledOLS(data.lwage, exog)
pooled_res = mod.fit()
print(pooled_res)

from linearmodels.panel import RandomEffects

mod = RandomEffects(data.lwage, exog)
re_res = mod.fit()
print(re_res)

from linearmodels.panel import BetweenOLS

exog_vars = ["black", "hisp", "exper", "married", "educ", "union"]
exog = sm.add_constant(data[exog_vars])
mod = BetweenOLS(data.lwage, exog)
be_res = mod.fit()
print(be_res)

from linearmodels.panel import PanelOLS

exog_vars = ["expersq", "union", "married", "year"]

exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(
    data.lwage, exog, entity_effects=True, time_effects=True,
    drop_absorbed=True
)

fe_res = mod.fit()

import warnings
# Capture the warning issued by the function
cw = []
with warnings.catch_warnings(record=True) as captured_warnings:
    fe_res = mod.fit()
    print(captured_warnings)
    for w in captured_warnings:
        if issubclass(w.category, AbsorbingEffectWarning):
            cw.append(str(w.message))



fe_res = mod.fit()
print(fe_res)
dir(mod)

fe_res._var_names
fe_res
exog_vars = ["expersq", "union", "married"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True, time_effects=True)
fe_te_res = mod.fit()
print(fe_te_res)

from linearmodels.panel.data import PanelData

exog_vars = ["expersq", "union", "married"]
exog = sm.add_constant(data[exog_vars])
# Use the `PanelData` structure to simplify constructing the time IDs
time_ids = pd.DataFrame(
    PanelData(data.lwage).time_ids, index=data.index, columns=["Other Effect"]
)

mod = PanelOLS(data.lwage, exog, entity_effects=True, other_effects=time_ids)
fe_oe_res = mod.fit()
print(fe_oe_res)

from linearmodels.panel import FirstDifferenceOLS

exog_vars = ["exper", "expersq", "union", "married"]
exog = data[exog_vars]
mod = FirstDifferenceOLS(data.lwage, exog)
fd_res = mod.fit()
print(fd_res)

from differences.models.lm.panel_results import PanelModelComparison

check = PanelModelComparison([fd_res])

check.f_statistic
check.rsquared

from linearmodels.panel import compare

print(compare({"BE": be_res, "RE": re_res, "Pooled": pooled_res, "PanelOLS": fe_res}))

fe_res.included_effects

fe_res.model.dependent.vars

oops = [
    fe_res.name,  # fe_res.estimator_method,
    fe_res.nobs,
    fe_res._cov_type,  # fe_res.cov_estimator,
    fe_res.rsquared,
    fe_res.rsquared_within,
    fe_res.rsquared_between,
    fe_res.rsquared_overall,
    fe_res.f_statistic,
]
