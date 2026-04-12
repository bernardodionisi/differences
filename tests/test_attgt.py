from __future__ import annotations

import pandas as pd

from differences import ATTgt


def test_attgt_init(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat")
    assert att.cohort_column == "first_treat"
    assert att.anticipation == 0
    assert att.base_period_type == "varying"


def test_attgt_fit_reg(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat")
    result = att.fit(formula="lemp", est_method="reg", control_group="never_treated")
    result_df = result.to_pandas()

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0
    assert any("ATT" in str(col) for col in result_df.columns)


def test_attgt_fit_dr(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat")
    result = att.fit(formula="lemp ~ lpop", est_method="dr", control_group="never_treated")
    result_df = result.to_pandas()

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_attgt_aggregate_event(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat")
    att.fit(formula="lemp", est_method="reg", control_group="never_treated")
    agg = att.aggregate("event")

    assert agg is not None


def test_attgt_aggregate_simple(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat")
    att.fit(formula="lemp", est_method="reg", control_group="never_treated")
    agg = att.aggregate("simple")

    assert agg is not None


def test_attgt_universal_base(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat", base_period="universal")
    result = att.fit(formula="lemp", est_method="reg", control_group="never_treated")
    result_df = result.to_pandas()

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_attgt_not_yet_treated(mpdta):
    att = ATTgt(data=mpdta, cohort_column="first_treat")
    result = att.fit(formula="lemp", est_method="reg", control_group="not_yet_treated")
    result_df = result.to_pandas()

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0
