from __future__ import annotations

from differences.models.tools.shared import parse_att_gt_titles, single_idx
import pandas as pd


def grouping_facet_difference(
        idx_names: list,
        grouping_name: str = None,
        facet_name: str = None,
        difference: bool = False,
        diff_name: str = None,
):
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

    if "sample_name" in idx_names and "stratum" in idx_names:
        grouping_name, facet_name = "sample_name", "stratum"

    elif "sample_name" in idx_names:
        grouping_name = "sample_name"

    elif "stratum" in idx_names:
        grouping_name = "stratum"

    if difference:
        if "stratum" in idx_names:
            grouping_name = "stratum"
            diff_name = "samples"

        if "sample_name" in idx_names:
            grouping_name = "sample_name"
            diff_name = "strata"

    return grouping_name, facet_name, diff_name


def determine_params(
        df: pd.DataFrame,
        grouping_name: str = None,
        facet_name: str = None,
        diff_name: str = None,
):
    title = parse_att_gt_titles(df)  # plot title: from the nt

    idx_names, df = single_idx(df)  # names of the table indexes: cohort-time...

    if "stratum" in list(df):
        df["stratum"] = df["stratum"].astype(str)

    difference = "difference_between" in idx_names

    grouping_name, facet_name, diff_name = grouping_facet_difference(
        idx_names=idx_names,
        grouping_name=grouping_name,
        facet_name=facet_name,
        diff_name=diff_name,
        difference=difference,
    )

    if difference:
        diff_name = diff_name if diff_name is not None else ""
        difference = (
                f"difference between {diff_name}: " + df["difference_between"].unique()[0]
        )

    params = {
        "data": df,
        "title": title,
        # 'subtitle': difference,
        "shape_by": grouping_name,
        "facet_group": facet_name,
    }

    return params
