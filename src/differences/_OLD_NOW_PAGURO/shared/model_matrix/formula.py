from __future__ import annotations


# useful issues:
# https://github.com/matthewwardrop/formulaic/issues/32
# https://github.com/matthewwardrop/formulaic/issues/60
# resolved after materialization: https://github.com/matthewwardrop/formulaic/pull/145

def split_formula(formula: str) -> tuple[str, list, bool]:
    """
    model_matrix: 'y ~ x + z | fe1 + fe2'.
    """
    regressors = []
    drop_intercept = False
    if "|" in formula:
        formula, regressors = formula.split("|")
        formula = formula.strip()

        if regressors.strip():
            regressors = parse_formula_regressors(regressors)
    if '~' not in formula:
        formula += '~ 1'
        drop_intercept = True

    return formula, regressors, drop_intercept


def parse_formula_regressors(formula_regressors: str) -> list:
    """inplace change of object/str dtype to categoricals."""
    try:
        regressors = [i.strip() for i in formula_regressors.split("+")]
        # todo: add interactions
    except ValueError:
        formula_regressors = formula_regressors.strip()
        if formula_regressors:
            # only one
            regressors = [formula_regressors]
        else:
            regressors = []

    return regressors


def parse_fe_from_formula(formula) -> tuple[str, list]:
    """
    basic extraction of fixed effects from model_matrix

    model_matrix: 'y ~ x + z | fe1 + fe2'
    """
    try:
        formula, fe = formula.split("|")
    except ValueError:
        return formula, []

    fe = [i.strip() for i in fe.split("+") if i.strip()]

    return formula, fe


def process_formula_twfe(
        formula: str,
        entity_column: str,
        time_column: str,
        stacked: bool = False,
        return_fe: bool = False,
):
    if "~" not in formula:
        formula = f"{formula} ~ 1"

    try:
        spec, fe = formula.split("|")

        if not fe.strip():  # not ''
            fe = []
            formula = spec.strip()
        else:
            fe = [i.strip() for i in fe.split("+") if i.strip()]

    except ValueError:  # if no | then use two-way fe

        fe = [entity_column, time_column]

        if stacked:
            formula = f"{formula} | {entity_column}_stack + {time_column}_stack"
        else:
            formula = f"{formula} | {entity_column} + {time_column}"

        # to use fe, then just specify a model_matrix with |

    if return_fe:
        return formula, fe

    return formula


def insert_covars_into_formula(formula: str, covariates: list):
    elements = parse_formula(formula)
    elements['covars'] += covariates
    return reconstruct_formula(**elements)


def parse_formula(formula: str, return_dict: bool = True) -> dict | tuple:
    """
    model_matrix: 'y ~ x + z | fe1 + fe2'.
    """
    try:
        formula, fe = formula.split("|")
        fe = [i.strip() for i in fe.split("+") if i.strip()]

    except ValueError:
        formula, fe = formula, []

    if not return_dict:
        return formula, fe

    if "~" in formula:
        dep, covars = formula.split("~")

        dep = dep.strip()
        covars = [i.strip() for i in covars.split("+") if i.strip()]
    else:
        dep, covars = formula.strip(), []

    return {"dependent": dep, "covars": covars, "fe": fe}


def reconstruct_formula(dependent: str, covars: list, fe: list):
    covars = ' + '.join(covars)
    fe = ' + '.join(fe)

    return f"{dependent} ~ {covars} | {fe}"
