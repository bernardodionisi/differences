from __future__ import annotations
import re

import polars as pl

from differences._OLD_NOW_PAGURO.shared.utils_expr import expr_all_output_names, expr_all_root_names


class DataMatrix:
    def __init__(
            self,
            data: pl.DataFrame,
            index_columns: str | list[str] = None,  # this can be for example [entity, time]
    ):

        self.data = data

        if isinstance(index_columns, str):
            index_columns = [index_columns]

        self.index_columns = index_columns
        # may not be unique (in that case be careful joining with other data)

        self.data_matrix = None

        self.y_column: str | None = None
        self.x_columns: list[str] | None = None

    # def __repr__(self):
    #     """
    #     DataMatrix representation.
    #     """
    #     # this could also be delegated
    #     four_spaces = "    "
    #
    #     data_repr = self.data.__repr__().replace('\n', f'\n{four_spaces}\t')
    #     data_matrix_repr = self.data_matrix.__repr__().replace('\n', f'\n{four_spaces}\t')
    #
    #     out = (
    #         f"\n{self.__class__.__qualname__} [at {hex(id(self))}]\n"
    #         f"{four_spaces}data = {data_repr}\n"
    #         f"{four_spaces}data_matrix = {data_matrix_repr}\n"
    #         f"{four_spaces}id_columns = {self.index_columns}\n"
    #         f"{four_spaces}y_column = {self.y_column}\n"
    #         f"{four_spaces}x_columns = {self.x_columns}\n"
    #     )
    #     return out

    def build_model_matrix(
            self,
            formula: str | list,
            include_intercept: bool = True,
            *,
            other_columns: list[str] | bool = True,
    ) -> DataMatrix:

        # either tuple[str, str] | tuple[list, list]

        y_from_formula, x_from_formula = parse_formula_arg(
            formula=formula,
            columns=self.data.columns,
            include_intercept=include_intercept
        )

        if isinstance(y_from_formula, str):
            data_matrix = self._get_formulaic_model_matrix(
                y_formula=y_from_formula, x_formula=x_from_formula
            )
        else:
            data_matrix = self._get_polars_model_matrix(
                y_formula=y_from_formula, x_formula=x_from_formula
            )

        # --------------------------------------------------------------

        if other_columns:
            if isinstance(other_columns, bool):
                other_columns = None

            # now data has no nulls for the columns in formula but could have for 'keep_columns'
            data_matrix = self._join_other_columns(
                data=data_matrix, other_columns=other_columns
            )

        # --------------------------------------------------------------

        self.data_matrix = data_matrix

        return self

    def _join_other_columns(
            self,
            data: pl.DataFrame,  # data_matrix
            other_columns: str | list[str] = None
    ) -> pl.DataFrame:
        # adding columns to model matrix just to keep them together instead of passing them
        # around separately. Joined only after the named of the Xs are saved.

        if other_columns is None:
            # all the columns in data
            other_columns = self.data.columns

        keep_columns = [
            i
            for i in other_columns
            # todo: if user passes other_columns keep even if here in formula?
            if (i not in self._formula_root_columns) or
               (i in self.index_columns)  # make sure id columns are always included
        ]

        data = (
            data
            .join(
                self.data
                .select(keep_columns),
                on=self.index_columns,
                # id these are not unique it's a problem, should use row_nr?
                how="left"
            )
        )

        return data

    def _join_other_data(self):
        # if other_data is not None:
        #     data_matrix = data_matrix.join(other_data, on=self.id_columns, how="left")
        pass

    # ------------------------------------------------------------------

    def _get_polars_model_matrix(
            self,
            y_formula: list[pl.Expr],
            x_formula: list[pl.Expr],
    ) -> pl.DataFrame:

        self._y_root_column = expr_all_root_names(*y_formula)  # first column is y
        self._x_root_columns = expr_all_root_names(*x_formula)  # list of remaining columns

        self._formula_root_columns = self._y_root_column + self._x_root_columns

        # save output column names
        self.y_column = expr_all_output_names(*y_formula)  # first column is y
        self.x_columns = expr_all_output_names(*x_formula)  # list of remaining columns

        # drop nulls here
        return self.data.select(self.index_columns + y_formula + x_formula).drop_nulls()

    def _get_formulaic_model_matrix(
            self,
            y_formula: str,
            x_formula: str,
    ) -> pl.DataFrame:
        try:
            import formulaic
            import pandas as pd
        except ImportError as e:
            raise e

        # todo: maybe just don't parse formula?
        # spec = f"{y_formula} + {x_formula}"
        spec = f"{y_formula} ~ {x_formula}"

        data = self.data.to_pandas()

        if self.index_columns is not None:
            data = data.set_index(self.index_columns)

        mm = formulaic.Formula(spec=spec).get_model_matrix(data)

        # save columns used to create the design_matrix
        self._y_root_column = list(mm.lhs.model_spec.variables_by_source['data'])
        self._x_root_columns = list(mm.rhs.model_spec.variables_by_source['data'])

        self._formula_root_columns = self._y_root_column + self._x_root_columns

        # save columns in the design_matrix, y and Xs
        self.y_column = list(mm.lhs.model_spec.feature_names)
        self.x_columns = list(mm.rhs.model_spec.feature_names)

        # if not pandas, even before pl, something strange was happening, can't remember what
        y_data, x_data = pd.DataFrame(mm.lhs), pd.DataFrame(mm.rhs)

        if self.index_columns is not None:
            y_data = y_data.reset_index()
            x_data = x_data.reset_index(drop=True)  # do not keep the id_columns twice

        return pl.concat([pl.from_pandas(y_data), pl.from_pandas(x_data)], how="horizontal")


# ----------------------------------------------------------------------

def parse_formula_arg(
        formula: str | list,
        columns: list,
        include_intercept: bool,
) -> tuple[str, str] | tuple[list[pl.Expr], list[pl.Expr]]:
    has_intercept, terms = True, None

    if isinstance(formula, str):
        has_intercept, terms = _check_formula(formula=formula, columns=columns)

        if terms is None:  # the model_matrix is not just a list of terms
            # strings to be used in formulaic, split them for now, recombine them later
            y_fml, x_fml = _split_formula_into_y_and_x(formula=formula)
            return y_fml, x_fml

        # if we are here terms is a list of strings

    elif isinstance(formula, list):
        terms = formula
    else:
        ValueError(f"Invalid type: {type(formula)}")

    out = []
    for t in terms:
        if isinstance(t, str):
            out.append(pl.col(t))
        else:
            out.append(t)  # is already an expression

    if has_intercept and include_intercept:
        out.insert(1, pl.lit(1).alias("Intercept"))

    return out[:1], out[1:]  # [y], [Xs]


def _check_formula(formula: str, columns: list) -> tuple[bool, list[str] | None]:
    """
    Determines whether the elements in the Wilkinson model_matrix are simple columns or
    more complex objects and checks if the model_matrix has an intercept.

    Parameters
    ----------
    formula : str
        The model_matrix in Wilkinson notation.
    columns : list
        List of column names.

    Returns
    -------
    tuple
        has_intercept (bool) : Indicates if the model_matrix has an intercept.
        list of terms (list[str] | None) : list of terms if the elements are all
        simple columns from the provided
        list, False otherwise.
    """

    # By default, the model_matrix includes an intercept unless specified with "-1"
    has_intercept = True

    # Split model_matrix into LHS and RHS, if no '~' then entire model_matrix is LHS
    if '~' not in formula:
        lhs = formula.strip()
        rhs = ''
    else:
        lhs, rhs = [side.strip() for side in formula.split('~')]

    # intercept as + 1 or - 1 (ignoring edge cases like "y ~ x +1 -1")
    if re.search(r'-\s*1', rhs):
        has_intercept = False

        # remove the "- 1" to simplify further checks
        rhs = re.sub(r'-\s*1', '', rhs).strip()

    elif re.search(r'\+\s*1', rhs) or re.search(r'~\s*1\s*(?=\+|$)', formula):
        rhs = re.sub(r'^\s*1\s*(?=\+|$)|\+\s*1\b', '', rhs).strip()

    # Checking LHS
    if lhs not in columns:
        return has_intercept, None

    # If there's no RHS, then it's just a simple column on the LHS
    if not rhs:
        return has_intercept, [lhs]

    # Split the RHS by '+' to get the individual terms
    terms = [
        term.strip()
        for term in rhs.split('+')
        if term.strip()  # ensure no empty terms
    ]

    # Check if each term is a simple column
    for term in terms:
        if term not in columns:
            return has_intercept, None

    return has_intercept, [lhs] + terms


def _split_formula_into_y_and_x(
        formula: str,
) -> tuple[str, str]:
    y, x = formula, None

    if "~" in formula:
        y, x = formula.split("~")
        y, x = y.strip(), x.strip()

    if x is None:  # if no X was passed then just use an intercept
        x = "1"

    return y, x  # y_formula and x_formula
