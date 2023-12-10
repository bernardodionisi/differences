
data_matrix = DataMatrix(
    pl.DataFrame({
        "entity": [1, 1, 1, 2, 2, 2],
        "y": [.1, .2, .3, .4, .5, .6],
        "x": [1, 2, 3, 4, 5, 6]
    }),
    id_columns="entity"
)

check = data_matrix.build_model_matrix(formula="y ~ -1 + x")

data_matrix.y_column
data_matrix.x_columns

parse_formula_arg("y ~ 1", columns=["y", "x"])
parse_formula_arg("y ~ 1", columns=["y", "x", "z"])

import pandas
from formulaic import model_matrix
import formulaic

df = pandas.DataFrame({
    'y': [0, 1, 2],
    'a': ['A', 'B', 'C'],
    'b': [0.3, 0.1, 0.2],
})

dir(mm.rhs.model_spec)

dir(mm.rhs.feature_names)
dir(mm.rhs.feature_names)

list(mm.lhs.model_spec.feature_names)
mm.rhs.model_spec.feature_names

mm.model_spec.variables_by_source

# model_matrix("y ~ -1 +  a + b + a:b", df)


mm.required_data_variables
