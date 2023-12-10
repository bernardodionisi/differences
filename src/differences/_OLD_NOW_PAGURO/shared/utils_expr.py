def expr_all_root_names(*expr, **named_expr) -> list[str]:
    """Extract all the root column names"""

    # expr_all_root_names(*(pl.col("a").alias("b"),), **{"c": pl.col("d")}) -> ['a', 'd']

    out = []
    if expr:
        # if the expr is ([e1, e2],) -> (e1, e2)
        expr = _expr_list_in_tuple(*expr)
        for e in expr:
            out.extend(e.meta.root_names())

    for v in named_expr.values():
        out.extend(v.meta.root_names())

    return out


def expr_all_output_names(*expr, **named_expr) -> list[str]:
    """Extract all the output column names"""

    # expr_all_output_names(*(pl.col("a").alias("b"),), **{"c": pl.col("d")}) -> ['b', 'c']

    out = []
    if expr:

        # if the expr is ([e1, e2],) -> (e1, e2)
        expr = _expr_list_in_tuple(*expr)

        for e in expr:
            out.append(e.meta.output_name())

    # the name of the output columns are the keys of named_expr
    out.extend(named_expr.keys())

    return out


def expr_all_root_and_output_names(*expr, **named_expr) -> dict[tuple, str]:
    """Extract all the {root: output} column names"""

    out = {}
    if expr:

        expr = _expr_list_in_tuple(*expr)

        for e in expr:
            root_names = tuple(e.meta.root_names())
            out[root_names] = e.meta.output_name()

    for k, v in named_expr.items():
        out[k] = v.meta.output_name()

    return out


def expr_all_trees(*expr, **named_expr) -> list[str]:
    """Extract all the tree_formats"""

    out = []
    if expr:

        # if the expr is ([e1, e2],) -> (e1, e2)
        expr = _expr_list_in_tuple(*expr)

        for e in expr:
            out.append(e.meta.tree_format(return_as_string=True))

    for k, v in named_expr.items():
        out.append(f"[named] {k} \n\n{v.meta.tree_format(return_as_string=True)}")

    return out


def _expr_list_in_tuple(*expr):
    # when the *args: expr is list[expr] then convert into tuple as a regular *args
    if len(expr) == 1 and isinstance(expr[0], list):
        expr = tuple(expr[0])
    # if the expr is ([e1, e2],) -> (e1, e2)
    return expr
