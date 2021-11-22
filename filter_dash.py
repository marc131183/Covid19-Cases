import pandas as pd
import numpy as np
import operator


def split_query(query):
    # remove spaces that are not between " characters
    temp = ""
    open_string = False
    for character in query:
        if character == '"':
            open_string = not open_string
            temp = temp + character
        elif character != " " or open_string:
            temp = temp + character

    query = temp
    brackets_open = 0

    # remove outer brackets, if they are from one pair, e.g. ((x > 0) or (y = 0)) is converted to (x > 0) or (y = 0)
    has_outer_brackets = True
    while has_outer_brackets:
        if len(query) > 2 and query[0] == "(" and query[-1] == ")":
            for i, character in enumerate(query):
                if character == "(":
                    brackets_open += 1
                elif character == ")":
                    brackets_open -= 1
                    if brackets_open == 0:
                        if i == len(query) - 1:
                            query = query[1:-1]
                        else:
                            has_outer_brackets = False
                            break
        else:
            has_outer_brackets = False

    brackets_open = 0
    sub_query = {"LHS": "", "RHS": "", "operator": "", "operator_type": "logical"}
    logical_operators = {"and": np.logical_and, "or": np.logical_or}
    quotation_marks = False

    for i, character in enumerate(query):
        if character == "(":
            brackets_open += 1
        elif character == ")":
            brackets_open -= 1
        elif character == '"':
            quotation_marks = not quotation_marks
        # if there are no brackets/quotation marks open, look for logical operators
        if brackets_open == 0 and not quotation_marks:
            for op in logical_operators.keys():
                # found logical operator that is not in brackets, split into LHS and RHS between it
                if i + len(op) < len(query) and query[i : i + len(op)] == op:
                    sub_query["LHS"] = query[:i]
                    sub_query["operator"] = logical_operators[op]
                    sub_query["RHS"] = query[i + len(op) :]

                    # if the LHS/RHS contains subqueries, we need to split further
                    if any([op in sub_query["LHS"] for op in logical_operators.keys()]):
                        sub_query["LHS"] = split_query(sub_query["LHS"])
                    # if it is a single query, we can parse it
                    else:
                        sub_query["LHS"] = parse_sub_query(sub_query["LHS"])

                    if any([op in sub_query["LHS"] for op in logical_operators.keys()]):
                        sub_query["RHS"] = split_query(sub_query["RHS"])
                    else:
                        sub_query["RHS"] = parse_sub_query(sub_query["RHS"])

                    return sub_query

    # we went through the whole query but didn't find any logical operators that weren't in brackets, so that means it is a single query
    sub_query["operator_type"] = "single"
    sub_query["LHS"] = query
    sub_query["LHS"] = parse_sub_query(sub_query["LHS"])

    return sub_query


def parse_sub_query(side):
    side = side.replace("(", "").replace(")", "")
    sub = {"LHS": None, "RHS": None, "operator": None, "operator_type": "relational"}
    valid_operators = {
        "<=": operator.le,
        ">=": operator.ge,
        "!=": operator.ne,
        "=": operator.eq,
        "<": operator.lt,
        ">": operator.gt,
    }

    # find operator and split into LHS/RHS
    for i in range(len(side)):
        for oper in valid_operators.keys():
            if len(side) > i + len(oper) and side[i : i + len(oper)] == oper:
                sub["LHS"] = side[:i]
                sub["RHS"] = side[i + len(oper) :]
                sub["operator"] = valid_operators[oper]
                return sub

    return sub


def parse_expression(df_, expression):
    valid_operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }
    valid_col_operators = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "max": np.max,
        "min": np.min,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1],
    }

    operators = []
    sub_expressions = [""]

    # split expression between valid operators
    for character in expression:
        if character in valid_operators:
            operators.append([character, 0])
            sub_expressions.append("")
        else:
            sub_expressions[-1] += character

    # assign priorities to operators according to brackets found
    brackets_open = 0
    for i in range(len(sub_expressions) - 1):
        # check if string is empty
        if sub_expressions[i] != "":
            # if there are brackets on opening and end, we can remove them
            while sub_expressions[i][0] == "(" and sub_expressions[i][-1] == ")":
                sub_expressions[i] = sub_expressions[i][1:-1]
            brackets_open += sub_expressions[i].count("(")
            brackets_open -= sub_expressions[i].count(")")
            # update priority of the corresponding operator
            operators[i][1] = brackets_open
            # we won't need the brackets anymore, so we can remove them
            sub_expressions[i] = sub_expressions[i].replace("(", "").replace(")", "")

            # also remove brackets from last sub_expression
            sub_expressions[-1] = sub_expressions[-1].replace("(", "").replace(")", "")

    # parse sub_expressions
    for i in range(len(sub_expressions)):
        # check if it's empty, then we assign 0 (for cases of the form x*-y pr x/-y)
        if sub_expressions[i] == "":
            sub_expressions[i] = 0
            # we have to increase the priority of the corresponding operator, because otherwise it will be incorrect
            operators[i][1] = np.inf
        # if it contains {}, then it's a column
        elif "{" in sub_expressions[i]:
            cur_op = lambda x: x
            # check if there is a valid col operator before
            index_first_bracket = sub_expressions[i].index("{")
            if index_first_bracket != 0:
                cur_op = valid_col_operators[sub_expressions[i][:index_first_bracket]]
            sub_expressions[i] = cur_op(
                df_[sub_expressions[i][index_first_bracket + 1 : -1]]
            )
        # if it contains "", then it's a string
        elif '"' in sub_expressions[i]:
            # " should be at beginning and end, remove them
            sub_expressions[i] = sub_expressions[i][1:-1]
        elif sub_expressions[i].isnumeric():
            sub_expressions[i] = float(sub_expressions[i])

    # apply operators according to priorities
    all_op_same = False
    while len(sub_expressions) > 1:
        # check if all operators have the same priority
        if all_op_same or all([operators[0][1] == elem[1] for elem in operators]):
            all_op_same = True
            # do * and / before + and -
            i = 0
            while i < len(operators):
                if operators[i][0] in ["*", "/"]:
                    sub_expressions[i] = valid_operators[operators[i][0]](
                        sub_expressions[i], sub_expressions[i + 1]
                    )

                    # remove operator and corresponding sub_expression (we combined 2 sub_expressions)
                    del operators[i]
                    del sub_expressions[i + 1]
                else:
                    i += 1

            i = 0
            while i < len(operators):
                if operators[i][0] in ["+", "-"]:
                    sub_expressions[i] = valid_operators[operators[i][0]](
                        sub_expressions[i], sub_expressions[i + 1]
                    )

                    # remove operator and corresponding sub_expression (we combined 2 sub_expressions)
                    del operators[i]
                    del sub_expressions[i + 1]
                else:
                    i += 1
        else:
            # find index of operator with maximum priority
            cur_max, cur_max_index = 0, 0
            for i in range(len(operators)):
                if operators[i][1] > cur_max:
                    cur_max, cur_max_index = operators[i][1], i

            # apply that operation
            sub_expressions[cur_max_index] = valid_operators[
                operators[cur_max_index][0]
            ](sub_expressions[cur_max_index], sub_expressions[cur_max_index + 1])

            # remove operator and corresponding sub_expression (we combined 2 sub_expressions)
            del operators[cur_max_index]
            del sub_expressions[cur_max_index + 1]

    return sub_expressions[0]


def resolve_query(df_, derived_query_structure):
    if derived_query_structure["operator_type"] == "relational":
        lhs = parse_expression(df_, derived_query_structure["LHS"])
        rhs = parse_expression(df_, derived_query_structure["RHS"])

        return derived_query_structure["operator"](lhs, rhs)

    if derived_query_structure["operator_type"] == "single":
        return resolve_query(df_, derived_query_structure["LHS"])

    if derived_query_structure["operator_type"] == "logical":
        return derived_query_structure["operator"](
            resolve_query(df_, derived_query_structure["LHS"]),
            resolve_query(df_, derived_query_structure["RHS"]),
        )


if __name__ == "__main__":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        parse_dates=["date"],
        nrows=1000,
    )

    query = "({new_cases} > (mean{new_cases} + 3 * std{new_cases})) or ({new_cases} < (mean{new_cases} - 3 * std{new_cases}))"
    should_be = (
        df["new_cases"] > df["new_cases"].mean() + 3 * df["new_cases"].std()
    ) | (df["new_cases"] < df["new_cases"].mean() - 3 * df["new_cases"].std())

    processed_query = split_query(query)

    print(np.array(should_be == resolve_query(df, processed_query)).all())
