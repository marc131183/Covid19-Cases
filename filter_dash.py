import dash
from dash.dependencies import Input, Output
from dash import dash_table
from dash import dcc
from dash import html
from numpy import split
import pandas as pd
import json
import numpy as np
import operator

df = pd.read_csv(
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
    parse_dates=["date"],
    nrows=1000,
)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Br(),
        dcc.Input(id="filter-query-input", placeholder="Enter filter query"),
        html.Hr(),
        dash_table.DataTable(
            id="datatable",
            columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
            data=df.to_dict("records"),
            page_size=15,
        ),
    ]
)


def split_query(query):
    # remove spaces that are not between " characters
    temp = ""
    open_string = False
    for character in query:
        if character == '"':
            open_string = not open_string
            temp = temp + character
        elif character != "" or open_string:
            temp = temp + character

    brackets_open = 0
    sub_query = {"LHS": "", "RHS": "", "operator": "", "operator_type": "logical"}
    split = []

    # remove outer brackets
    while (
        len(query) > 4
        and query[0] == "("
        and query[1] == "("
        and query[-1] == ")"
        and query[-2] == ")"
    ):
        query = query[1:-1]

    # query should have a LHS and RHS marked by brackets and the operator is between them
    # query should look like this: (*) and/or *
    for i, character in enumerate(query):
        if character == "(":
            if brackets_open == 0 and len(split) == 1:
                split.append(i)
                break
            brackets_open += 1
        elif character == ")":
            brackets_open -= 1
            if brackets_open == 0:
                split.append(i)

    # if it only contains a LHS
    if len(split) == 1:
        sub_query["operator_type"] = "single"
        sub_query["LHS"] = query
        sub_query["LHS"] = parse_sub_query(sub_query["LHS"])
    else:
        sub_query["LHS"] = query[: split[0] + 1]
        sub_query["RHS"] = query[split[1] :]
        sub_query["operator"] = query[split[0] + 1 : split[1]]

        # if the query contains subqueries, we need to split further
        if sub_query["LHS"].count("(") > 1:
            sub_query["LHS"] = split_query(sub_query["LHS"])
        # it's now a single subquery, we can parse it
        else:
            sub_query["LHS"] = parse_sub_query(sub_query["LHS"])

        if sub_query["RHS"].count("(") > 1:
            sub_query["RHS"] = split_query(sub_query["RHS"])
        else:
            sub_query["RHS"] = parse_sub_query(sub_query["RHS"])

    return sub_query


def parse_sub_query(side):
    side = side.replace("(", "").replace(")", "")
    sub = {"LHS": None, "RHS": None, "operator": None, "operator_type": "relational"}
    valid_operators = {
        "<=": operator.le,
        ">=": operator.ge,
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


# parse expression from left to right, ignoring any other rules
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
        "max": np.max,
        "min": np.min,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1],
    }
    cur_op = None
    out = None

    i = 0
    while i < len(expression):
        if expression[i] in valid_operators:
            cur_op = valid_operators[expression[i]]
            i += 1
        # found opening of column
        elif expression[i] == "{":
            op_to_apply = lambda x: x
            # check if there is a columnn operator before
            for col_op in valid_col_operators.keys():
                if i - len(col_op) > 0 and expression[i - len(col_op) : i] == col_op:
                    op_to_apply = valid_col_operators[col_op]
            # find closing of columm between i and the end of the string
            end_col = expression[i:].index("}") + i
            col_name = expression[i + 1 : end_col]
            if out is None:
                out = op_to_apply(df_[col_name])
            else:
                out = cur_op(out, op_to_apply(df_[col_name]))
            # set i to be behind closing of {column}
            i = end_col + 1
        # found opening of string value
        elif expression[i] == '"':
            # find closing of string
            end_string = expression[i + 1 :].index('"') + i + 1
            found_string = expression[i + 1 : end_string]
            if out is None:
                out = found_string
            else:
                out = cur_op(out, found_string)
            # set i to be behind closing of string
            i = end_string + 1
        elif expression[i].isnumeric():
            # if this is the end of the expression, this is also the end of this number
            if i + 1 == len(expression):
                number = float(expression[i])
                if out is None:
                    out = number
                else:
                    out = cur_op(out, number)
                i += 1
            # check how long the expression is numeric
            for j in range(i + 1, len(expression) + 1):
                if j == len(expression) or not expression[j].isnumeric():
                    number = float(expression[i:j])

                    if out is None:
                        out = number
                    else:
                        out = cur_op(out, number)
                    i = j
                    break
        else:
            i += 1

    return out


def resolve_query(df_, derived_query_structure):
    if derived_query_structure["operator_type"] == "relational":
        lhs = parse_expression(df_, derived_query_structure["LHS"])
        rhs = parse_expression(df_, derived_query_structure["RHS"])

        return derived_query_structure["operator"](lhs, rhs)

    if derived_query_structure["operator_type"] == "single":
        return resolve_query(df_, derived_query_structure["LHS"])

    if derived_query_structure["operator_type"] == "logical":
        if derived_query_structure["operator"] == "and":
            return np.logical_and(
                resolve_query(df_, derived_query_structure["LHS"]),
                resolve_query(df_, derived_query_structure["RHS"]),
            )
        elif derived_query_structure["operator"] == "or":
            return np.logical_or(
                resolve_query(df_, derived_query_structure["LHS"]),
                resolve_query(df_, derived_query_structure["RHS"]),
            )


@app.callback(
    Output("datatable", "data"),
    Output("datatable", "columns"),
    Input("filter-query-input", "value"),
)
def display_query(query):
    if query != None:
        processed_query = split_query(query.replace(" ", ""))
        df.groupby("location").apply(lambda x: resolve_query(x, processed_query))
        # print(json.dumps(processed_query, sort_keys=True, indent=4))

    return df.to_dict("records"), [
        {"name": i, "id": i, "deletable": True} for i in df.columns
    ]


if __name__ == "__main__":
    app.run_server(debug=True)
