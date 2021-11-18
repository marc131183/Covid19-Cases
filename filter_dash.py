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
    query = query.replace(" ", "")
    brackets_open = 0
    sub_query = {"LHS": "", "RHS": "", "operator": "", "operator_type": "logical"}
    subs = []

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
    for i, character in enumerate(query):
        if character == "(":
            if brackets_open == 0:
                subs.append([i + 1, None])
            brackets_open += 1
        elif character == ")":
            brackets_open -= 1
            if brackets_open == 0:
                subs[-1][1] = i

    # if it only contains a LHS
    if len(subs) == 1:
        sub_query["operator_type"] = "single"
        sub_query["LHS"] = query
        sub_query["LHS"] = parse_sub_query(sub_query["LHS"])
    else:
        sub_query["LHS"] = query[subs[0][0] : subs[0][1]]
        sub_query["RHS"] = query[subs[1][0] : subs[1][1]]
        sub_query["operator"] = query[subs[0][1] + 1 : subs[1][0] - 1]

        # if the query contains subqueries, we need to split further
        if sub_query["LHS"].count("(") > 0:
            sub_query["LHS"] = split_query(sub_query["LHS"])
        # it's now a single subquery, we can parse it
        else:
            sub_query["LHS"] = parse_sub_query(sub_query["LHS"])

        if sub_query["RHS"].count("(") > 0:
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
                sub["LHS"] = parse_value(side[:i])
                sub["RHS"] = parse_value(side[i + len(oper) :])
                sub["operator"] = valid_operators[oper]
                return sub

    return sub


def parse_value(value):
    # check if value is a column
    out = {"func": lambda x: x, "value": None, "type": None}
    valid_funcs = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1],
    }
    if "{" in value:
        first = value.index("{")
        last = value.index("}")
        out["type"] = "column"
        out["value"] = value[first + 1 : last]
        out["func"] = valid_funcs[value[:first]] if value[:first] != "" else lambda x: x
    else:
        out["type"] = "value"
        out["value"] = float(value) if value.isnumeric() else value

    return out


def resolve_query(df_, derived_query_structure):
    if derived_query_structure["operator_type"] == "relational":
        if derived_query_structure["LHS"]["type"] == "column":
            lhs = derived_query_structure["LHS"]["func"](
                df_[derived_query_structure["LHS"]["value"]]
            )
        else:
            lhs = derived_query_structure["LHS"]["value"]
        if derived_query_structure["RHS"]["type"] == "column":
            rhs = derived_query_structure["RHS"]["func"](
                df_[derived_query_structure["RHS"]["value"]]
            )
        else:
            rhs = derived_query_structure["RHS"]["value"]

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
