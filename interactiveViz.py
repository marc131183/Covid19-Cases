import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import json
import numpy as np
import filter_dash

from data_processing import getData
from predict import Model

app = dash.Dash(__name__)

colors = {"background": "#ede6d8", "text": "#1405eb"}

df = getData()

# load only some of the data for faster updating
# df = pd.DataFrame(
#    {"location": np.arange(10), "new_cases": np.arange(10), "continent": np.arange(10)}
# )

model = Model(df)
df_predict = model.predict("Norway")

fig = px.bar(df, x="location", y="new_cases", color="continent", barmode="group")

fig.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font_color=colors["text"],
)

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Overview of Covid-19",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="Historical data of Covid-19 cases.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        "Filter query:",
        dcc.Input(
            id="filter-query-input",
            placeholder="Enter filter query",
            style={"width": "15%", "height": "30px",},
        ),
        html.Div(
            [
                "Select plot type:",
                dcc.RadioItems(
                    id="plot_type",
                    options=[
                        {"label": i, "value": i}
                        for i in ["Scatter", "Bar", "Line", "Predict"]
                    ],
                    value="Scatter",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "30%", "float": "right", "display": "inline-block",},
        ),
        html.Div(
            [
                "Color by:",
                dcc.Dropdown(
                    id="grouping",
                    placeholder="Choose how to color",
                    options=[
                        {"label": elem[0].upper() + elem[1:], "value": elem}
                        for elem in df.columns
                    ],
                ),
            ],
            style={
                "width": "30%",
                "float": "right",
                "display": "inline-block",
                "marginRight": "10px",
            },
        ),
        html.Div(
            [
                "Y-parameter:",
                dcc.Dropdown(
                    id="yaxis-column",
                    placeholder="Choose a y-parameter",
                    options=[
                        {"label": elem[0].upper() + elem[1:], "value": elem}
                        for elem in df.columns
                    ],
                    # multi=True,
                ),
                dcc.RadioItems(
                    id="yaxis-type",
                    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                    value="Linear",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "48%", "float": "down", "display": "inline-block"},
        ),
        html.Div(
            [
                "X-parameter:",
                dcc.Dropdown(
                    id="xaxis-column",
                    placeholder="Choose a x-parameter",
                    options=[
                        {"label": elem[0].upper() + elem[1:], "value": elem}
                        for elem in df.columns
                    ],
                    # multi=True,
                ),
                dcc.RadioItems(
                    id="xaxis-type",
                    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                    value="Linear",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "48%", "float": "right", "display": "inline-block"},
        ),
        dcc.Graph(id="indicator-graphic"),
    ],
)


@app.callback(
    Output("indicator-graphic", "figure"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("xaxis-type", "value"),
    Input("yaxis-type", "value"),
    Input("plot_type", "value"),
    Input("grouping", "value"),
    Input("filter-query-input", "value"),
)
def update_graph(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    plot_type,
    grouping,
    query,
):

    if xaxis_column_name == None or yaxis_column_name == None:
        return {}

    if query is None:
        df_ = df
    else:
        derived_query_structure = filter_dash.split_query(query)
        df_ = df.groupby("location").apply(
            lambda x: x[filter_dash.resolve_query(x, derived_query_structure)]
        )

    if plot_type == "Scatter":  # remember: only make some parameters available?

        fig = px.scatter(
            df_,
            x=df_[xaxis_column_name],  # remember: make available multiple parameters
            y=df_[yaxis_column_name],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
            color=(df[grouping] if grouping != None else None),
        )

    elif plot_type == "Bar":
        fig = px.bar(
            df_,
            x=df_[xaxis_column_name],
            y=df_[yaxis_column_name],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
            color=(df[grouping] if grouping != None else None),
        )

    elif plot_type == "Line":
        fig = px.line(
            df_,
            x=df_[xaxis_column_name],
            y=df_[yaxis_column_name],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
            color=(df[grouping] if grouping != None else None),
        )

    else:
        fig = px.line(  # remember: fix prediction
            df_predict,
            x=df_predict["date"],
            y=df_predict["new_cases"],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
        )

    fig.update_layout(margin={"l": 40, "b": 40, "t": 20, "r": 0})

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
