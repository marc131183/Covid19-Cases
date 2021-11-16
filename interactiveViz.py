import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import json

from data_processing import getData

app = dash.Dash(__name__)

colors = {"background": "#ede6d8", "text": "#1405eb"}

df = getData()

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
        html.Div(
            [
                dcc.Dropdown(
                    id="xaxis-column",
                    options=[
                        {"label": elem[0].upper() + elem[1:], "value": elem}
                        for elem in df.columns
                    ],
                    value="location",
                ),
                dcc.RadioItems(
                    id="xaxis-type",
                    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                    value="Linear",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "48%", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="yaxis-column",
                    options=[
                        {"label": elem[0].upper() + elem[1:], "value": elem}
                        for elem in df.columns
                    ],
                    value="new_cases",
                ),
                dcc.RadioItems(
                    id="yaxis-type",
                    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                    value="Linear",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "48%", "float": "right", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.RadioItems(
                    id="plot_type",
                    options=[
                        {"label": i, "value": i} for i in ["Scatter", "Bar", "Line"]
                    ],
                    value="Scatter",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "48%", "float": "down", "display": "inline-block"},
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
)
def update_graph(
    xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, plot_type
):

    if plot_type == "Scatter":

        fig = px.scatter(
            df,
            x=df[xaxis_column_name],
            y=df[yaxis_column_name],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
        )

    elif plot_type == "Bar":
        fig = px.bar(
            df,
            x=df[xaxis_column_name],
            y=df[yaxis_column_name],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
            color="continent",  # make the user choose how to group
        )

    else:
        fig = px.line(
            df,
            x=df[xaxis_column_name],
            y=df[yaxis_column_name],
            log_x=xaxis_type == "Log",
            log_y=yaxis_type == "Log",
        )

    fig.update_layout(margin={"l": 40, "b": 40, "t": 20, "r": 0})

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
