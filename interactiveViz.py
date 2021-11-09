import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import json

from data_processing import getData

app = dash.Dash(__name__)

colors = {"background": "#111111", "text": "#7FDBFF"}

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
            children="Overview of covid 19",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="Historical data of covid 19 cases.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        dcc.Graph(id="clientside-graph-px", figure=fig),
        dcc.Store(id="clientside-figure-store-px"),
        # select y-parameter
        "Indicator",
        dcc.Dropdown(
            id="clientside-graph-indicator-px",
            options=[
                {"label": "New Cases", "value": "new_cases"},
                {"label": "Total Cases", "value": "total_cases"},
                {"label": "Date", "value": "date"},
            ],
            value="new_cases",
        ),
        # select x-parameter
        "Country",
        dcc.Dropdown(
            id="clientside-graph-country-px",
            options=[
                {"label": "New Cases", "value": "new_cases"},
                {"label": "Total Cases", "value": "total_cases"},
                {"label": "Location", "value": "location"},
            ],
            value="location",
        ),
        html.Hr(),
        html.Details(
            [
                html.Summary("Contents of figure storage"),
                dcc.Markdown(id="clientside-figure-json-px"),
            ]
        ),
    ],
)


@app.callback(
    Output("clientside-figure-store-px", "data"),
    Input("clientside-graph-indicator-px", "value"),
    Input("clientside-graph-country-px", "value"),
)
def update_store_data(indicator, country):
    # dff = df[df['location'] == country]
    return [{"x": df[indicator], "y": df[country], "mode": "markers"}]


app.clientside_callback(
    """
    function(figure, scale) {
        if(figure === undefined) {
            return {'data': [], 'layout': {}};
        }
        const fig = Object.assign({}, figure, {
            'layout': {
                ...figure.layout,
                'yaxis': {
                    ...figure.layout.yaxis, type: scale
                }
             }
        });
        return fig;
    }
    """,
    Output("clientside-graph-px", "figure"),
    Input("clientside-figure-store-px", "data"),
    Input("clientside-graph-scale-px", "value"),
)


@app.callback(
    Output("clientside-figure-json-px", "children"),
    Input("clientside-figure-store-px", "data"),
)
def generated_px_figure_json(data):
    return "```\n" + json.dumps(data, indent=2) + "\n```"


if __name__ == "__main__":
    app.run_server(debug=True)
