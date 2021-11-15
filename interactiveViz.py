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
            children="Overview of covid 19",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="Historical data of covid 19 cases.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[
                {"label": elem[0].upper() + elem[1:], "value": elem} for elem in df.columns
                ],
                value='new_cases'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[
                {"label": elem[0].upper() + elem[1:], "value": elem} for elem in df.columns
                ],
                value='location'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

        html.Hr(),
        html.Details(
            [
                html.Summary("Contents of figure storage"),
                dcc.Markdown(id="clientside-figure-json-px"),
            ]
        ),
        dcc.Graph(id='indicator-graphic')
    ]
)

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):

    fig = px.scatter(x=df[xaxis_column_name],
                     y=df[yaxis_column_name])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    return fig


#@app.callback(
    Output("clientside-figure-store-px", "data"),
    Input("clientside-graph-indicator-px", "value"),
    Input("clientside-graph-country-px", "value"),
#)
#def update_store_data(indicator, country):
    # dff = df[df['location'] == country]
    return [{"x": df[indicator], "y": df[country], "mode": "markers"}]


#app.clientside_callback(
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
#)


#@app.callback(
    Output("clientside-figure-json-px", "children"),
    Input("clientside-figure-store-px", "data"),
#)
#def generated_px_figure_json(data):
    return "```\n" + json.dumps(data, indent=2) + "\n```"


if __name__ == "__main__":
    app.run_server(debug=True)
