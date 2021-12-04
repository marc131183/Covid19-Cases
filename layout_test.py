import dash
from dash import dcc
from dash import html
import plotly.express as px

import numpy as np
import pandas as pd

colors = {"bg": "#282b38", "bg_bright": "#3b3f53", "text": "#979A9C"}

df = pd.DataFrame(
    {
        "location": np.arange(10),
        "new_cases": np.arange(10),
        "continent": np.arange(10),
        "date": np.arange(10),
    }
)

# Initialise the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div(
    children=[
        html.Div(
            children="Covid-19 Data Explorer",
            style={
                "color": colors["text"],
                "fontSize": 60,
                "textAlign": "center",
                "padding": "20px",
                "backgroundColor": colors["bg_bright"],
                "border-radius": 10,
            },
        ),
        html.Div(className="row", style={"height": "10px"}),
        html.Div(
            className="row",
            children=[
                html.Div(
                    "Plot Type:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "20%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                html.Div(
                    "Filter Query:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "55.3%",
                        "float": "left",
                    },
                ),
                html.Div(
                    "Color by:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "20%",
                        "float": "right",
                    },
                ),
            ],
            style={"height": "20px"},
        ),
        html.Div(className="row", style={"height": "10px"}),
        html.Div(
            className="row",
            children=[
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id="plot_type",
                            value="Scatter",
                            options=[
                                {"label": elem, "value": elem}
                                for elem in ["Scatter", "Bar", "Line", "Predict"]
                            ],
                            style={
                                "color": colors["text"],
                                "backgroundColor": colors["bg_bright"],
                                "fontSize": 25,
                                "border-color": "#ffffff",
                                "border-radius": 5,
                            },
                        ),
                    ],
                    style={
                        "width": "20%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                dcc.Input(
                    id="filter-query-input",
                    placeholder="Enter filter query!",
                    style={
                        "width": "55.3%",
                        "color": colors["text"],
                        "fontSize": 25,
                        "backgroundColor": colors["bg_bright"],
                        "border-color": "#ffffff",
                        "border-radius": 5,
                        "float": "left",
                    },
                ),
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id="grouping",
                            placeholder="Choose how to color",
                            options=[
                                {"label": elem, "value": elem} for elem in df.columns
                            ],
                            style={
                                "color": colors["text"],
                                "backgroundColor": colors["bg_bright"],
                                "fontSize": 25,
                                "border-color": "#ffffff",
                                "border-radius": 5,
                            },
                        )
                    ],
                    style={
                        "width": "20%",
                        "float": "right",
                    },
                ),
            ],
            style={"height": "30px"},
        ),
        html.Div(className="row", style={"height": "20px"}),
        html.Div(
            className="row",
            children=[
                html.Div(
                    "Y-Axis:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "29.5%",
                        "float": "left",
                    },
                ),
                dcc.RadioItems(
                    id="yaxis-type",
                    options=[
                        {"label": label, "value": value}
                        for label, value in zip(
                            ["Linear Scale", "Logarithmic Scale"], ["Linear", "Log"]
                        )
                    ],
                    value="Linear",
                    labelStyle={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "10%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "1.5%", "height": "1px", "float": "left"}),
                html.Div(
                    "X-Axis:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "10%",
                        "float": "left",
                    },
                ),
                dcc.RadioItems(
                    id="xaxis-type",
                    options=[
                        {"label": label, "value": value}
                        for label, value in zip(
                            ["Logarithmic Scale", "Linear Scale"], ["Log", "Linear"]
                        )
                    ],
                    value="Linear",
                    labelStyle={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "9.5%",
                        "float": "right",
                    },
                ),
            ],
            style={"height": "20px"},
        ),
        html.Div(className="row", style={"height": "10px"}),
        html.Div(
            className="row",
            children=[
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id="yaxis-column",
                            placeholder="Choose which column to visualize for y-axis",
                            options=[
                                {"label": elem, "value": elem} for elem in df.columns
                            ],
                            style={
                                "color": colors["text"],
                                "backgroundColor": colors["bg_bright"],
                                "fontSize": 25,
                                "border-color": "#ffffff",
                                "border-radius": 5,
                            },
                            multi=True,
                        ),
                    ],
                    style={
                        "width": "49%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id="xaxis-column",
                            placeholder="Choose which column to visualize for x-axis",
                            options=[
                                {"label": elem, "value": elem} for elem in df.columns
                            ],
                            style={
                                "color": colors["text"],
                                "backgroundColor": colors["bg_bright"],
                                "fontSize": 25,
                                "border-color": "#ffffff",
                                "border-radius": 5,
                            },
                        ),
                    ],
                    style={
                        "width": "49%",
                        "float": "right",
                    },
                ),
            ],
            style={"height": "30px"},
        ),
    ]
)


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
