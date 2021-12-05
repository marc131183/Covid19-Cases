import dash
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import pandas as pd
from pandas.api.types import is_numeric_dtype
from dash.dependencies import Input, Output
import json
import numpy as np
import filter_dash
import datetime
from dash.exceptions import PreventUpdate

from data_processing import getData
from predict import Model

app = dash.Dash(__name__)

colors = {
    "bg": "#282b38",
    "bg_bright": "#3b3f53",
    "text": "#979A9C",
    "buttonColor": "#4fc6e9",
}

df = getData()

df = df.rename(columns={elem: elem[0].upper() + elem[1:] for elem in df.columns})
df = df.sort_index(axis=1)

model = Model(df)

# add columns divided by population
cols_to_divide_by_population = [
    "ConfirmedDeaths",
    "New_cases",
    "New_deaths",
    "New_tests",
    "New_vaccinations",
    "People_fully_vaccinated",
    "People_vaccinated",
    "Total_cases",
    "Total_tests",
    "Total_vaccinations",
]

for col in cols_to_divide_by_population:
    df["{}_per_thousand".format(col)] = df[col] / (df["Population"] / 1e3)


# load only some of the data for faster updating
# df = pd.DataFrame(
#    {"location": np.arange(10), "new_cases": np.arange(10), "continent": np.arange(10)}
# )

app.layout = html.Div(
    style={"backgroundColor": colors["bg"]},
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
                        "width": "17%",
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
                    style={"color": colors["buttonColor"]},
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                html.Div(
                    "Country:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "22.5%",
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
                        "width": "10%",
                        "float": "right",
                    },
                    style={"color": colors["buttonColor"]},
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
                        "width": "37%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id="country",
                            placeholder="Choose which country to visualize",
                            options=[
                                {"label": elem, "value": elem} for elem in sorted(set(df["Location"]))
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
                        "width": "22%",
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
                        "width": "37%",
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
                html.Div(style={"width": "1%", "height": "1px", "float": "left"}),
                html.Div(
                    "Date Range:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "20%",
                        "float": "left",
                    },
                ),
                html.Div(
                    id="date-range-container",
                    children="{} - {}".format(
                        df["Date"].min().strftime("%d %b %Y"),
                        df["Date"].max().strftime("%d %b %Y"),
                    ),
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "13%",
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
                dcc.RangeSlider(
                    id="daterange",
                    min=0,
                    max=len(df["Date"].unique()) - 1,
                    step=1,
                    allowCross=False,
                    value=[0, len(df["Date"].unique()) - 1],
                    # style={"backgroundColor": colors["buttonColor"]},
                )
            ],
            style={"height": "30px"},
        ),
        html.Div(className="row", style={"height": "15px"}),
        dcc.Graph(id="indicator-graphic"),
        html.Div(
            className="row",
            children=[
                html.Div(
                    [
                        dcc.Markdown(
                            """
                    **Hover Data**
                    Mouse over values in the graph.
                """
                        ),
                        html.Pre(
                            id="hover-data",
                            style={
                                "border": "thin lightgrey solid",
                                "overflowX": "scroll",
                            },
                        ),
                    ],
                    className="three columns",
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "color": colors["text"],
                    },
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            """
                    **Click Data**
                    Click on points in the graph.
                """
                        ),
                        html.Pre(
                            id="click-data",
                            style={
                                "border": "thin lightgrey solid",
                                "overflowX": "scroll",
                            },
                        ),
                    ],
                    className="three columns",
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "color": colors["text"],
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Click here to see filter query examples", id="show-secret"
                        ),
                        html.Div(id="body-div"),
                    ],
                    style={
                        "color": colors["text"],
                        "fontSize": 15,
                        "width": "50%",
                        "display": "inline-block",
                        "float": "right",
                    },
                ),
            ],
        ),
    ],
)


@app.callback(
    Output(component_id="body-div", component_property="children"),
    Input(component_id="show-secret", component_property="n_clicks"),
)
def update_output(n_clicks):
    if n_clicks is None or (n_clicks % 2 == 0):
        return {}
    else:
        return [
            '{Continent} = "Asia": Get all countries that are in Asia:',
            html.Br(),
            '{Continent} = "Europe" or {Population} >= 10000000: Get all countries that are either in Europe or have a population of 10000000 or bigger',
            html.Br(),
            '({Continent} = "North America" or {Continent} = "South America") and {Human_development_index} < 30: Get all countries that are either in North- or South America and have a human development index of smaller than 30',
            html.Br(),
            "mean{New_cases} >= 300 and max{New_deaths} < 100: Get all countries that have an average number of new cases of 300 or higher and the maxmimum number of new deaths should be below 100",
            html.Br(),
            "{New_cases} > mean{New_cases} + 3 * std{New_cases} or {New_cases} < mean{New_cases} - 3 * std{New_cases}: Get all rows of countries that have a value of new cases that is either below or higher than the mean +- 3*std (outliers)",
        ]


@app.callback(
    Output("hover-data", "children"),
    Input("indicator-graphic", "hoverData"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
)
def display_hover_data(hoverData, xaxis_column, yaxis_column):
    if hoverData is None:
        return json.dumps(hoverData, indent=2)
    else:
        return json.dumps(
            {
                xaxis_column: hoverData["points"][0]["x"],
                "y": hoverData["points"][0]["y"],
            },
            indent=2,
        )


@app.callback(
    Output("click-data", "children"),
    Input("indicator-graphic", "clickData"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
)
def display_click_data(clickData, xaxis_column, yaxis_column):
    if clickData is None:
        return json.dumps(clickData, indent=2)
    else:
        return json.dumps(
            {
                xaxis_column: clickData["points"][0]["x"],
                "y": clickData["points"][0]["y"],
            },
            indent=2,
        )


@app.callback(
    [
        Output("indicator-graphic", "figure"),
        Output("xaxis-type", "options"),
        Output("xaxis-type", "value"),
        Output("yaxis-type", "options"),
        Output("yaxis-type", "value"),
        Output("date-range-container", "children"),
    ],
    [
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
        Input("xaxis-type", "value"),
        Input("yaxis-type", "value"),
        Input("plot_type", "value"),
        Input("grouping", "value"),
        Input("filter-query-input", "value"),
        Input("daterange", "value"),
        Input("country", "value"),
    ],
)
def update_graph(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    plot_type,
    grouping,
    query,
    date_slider,
    country,
):
    x_radio_options = [
        {"label": label, "value": value}
        for label, value in zip(
            ["Linear Scale", "Logarithmic Scale"], ["Linear", "Log"]
        )
    ]
    y_radio_options = [
        {"label": label, "value": value}
        for label, value in zip(
            ["Logarithmic Scale", "Linear Scale"], ["Log", "Linear"]
        )
    ]
    # check if xaxis_colum in non-numerical, if yes disable log selection
    if xaxis_column_name != None and not is_numeric_dtype(df[xaxis_column_name]):
        x_radio_options[1]["disabled"] = True
        xaxis_type = "Linear"
    if yaxis_column_name != None and not all(
        [is_numeric_dtype(df[elem]) for elem in yaxis_column_name]
    ):
        y_radio_options[1]["disabled"] = True
        yaxis_type = "Linear"

    # filter data according to daterange and filter query
    df_ = df.copy()
    date_range = sorted(df_["Date"].unique())
    min_date, max_date = pd.to_datetime(date_range[date_slider[0]]), pd.to_datetime(
        date_range[date_slider[1]]
    )
    df_ = df_[(df_["Date"] >= min_date) & (df_["Date"] <= max_date)]

    if (xaxis_column_name == None or yaxis_column_name == None) and plot_type != "Predict" or (country == None and plot_type == "Predict"):
        return (
            {},
            x_radio_options,
            xaxis_type,
            y_radio_options,
            yaxis_type,
            "{} - {}".format(
                min_date.strftime("%d %b %Y"),
                max_date.strftime("%d %b %Y"),
            ),
        )

    if not (
        query is None or query == ""
    ):  # remember: add options for filter and a country filter
        derived_query_structure = filter_dash.split_query(query)
        df_ = df_.groupby("Location").apply(
            lambda x: x[filter_dash.resolve_query(x, derived_query_structure)]
        )

    if len(yaxis_column_name) > 1 and yaxis_column_name != None:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        cols_yprimary = str(yaxis_column_name[0])
        cols_ysecondary = str(yaxis_column_name[1])

        xaxis_data = df_[xaxis_column_name]
        magnitude_primary, magnitude_secondary = (
            df_[yaxis_column_name[0]].mean(),
            df_[yaxis_column_name[1]].mean(),
        )
        symbols = SymbolValidator().values
        # markers https://plotly.com/python/marker-style/
        for i, yaxis in enumerate(yaxis_column_name):
            cur_data = df_[yaxis]
            if i == 0 or i == 1:
                add_to_secondary = i == 1
            else:
                # add axis to the one that's magnitude is closer
                magnitude = cur_data.mean()
                add_to_secondary = True
                if np.linalg.norm(magnitude - magnitude_primary) < np.linalg.norm(
                    magnitude - magnitude_secondary
                ):
                    add_to_secondary = False
                if add_to_secondary:
                    magnitude_secondary = (
                        magnitude
                        if magnitude > magnitude_secondary
                        else magnitude_secondary
                    )
                    cols_ysecondary += " / {}".format(yaxis)
                else:
                    magnitude_primary = (
                        magnitude
                        if magnitude > magnitude_primary
                        else magnitude_primary
                    )
                    cols_yprimary += " / {}".format(yaxis)

            # Add traces
            if plot_type == "Scatter":
                fig.add_trace(
                    go.Scatter(
                        x=xaxis_data,
                        y=cur_data,
                        name=str(yaxis),
                        mode="markers",
                        marker_symbol=symbols[i],
                    ),
                    secondary_y=add_to_secondary,
                )
                # marker_color=(df_[grouping] if grouping != None else None),
            elif plot_type == "Bar":
                fig.add_trace(
                    go.Bar(
                        x=xaxis_data,
                        y=cur_data,
                        name=str(yaxis),
                        offsetgroup=i,
                    ),
                    secondary_y=add_to_secondary,
                )
            elif plot_type == "Line":
                fig.add_trace(
                    go.Line(
                        x=xaxis_data,
                        y=df_[yaxis],
                        name=str(yaxis),
                    ),
                    secondary_y=add_to_secondary,
                )

        # Set x-axis title
        if xaxis_type == "Log":
            fig.update_xaxes(
                title_text=str(xaxis_column_name), type="log", row=1, col=1
            )
        else:
            fig.update_xaxes(title_text=str(xaxis_column_name))

        # Set y-axes titles
        fig.update_yaxes(
            title_text=cols_yprimary,
            secondary_y=False,
            type="log" if yaxis_type == "Log" else "linear",
            col=1,
            row=1,
        )
        fig.update_yaxes(
            title_text=cols_ysecondary,
            secondary_y=True,
            type="log" if yaxis_type == "Log" else "linear",
            col=1,
            row=1,
        )
    else:
        df_with_country = df_[df_["Location"] == country] if country != None else df_
        if plot_type == "Scatter":  # remember: only make some parameters available?
            fig = px.scatter(
                df_with_country,
                x=df_with_country[xaxis_column_name],
                y=[df_with_country[ycolumn] for ycolumn in yaxis_column_name],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=(df_with_country[grouping] if grouping != None else None),
            )
        elif plot_type == "Bar":
            fig = px.bar(
                df_with_country,
                x=df_with_country[xaxis_column_name],
                y=[df_with_country[ycolumn] for ycolumn in yaxis_column_name],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=(df_with_country[grouping] if grouping != None else None),
            )
        elif plot_type == "Line":
            fig = px.line(
                df_with_country,
                x=df_with_country[xaxis_column_name],
                y=[df_with_country[ycolumn] for ycolumn in yaxis_column_name],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=(df_with_country[grouping] if grouping != None else None),
            )
        elif plot_type == "Predict": # remember: fix prediction
            df_predict = model.predict(country)

            fig = px.line( 
                df_predict,
                x=df_predict[2],
                y=df_predict[3],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
            ) 
            fig.add_trace(go.Line( 
                x=df_predict[0],
                y=df_predict[1],
                name="Prediction",),
            )

    fig.update_layout(
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["bg"],
        font_color=colors["text"],
        clickmode="event+select",
        margin={"l": 40, "b": 40, "t": 20, "r": 0},
    )

    return (
        fig,
        x_radio_options,
        xaxis_type,
        y_radio_options,
        yaxis_type,
        "{} - {}".format(
            min_date.strftime("%d %b %Y"),
            max_date.strftime("%d %b %Y"),
        ),
    )


if __name__ == "__main__":
    app.run_server(debug=True)
