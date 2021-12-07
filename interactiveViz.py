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

from data_processing import getData
from predict import Model

app = dash.Dash(__name__)

plot_colors = np.array(px.colors.qualitative.Plotly)
symbols = ["circle", "star", "cross", "triangle-up", "bowtie", "diamond-cross"]
patterns = ["", ".", "x", "+", "/", "\\"]
line_styles = [
    "solid",
    "dot",
    "dash",
    "longdash",
    "dashdot",
    "longdashdot",
]

colors = {
    "bg": "#282b38",
    "bg_bright": "#3b3f53",
    "text": "#979A9C",
    "buttonColor": "#4fc6e9",
}

df = getData()

df_column_desc = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-codebook.csv")

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

cols_to_sum_up = cols_to_divide_by_population
cols_to_sum_up_with_population = [
    "{}_per_thousand".format(elem) for elem in cols_to_divide_by_population
]
cols_to_ignore = [
    "Date",
    "CountryCode",
    "Continent",
    "Location",
    "Weekday",
    "Population",
]
cols_to_mean_up = list(
    df.columns.difference(
        cols_to_sum_up + cols_to_sum_up_with_population + cols_to_ignore
    )
)

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
                        "width": "14%",
                        "float": "left",
                    },
                ),
                dcc.RadioItems(
                    id="yaxis-type",
                    options=[
                        {"label": label, "value": value}
                        for label, value in zip(
                            ["Logarithmic", "Linear"], ["Log", "Linear"]
                        )
                    ],
                    value="Linear",
                    labelStyle={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "10%",
                        "float": "left",
                        "textAlign": "right",
                    },
                    style={"color": colors["buttonColor"]},
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                html.Div(
                    "Country:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "34%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "2%", "height": "1px", "float": "left"}),
                html.Div(
                    "X-Axis:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "8%",
                        "float": "left",
                    },
                ),
                dcc.RadioItems(
                    id="xaxis-type",
                    options=[
                        {"label": label, "value": value}
                        for label, value in zip(
                            ["Logarithmic", "Linear"], ["Log", "Linear"]
                        )
                    ],
                    value="Linear",
                    labelStyle={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "10%",
                        "float": "left",
                        "textAlign": "right",
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
                                {"label": elem, "value": elem}
                                for elem in df.columns
                                if not elem in cols_to_ignore
                            ]
                            + [{"label": "Population", "value": "Population"}],
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
                        "width": "34%",
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
                                {"label": elem, "value": elem}
                                for elem in sorted(df["Location"].unique())
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
                        "width": "34%",
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
                        "width": "28%",
                        "float": "right",
                    },
                ),
            ],
            style={"height": "30px"},
        ),
        html.Div(className="row", style={"height": "30px"}),
        html.Div(
            className="row",
            children=[
                html.Div(style={"width": "1%", "height": "1px", "float": "left"}),
                html.Div(
                    "Date Range:",
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "25%",
                        "float": "left",
                    },
                ),
                html.Div(style={"width": "1%", "height": "1px", "float": "right"}),
                html.Div(
                    id="date-range-container",
                    children="{} - {}".format(
                        df["Date"].min().strftime("%d %b %Y"),
                        df["Date"].max().strftime("%d %b %Y"),
                    ),
                    style={
                        "color": colors["text"],
                        "fontSize": 20,
                        "width": "25%",
                        "float": "right",
                        "textAlign": "right",
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
                )
            ],
            style={"height": "30px"},
        ),
        html.Div(className="row", style={"height": "15px"}),
        dcc.Graph(id="indicator-graphic"),
        html.Div(
            children=[
                dcc.Dropdown(
                    id="additional_information",
                    placeholder="Show additional information",
                    options=[
                        {"label": elem, "value": elem}
                        for elem in ["Hover/click information", "Filter query examples", "Column description"]
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
        html.Div(style={"width": "10%", "height": "1px", "float": "left"}),
        html.Div(
            id="hover_click",
            className="row",
            children=[
                html.Div(
                    [
                        dcc.Markdown(
                            """
                    Mouse over values in the graph
                """
                        ),
                        html.Pre(
                            id="hover-data",
                            style={
                                # "border": "thin lightgrey solid",
                                #"overflowX": "scroll",
                            },
                        ),
                    ],
                    className="three columns",
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "color": colors["text"],
                    },
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            """
                    Click on points in the graph
                """
                        ),
                        html.Pre(
                            id="click-data",
                            style={
                                # "border": "thin lightgrey solid",
                                #"overflowX": "scroll",
                            },
                        ),
                    ],
                    className="three columns",
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "color": colors["text"],
                    },
                ),
            ]
        ),
        html.Div(
            id="filter_examples",
            children = [
                html.P(
                    children=[
                        html.Span("1. "),
                        html.Strong('{Continent} = "Asia"'),
                        html.Br(),
                        html.Span("Get all countries that are in Asia"),
                    ]
                ),
                html.P(
                    children=[
                        html.Span("2. "),
                        html.Strong('{Continent} = "Europe" or {Population} >= 10000000'),
                        html.Br(),
                        html.Span(
                            "Get all countries that are either in Europe or have a population of 10000000 or bigger"
                        ),
                    ]
                ),
                html.P(
                    children=[
                        html.Span("3. "),
                        html.Strong(
                            '({Continent} = "North America" or {Continent} = "South America") and {Human_development_index} < 30'
                        ),
                        html.Br(),
                        html.Span(
                            "Get all countries that are either in North- or South America and have a human development index of smaller than 30"
                        ),
                    ]
                ),
                html.P(
                    children=[
                        html.Span("4. "),
                        html.Strong("mean{New_cases} >= 300 and max{New_deaths} < 100"),
                        html.Br(),
                        html.Span(
                            "Get all countries that have an average number of new cases of 300 or higher and the maxmimum number of new deaths below 100"
                        ),
                    ]
                ),
                html.P(
                    children=[
                        html.Span("5. "),
                        html.Strong(
                            "{New_cases} > mean{New_cases} + 3 * std{New_cases} or {New_cases} < mean{New_cases} - 3 * std{New_cases}"
                        ),
                        html.Br(),
                        html.Span(
                            "Get all rows of countries that have a value of new cases that is either below or higher than the mean +- 3*std (outliers)"
                        ),
                    ]
                ),
            ],
            
            style={
                "color": colors["text"],
                "fontSize": 14,
                "width": "50%",
                "float": "left",
            },
        ),
        html.Div(
            id="column",
            children = [
                
            ],
            
            style={
                "color": colors["text"],
                "fontSize": 14,
                "width": "50%",
                "float": "left",
            },
        ),
    ],
)

@app.callback(
    Output(component_id="column", component_property="children"),
    Input(component_id="xaxis-column", component_property="value"),
    Input(component_id="yaxis-column", component_property="value"),
)
def show_column_desc(xaxis, yaxis):
    total_descs = []
    if xaxis != None:
        xaxis_descs = df_column_desc[df_column_desc["column"] == xaxis.lower()]
        if len(xaxis_descs) != 0:
            total_descs.append((xaxis, xaxis_descs.iloc[0]["description"]))
        else: 
            total_descs.append((xaxis, "No description"))
    
    if yaxis != None:
        for yaxis_el in yaxis:
            yaxis_descs = df_column_desc[df_column_desc["column"] == yaxis_el.lower()]
            if len(yaxis_descs) != 0:
                total_descs.append((yaxis_el, yaxis_descs.iloc[0]["description"]))
            else: 
                total_descs.append((yaxis_el, "No description"))
    
    return [html.Div(children = column + ": " + desc) for column, desc in total_descs]



@app.callback(
    Output(component_id="column", component_property="hidden"),
    Output(component_id="hover_click", component_property="hidden"),
    Output(component_id="filter_examples", component_property="hidden"),
    Input(component_id="additional_information", component_property="value"),
)
def update_information(value):
    column_desc_vis = value != "Column description"
    hover_info_vis = value != "Hover/click information"
    filter_examples_vis = value != "Filter query examples"

    return column_desc_vis, hover_info_vis, filter_examples_vis

@app.callback(
    Output("hover-data", "children"),
    Input("indicator-graphic", "hoverData"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("plot_type", "value"),
)
def display_hover_data(hoverData, xaxis_column, yaxis_column, plot_type):
    if hoverData is None:
        return json.dumps(hoverData, indent=2)
    elif plot_type == "Predict":
        return json.dumps(
            {
                "Date": hoverData["points"][0]["x"],
                "New cases": round(hoverData["points"][0]["y"]),
            },
            indent=2,
        )
    else:
        return json.dumps(
            {
                xaxis_column: hoverData["points"][0]["x"],
                yaxis_column[0]
                if len(yaxis_column) == 1
                else "y": hoverData["points"][0]["y"],
            },
            indent=2,
        )


@app.callback(
    Output("click-data", "children"),
    Input("indicator-graphic", "clickData"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("plot_type", "value"),
)
def display_click_data(
    clickData,
    xaxis_column,
    yaxis_column,
    plot_type,
):
    if clickData is None:
        return json.dumps(clickData, indent=2)
    elif plot_type == "Predict":
        return json.dumps(
            {
                "Date": clickData["points"][0]["x"],
                "New_cases": round(clickData["points"][0]["y"]),
            },
            indent=2,
        )
    else:
        return json.dumps(
            {
                xaxis_column: clickData["points"][0]["x"],
                yaxis_column[0]
                if len(yaxis_column) == 1
                else "y": clickData["points"][0]["y"],
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
        for label, value in zip(["Linear", "Logarithmic"], ["Linear", "Log"])
    ]
    y_radio_options = [
        {"label": label, "value": value}
        for label, value in zip(["Linear", "Logarithmic"], ["Linear", "Log"])
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
    min_date, max_date = (
        pd.to_datetime(date_range[date_slider[0]]),
        pd.to_datetime(date_range[date_slider[1]]),
    )
    df_ = df_[(df_["Date"] >= min_date) & (df_["Date"] <= max_date)]

    if (
        (xaxis_column_name == None or yaxis_column_name == None)
        and plot_type != "Predict"
        or (country == None and plot_type == "Predict")
    ):
        fig = go.Figure()
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

    if not (
        query is None or query == ""
    ):  # remember: add options for filter and a country filter
        derived_query_structure = filter_dash.split_query(query)
        df_ = df_.groupby("Location").apply(
            lambda x: x[filter_dash.resolve_query(x, derived_query_structure)]
        )

    if country != None and country != []:
        df_ = df_[df_["Location"].isin(country)]

    if plot_type != "Predict":
        # aggregate data according to xaxis and the color-by (if given)
        cols_to_mean_up_present = list(set(cols_to_mean_up) & set(yaxis_column_name))
        cols_to_mean_up_present_new_names = []
        for i, col in enumerate(cols_to_mean_up_present):
            df_["{}_times_population".format(col)] = df_[col] * df_["Population"]
            cols_to_mean_up_present_new_names.append("{}_times_population".format(col))

        cols = (
            list(set(yaxis_column_name) - set(cols_to_mean_up_present))
            + cols_to_mean_up_present_new_names
        )

        sum_divide_population = list(
            set(cols_to_sum_up_with_population) & set(yaxis_column_name)
        )
        for col in sum_divide_population:
            cols.append(col[:-13])
        cols = list(set(cols))
        if grouping != None:
            df_ = (
                df_[set(cols + [xaxis_column_name, grouping, "Population"])]
                .groupby([xaxis_column_name, grouping])
                .sum()
            )
        else:
            df_ = (
                df_[set(cols + [xaxis_column_name, "Population"])]
                .groupby(xaxis_column_name, as_index=False)
                .sum()
            )

        df_.reset_index(inplace=True)

        if cols_to_mean_up_present != []:
            df_[cols_to_mean_up_present] = (
                df_[cols_to_mean_up_present_new_names]
                / df_["Population"].to_numpy()[:, np.newaxis]
            )

        if sum_divide_population != []:
            df_[sum_divide_population] = df_[
                [elem[:-13] for elem in sum_divide_population]
            ] / (df_["Population"].to_numpy()[:, np.newaxis] / 1e3)

    if yaxis_column_name != None and len(yaxis_column_name) > 1:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        cols_yprimary = str(yaxis_column_name[0])
        cols_ysecondary = str(yaxis_column_name[1])

        xaxis_data = df_[xaxis_column_name]
        magnitude_primary, magnitude_secondary = (
            df_[yaxis_column_name[0]].max(),
            df_[yaxis_column_name[1]].max(),
        )
        if grouping != None:
            color_groups = df_[grouping].unique()
            color_mapping = dict(zip(color_groups, np.arange(len(color_groups))))
            group_data = np.array([color_mapping[elem] for elem in df_[grouping]])
        # markers https://plotly.com/python/marker-style/
        for i, yaxis in enumerate(yaxis_column_name):
            cur_data = df_[yaxis]
            if i == 0 or i == 1:
                add_to_secondary = i == 1
            else:
                # add axis to the one that's magnitude is closer
                magnitude = cur_data.max()
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
                if grouping != None:
                    for j, group in enumerate(color_groups):
                        data = df_[df_[grouping] == group]
                        fig.add_trace(
                            go.Scatter(
                                x=data[xaxis_column_name],
                                y=data[yaxis],
                                name="{}_{}".format(yaxis, group),
                                mode="markers",
                                marker_symbol=symbols[i],
                                marker_color=plot_colors[j],
                            ),
                            secondary_y=add_to_secondary,
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=xaxis_data,
                            y=df_[yaxis],
                            name=str(yaxis),
                            mode="markers",
                            marker_symbol=symbols[i],
                        ),
                        secondary_y=add_to_secondary,
                    )
            elif plot_type == "Bar":
                fig.add_trace(
                    go.Bar(
                        x=xaxis_data,
                        y=cur_data,
                        name=str(yaxis),
                        offsetgroup=i,
                        marker_pattern_shape=patterns[i],
                        marker_color=plot_colors[group_data]
                        if grouping != None
                        else None,
                        showlegend=grouping == None,
                    ),
                    secondary_y=add_to_secondary,
                )
                if grouping != None:
                    for j, group in enumerate(color_groups):
                        fig.add_trace(
                            go.Bar(
                                x=[None],
                                y=[None],
                                name="{}_{}".format(yaxis, group),
                                marker_pattern_shape=patterns[i],
                                marker_color=plot_colors[j],
                                showlegend=True,
                            )
                        )
            elif plot_type == "Line":
                if grouping != None:
                    for j, group in enumerate(color_groups):
                        data = df_[df_[grouping] == group]
                        fig.add_trace(
                            go.Scatter(
                                x=data[xaxis_column_name],
                                y=data[yaxis],
                                name="{}_{}".format(yaxis, group),
                                line=dict(dash=line_styles[i]),
                                marker_color=plot_colors[j],
                            ),
                            secondary_y=add_to_secondary,
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=xaxis_data,
                            y=df_[yaxis],
                            name=str(yaxis),
                            line=dict(dash=line_styles[i]),
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
        if plot_type == "Scatter":  # remember: only make some parameters available?
            fig = px.scatter(
                df_,
                x=xaxis_column_name,
                y=yaxis_column_name,
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=grouping,
            )
        elif plot_type == "Bar":
            fig = px.bar(
                df_,
                x=xaxis_column_name,
                y=yaxis_column_name,
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=grouping,
            )
        elif plot_type == "Line":
            fig = px.line(
                df_,
                x=xaxis_column_name,
                y=yaxis_column_name,
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=grouping,
            )
        elif plot_type == "Predict":
            fig = go.Figure()

            for i, con in enumerate(country):
                pred_x, pred_y, hist_x, hist_y = model.predict(con)

                fig.add_trace(
                    go.Scatter(
                        x=hist_x,
                        y=hist_y,
                        name="History {}".format(con),
                        line=dict(width=4, color=px.colors.qualitative.Plotly[i]),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pred_x,
                        y=pred_y,
                        name="Prediction {}".format(con),
                        line=dict(
                            width=4, dash="dot", color=px.colors.qualitative.Plotly[i]
                        ),
                    )
                )

            if yaxis_type == "Log":
                fig.update_yaxes(type="log")

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
