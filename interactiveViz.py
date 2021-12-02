import dash
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import pandas as pd
from dash.dependencies import Input, Output
import json
import numpy as np
import filter_dash

from data_processing import getData
from predict import Model

app = dash.Dash(__name__)

colors = {"background": "#282b38", "text": "#979A9C"}  # 282b38, a5b1cd

df = getData()

df = df.rename(columns={elem: elem[0].upper() + elem[1:] for elem in df.columns})
df = df.sort_index(axis=1)


# load only some of the data for faster updating
# df = pd.DataFrame(
#    {"location": np.arange(10), "new_cases": np.arange(10), "continent": np.arange(10)}
# )

# model = Model(df)
# df_predict = model.predict("Norway")

fig = px.bar(df, x="Location", y="New_cases", color="Continent", barmode="group")

fig.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font_color=colors["text"],
)

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            html.A(
                                "Covid-19 Data Explorer",
                                href="https://github.com/marc131183/Covid19-Cases",
                                style={
                                    "text-decoration": "none",
                                    "color": colors["text"],
                                    "fontSize": 50,
                                },
                            )
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
            ],
        ),
        "Filter query:",
        dcc.Input(
            id="filter-query-input",
            placeholder="Enter filter query",
            style={
                "width": "30%",
                "height": "30px",
                "color": colors["text"],
            },
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
                    labelStyle={"display": "inline-block", "color": colors["text"]},
                ),
            ],
            style={
                "width": "30%",
                "float": "right",
                "display": "inline-block",
            },
        ),
        html.Div(
            [
                "Color by:",
                dcc.Dropdown(
                    id="grouping",
                    placeholder="Choose how to color",
                    options=[{"label": elem, "value": elem} for elem in df.columns],
                ),
            ],
            style={
                "width": "30%",
                "float": "right",
                "display": "inline-block",
                "marginRight": "10px",
                "color": colors["text"],
            },
        ),
        html.Div(
            [
                "Y-parameter:",
                dcc.Dropdown(
                    id="yaxis-column",
                    placeholder="Choose a y-parameter",
                    options=[{"label": elem, "value": elem} for elem in df.columns],
                    multi=True,
                ),
                dcc.RadioItems(
                    id="yaxis-type",
                    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                    value="Linear",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={
                "width": "48%",
                "float": "down",
                "display": "inline-block",
                "color": colors["text"],
            },
        ),
        html.Div(
            [
                "X-parameter:",
                dcc.Dropdown(
                    id="xaxis-column",
                    placeholder="Choose a x-parameter",
                    options=[{"label": elem, "value": elem} for elem in df.columns],
                    # multi=True,
                ),
                dcc.RadioItems(
                    id="xaxis-type",
                    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                    value="Linear",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={
                "width": "48%",
                "float": "right",
                "display": "inline-block",
                "color": colors["text"],
            },
        ),
        dcc.Graph(id="indicator-graphic"),
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
                    style={"border": "thin lightgrey solid", "overflowX": "scroll"},
                ),
            ],
            className="three columns",
        ),
    ],
)


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

    if (
        query is None or query == ""
    ):  # remember: add options for filter and a country filter
        df_ = df.copy()
    else:
        derived_query_structure = filter_dash.split_query(query)
        df_ = (
            df.copy()
            .groupby("Location")
            .apply(lambda x: x[filter_dash.resolve_query(x, derived_query_structure)])
        )

    if len(yaxis_column_name) > 1:
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
                    cols_ysecondary += " / {}".format(yaxis)
                else:
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
        if plot_type == "Scatter":  # remember: only make some parameters available?
            fig = px.scatter(
                df_,
                x=df_[xaxis_column_name],
                y=[df_[ycolumn] for ycolumn in yaxis_column_name],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=(df_[grouping] if grouping != None else None),
            )

        elif plot_type == "Bar":
            fig = px.bar(
                df_,
                x=df_[xaxis_column_name],
                y=[df_[ycolumn] for ycolumn in yaxis_column_name],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=(df_[grouping] if grouping != None else None),
            )

        elif plot_type == "Line":
            fig = px.line(
                df_,
                x=df_[xaxis_column_name],
                y=[df_[ycolumn] for ycolumn in yaxis_column_name],
                log_x=xaxis_type == "Log",
                log_y=yaxis_type == "Log",
                color=(df_[grouping] if grouping != None else None),
            )

        # elif plot_type == "Predict":
        #     fig = px.line(  # remember: fix prediction
        #         df_predict,
        #         x=df_predict["date"],
        #         y=df_predict["new_cases"],
        #         log_x=xaxis_type == "Log",
        #         log_y=yaxis_type == "Log",
        #     )

    fig.update_layout(margin={"l": 40, "b": 40, "t": 20, "r": 0})

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
