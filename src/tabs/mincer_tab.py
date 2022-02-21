from dash import dcc, html
import dash_daq as daq
from dash import dash_table
from src.mincer import *
import dash_bootstrap_components as dbc

mincer = html.Div([dbc.Container([dcc.Dropdown(
    options=[
        {'label': 'Ordinary Least Squares', 'value': 'ols'},
        {'label': 'Random Forest', 'value': 'rf'},
        {'label': 'Support Vector Machine', 'value': 'svr'},
    ],
    value='ols', id="mincer-model-dropdown"

), daq.BooleanSwitch(id='mincer-log-switch', on=False, label="Logarithm?", labelPosition="top")]),
#)]),

    dbc.Container([
        dcc.Loading([dcc.Graph(id='mincer-output-graph', clear_on_unhover=True)], fullscreen=False, type='dot',
                    color="#119DFF"),
        dcc.Tooltip(id="graph-tooltip"),
        html.Div(id='mincer-output-container')])]
)
