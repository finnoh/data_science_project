from dash import dcc, html
import dash_daq as daq
from dash import dash_table
from src.mincer import *
import dash_bootstrap_components as dbc
import recommmendation_engine
from src.utils_dash import _player_selector
from dash import dash_table

player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()
player_selector = _player_selector()

top_card_rec = dbc.Card(
    [
        dbc.CardImg(id='prediction-playerRep-image', top=True),
        dbc.CardBody(
            html.Div(
                [dcc.Dropdown(
                id='prediction-teamRec-starting5-dropdown',
                options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                         enumerate(player_selector.label.unique())],
                placeholder='Select a Player',
                value=203500
            )], style={'width': '100%'}
            ),
        ),
    ],
    style={"width": "15rem", "align": "auto"}
)

bottom_card_rec = dbc.Card(
    [
        dbc.CardImg(id='prediction-playerRep-image2', top=True),
        dbc.CardBody(
            html.Div(
                [dcc.Dropdown(
                id='prediction-teamRec-starting5-dropdown2',
                options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                         enumerate(player_selector.label.unique())],
                placeholder='Select a Player',
                value=203500
            )], style={'width': '100%'}
            ),
        ),
    ],
    style={"width": "15rem", "align": "auto"}
)

team_card_rec = dbc.Card(
    [
        dbc.CardImg(id='prediction-teamRep-image', top=True),
        dbc.CardBody(
            [dcc.Dropdown(
                id='prediction-teamRec-select-dropdown',
                options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
                         enumerate(team_data['abbreviation'])],
                placeholder='Select a Team',
                value='LAL'

            )])
    ],
    style={"width": "20rem", "align": "auto"})

validation_trade = html.Div([
    dbc.Container([
        dcc.Loading([dcc.Graph(id='prediction-output-graph_trade', clear_on_unhover=True),
                     dcc.Graph(id='prediction-output-graph2_trade', clear_on_unhover=True),
                     dash_table.DataTable(id='prediction-output-table_trade')], fullscreen=False, type='dot',
                    color="#119DFF")])])


simulation_trade = html.Div([
    dbc.Container([
        dcc.Loading([dcc.Graph(id='prediction-output-graphb_trade', clear_on_unhover=True),
                     dcc.Graph(id='prediction-output-graph2b_trade', clear_on_unhover=True),
                     dash_table.DataTable(id='prediction-output-tableb_trade')], fullscreen=False, type='dot',
                    color="#119DFF")])])

prediction = html.Div(children=[dbc.Container([html.Button('Submit', id='prediction-submit_trade', n_clicks=0)]), top_card_rec, bottom_card_rec, team_card_rec,
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Validation Trade', value='tab-1-example-graph-trade', children=[validation_trade, dash_table.DataTable(id='prediction-validation-table')]),
        dcc.Tab(label='Season Prediction Trade', value='tab-2-example-graph-trade', children=simulation_trade)])])