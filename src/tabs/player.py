from dash import dcc, html
from dash import dash_table
from src.utils_dash import _player_selector
import dash_bootstrap_components as dbc

player_selector = _player_selector()

player_acc = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [dash_table.DataTable(id='playerselect-table')], title="Career Stats"
            ),
            dbc.AccordionItem(
                html.Div(id='playerselect-output-container-wiki'), title="Player-Bio"
            )
        ],
        flush=False,
    )
)

player_info_card = html.Div(
    dbc.Container(
        [
            html.H2(id='playerselect-name-container', className="display-3"),
            html.Div(
                [html.Div([html.Img(id='playerselect-image', style={'width': '100%'})],
                          style={'display': 'inline-block'}),
                 html.Div([html.Img(id='teamSel-image', style={'width': '50%'})], style={'display': 'inline-block'})],
                style={'width': '200%', 'display': 'inline-block'}),
            html.Div([dcc.Dropdown(
                id='playerselect-dropdown',
                options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                         enumerate(player_selector.label.unique())],
                placeholder='Select a Player',
                value=203500
            )], style={'width': '33%'}),
            html.Hr(className="my-2"),
            html.Div(id='playerselect-output-container'),
            html.Div(id='playerselect-score')
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-light rounded-3"
)

player_selector = _player_selector()

left_player = dbc.Col([player_info_card,
                       player_acc
                       ], md=6)

right_player = dbc.Col([dbc.Container([
    dcc.Graph(id='hotzone-graph')]), dbc.Container([dcc.Graph(id='playerselect-graph1')])
], md=6)

jumbotron_player = dbc.Row(
    [left_player, right_player],
    className="align-items-md-stretch"
)