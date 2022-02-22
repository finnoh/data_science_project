from dash import dcc, html
from dash import dash_table
from src.utils_dash import _player_selector
import dash_bootstrap_components as dbc
from src.tabs.mincer_tab import mincer

player_selector = _player_selector()

player_acc = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [dash_table.DataTable(id='playerselect-table')], title="Career Stats"
            )
        ],
        flush=False, start_collapsed=False
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
            html.Hr(className="my-2"),
            html.Div([dcc.Dropdown(
                id='playerselect-dropdown',
                options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                         enumerate(player_selector.label.unique())],
                placeholder='Select a Player',
                value=203500
            )], style={'width': '100%'})

        ],
        fluid=True,
    ),
    className="p-3 bg-light rounded-3"
)

player_selector = _player_selector()


wikitext = dbc.Alert(
    [
        html.H4("Player-Bio", className="alert-heading"),
        html.P(id='playerselect-output-container-wiki'
        ),
        html.Hr(),
        html.Span(
            [
                dbc.Badge(id='playerselect-draft', color="info", className="me-1", pill=True),
                dbc.Badge(id='playerselect-bio', color="info", className="me-1", pill=True),
                dbc.Badge(id='playerselect-score', color="info", className="me-1", pill=True),
                dbc.Badge(id='playerselect-output-container', color="info", className="me-1", pill=True)
            ]
        )
    ], color="light"
)

left_jumbotron_info = dbc.Col(
    html.Div(
        [player_info_card],
        className="h-100 p-5 text-black bg-white rounded-3",
    ),
    md=8,
)

right_jumbotron_info = dbc.Col(
    html.Div(
        [wikitext],
        className="h-100 p-5 text-black bg-white rounded-3",
    ),
    md=4,
)

left_player = dbc.Row(
    [left_jumbotron_info, right_jumbotron_info, player_acc],
    className="align-items-md-stretch",
)


right_player = dcc.Tabs(id="tabs-player-graph", value='tab-1-player-graph', children=[
    dcc.Tab(label='Hotzone', value='tab-1-player-graph', children=[dcc.Graph(id='hotzone-graph', style={'width': '100%', 'height': '100%'})]),
    dcc.Tab(label='Salary', value='tab-2-player-graph', children=[dcc.Graph(id='playerselect-graph2', style={'width': '100%', 'height': '100%'})]),
    dcc.Tab(label='Player Score', value='tab-3-player-graph', children=[dcc.Graph(id='playerselect-graph1', style={'width': '100%', 'height': '100%'})]),
    dcc.Tab(label='Player Score (interaction)', value='tab-4-player-graph', children=[dcc.Graph(id='playerselect-graph3', style={'width': '100%', 'height': '100%'})])
])

jumbotron_player = dbc.Container(
    [left_player, right_player]
)

top_players = dbc.Row([dash_table.DataTable(id='playerselect-topplayer',
                                            filter_action="native",
                                            sort_action="native",
                                            sort_mode="multi",
                                            page_action="native",
                                            page_current=0,
                                            page_size=25,
                                            style_cell={'textAlign': 'center'},
                                            style_as_list_view=True
                                            )])

draft_pick_performance = html.Div([
    html.P("Draft Pick:"),
    html.Div([
    dcc.RangeSlider(min=1, max=60, step=1, value=[1, 5], id='pick', tooltip = { 'always_visible': True })]),
    dcc.Graph(id="graph")
])

player_mincer_coefs = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [top_players, draft_pick_performance], title="Top Players and Draft Picks"
            ),
            dbc.AccordionItem(
                [mincer], title="Mincer - Salary Evaluation"
            )
        ],
        flush=False, start_collapsed=True
    )
)
