import recommmendation_engine
from dash import dcc, html
import dash_bootstrap_components as dbc

player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()

top_card_rec = dbc.Card(
    [
        dbc.CardImg(id='playerRep-image', top=True),
        dbc.CardBody(
            html.Div(
                [dcc.Dropdown(
                    id='teamRec-starting5-dropdown',
                    placeholder='Select a player',
                    value='LeBron James'

                )], style={'width': '100%'}
            ),
        ),
    ],
    style={"width": "15rem", "align": "auto"}
)

bottom_card_rec = dbc.Card(
    [
        dbc.CardImg(id='playerRec-image', top=True),
        dbc.CardBody(html.H4(id='teamRec-player-dropdown', className="card-text"))
    ],
    style={"width": "15rem", "align": "auto"}
)

team_card_rec = dbc.Card(
    [
        dbc.CardImg(id='teamRep-image', top=True),
        dbc.CardBody(
            [dcc.Dropdown(
                id='teamRec-select-dropdown',
                options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
                         enumerate(team_data['abbreviation'])],
                placeholder='Select a Team',
                value='LAL'

            )])
    ],
    style={"width": "20rem", "align": "auto"})

method_select_rec = html.Div([
    dbc.Label("Method"),
    dbc.RadioItems(
        options=[{'label': 'Similar player', 'value': 'Similar'},
                 {'label': 'Complementary player', 'value': 'Fit'}],
        value='Similar',
        id='recommendation-type'
    ),
])

cards_rec = dbc.Row(
    [method_select_rec,
     dbc.Col(team_card_rec, width="auto"),
     dbc.Col(top_card_rec, width="auto"),
     dbc.Col(dcc.Loading([bottom_card_rec], fullscreen=False, type='dot', color="#119DFF"), width="auto")
     ], className="align-items-md-stretch"
)

dim_red = dbc.Col([html.Div(
    [dcc.Dropdown(
        id='dimreduction-dropdown',
        options=[{'label': 'Spectral Embedding', 'value': 'spectral'},
                 {'label': 'TSNE', 'value': 'tsne'},
                 {'label': 'UMAP', 'value': 'umap'},
                 {'label': 'PCA', 'value': 'pca'}],
        placeholder='Select a dimensionality reduction technique',
        value='spectral'
    )], style={'width': '60%'}
),
    dbc.Container([
        dcc.Loading(children=[dcc.Graph(id='dimreduction-graph1')], fullscreen=False, type='dot',
                    color="#119DFF")
    ])], md=12)
