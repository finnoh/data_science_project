import dash
import wikipediaapi
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from src.get_data import get_clean_player_data
from src.utils_dash import _player_selector
import recommmendation_engine
from hotzone import hotzone

import dash_bootstrap_components as dbc

app = dash.Dash(__name__, title="NBA GM")

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# SETUP STATIC DATA
player_selector = _player_selector()
player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()

# APP LAYOUT

app.layout = html.Div(children=[
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Player Bio', value='tab-1', children=[
            html.H1(children='NBA GM'),
            dbc.Row([dbc.Col(html.Div([html.Div([html.Img(id='playerselect-image')])], style={'width': '49%'})),
                    dbc.Col(html.Div([html.Div([html.Img(id='teamSel-image')])], style={'width': '49%'}))
                    ]
                ),
            html.Div([dcc.Dropdown(
                    id='playerselect-dropdown',
                    options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                             enumerate(player_selector.label.unique())],
                    placeholder='Select a Player',
                    value=203500
                )], style={'width': '20%'}
            ),
            html.Div(id='playerselect-output-container'),
            html.Div(children=[html.Div(id='playerselect-output-container-wiki')], style={'width': '49%', 'display': 'inline-block'}),
            dbc.Container([dash_table.DataTable(id='playerselect-table'),
                            dbc.Alert(id='tbl_out')]),
                            dbc.Container([dcc.Graph(id='playerselect-graph1')]
            ),
            dbc.Container([
                dcc.Graph(id='hotzone-graph')
            ])

        ]),

        dcc.Tab(label='Recommendation engine', value='tab-3', children=[
            html.H1(children='Recommendation Engine for NBA players'),
            html.Div([html.Div(
                [html.Img(id='teamRep-image')]
            )], style={'height':'5%', 'width':'5%'}),
            html.Div(
                [dcc.Dropdown(
                    id='teamRec-select-dropdown',
                    options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in enumerate(team_data['abbreviation'])],
                    placeholder='Select a Team',
                    value='LAL'
                )], style={'width': '20%'}            
                ),
            html.Div([html.Div(
                [html.Img(id='playerRep-image')]
            )], style={'width': '49%'}),
            html.Div(
                [dcc.Dropdown(
                    id='teamRec-starting5-dropdown',
                    placeholder='Select a player',
                    value='LeBron James'
                )], style={'width': '20%'}            
                ),
            dcc.Loading(children = [html.Div(id='teamRec-player-dropdown')], fullscreen=False, type='dot', color="#119DFF"),
            html.Div([html.Div(
                [html.Img(id='playerRec-image')]
            )], style={'width': '49%'})
            ]
        ),

        dcc.Tab(label='Dimensionality Reduction', value='tab-4', children=[
            html.H1(children='Projections of active NBA players into 2D'), 
            html.Div(
                [dcc.Dropdown(
                    id='dimreduction-dropdown',
                    options=[{'label': 'Sepectral Embedding', 'value': 'spectral'},
                             {'label': 'TSNE', 'value': 'tsne'},
                             {'label': 'UMAP', 'value': 'umap'},
                             {'label': 'PCA', 'value': 'pca'},],
                    placeholder='Select a dimensionality reduction technique',
                    value='spectral'
                )], style={'width': '20%'}
            ),
            dbc.Container([
                            dcc.Loading(children = [dcc.Graph(id='dimreduction-graph1')], fullscreen=False, type='dot', color="#119DFF")
            ])
            ]
        )
    ])

])


# APP CALLBACKS

@app.callback(
    Output('playerselect-output-container', 'children'),
    Input('playerselect-dropdown', 'value'))
def update_output(value):
    return f'Player has the ID: {value}'

@app.callback(
    Output('teamSel-image', 'src'),
    [Input('playerselect-dropdown', 'value')])
def update_image_selTeam(value):
    team_abb = list(player_data[player_data['id'] == value]['team'])[0]
    return f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{team_abb.lower()}.png"

@app.callback(
    [Output('playerselect-table', 'data'),
     Output('playerselect-table', 'columns'),
     Output('playerselect-graph1', 'figure')],

    [Input('playerselect-dropdown', 'value')])
def update_player(value):
    # make api call
    df = get_clean_player_data(player_id=value)

    # get objects
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')

    # get figure
    fig = px.line(df, x="SEASON_ID", y="PTS")
    fig.update_layout(transition_duration=500)

    return data, columns, fig

@app.callback(
    Output('playerselect-image', 'src'),
    [Input('playerselect-dropdown', 'value')])
def update_image_src(value):
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(value)}.png"

@app.callback(
    Output('playerselect-output-container-wiki', 'children'),
    [Input('playerselect-dropdown', 'value')])
def _player_wiki_summary(value):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    df = player_selector
    name = list(df[df['value'] == value]['label'])[0]
    page_py = wiki_wiki.page(name)
    if page_py.exists():
        return page_py.summary
    else:
        return f"No Wikipedia page found for {str(name)}"

@app.callback(
    Output('hotzone-graph', 'figure'),
    [Input('playerselect-dropdown', 'value')])
def hotzone_graph(value):
    shots = hotzone(value)
    fig = px.scatter(shots, x="LOC_X", y="LOC_Y", color = 'SHOT_MADE_FLAG', title= "Hot zone",
                    width=1200, height=1000)
    fig.update_layout(transition_duration=500, template='simple_white')
    return fig


@app.callback(
    Output('dimreduction-graph1', 'figure'),
    [Input('dimreduction-dropdown', 'value')])
def get_emb(value):
    players_stats, _, positions, data_names, player_stats = recommmendation_engine.embeddings(value)
    name_emb = {'spectral': 'Sepectral Embedding', 'tsne': 'TSNE', 'umap': 'UMAP', 'pca': 'PCA'}
#    player_stats.positions = positions
#    player_stats.head()
    fig = px.scatter(players_stats, x="embedding_1", y="embedding_2", color = positions, hover_name = data_names, 
                    hover_data={'embedding_1':False, # remove species from hover data
                                'embedding_2':False, # customize hover for column of y attribute,
                              #  'positions': False,
                               # 'Name': data_names,
                                'Position': positions, # add other column, default formatting
                                'Age': player_stats['PLAYER_AGE'],
                                'Points': player_stats['GP'],
                                '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                                'Assists': (':.3f', player_stats['AST']),
                                'Rebounds': (':.3f', player_stats['REB'])
                                },
                    labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2"}, title=f"{name_emb[str(value)]} representation of NBA players")
    fig.update_layout(transition_duration=500, template='simple_white')
    return fig
# (['PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       #'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       #'BLK', 'TOV', 'PF', 'PTS'],

@app.callback(
    Output('teamRec-starting5-dropdown', 'options'),
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(value):
    players_team = recommmendation_engine.starting_five(value, names=True)
    return [{'label': i, 'value': i} for i in players_team.keys()]

@app.callback(
    Output('playerRep-image', 'src'),
    [Input('teamRec-starting5-dropdown', 'value')])
def update_image_repPlayer(value):
    player_id = list(player_data[player_data['player_names'] == value]['id'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"

@app.callback(
    Output('teamRep-image', 'src'),
    [Input('teamRec-select-dropdown', 'value')])
def update_image_repTeam(value):
    return f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{value.lower()}.png"

@app.callback(
    Output('teamRec-player-dropdown', 'children'),
    Input('teamRec-starting5-dropdown', 'value'))
def selected_player(value):
    data_emb, emb, _, _, _ = recommmendation_engine.embeddings('umap')
    sample_recommendation = recommmendation_engine.RecommendationEngine(data_emb, value, emb, 'Similar')
    r = sample_recommendation.recommend()
    return r

#@app.callback(
#    Output('teamRec-player-name', 'children'),
#    Input('teamRec-player-dropdown', 'data'))
#def print_recommended_player(value):
#    return f"{value} was recommended."

@app.callback(
    Output('playerRec-image', 'src'),
    Input('teamRec-player-dropdown', 'children'))
def update_image_recPlayer(children):
    player_id = list(player_data[player_data['player_names'] == children]['id'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"


if __name__ == '__main__':
    app.run_server(debug=True)


# Recommendation Tab:
# which embedding, fit/similar, outputs
# only umap works in recommendation engine
# add correct output (incl. graphs) from recommendation
# add dropdown for similar or fit
# option: age included, age excluded for best fit
# which player fit best to strategy
# cut off for minutes played; player obere bubble?
# age rausnehmen?
# box für user mit attributen zur auswahl

# Finn: 
# 5 wichtigsten attribute pro spieler angeben -> wie angeben @ Finn?
# performance plot bei steph curry?
# finns parameter einbauen

# Präsi:
# story kommunizieren, data management (wie daten, wie transformiert?), prozess dokumentieren, auch methodik zeigen, was sind unsere Fragen? -> wie passen Modelle zusammen
# wird nicht benotet
# columns for projection in präsi
# skizze for ideal präsi
# max. 15 min präsi

# Done:
# add Spinner (via output von model?) https://www.youtube.com/watch?v=t1bKNj021do
# performance & hot zone, logo player -> player
# add picture of player below selection
# add "loading" button? https://stackoverflow.com/questions/54439548/display-loading-symbol-while-waiting-for-a-result-with-plot-ly-dash
# https://community.plotly.com/t/updating-a-dropdown-menus-contents-dynamically/4920
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/
# NBA: difference mit weights scales

# additional scraping of: https://www.basketball-reference.com/
# steph curry -> louis williams?
# # IDs in file als string importen
# dim reduction (search for player)
