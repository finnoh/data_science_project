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
            html.Div([html.Div(
                [html.Img(id='playerselect-image')]
            )], style={'width': '49%'}
            ), html.Div(
                [dcc.Dropdown(
                    id='playerselect-dropdown',
                    options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                             enumerate(player_selector.label.unique())],
                    placeholder='Select a Player',
                    value=203500
                )], style={'width': '20%'}
            ),
            html.Div(id='playerselect-output-container'),
            html.Div(children=[html.Div(id='playerselect-output-container-wiki')], style={'width': '49%', 'display': 'inline-block'})

        ]),
        dcc.Tab(label='Performance', value='tab-2', children=[dbc.Container([
            dash_table.DataTable(
                id='playerselect-table'
            ),
            dbc.Alert(id='tbl_out')]),

            dbc.Container([
                dcc.Graph(id='playerselect-graph1')
            ])
        ]),

        dcc.Tab(label='Recommendation engine', value='tab-3', children=[
            html.H1(children='Recommendation Engine for NBA players'),
            html.Div(
                [dcc.Dropdown(
                    id='teamRec-select-dropdown',
                    options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in enumerate(team_data['abbreviation'])],
                    placeholder='Select a Team',
                    value='LAL'
                )], style={'width': '20%'}            
                ),
            html.Div(
                [dcc.Dropdown(
                    id='teamRec-starting5-dropdown',
                    placeholder='Select a player',
                    value='LeBron James'
                )], style={'width': '20%'}            
                ),
            html.Div(id='teamRec-player-dropdown')
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
                dcc.Graph(id='dimreduction-graph1')
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
    Output('dimreduction-graph1', 'figure'),
    [Input('dimreduction-dropdown', 'value')])
def get_emb(value):
    players_stats, _, positions, data_names = recommmendation_engine.embeddings(value)
    fig = px.scatter(players_stats, x="embedding_1", y="embedding_2", color = positions, hover_name = data_names)
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('teamRec-starting5-dropdown', 'options'),
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(value):
    players_team = recommmendation_engine.starting_five(value, names=True)
    return [{'label': i, 'value': i} for i in players_team.keys()]


@app.callback(
    Output('teamRec-player-dropdown', 'children'),
    [Input('teamRec-starting5-dropdown', 'value')])
def selected_player(value):
    data_emb, emb, _, _ = recommmendation_engine.embeddings('umap')
    sample_recommendation = recommmendation_engine.RecommendationEngine(data_emb, value, emb, 'Similar')
    r = sample_recommendation.recommend()
    return f"{r} was returned"

if __name__ == '__main__':
    app.run_server(debug=True)
