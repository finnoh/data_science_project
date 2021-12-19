import dash
import wikipediaapi
import requests
from PIL import Image
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from src.get_data import get_clean_player_data, get_team_image
from src.utils_dash import _player_selector, _team_selector, _link_team_website, _team_full_name, _get_team_id, \
    _get_mvp_id_team, _player_full_name, _mvp_descr_builder
import dash
import wikipediaapi
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
from src.get_data import get_clean_player_data
from src.utils_dash import _player_selector
import recommmendation_engine

import dash_bootstrap_components as dbc

app = dash.Dash(__name__, title="NBA GM", external_stylesheets=[dbc.themes.LUX])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# SETUP STATIC DATA
player_selector = _player_selector()
team_selector = _team_selector()
player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()

# APP ELEMENTS

offcanvas = html.Div(
    [
        dbc.Button("Further Team Information", id="teamselect-open-offcanvas", n_clicks=0, color='light'),
        dbc.Offcanvas([html.Div(
            html.P(id='teamselect-output-wiki'
                   )),

            dbc.Button(id="teamselect-link-button",
                       target="_blank",
                       color="info", external_link=True)],
            backdrop=True,
            scrollable=True,
            id="offcanvas",
            is_open=False,
            placement='top'
        ),
    ]
)

col_teamname = dbc.Col(html.Div(
    [html.H2(id='teamselect-output-container',
             className="display-3", style={'margin': 'auto', 'width': '100%', 'display': 'inline-block'}), offcanvas]),
    md=10)

col_logo = dbc.Col(html.Div(
    [html.Img(id='teamselect-image', style={'margin': 'auto', 'width': '100%', 'display': 'inline-block'})]), md=2)

starplayer = dbc.Alert(
    [
        html.H4("Starplayer", className="alert-heading"),
        html.P(id='teamselect-mvp-descr'),
        html.Hr(),
        html.P(
            id='teamselect-mvp-name',
            className="mb-0",
        ),
    ]
)

right_part = dbc.Col([html.Div(
    [html.Img(id='teamselect-mvp-image', style={'margin': 'auto', 'width': '50%', 'display': 'inline-block'})]),
    starplayer],
    md=4)

left_jumbotron = dbc.Col([dbc.Row([col_teamname, col_logo], className="align-items-md-stretch"),
                          html.Hr(className="my-2"),
                          dcc.Dropdown(
                              id='teamselect-dropdown',
                              options=[{'label': team,
                                        'value': team_selector.iloc[
                                            i, 1]} for i, team in
                                       enumerate(
                                           team_selector.label.unique())],
                              value='ATL'
                          ), html.Hr(className="my-2"),
                          dbc.Container([
                              dcc.Graph(id='teamselect-capspace-graph')
                          ])], md=8, className="h-100 p-5 bg-light border rounded-3")

jumbotron = dbc.Row(
    [left_jumbotron, right_part],
    className="align-items-md-stretch",
)

# APP LAYOUT

app.layout = html.Div(children=[
    dcc.Tabs(id='tabs-example', value='tab-1', vertical=False, children=[
        dcc.Tab(label='Player Bio', value='tab-1', children=[
            html.H2(children='Player', className="display-3"),
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
            html.Div(children=[html.Div(id='playerselect-output-container-wiki')],
                     style={'width': '49%', 'display': 'inline-block'})

        ]),
        dcc.Tab(label='Performance', value='tab-2', children=[dbc.Container([
            dash_table.DataTable(
                id='playerselect-table'
            ),
            dbc.Alert(id='tbl_out')]),

            dbc.Container([
                dcc.Graph(id='playerselect-graph1')
            ])]),
        dcc.Tab(label='Recommendation engine', value='tab-3', children=[
            html.H2(children='Recommendation Engine for NBA players', className="display-3"),
            html.Div(
                [dcc.Dropdown(
                    id='teamRec-select-dropdown',
                    options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
                             enumerate(team_data['abbreviation'])],
                    placeholder='Select a Team',
                    value='LAL'
                )], style={'width': '80%'}
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
            html.H2(children='Projections of active NBA players into 2D', className="display-3"),
            html.Div(
                [dcc.Dropdown(
                    id='dimreduction-dropdown',
                    options=[{'label': 'Sepectral Embedding', 'value': 'spectral'},
                             {'label': 'TSNE', 'value': 'tsne'},
                             {'label': 'UMAP', 'value': 'umap'},
                             {'label': 'PCA', 'value': 'pca'}, ],
                    placeholder='Select a dimensionality reduction technique',
                    value='spectral'
                )], style={'width': '20%'}
            ),
            dbc.Container([
                dcc.Graph(id='dimreduction-graph1')
            ])
        ]
                ),

        dcc.Tab(label='Team', value='tab-5', children=[jumbotron
                                                       ])
    ], colors={
        "border": "white",
        "primary": "#17408b",
        "background": "white"})
])


# APP CALLBACKS

@app.callback(
    Output('playerselect-output-container', 'children'),
    Input('playerselect-dropdown', 'value'))
def update_output(value):
    return f'Player has the ID: {value}'


@app.callback(
    [Output('teamselect-output-container', 'children'),
     Output('offcanvas', 'title')],
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return _team_full_name(value), _team_full_name(value)


@app.callback(
    [dash.dependencies.Output('teamselect-mvp-image', 'src'),
     dash.dependencies.Output('teamselect-mvp-descr', 'children'),
     dash.dependencies.Output('teamselect-mvp-name', 'children')],
    [dash.dependencies.Input('teamselect-dropdown', 'value')])
def update_output(value):
    team_id = _get_team_id(value)
    mvp_data, url_image = _get_mvp_id_team(team_id=team_id, season='2020-21')
    mvp_name, mvp_pos = _player_full_name(player_id=mvp_data[0])
    descr = _mvp_descr_builder(mvp_name=mvp_name, mvp_position=mvp_pos, mvp_data=mvp_data)

    return url_image, descr, mvp_name


@app.callback(
    Output('teamselect-link-button', 'children'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    full_name = _team_full_name(value)
    return f'Visit the {full_name}\'s website.'


@app.callback(
    Output('teamselect-link-button', 'href'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    nickname = _link_team_website(value)
    return f"https://www.nba.com/{nickname}/"


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
    dash.dependencies.Output('playerselect-image', 'src'),
    [dash.dependencies.Input('playerselect-dropdown', 'value')])
def update_image_src(value):
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(value)}.png"


@app.callback(
    dash.dependencies.Output('teamselect-image', 'src'),
    [dash.dependencies.Input('teamselect-dropdown', 'value')])
def get_team_image(value, season: str = '2021-22'):
    """

    :param team_abb:
    :return:
    """
    url = f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{value.lower()}.png"
    return url


@app.callback(
    dash.dependencies.Output('teamselect-output-wiki', 'children'),
    [dash.dependencies.Input('teamselect-dropdown', 'value')])
def _team_wiki_summary(value):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    full_name = _team_full_name(value)
    page_py = wiki_wiki.page(full_name)
    return page_py.summary


@app.callback(
    Output("offcanvas", "is_open"),
    Input("teamselect-open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


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
    fig = px.scatter(players_stats, x="embedding_1", y="embedding_2", color=positions, hover_name=data_names)
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


@app.callback(
    Output('teamselect-capspace-graph', 'figure'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return recommmendation_engine.visualize_capspace_team_plotly(value)


if __name__ == '__main__':
    app.run_server(debug=True)

# starting five, player scores, cap space, visualisierung cap space, get team salaries, design
