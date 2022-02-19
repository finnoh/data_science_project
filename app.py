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
#import recommmendation_engine
from hotzone import hotzone

import copy
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
import recommmendation_engine as recommmendation_engine

import dash_bootstrap_components as dbc


import flask
server = flask.Flask(__name__)
app = dash.Dash(__name__, title="NBA GM", external_stylesheets=[dbc.themes.LUX], server = server)


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# SETUP STATIC DATA
player_selector = _player_selector()
team_selector = _team_selector()
player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()
players_stats = recommmendation_engine.get_players_stats()
boxscores_20_21 = recommmendation_engine.get_boxscores('20_21')



def highlight_max_col(df):
    df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
    colors = ['#BDF6BB', '#9CF599', '#28B463', '#FBA898', '#F8775E', '#FF0000'] #https://htmlcolorcodes.com/
    styles = []
    for col in df_numeric_columns.keys():
        rep_value = list(df[col])[0]
        comparison_values = [rep_value+perc*rep_value for perc in [0.00001, 0.2, 0.5, -0.0001, -0.2, -0.5]]
        if col in ['distance', 'luxury_tax', 'PF', 'TOV']:
            for branch in range(len(comparison_values)):
                if branch < 3:
                    styles.append({
                        'if': {
                            'filter_query': '{{{col}}} < {value} && {{{col}}} < {rep_value}'.format(col = col, value = comparison_values[branch], rep_value = rep_value),
                            'column_id': col
                        },
                        'color': colors[branch],
                        #'color': 'white'
                    })
                else:
                    styles.append({
                        'if': {
                            'filter_query': '{{{col}}} > {value} && {{{col}}} > {rep_value}'.format(col = col, value = comparison_values[branch], rep_value = rep_value),
                            'column_id': col
                        },
                       # 'backgroundColor': colors[branch],
                        'color': colors[branch] #'white'
                    })
        else:
            for branch in range(len(comparison_values)):
                if branch < 3:
                    styles.append({
                        'if': {
                            'filter_query': '{{{col}}} > {value} && {{{col}}} > {rep_value}'.format(col = col, value = comparison_values[branch], rep_value = rep_value),
                            'column_id': col
                        },
                       # 'backgroundColor': colors[branch],
                        'color': colors[branch] #'white'
                    })
                else:
                    styles.append({
                        'if': {
                            'filter_query': '{{{col}}} < {value} && {{{col}}} < {rep_value}'.format(col = col, value = comparison_values[branch], rep_value = rep_value),
                            'column_id': col
                        },
                        #'backgroundColor': colors[branch],
                        'color': colors[branch] #'white'
                    })
    return styles


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
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Player Bio', value='tab-1', children=[
            html.H2(children='Player', className="display-3"), #html.H1(children='NBA GM'),
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
            html.Div(children=[html.Div(id='playerselect-output-container-wiki')],
                     style={'width': '49%', 'display': 'inline-block'}),
            dbc.Container([dash_table.DataTable(id='playerselect-table'),
                           dbc.Alert(id='tbl_out2')]),
            dbc.Container([dcc.Graph(id='playerselect-graph1')]
                          ),

            dbc.Container([
                dcc.Graph(id='hotzone-graph')
            ])

        ]),

        dcc.Tab(label='Recommendation engine', value='tab-3', children=[
            html.H2(children='Recommendation Engine for NBA players', className="display-3"),
            dbc.Row([dbc.Col(html.Div(
                                        [dcc.RadioItems( # Dropdown
                                            id='recommendation-type',
                                            options=[{'label': ' Similar player', 'value': 'Similar'},
                                                    {'label': ' Complementary player', 'value': 'Fit'}],
                                        # placeholder='Select a recommendation technique',
                                            value='Similar'
                                        )], style={'width': '30%'}
                                    )),
                     dbc.Col(html.Div(
                                        [dcc.RadioItems(
                                            id='recommendation-distance',
                                            options=[{'label': ' L1 distance', 'value': 'L1'},
                                                    {'label': ' L2 distance', 'value': 'L2'}],
                                            #placeholder='Select a distance measure',
                                            value='L2'
                                        )], style={'width': '20%'}
                                    )),
                     ]),
            
            
            html.Plaintext("Weights (in percent) for the seasons 2018/2019, 2019/2020 and 2020/2021, respectively"),
            html.Div([
                dcc.Input(id="weight1", autoComplete="off", type="number", placeholder = 'Season 2020/21', min=0, max=100, value=70),
                dcc.Input(id="weight2", autoComplete="off", type="number", placeholder = 'Season 2019/20', min=0, max=100, value=20),
                dcc.Input(id="weight3", autoComplete="off", type="number", placeholder = 'Season 2018/19', min=0, max=100, value=10),
               # html.Button('Calculate', id='btn_weights')
            ]), 
            html.Div(
                [dcc.Checklist(
                        id="checklist-allColumns",
                        options=[{"label": "All attributes", "value": "All"}, {"label": "Offensive attributes", "value": "Off"}, {"label": "Defensive attributes", "value": "Def"}],
                        value=["All"],
                        labelStyle={"display": "inline-block"},
                    ),
                    dcc.Checklist(
                        id="checklist-columns",
                        options=[
                       #     {"label": "Player Age", "value": "PLAYER_AGE"},
                       #     {"label": "Games Play", "value": "GP"},
                       #     {"label": "GS", "value": "GS"},
                       #     {"label": "MIN", "value": "MIN"},
                            {"label": "FGM", "value": "FGM"},
                            {"label": "FGA", "value": "FGA"},
                            {"label": "FG_PCT", "value": "FG_PCT"},
                            {"label": "FG3M", "value": "FG3M"},
                            {"label": "FG3A", "value": "FG3A"},
                            {"label": "FG3_PCT", "value": "FG3_PCT"},
                            {"label": "FTM", "value": "FTM"},
                            {"label": "FTA", "value": "FTA"},
                            {"label": "FT_PCT", "value": "FT_PCT"},
                            {"label": "Offensive Rebounds", "value": "OREB"},
                            {"label": "Defensive Rebounds", "value": "DREB"},
                            {"label": "Rebounds", "value": "REB"},
                            {"label": "Assists", "value": "AST"},
                            {"label": "Steals", "value": "STL"},
                            {"label": "Blocks", "value": "BLK"},
                            {"label": "Turnovers", "value": "TOV"},
                            {"label": "Personal Fouls", "value": "PF"},
                            {"label": "Points", "value": "PTS"},
                        ],
                        value=[],
                        labelStyle={"display": "inline-block"}, #'flex'
                    ),
                ]),


#Index(['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION',
#       'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
#       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
#       'BLK', 'TOV', 'PF', 'PTS'],

            html.Div([html.Div(
                [html.Img(id='teamRep-image')]

            )], style={'height': '5%', 'width': '5%'}),

            html.Div(
                [dcc.Dropdown(
                    id='teamRec-select-dropdown',
                    options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in enumerate(team_data['abbreviation'])],
                    placeholder='Select a Team',
                    value='LAL'

                )], style={'width': '20%'}
            ),
            ######
            html.Div([
                dbc.Row([dbc.Col(html.Div([html.Img(id='playerRep-image_1')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_2')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_3')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_4')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_5')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}))
                         ]),
                dbc.Row([dbc.Col(html.Div(id='playerRep-str_1', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRep-str_2', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRep-str_3', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRep-str_4', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRep-str_5', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),  
                         ]),
                dbc.Row([dbc.Col(html.Div([html.Button('Excute this trade!', id='btn_1')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Excute this trade!', id='btn_2')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Excute this trade!', id='btn_3')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Excute this trade!', id='btn_4')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Excute this trade!', id='btn_5')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),  
                         ]),
                dbc.Row([dbc.Col(html.Div([html.Img(id='playerRec-image_1')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_2')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_3')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_4')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_5')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}))
                         ]),
                dbc.Row([dbc.Col(html.Div(id='playerRec-caption_1', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRec-caption_2', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRec-caption_3', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRec-caption_4', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div(id='playerRec-caption_5', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),  
                         ])      
            ]),
            dcc.Store(id='pos_img'),


#html.Div(children=[dcc.Graph(id='plot1')] if condition else [dcc.Graph(id='plot2')])


            ######
            
            dcc.Loading(children = [html.Div(id='teamRec-player-dropdown', style = {'display': 'none'})], fullscreen=False, type='dot', color="#119DFF"),
            html.Div([html.Div(
                [html.Img(id='playerRec-image')]
            )], style={'width': '49%'}),

            html.Div(id='playerRec-table'),

            #dbc.Container([dash_table.DataTable(id='playerRec-table', style_data_conditional=highlight_max_row(pd.DataFrame(data))),
            #               dbc.Alert(id='tbl_out')])
           # html.Div([html.Div(
           #     [html.Table(id='playerRec-table')]
           # )]),

            #dcc.Loading(children = [html.Div(id='players-recommended', style = {'display': 'none'})], fullscreen=False, type='dot', color="#119DFF"),
            dcc.Store(id='players-recommended'),
            
            html.Div(
                [dcc.Dropdown(
                    id='rec-dimreduction-type',
                    options=[{'label': 'Sepectral Embedding', 'value': 'spectral'},
                             {'label': 'TSNE', 'value': 'tsne'},
                             {'label': 'UMAP', 'value': 'umap'},
                             {'label': 'PCA', 'value': 'pca'}],
                    placeholder='Select a dimensionality reduction technique',
                    value='spectral'
                )], style={'width': '20%'}
            ),
            html.Div(
                [dcc.Dropdown(
                    id='rec-dimreduction-dim',
                    options=[{'label': '2D', 'value': 2},
                             {'label': '3D', 'value': 3}],
                    placeholder='Select the reduced number of dimensions',
                    value='2'
                )], style={'width': '20%'}
            ),
            dbc.Container([
                dcc.Loading(children=[dcc.Graph(id='rec-dimreduction-graph1', responsive='auto')], fullscreen=False, type='dot',
                            color="#119DFF")

            ]),
        ]),


        dcc.Tab(label='Dimensionality Reduction', value='tab-4', children=[

            html.H2(children='Projections of active NBA players into 2D', className="display-3"),

            html.Div(
                [dcc.Dropdown(
                    id='dimreduction-type',
                    options=[{'label': 'Sepectral Embedding', 'value': 'spectral'},
                             {'label': 'TSNE', 'value': 'tsne'},
                             {'label': 'UMAP', 'value': 'umap'},
                             {'label': 'PCA', 'value': 'pca'}],
                    placeholder='Select a dimensionality reduction technique',
                    value='spectral'
                )], style={'width': '20%'}
            ),
            html.Div(
                [dcc.Dropdown(
                    id='dimreduction-dim',
                    options=[{'label': '2D', 'value': 2},
                             {'label': '3D', 'value': 3}],
                    placeholder='Select the reduced number of dimensions',
                    value='2'
                )], style={'width': '20%'}
            ),
            html.Div(
                [dcc.Dropdown(
                    id='playerselect-dropdown_dimreduction',
                    options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                            enumerate(player_selector.label.unique())],
                    placeholder='Select a Player',
                    value= 2544
                )], style={'width': '20%'}
            ),
            dbc.Container([

                dcc.Loading(id='rec-loading', children=[dcc.Graph(id='dimreduction-graph1', responsive='auto')], fullscreen=False, type='dot',
                            color="#119DFF")

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
    Output('hotzone-graph', 'figure'),
    [Input('playerselect-dropdown', 'value')])
def hotzone_graph(value):
    shots = hotzone(value)

    #import plotly.figure_factory as ff
    #fig = ff.create_hexbin_mapbox(
    #    data_frame=shots, lat="LOC_X", lon="LOC_Y",
    #    nx_hexagon=100, opacity=0.9, labels={"color": "SHOT_MADE_FLAG"})
    #fig.update_layout(mapbox_style = "white-bg", margin=dict(b=0, t=0, l=0, r=0), template='simple_white')
    shots = shots[shots['SHOT_MADE_FLAG'] == 1]
    fig = px.density_heatmap(shots, x="LOC_X", y="LOC_Y", range_x = [-275, 275], range_y = [-20, 400], nbinsx=20, nbinsy=20, color_continuous_scale="Viridis")
    #fig = px.scatter(shots, x="LOC_X", y="LOC_Y", color = 'SHOT_MADE_FLAG', title= "Hot zone",
    #                width=1200, height=1000)
    fig.update_layout(transition_duration=500, template='simple_white')
    return fig
# or: https://plotly.com/python/hexbin-mapbox/

@app.callback(
    Output('dimreduction-graph1', 'figure'),
    [Input('dimreduction-type', 'value'), Input('dimreduction-dim', 'value'), Input('playerselect-dropdown_dimreduction', 'value')])
def get_emb(dim_type, dim, player):
    stats_agg, stats_agg_notTransformed = recommmendation_engine.aggregate_data(players_stats, w = [7/10, 2/10, 1/10]) 
    players_stats_emb, _, positions, data_names, player_stats = recommmendation_engine.embeddings(dim_type, stats_agg, stats_agg_notTransformed, int(dim)) # player
    name_emb = {'spectral': 'Sepectral Embedding', 'tsne': 'TSNE', 'umap': 'UMAP', 'pca': 'PCA'}
#    player_stats.positions = positions
#    player_stats.head()

    ind_player = list(player_data.index[player_data['id'] == player])[0] ### Implement selection of players!!
    labels = copy.deepcopy(list(positions))
    labels[ind_player] = list(player_data[player_data['id'] == player]['player_names'])[0]

    if int(dim) == 2:
        fig = px.scatter(players_stats_emb, x="embedding_1", y="embedding_2", color = labels, hover_name = data_names, 
                        color_discrete_sequence=["red", "green", "blue", "yellow"],
                        hover_data={'embedding_1':False, 
                                    'embedding_2':False, 
                                    'Position': positions,
                                    'Age': player_stats['PLAYER_AGE'],
                                    'Points': player_stats['PTS'],
                                    '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                                    'Assists': (':.3f', player_stats['AST']),
                                    'Rebounds': (':.3f', player_stats['REB'])
                                    },
                        labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2"}, title=f"{name_emb[str(dim_type)]} representation of NBA players")
        fig.update_layout(transition_duration=500, template='simple_white')
        return fig
    
    if int(dim) == 3:
        fig = px.scatter_3d(players_stats_emb, x="embedding_1", y="embedding_2", z = "embedding_3", color = labels, hover_name = data_names, 
                        color_discrete_sequence=["red", "green", "blue", "yellow"],
                        hover_data={'embedding_1':False, 
                                    'embedding_2':False, 
                                    'embedding_3':False, 
                                    'Position': positions,
                                    'Age': player_stats['PLAYER_AGE'],
                                    'Points': player_stats['PTS'],
                                    '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                                    'Assists': (':.3f', player_stats['AST']),
                                    'Rebounds': (':.3f', player_stats['REB'])
                                    },
                        labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2", "embedding_3": "Embedding Dimension 2"}, title=f"{name_emb[str(dim_type)]} representation of NBA players")
        fig.update_layout(transition_duration=500, template='simple_white')
        return fig


##########################################
@app.callback(
    [Output('playerRep-image_1', 'src'),
     Output('playerRep-str_1', 'children')],
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(team):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[3]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})"

@app.callback(
    [Output('playerRep-image_2', 'src'),
     Output('playerRep-str_2', 'children')],
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(team):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[4]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})"

@app.callback(
    [Output('playerRep-image_3', 'src'),
     Output('playerRep-str_3', 'children')],
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(team):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[0]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})"

@app.callback(
    [Output('playerRep-image_4', 'src'),
     Output('playerRep-str_4', 'children')],
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(team):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[1]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})"

@app.callback(
    [Output('playerRep-image_5', 'src'),
     Output('playerRep-str_5', 'children')],
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(team):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[2]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})"

##########################################

@app.callback(
    Output('teamRep-image', 'src'),
    [Input('teamRec-select-dropdown', 'value')])
def update_image_repTeam(value):
    return f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{value.lower()}.png"

@app.callback(
    [Output('teamRec-player-dropdown', 'children'),
    #Output('playerRec-table', 'data'),
    Output('playerRec-table', 'children'),
    Output('players-recommended', 'data'),
    Output('btn_1', 'n_clicks'), Output('btn_2', 'n_clicks'), Output('btn_3', 'n_clicks'), Output('btn_4', 'n_clicks'), Output('btn_5', 'n_clicks'), Output('pos_img', 'data')],
    [Input('teamRec-select-dropdown', 'value'), State('recommendation-type', 'value'), State("checklist-columns", "value"), State('recommendation-distance', 'value'),
     Input('btn_1', 'n_clicks'), Input('btn_2', 'n_clicks'), Input('btn_3', 'n_clicks'), Input('btn_4', 'n_clicks'), Input('btn_5', 'n_clicks')],
     State('weight1', 'value'), State('weight2', 'value'), State('weight3', 'value')
     )
def selected_player(team, rec_type, cols, dist_m, b1, b2, b3, b4, b5, w1, w2, w3):  
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    weights = [w1/100, w2/100, w3/100] #[7/10, 2/10, 1/10]

    if w1 + w2 + w3 != 100: # add??
        print('Error:', weights)

    if b1 is not None:
        rep_player = list(players_team.keys())[3]
        b1 = None
        pos = 1
    elif b2 is not None:
        rep_player = list(players_team.keys())[4]
        b2 = None
        pos = 2
    elif b3 is not None:
        rep_player = list(players_team.keys())[0]
        b3 = None
        pos = 3
    elif b4 is not None:
        rep_player = list(players_team.keys())[1]
        b4 = None
        pos = 4
    elif b5 is not None:
        rep_player = list(players_team.keys())[2]  
        b5 = None
        pos = 5
    else:
        rep_player = list(players_team.keys())[3] 
        pos = 1
        

    if 'PLAYER_AGE' in cols:
        sel_col = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE' 'GP', 'GS', 'MIN']
        cols.remove('PLAYER_AGE')
    else:
        sel_col = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'GS', 'MIN']
    stats_agg, _ = recommmendation_engine.aggregate_data(players_stats, w = weights, cols = sel_col+cols)

    #data_emb, emb, _, _, _ = recommmendation_engine.embeddings('umap', stats_agg, stats_agg_notTransformed)
    sample_recommendation = recommmendation_engine.RecommendationEngine(stats_agg, rep_player, rec_type, distance_measure = dist_m, w = weights, cols_sel = sel_col+cols) # 'Similar'
    r, result_table = sample_recommendation.recommend()
    result_table.distance = result_table.distance.round(2)
    result_table['id'] = result_table.index ##
    columns = [{"name": i, "id": i} for i in result_table.columns]
    data = result_table.to_dict('records')

    dt = dash_table.DataTable(
            data = data,
            columns=[{'name': i, 'id': i} for i in result_table.columns if i != 'id'],
            style_data_conditional=highlight_max_col(result_table),
            sort_action="native"
        )

    players_plot = list(result_table['player'])
    #players_plot.insert(0, rep_player)

    #print(result_table)
    #print(players_plot)
    #print()

    return r, dt, players_plot, b1, b2, b3, b4, b5, pos # data, columns

#stats_agg, stats_agg_notTransformed = aggregate_data(players_stats, [7/10, 2/10, 1/10], ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT'])

#@app.callback(
#    Output('teamRec-player-name', 'children'),
#    Input('teamRec-player-dropdown', 'data'))
#def print_recommended_player(value):
#    return f"{value} was recommended."

@app.callback(
    [Output('playerRec-image_1', 'src'),Output('playerRec-image_2', 'src'),Output('playerRec-image_3', 'src'),Output('playerRec-image_4', 'src'),Output('playerRec-image_5', 'src'),
     Output('playerRec-caption_1', 'children'),Output('playerRec-caption_2', 'children'),Output('playerRec-caption_3', 'children'),Output('playerRec-caption_4', 'children'),Output('playerRec-caption_5', 'children')],
    [Input('teamRec-player-dropdown', 'children'),
     Input('pos_img', 'data')])
def update_image_recPlayer(player, pos):
    player_id = list(player_data[player_data['player_names'] == player]['id'])[0]
    player_team = list(player_data[player_data['player_names'] == player]['team'])[0]
    player_pos = list(player_data[player_data['player_names'] == player]['position'])[0]

    pos = int(pos)
    img1 = ''
    img2 = ''
    img3 = ''
    img4 = ''
    img5 = ''

    cap1 = ''
    cap2 = ''
    cap3 = ''
    cap4 = ''
    cap5 = ''

    output_str = f'{player} ({player_team}, {player_pos})' 

    if pos == 1:
        img1 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap1 = output_str
    elif pos == 2:
        img2 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap2 = output_str
    elif pos == 3:
        img3 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap3 = output_str
    elif pos == 4:
        img4 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap4 = output_str
    else:
        img5 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap5 = output_str

    return img1, img2, img3, img4, img5, cap1, cap2, cap3, cap4, cap5


@app.callback(
    Output('teamselect-capspace-graph', 'figure'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return recommmendation_engine.visualize_capspace_team_plotly(value)

@app.callback(
    Output("checklist-columns", "value"),
    [Input("checklist-allColumns", "value")],
    [State("checklist-columns", "options")],
)
def select_all_none(all_selected, options):
    if len(all_selected) == 0:
        return []
    else:
        attributes = []
        if 'All' in all_selected:
            for option in options:
                attributes.append(option["value"])
        if 'Off' in all_selected:
            off_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'AST', 'PTS']
            for option in options:
                if option['value'] in off_cols:
                    attributes.append(option["value"])
            #attributes = [option["value"] for option in options if option['value'] in off_cols]
        if 'Def' in all_selected:
            def_cols = ['OREB', 'DREB', 'REB', 'STL', 'BLK', 'TOV', 'PF']
            for option in options:
                if option['value'] in def_cols:
                    attributes.append(option["value"])
            #attributes = attributes = [option["value"] for option in options if option['value'] in def_cols]
    
        return list(set(attributes))


@app.callback(
    Output('rec-dimreduction-graph1', 'figure'),
    [Input('rec-dimreduction-type', 'value'), Input('rec-dimreduction-dim', 'value'), Input('players-recommended', 'data')])
def get_emb(dim_type, dim, players):
    stats_agg, stats_agg_notTransformed = recommmendation_engine.aggregate_data(players_stats, w = [7/10, 2/10, 1/10]) 
    players_stats_emb, _, positions, data_names, player_stats = recommmendation_engine.embeddings(dim_type, stats_agg, stats_agg_notTransformed, int(dim)) # player
    name_emb = {'spectral': 'Sepectral Embedding', 'tsne': 'TSNE', 'umap': 'UMAP', 'pca': 'PCA'}

    labels = copy.deepcopy(list(positions))
    for i, player in enumerate(players):
        ind_player = list(player_data.index[player_data['player_names'] == player])[0]
        if i == 0:
            labels[ind_player] = player
        else:
            labels[ind_player] = 'Recommendations'


    if int(dim) == 2:
        fig = px.scatter(players_stats_emb, x="embedding_1", y="embedding_2", color = labels, hover_name = data_names, 
                        color_discrete_sequence=["red", "green", "blue", "yellow", "black"],
                        hover_data={'embedding_1':False, 
                                    'embedding_2':False, 
                                    'Position': positions,
                                    'Age': player_stats['PLAYER_AGE'],
                                    'Points': player_stats['PTS'],
                                    '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                                    'Assists': (':.3f', player_stats['AST']),
                                    'Rebounds': (':.3f', player_stats['REB'])
                                    },
                        labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2"}, title=f"{name_emb[str(dim_type)]} representation of NBA players")
        fig.update_layout(transition_duration=500, template='simple_white')
        return fig
    
    if int(dim) == 3:
        fig = px.scatter_3d(players_stats_emb, x="embedding_1", y="embedding_2", z = "embedding_3", color = labels, hover_name = data_names, 
                        color_discrete_sequence=["red", "green", "blue", "yellow", "black"],
                        hover_data={'embedding_1':False, 
                                    'embedding_2':False, 
                                    'embedding_3':False, 
                                    'Position': positions,
                                    'Age': player_stats['PLAYER_AGE'],
                                    'Points': player_stats['PTS'],
                                    '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                                    'Assists': (':.3f', player_stats['AST']),
                                    'Rebounds': (':.3f', player_stats['REB'])
                                    },
                        labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2", "embedding_3": "Embedding Dimension 2"}, title=f"{name_emb[str(dim_type)]} representation of NBA players")
        fig.update_layout(transition_duration=500, template='simple_white')
        return fig


if __name__ == '__main__':
    app.run_server(debug=True) # set equal to False

