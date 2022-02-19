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


############


recommendation_tab = dcc.Tab(label='Recommendation engine', value='tab-3', children=[
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
          #  html.Div([html.Div(
          #      [html.Img(id='playerRec-image')]
          #  )], style={'width': '49%'}),

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
        ])



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

'''
dbc.Card(
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
    '''