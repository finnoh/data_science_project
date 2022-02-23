import recommmendation_engine
from dash import dcc, html
import dash_bootstrap_components as dbc

player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()


text_offcanvas = """
This engine automatically finds the best trade recommendation for a given input player whereas the players presented per team are their current starting five heading into the season.
Essentially, this model uses the selected attributes (including our own player score) to find the most similar or complementary player in the high-dimensional space.

**1. Similar Player:** in this option, the closest player to the input player in the high-dimensional space is selected.

**2. Complementary Player:** here, the best four teams from the past four seasons are used as role models to determine the optimal player given the remaining four players of the starting five.

To adjust this decision process, several parameters can be selected which have an influence on the recommendation.
Apart from the typical L1- and L2-distance measures, the weights for the past three seasons can be selected based on which the attributes are first aggregated and then, normalized to zero mean and unit veriance.
While the model uses the normalized values for finding the optimal recommmendation, the output table below features the weighted values for 36 minutes (typical game time for a starter)

As a further result of the trade, the season is predicted with the effect of this trade, similar to the *Season Prediction Tab*.
Finally, the selected attributes are used to compute different lower-dimensional embeddings which are then visualized in a 2D-/ 3D-plot.

One recommendation using the *similar* option takes roughly 30 seconds while the *complementary* option requires roughly one minute.
"""



offcanvas = html.Div([html.Div([dbc.Button("Information regarding the usage of the recommendation engine", id="rec-infocanvas", n_clicks=0, style={'background-color': '#17408b', 'color': 'white', 'margin-bottom': '10px'})], 
                                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                      html.Div([dbc.Offcanvas([dcc.Markdown(text_offcanvas)],
                                backdrop=True,
                                scrollable=True,
                                id="infocanvas",
                                title = 'Information regarding Recommendation Engine',
                                is_open=False,
                                placement='end')], 
                                style = {'width': '100%', 'display': 'flex', 'justify-content': 'top'})          
                     ])


recommendation_type = html.Div([dcc.RadioItems(
                                            id='recommendation-type',
                                            options=[{'label': ' Similar player', 'value': 'Similar'},
                                                     {'label': ' Complementary player', 'value': 'Fit'}],
                                            value='Similar'
                                        )], style={'width': '30%'}
                                    )

recommendation_distance = html.Div([dcc.RadioItems(
                                            id='recommendation-distance',
                                            options=[{'label': ' L1 distance', 'value': 'L1'},
                                                     {'label': ' L2 distance', 'value': 'L2'}],
                                            value='L2'
                                        )], style={'width': '30%'}
                                    )


recommendation_weights =  html.Div([dcc.Input(id="weight1", autoComplete="off", type="number", placeholder = '2020/21', min=0, max=100, value=70, style={'width':'20%'}),
                                    dcc.Input(id="weight2", autoComplete="off", type="number", placeholder = '2019/20', min=0, max=100, value=20, style={'width':'20%'}),
                                    dcc.Input(id="weight3", autoComplete="off", type="number", placeholder = '2018/19', min=0, max=100, value=10, style={'width':'20%'}),
                                  ])

recommendation_checklist = html.Div([dcc.Checklist(
                            id="checklist-allColumns",
                            options=[{"label": " All attributes", "value": "All"}, {"label": " Offensive attributes", "value": "Off"}, {"label": " Defensive attributes", "value": "Def"}],
                          #  value=["All"],
                            labelStyle={"display": "inline-block", 'cursor': 'pointer', 'margin-right':'30px'},
                        ),
                        dcc.Checklist(
                            id="checklist-columns",
                            options=[
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
                            labelStyle={"display": "inline-block", 'cursor': 'pointer', 'margin-right':'30px'},
                        ),
                    ])

recommendation_checklist_groups = dbc.Row([dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-all",
                                                options=[
                                                    {"label": " All attributes", "value": "All"}
                                                ],
                                                #value=['All'],
                                                labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px', 'font-weight': 'bold'},
                                            ))], width=3),
                                            dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-off",
                                                options=[
                                                    {"label": " Offense", "value": "Off"},
                                                ],
                                                value=[],
                                                labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px', 'font-weight': 'bold'}
                                                ))], width=6),
                                            dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-def",
                                                options=[
                                                    {"label": " Defense", "value": "Def"},
                                                ],
                                                value=[],
                                                labelStyle={'cursor': 'pointer', 'margin-right':'30px', 'font-weight': 'bold'}
                                            ))], width=3)
                            ])

recommendation_checklist_details = dbc.Row([dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-all-details",
                                                #value=['PLAYER_AGE', 'WEIGHT', 'HEIGHT', 'EXPERIENCE', 'Score', 'Athleticism', 'Playmaking'],
                                                labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'},
                                            ))], width=3),
                                    dbc.Col([html.Div(dcc.Checklist(
                                        id="checklist-off-details",
                                        options=[
                                            {"label": " Field Goals Made", "value": "FGM"},
                                            {"label": " Field Goals Attempted", "value": "FGA"},
                                            {"label": " Field Goals Percentage", "value": "FG_PCT"},
                                            {"label": " 3-point Shots Made", "value": "FG3M"},
                                            {"label": " 3-point Shots Attempted", "value": "FG3A"},
                                            {"label": " 3-point Percentage", "value": "FG3_PCT"},
                                            {"label": " Free Throws Made", "value": "FTM"}
                                        ],
                                        #value=['FGM', 'FG3_PCT', 'FGA', 'FG3M', 'FTM', 'FG3A', 'FG_PCT'],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}))], width=3),
                                    dbc.Col([html.Div(dcc.Checklist(
                                        id="checklist-off2-details",
                                        options=[
                                            {"label": " Free Throws Attempted", "value": "FTA"},
                                            {"label": " Free Throws Percentage", "value": "FT_PCT"},
                                            {"label": " Offensive Rebounds", "value": "OREB"},
                                            {"label": " Assists", "value": "AST"},
                                            {"label": " Points", "value": "PTS"},
                                            {"label": " Turnovers", "value": "TOV"},
                                        ],
                                       # value=['FT_PCT', 'TOV', 'FTA', 'AST','PTS', 'OREB'],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}))], width=3),
                                    dbc.Col([html.Div(dcc.Checklist(
                                        id="checklist-def-details",
                                        options=[
                                            {"label": " Defensive Rebounds", "value": "DREB"},
                                            {"label": " Total Rebounds", "value": "REB"},
                                            {"label": " Steals", "value": "STL"},
                                            {"label": " Blocks", "value": "BLK"},
                                            {"label": " Personal Fouls", "value": "PF"},
                                        ],
                                       # value=['DREB', 'PF', 'STL', 'BLK', 'REB'],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}
                                    ))], width=3)
                    ])



mincer = html.Div([dcc.Dropdown(options=[{'label': 'Ordinary Least Squares', 'value': 'ols'},
                                         {'label': 'Random Forest', 'value': 'rf'},
                                         {'label': 'Support Vector Machine', 'value': 'svr'}],
                                        value='ols', id="mincer-rec-dropdown"),
                    dcc.Store(id = 'mincer-output-rec')])


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
    

player_images = dbc.Row([dbc.Col(html.Div([html.Img(id='playerRep-image_1')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_2')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_3')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_4')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRep-image_5')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}))
                         ])

player_text = dbc.Row([dbc.Col(html.Div(id='playerRep-str_1', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                         dbc.Col(html.Div(id='playerRep-str_2', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                         dbc.Col(html.Div(id='playerRep-str_3', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                         dbc.Col(html.Div(id='playerRep-str_4', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                         dbc.Col(html.Div(id='playerRep-str_5', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),  
                         ])

player_overpriced = dbc.Row([dbc.Col(html.Div(id='playerRep-price_1', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                            dbc.Col(html.Div(id='playerRep-price_2', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                            dbc.Col(html.Div(id='playerRep-price_3', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                            dbc.Col(html.Div(id='playerRep-price_4', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                            dbc.Col(html.Div(id='playerRep-price_5', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),  
                         ]) 

alert_weights = html.Div([dbc.Alert([html.H4("You selected weights which do not sum up to 100%.", className="alert-heading"),
                                     html.P("Please be aware of that when interpreting the results.")], color="warning", dismissable = True, is_open=False, id="alert-weights"),
                          dcc.Store(id = 'alert-triggered')])

alert_features = html.Div([dbc.Alert([html.H4("You selected no attributes at all.", className="alert-features-h"),
                                     html.P("Please select at least one of the features from above such that an recommended trade can be computed.")], color="danger", dismissable = False, is_open=False, id="alert-features"),
                          dcc.Store(id = 'alert-features-triggered')])

alert_emb = html.Div([dbc.Alert([html.H4("You selected only one attribute.", className="alert-features-h"),
                                     html.P("Please select at least two features from above to visualize to visualize the NBA players according to those attributes.")], color="danger", dismissable = False, is_open=False, id="alert-emb"),
                          dcc.Store(id = 'alert-emb-triggered')])


buttons_trade = dbc.Row([dbc.Col(html.Div([html.Button('Trade!', id='btn_1')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Trade!', id='btn_2')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Trade!', id='btn_3')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Trade!', id='btn_4')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Button('Trade!', id='btn_5')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),  
                         ], style = {'margin-bottom': '10px'})

recommended_image = dbc.Row([dbc.Col(html.Div([html.Img(id='playerRec-image_1')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_2')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_3')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_4')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})),
                         dbc.Col(html.Div([html.Img(id='playerRec-image_5')], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}))
                         ])

recommended_text = dbc.Row([dbc.Col(html.Div(id='playerRec-caption_1', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='playerRec-caption_2', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='playerRec-caption_3', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='playerRec-caption_4', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='playerRec-caption_5', style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "fontWeight": "bold"})),  
                         ]) 


recommended_vis = html.Div([html.Div(
                                [dcc.Dropdown(
                                    id='rec-dimreduction-dim',
                                    options=[{'label': '2D', 'value': 2},
                                             {'label': '3D', 'value': 3}],
                                    placeholder='Select the reduced number of dimensions',
                                    value='2'
                                )], style={'width': '20%'}
                                ),
                            html.Div([dcc.Dropdown(
                            id='rec-dimreduction-type',
                            options=[{'label': 'Spectral Embedding', 'value': 'spectral'},
                                     {'label': 'TSNE', 'value': 'tsne'},
                                     {'label': 'UMAP', 'value': 'umap'},
                                     {'label': 'PCA', 'value': 'pca'}],
                            placeholder='Select a dimensionality reduction technique',
                            value='spectral'
                            )], style={'width': '20%'}),
                            
                            dbc.Container([
                            dcc.Loading(children=[dcc.Graph(id='rec-dimreduction-graph1', responsive='auto', clear_on_unhover=True)], fullscreen=False, type='dot',
                                    color="#119DFF"),
                            dcc.Tooltip(id="graph-rec-tooltip"),
                            #dcc.Store(id = "rec-dimreduction-data")
                                    ])
                            ])


recommendation_outputs = html.Div(
                            dcc.Tabs( value = 'tab 1', children = 
                                [ 
                                    dcc.Tab(label = "Details on recommendation result", value = 'tab 1', children = [html.Div(id='playerRec-table')]),
                                    dcc.Tab(label = "Season prediction based on trade", value = 'tab 2', children = [html.Div(dcc.Graph(id='s-prediction-output-graph_trade', clear_on_unhover=True))]),
                                    dcc.Tab(label = "Embeddings of recommended players", value = 'tab 3', children = [alert_emb, recommended_vis])
                              ],
                               # flush=False,
                               # start_collapsed=True
                            )
                        )

recommendation_tab = html.Div([
                        html.H2(children='Trade Recommendation Engine', className="display-3"),
                        dbc.Row([dbc.Col(team_card_rec, width=3),
                                 dbc.Col([offcanvas,
                                          dbc.Row([dbc.Col(recommendation_type),
                                                   dbc.Col(recommendation_distance)], style = {'margin-bottom': '5px'}),
                                          dbc.Row([dbc.Col(html.Div("Season weights (in %) for 2020/21, 2019/20, 2018/19:", style = {"fontWeight": "bold"})),
                                                   dbc.Col(recommendation_weights)],
                                                   style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '5px'}),
                                          recommendation_checklist_groups,
                                          recommendation_checklist_details, 
                                          dbc.Row([dbc.Col(html.Div("Estimation model for the Mincer salary evaluation:", style = {"fontWeight": "bold"})),
                                                   dbc.Col(mincer)],
                                                   style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '10px'})], 
                                            width=9)
                                 ]),

                        html.Div([player_images,
                                  player_text,
                                  player_overpriced,
                                  buttons_trade,
                                  alert_weights,
                                  alert_features,
                                  recommended_image,
                                  recommended_text,
                                  ]),
                        dcc.Store(id='pos_img'),
                        dcc.Loading(children=[html.Div(id='teamRec-player-dropdown', style = {'display': 'none'})], fullscreen=False, type='dot', color="#119DFF"),
                        dcc.Store(id='players-recommended'),
                        dcc.Store(id = 'playerRec-stats'), 
                        dcc.Store(id = 'rec-cols-sel'),
                        recommendation_outputs
                        #dcc.Loading(children = [html.Div(id='players-recommended', style = {'display': 'none'})], fullscreen=False, type='dot', color="#119DFF"),
                    ])



def highlight_max_col(df):
    df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
    colors = ['#BDF6BB', '#9CF599', '#28B463', '#FBA898', '#F8775E', '#FF0000'] #https://htmlcolorcodes.com/
    styles = [{"if": {"row_index": 0}, "fontWeight": "bold"}]

    for col in df_numeric_columns.keys():
        rep_value = list(df[col])[0]
        comparison_values = [rep_value+perc*rep_value for perc in [0.00001, 0.2, 0.5, -0.0001, -0.2, -0.5]]
        if col in ['Distance', 'Luxury Tax', 'PF', 'TOV', 'Priced', 'Age', 'Weight']:
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
