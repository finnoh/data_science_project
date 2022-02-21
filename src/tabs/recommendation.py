import recommmendation_engine
from dash import dcc, html
import dash_bootstrap_components as dbc

player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()

'''


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

'''
############

text_offcanvas = """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. 
At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur 
sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. 
Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut
labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum 
dolor sit amet
"""

'''
offcanvas = html.Div(
    [
        dbc.Button("Information regarding the usage of the recommendation engineee", id="rec-infocanvas", n_clicks=0, style={"color": "#FEC700",
                   }),#color='danger'),
        dbc.Offcanvas([html.Plaintext(text_offcanvas)],
            backdrop=True,
            scrollable=True,
            id="infocanvas",
            is_open=False,
            placement='top'
        ),
    ], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
)
'''


offcanvas = html.Div([html.Div([dbc.Button("Information regarding the usage of the recommendation engine", id="rec-infocanvas", n_clicks=0, style={'background-color': '#17408b', 'color': 'white', 'margin-bottom': '10px'})], 
                                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                      html.Div([dbc.Offcanvas([html.P(text_offcanvas)],
                                backdrop=True,
                                scrollable=True,
                                id="infocanvas",
                                title = 'Title',
                                is_open=False,
                                placement='top')], 
                                style = {'width': '100%', 'display': 'flex', 'justify-content': 'top'})          
                     ])


recommendation_type = html.Div([dcc.RadioItems( # Dropdown
                                            id='recommendation-type',
                                            options=[{'label': ' Similar player', 'value': 'Similar'},
                                                    {'label': ' Complementary player', 'value': 'Fit'}],
                                        # placeholder='Select a recommendation technique',
                                            value='Similar'
                                        )], style={'width': '50%'}
                                    )

recommendation_distance = html.Div([dcc.RadioItems(
                                            id='recommendation-distance',
                                            options=[{'label': ' L1 distance', 'value': 'L1'},
                                                     {'label': ' L2 distance', 'value': 'L2'}],
                                            #placeholder='Select a distance measure',
                                            value='L2'
                                        )], style={'width': '30%'}
                                    )
# html.Plaintext("Season weights (in percent) for 2018/2019, 2019/2020, 2020/2021"),

recommendation_weights =  html.Div([dcc.Input(id="weight1", autoComplete="off", type="number", placeholder = 'Season 2020/21', min=0, max=100, value=70, style={'width':'20%'}),
                                    dcc.Input(id="weight2", autoComplete="off", type="number", placeholder = 'Season 2019/20', min=0, max=100, value=20, style={'width':'20%'}),
                                    dcc.Input(id="weight3", autoComplete="off", type="number", placeholder = 'Season 2018/19', min=0, max=100, value=10, style={'width':'20%'}),
                                # html.Button('Calculate', id='btn_weights')
                                  ])

recommendation_checklist = html.Div([dcc.Checklist(
                            id="checklist-allColumns",
                            options=[{"label": " All attributes", "value": "All"}, {"label": " Offensive attributes", "value": "Off"}, {"label": " Defensive attributes", "value": "Def"}],
                            value=["All"],
                            labelStyle={"display": "inline-block", 'cursor': 'pointer', 'margin-right':'30px'},
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
                            labelStyle={"display": "inline-block", 'cursor': 'pointer', 'margin-right':'30px'},
                        ),
                    ])

recommendation_checklist_groups = dbc.Row([dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-all",
                                                options=[
                                                    {"label": " All attributes", "value": "All"}
                                                ],
                                                value=['All'],
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
                                                options=[
                                                 {"label": " Player Age", "value": "PLAYER_AGE"},
                                                 {"label": " Weight", "value": "WEIGHT"},
                                                 {"label": " Height", "value": "HEIGHT"},
                                                 {"label": " Experience", "value": "EXPERIENCE"},
                                                 {"label": " Player Score", "value": "Score"},
                                                 {"label": " Experience (2k)", "value": "Athleticism"},
                                                 {"label": " Playmaking (2k)", "value": "Playmaking"},
                                            #     {"label": "Games Play", "value": "GP"},
                                            #     {"label": "GS", "value": "GS"},
                                            #     {"label": "MIN", "value": "MIN"},
                                                ],
                                                value=['PLAYER_AGE', 'WEIGHT', 'HEIGHT', 'EXPERIENCE', 'Score', 'Athleticism', 'Playmaking'],
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
                                        value=['FGM', 'FG3_PCT', 'FGA', 'FG3M', 'FTM', 'FG3A', 'FG_PCT'],
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
                                        value=['FT_PCT', 'TOV', 'FTA', 'AST','PTS', 'OREB'],
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
                                        value=['DREB', 'PF', 'STL', 'BLK', 'REB'],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}
                                    ))], width=3)
                    ])

#Index(['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION',
#       'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
#       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
#       'BLK', 'TOV', 'PF', 'PTS'],

#team_sel_image = html.Div([html.Div([html.Img(id='teamRep-image')])], style={'height': '5%', 'width': '5%'})
#team_sel_dropdown = html.Div([dcc.Dropdown(
#                    id='teamRec-select-dropdown',
#                    options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in enumerate(team_data['abbreviation'])],
#                    placeholder='Select a Team',
#                    value='LAL'
#
#                )], style={'width': '20%'}
#            )

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

alert_features = html.Div([dbc.Alert([html.H4("You selected the 'Complementary Player' option in conjunction with one of the following attributes: 'Playmaking', 'Athleticism', 'Score'", className="alert-features-h"),
                                     html.P("These features can only be used with the 'Similar Player' option because they require unavailable historic data. Please be change either your set of selected attributes or the recommendation strategy.")], color="danger", dismissable = False, is_open=False, id="alert-features"),
                          dcc.Store(id = 'alert-features-triggered')])


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


recommended_vis = html.Div([html.Div([dcc.Dropdown(
                            id='rec-dimreduction-type',
                            options=[{'label': 'Sepectral Embedding', 'value': 'spectral'},
                                    {'label': 'TSNE', 'value': 'tsne'},
                                    {'label': 'UMAP', 'value': 'umap'},
                                    {'label': 'PCA', 'value': 'pca'}],
                            placeholder='Select a dimensionality reduction technique',
                            value='spectral'
                            )], style={'width': '20%'}),
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
                            #dcc.Loading(children=[dcc.Graph(id='rec-dimreduction-graph1', responsive='auto')], fullscreen=False, type='dot',
                            dcc.Loading(children=[dcc.Graph(id='rec-dimreduction-graph1', responsive='auto', clear_on_unhover=True)], fullscreen=False, type='dot',
                                    color="#119DFF"),
                            dcc.Tooltip(id="graph-rec-tooltip"),
                            #dcc.Store(id = "rec-dimreduction-data")
                                    ])
                            ])


recommendation_outputs = html.Div(
                            dbc.Accordion(
                                [ 
                                    dbc.AccordionItem(
                                        html.Div(id='playerRec-table'), title="Details on recommendation result"
                                    ),
                                     dbc.AccordionItem(
                                       #html.Div(id='playerselect-output-container-wiki'), title="Season prediction based on trade" # SEASON PREDICTION
                                       html.Div('PLACEHOLDER'), title="Season prediction based on trade"
                                    ),
                                    dbc.AccordionItem(
                                        recommended_vis, title="Embeddings of recommended players"
                                    )
                                ],
                                flush=False,
                                start_collapsed=True
                            )
                        )

recommendation_tab = html.Div([
                        html.H2(children='Trade Recommendation Engine', className="display-3"),
                        dbc.Row([dbc.Col(team_card_rec, width=3),
                                 dbc.Col([offcanvas,
                                          dbc.Row([dbc.Col(recommendation_type),
                                                   dbc.Col(recommendation_distance)], style = {'margin-bottom': '5px'}),
                                          dbc.Row([dbc.Col(html.Div("Season weights (in %) for 2018/19, 2019/20, 2020/21:", style = {"fontWeight": "bold"})),
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
                        dcc.Loading(children = [html.Div(id='teamRec-player-dropdown', style = {'display': 'none'})], fullscreen=False, type='dot', color="#119DFF"),
                        dcc.Store(id='players-recommended'),
                        dcc.Store(id = 'playerRec-stats'), 
                        recommendation_outputs
                        #dcc.Loading(children = [html.Div(id='players-recommended', style = {'display': 'none'})], fullscreen=False, type='dot', color="#119DFF"),
                    ])



def highlight_max_col(df):
    df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
    colors = ['#BDF6BB', '#9CF599', '#28B463', '#FBA898', '#F8775E', '#FF0000'] #https://htmlcolorcodes.com/
    styles = [{"if": {"row_index": 0},
               "fontWeight": "bold"}]
    for col in df_numeric_columns.keys():
        rep_value = list(df[col])[0]
        comparison_values = [rep_value+perc*rep_value for perc in [0.00001, 0.2, 0.5, -0.0001, -0.2, -0.5]]
        if col in ['distance', 'Luxury Tax', 'PF', 'TOV', 'Priced', 'Age', 'Weight']:
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