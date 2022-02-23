from src.utils_dash import _player_selector, _team_selector, _link_team_website, _team_full_name, _get_team_id
from dash import dcc, html
import dash_bootstrap_components as dbc

team_selector = _team_selector()

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
            placement='end'
        ),
    ]
)

col_teamname = dbc.Col(html.Div(
    [html.H2(id='teamselect-output-container',
             className="display-3", style={'margin': 'auto', 'width': '100%', 'display': 'inline-block'})]),
    md=9)

col_logo = dbc.Col(html.Div(
    [html.Img(id='teamselect-image', style={'margin': 'auto', 'width': '120%', 'display': 'inline-block'})]), md=3)

starplayer = dbc.Alert(
    [
        html.H4(["Starplayer"], className="alert-heading"),
        html.P(id='teamselect-mvp-descr'),
        #html.Hr(),
        #html.P(
        #    id='teamselect-mvp-name',
        #    className="mb-0",
        #),
    ]
)


team_info = html.Div([dbc.Row([html.Div([html.Div('Nickname:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-1')], style = {'display':'flex', 'margin-top': '10px'})]),
                      dbc.Row([html.Div([html.Div('Year founded:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-2')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('Location:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-3')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('NBA championships:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-4')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('Arena:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-5')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('Owner:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-6')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('General Manager:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-7')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('Head Coach:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-8')], style = {'display':'flex', 'margin-top': '5px'})]),
                      dbc.Row([html.Div([html.Div('D-League Team:', style = {'margin-right': '10px', "fontWeight": "bold"}),
                                         html.Div(id='team_info-9')], style = {'display':'flex', 'margin-top': '5px', 'margin-bottom': '10px'})]),
                    ])

right_part = dbc.Col([html.Div(
    [html.Img(id='teamselect-mvp-image', style={'margin': 'auto', 'display': 'inline-block', 'margin-left': '30%'})]),
    starplayer,
    dbc.Container([dcc.Graph(id='teamselect-capspace-graph',  style={"height" : "40vh"})],
                  )],
    md=6)


left_jumbotron = dbc.Col([dbc.Row([col_teamname, col_logo], className="align-items-md-stretch"),
                          html.Hr(className="my-2"),
                          dcc.Dropdown(
                              id='teamselect-dropdown',
                              options=[{'label': team,
                                        'value': team_selector.iloc[
                                            i, 1]} for i, team in
                                       enumerate(
                                           team_selector.label.unique())],
                              value='ATL'), 
                          team_info, #html.H6(id = 'team-info'),
                          offcanvas,
                          #html.Hr(className="my-2")
                          ], md=6, className="h-100 p-5 bg-light border rounded-3")

                        
                # statistics of the last season (2020/2021)


checklist_cols_groups = dbc.Row([dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-team-all",
                                                options=[
                                                    {"label": " All attributes", "value": "All"}
                                                ],
                                                #value=['All'],
                                                labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px', 'font-weight': 'bold'},
                                            ))], width=3),
                                           # dbc.Col([html.Div(dcc.Checklist(
                                           #     id="checklist-team-salary",
                                           #     options=[
                                           #         {"label": " Offense", "value": "Off"},
                                           #     ],
                                           #     value=[],
                                           #     labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px', 'font-weight': 'bold'}
                                           #     ))], width=6),
                                            dbc.Col([html.Div(dcc.Checklist(
                                                id="checklist-team-stats",
                                                options=[
                                                    {"label": " Statistics from 2020/2021", "value": "AllStats"},
                                                ],
                                                value=[],
                                                labelStyle={'cursor': 'pointer', 'margin-right':'30px', 'font-weight': 'bold'}
                                            ))], width=9)
                            ])

checklist_cols = dbc.Row([dbc.Col([html.Div(dcc.Checklist(
                                                id="team-checklist-general",
                                                options = [
                                                    {"label": " Age", "value": "PLAYER_AGE"},
                                                    {"label": " Weight", "value":'Weight'},
                                                    {"label": " Height", "value":'Height'},
                                                    {"label": " Experience", "value":'Experience'},
                                                    {"label": " Salary 21/22", "value":'Salary 21/22'},
                                                    {"label": " Salary 22/23", "value":'Salary 22/23'},
                                                    {"label": " Salary 23/24", "value":'Salary 23/24'},
                                                    {"label": " Salary 24/25", "value":'Salary 24/25'}
                                                ],
                                                #value=['PLAYER_AGE', 'WEIGHT', 'HEIGHT', 'EXPERIENCE', 'Score', 'Athleticism', 'Playmaking'],
                                                labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'},
                                            ))], width=3),
                                    dbc.Col([html.Div(dcc.Checklist(
                                        id="team-checklist-stats1",
                                        options=[
                                            {"label": " Games Played", "value": "GP"},
                                            {"label": " Games Started", "value": "GS"},
                                            {"label": " Minutes Played", "value": "MIN"},
                                            {"label": " Field Goals Made", "value": "FGM"},
                                            {"label": " Field Goals Attempted", "value": "FGA"},
                                            {"label": " Field Goals Percentage", "value": "FG_PCT"},
                                            {"label": " 3-point Shots Made", "value": "FG3M"},
                                        ],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}))], width=3),
                                    dbc.Col([html.Div(dcc.Checklist(
                                        id="team-checklist-stats2",
                                        options=[
                                            {"label": " 3-point Shots Attempted", "value": "FG3A"},
                                            {"label": " 3-point Percentage", "value": "FG3_PCT"},
                                            {"label": " Free Throws Made", "value": "FTM"},
                                            {"label": " Free Throws Attempted", "value": "FTA"},
                                            {"label": " Free Throws Percentage", "value": "FT_PCT"},
                                            {"label": " Offensive Rebounds", "value": "OREB"},
                                            {"label": " Defensive Rebounds", "value": "DREB"},
                                        ],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}))], width=3),
                                    dbc.Col([html.Div(dcc.Checklist(
                                        id="team-checklist-stats3",
                                        options=[
                                            {"label": " Rebounds", "value": "REB"},
                                            {"label": " Assists", "value": "AST"},
                                            {"label": " Steals", "value": "STL"},
                                            {"label": " Blocks", "value": "BLK"},
                                            {"label": " Turnovers", "value": "TOV"},
                                            {"label": " Personal Fouls", "value": "PF"},
                                            {"label": " Points", "value": "PTS"},
                                        ],
                                        labelStyle={"display": "block", 'cursor': 'pointer', 'margin-right':'30px'}
                                    ))], width=3)
                    ])

data_table = html.Div([html.H2("Current roster"),
                        checklist_cols_groups,
                        checklist_cols,
                        html.Div(id='team-table', style = {'margin-left': '10px', 'margin-top': '15px'})], 
                        style = {'margin-left': '10px', 'margin-top': '1px'})

jumbotron = html.Div([dbc.Row(
    [left_jumbotron, right_part], className="align-items-md-stretch"),
    data_table]
)


def highlight_max_col(df):
    df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
    colors = ['#28B463', '#FF0000'] #https://htmlcolorcodes.com/
    styles = []

    for col in df_numeric_columns.keys():
        if col in ['Age', 'TOV', 'PF', 'Salary 21/22', 'Salary 22/23', 'Salary 23/24', 'Salary 24/25']:
            styles.append({'if': {
                                'filter_query': '{{{col}}} = {value}'.format(col = col, value = df[col].max()),
                                'column_id': col
                            },
                            'backgroundColor': colors[1],
                            'color': 'white'})
            styles.append({'if': {
                                'filter_query': '{{{col}}} = {value}'.format(col = col, value = df[col].min()),
                                'column_id': col
                            },
                            'backgroundColor': colors[0],
                            'color': 'white'})
        else:
            styles.append({'if': {
                                'filter_query': '{{{col}}} = {value}'.format(col = col, value = df[col].max()),
                                'column_id': col
                            },
                            'backgroundColor': colors[0],
                            'color': 'white'})
            styles.append({'if': {
                                'filter_query': '{{{col}}} = {value}'.format(col = col, value = df[col].min()),
                                'column_id': col
                            },
                            'backgroundColor': colors[1],
                            'color': 'white'})

    return styles