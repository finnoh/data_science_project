from dash import dcc, html
import dash_daq as daq
from dash import dash_table
from src.mincer import *
import dash_bootstrap_components as dbc
import recommmendation_engine
from src.utils_dash import _player_selector
from dash import dash_table

player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()
player_selector = _player_selector()

pred_player_images = dbc.Row([dbc.Col(html.Div(html.Img(id='pred-playerRep-image_1'),
                                               style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                      'justify-content': 'center'})),
                              dbc.Col(html.Div(html.Img(id='pred-playerRep-image_2'),
                                               style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                      'justify-content': 'center'})),
                              dbc.Col(html.Div(html.Img(id='pred-playerRep-image_3'),
                                               style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                      'justify-content': 'center'})),
                              dbc.Col(html.Div(html.Img(id='pred-playerRep-image_4'),
                                               style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                      'justify-content': 'center'})),
                              dbc.Col(html.Div(html.Img(id='pred-playerRep-image_5'),
                                               style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                      'justify-content': 'center'}))
                              ])

pred_player_text = dbc.Row([dbc.Col(html.Div(id='pred-playerRep-str_1',
                                             style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                    'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='pred-playerRep-str_2',
                                             style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                    'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='pred-playerRep-str_3',
                                             style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                    'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='pred-playerRep-str_4',
                                             style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                    'justify-content': 'center', "fontWeight": "bold"})),
                            dbc.Col(html.Div(id='pred-playerRep-str_5',
                                             style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                    'justify-content': 'center', "fontWeight": "bold"})),
                            ])
# buttons_trade = dbc.Row([dbc.Col(html.Div([html.Button('Trade!', id='pred-btn_1')],
#                                           style={'width': '100%', 'display': 'flex', 'align-items': 'center',
#                                                  'justify-content': 'center'})),
#                          dbc.Col(html.Div([html.Button('Trade!', id='pred-btn_2')],
#                                           style={'width': '100%', 'display': 'flex', 'align-items': 'center',
#                                                  'justify-content': 'center'})),
#                          dbc.Col(html.Div([html.Button('Trade!', id='pred-btn_3')],
#                                           style={'width': '100%', 'display': 'flex', 'align-items': 'center',
#                                                  'justify-content': 'center'})),
#                          dbc.Col(html.Div([html.Button('Trade!', id='pred-btn_4')],
#                                           style={'width': '100%', 'display': 'flex', 'align-items': 'center',
#                                                  'justify-content': 'center'})),
#                          dbc.Col(html.Div([html.Button('Trade!', id='pred-btn_5')],
#                                           style={'width': '100%', 'display': 'flex', 'align-items': 'center',
#                                                  'justify-content': 'center'})),
#                          ], style={'margin-bottom': '10px'})

buttons_trade = dbc.Row([dbc.Col(html.Div([dcc.Dropdown(
    id='pred-dd1',
    options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
             enumerate(player_selector.label.unique())],
    placeholder='Select a Player',
    value="No Trade", style={'width': '100%'}
)],
    style={'width': '100%', 'display': 'flex', 'align-items': 'center',
           'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd2',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='Select a Player',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd3',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='Select a Player',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd4',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='Select a Player',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd5',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='Select a Player',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
], style={'margin-bottom': '10px'})

execute_trade = html.Button('GO', id='exec-btn_1')

# top_card_rec = dbc.Card(
#     [dcc.Dropdown(
#         id='pred-teamRec-player-dropdown',
#         options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
#                  enumerate(player_selector.label.unique())],
#         placeholder='Select a Player',
#         value=203500, style={'width': '30%'}
#     )]
# )

# bottom_card_rec = html.Div(
#     [html.Img(id="prediction-teamRec-starting5-dropdown2-img", style={'width': '50%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
#         dbc.Container([
#         dcc.Dropdown(
#             id='prediction-teamRec-starting5-dropdown2',
#             options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
#                      enumerate(player_selector.label.unique())],
#             placeholder='Select a Player',
#             value=203500
#         )])
#     ]
# )

team_card_rec = dbc.Card(
    [html.Div([dbc.CardImg(id='prediction-teamRep-image')],
              style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
     dbc.CardBody(
         [dcc.Dropdown(
             id='pred-teamRec-select-dropdown',
             options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
                      enumerate(team_data['abbreviation'])],
             placeholder='Select a Team',
             value='LAL'

         ), dcc.RadioItems(id='prediction-mode-dropdown', options=[{'label': 'Validation', 'value': 'validation'},
                                                                   {'label': 'Simulation', 'value': 'simulation'}],
                           value='validation')])
     ],
    style={"width": "20rem", "align": "auto"})

# team_card_rec = html.Div([
#     html.Img(id='prediction-teamRep-image', style={'width': '50%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
#         dbc.Container([
#             dcc.Dropdown(
#                 id='pred-teamRec-select-dropdown',
#                 options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
#                          enumerate(team_data['abbreviation'])],
#                 placeholder='Select a Team',
#                 value='LAL'
#
#             ), dcc.RadioItems(id='prediction-mode-dropdown', options=[{'label': 'Validation', 'value': 'validation'},
#                                                                    {'label': 'Simulation', 'value': 'simulation'}],
#                            value='validation')])
# ])

# row_selector = dbc.Row([dbc.Col([dbc.Container([
#     dcc.Dropdown(
#         id='pred-teamRec-select-dropdown',
#         options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
#                  enumerate(team_data['abbreviation'])],
#         placeholder='Select a Team',
#         value='LAL'
#
#     ), dcc.RadioItems(id='prediction-mode-dropdown', options=[{'label': 'Validation', 'value': 'validation'},
#                                                               {'label': 'Simulation', 'value': 'simulation'}],
#                       value='validation')])]), dbc.Col([dbc.Container([
#     dcc.Dropdown(
#         id='prediction-teamRec-starting5-dropdown2',
#         options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
#                  enumerate(player_selector.label.unique())],
#         placeholder='Select a Player',
#         value=203500
#     )])])])
#
# row_image = dbc.Row([dbc.Col([html.Img(id='prediction-teamRep-image',
#                                        style={'width': '50%',
#                                               'display': 'flex',
#                                               'align-items': 'center',
#                                               'justify-content': 'center',
#                                               'textAlign': 'center'})], style={'textAlign': 'center'}), dbc.Col([html.Img(
#     id="prediction-teamRec-starting5-dropdown2-img",
#     style={'width': '50%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'textAlign': 'center'})], style={'textAlign': 'center'})])
#
# jumbotron_header = dbc.Container([row_image, row_selector])

text_offcanvas = """Select your team (left-side) and the player you want to trade to your team. 
Then select one of your starting players to trade against the selcted player."
"""

offcanvas = html.Div([html.Div([dbc.Button("Information regarding the usage of the season prediction", id="rec-infocanvas-pred", n_clicks=0, style={'background-color': '#17408b', 'color': 'white', 'margin-bottom': '10px'})],
                                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                      html.Div([dbc.Offcanvas([html.P(text_offcanvas)],
                                backdrop=True,
                                scrollable=True,
                                id="infocanvas-pred",
                                title = 'Title',
                                is_open=False,
                                placement='top')],
                                style = {'width': '100%', 'display': 'flex', 'justify-content': 'top'})
                     ])

validation_trade = html.Div([
    dbc.Container([
        dcc.Loading([dcc.Graph(id='prediction-output-graph_trade', clear_on_unhover=True),
                     dcc.Graph(id='prediction-output-graph_trade3', clear_on_unhover=True),
                     dash_table.DataTable(id='prediction-output-table_trade',
                                          filter_action="native",
                                          sort_action="native",
                                          sort_mode="multi",
                                          page_action="native",
                                          page_current=0,
                                          page_size=25,
                                          style_cell={'textAlign': 'center'},
                                          style_as_list_view=True
                                          )], fullscreen=False, type='dot',
                    color="#119DFF")]), html.H2(id="prediction-mae")])


# left_jumbotron_header = dbc.Col(
#     html.Div(
#         [
#             html.H2("Select Team", className="display-3"),
#             html.Hr(className="my-2"),
#             team_card_rec
#         ],
#         className="h-100 p-5 bg-light border rounded-3",
#     ),
#     md=6,
# )
#
# right_jumbotron_header = dbc.Col(
#     html.Div(
#         [
#             html.H2("Trade Target", className="display-3"),
#             html.Hr(className="my-2"),
#             bottom_card_rec
#         ],
#         className="h-100 p-5 bg-light border rounded-3",
#     ),
#     md=6,
# )

# jumbotron_header = dbc.Row(
#     [left_jumbotron_header, right_jumbotron_header],
#     className="align-items-md-stretch",
# )

prediction = html.Div(
    children=[offcanvas,
              execute_trade,
              team_card_rec,
              pred_player_images,
              pred_player_text,
              buttons_trade,
              validation_trade,
              html.Div(dash_table.DataTable(id='prediction-validation-table'))])
