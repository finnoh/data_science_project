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
    placeholder='No Trade',
    value="No Trade", style={'width': '100%'}
)],
    style={'width': '100%', 'display': 'flex', 'align-items': 'center',
           'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd2',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='No Trade',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd3',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='No Trade',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd4',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='No Trade',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
    dbc.Col(html.Div([dcc.Dropdown(
        id='pred-dd5',
        options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                 enumerate(player_selector.label.unique())],
        placeholder='No Trade',
        value="No Trade", style={'width': '100%'}
    )],
        style={'width': '100%', 'display': 'flex', 'align-items': 'center',
               'justify-content': 'center'})),
], style={'margin-bottom': '10px'})

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


# team_card_rec = html.Div(
#     dbc.Container(
#         [
#             html.H1("Jumbotron", className="display-3"),
#             html.Div([dbc.CardImg(id='prediction-teamRep-image')],
#                      style={'width': '33%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
# ,
#             html.Hr(className="my-2"),
#             select_go
#         ],
#         fluid=True,
#         className="py-3",
#     ),
#     className="p-3 bg-light rounded-3",
# )


left_jumbotron = dbc.Col(
    html.Div(
        [dbc.CardImg(id='prediction-teamRep-image', style={'width': '33%', 'margin-left': '30%', 'textAlign': 'center'}),
            html.Hr(className="my-2"),

         dcc.Dropdown(
             id='pred-teamRec-select-dropdown',
             options=[{'label': list(team_data['full_name'])[i], 'value': abb} for i, abb in
                      enumerate(team_data['abbreviation'])],
             placeholder='Select a Team',
             value='LAL'
         ),
        ],
        className="h-100 p-5 bg-light border rounded-3",
    ),
    md=6,
)

# dcc.RadioItems(
#                           options=[
#                               {'label': '10 sims (10s)', 'value': 10},
#                               {'label': '100 sims (30s)', 'value': 100},
#                               {'label': '1000 sims (180s)', 'value': 1000},
#                           ],
#                           value=10, id='slider-sim'
#                       ),


instructions_sim = """

Pick your team on the left side and pick trade partners for the presented players using the dropdown menu's below.

A higher number of simulations leads to less variance in the predictions i.e. more narrow boxplots below, but comes at the cost of longer computing time. 10 simulations take 10 seconds, while 1000 simulations take roughly 2:30 minutes.

"""

right_jumbotron = dbc.Col(
    html.Div(
        [
         html.Div(
             [html.P(instructions_sim),
                 dcc.Slider(min=10, max=1000, step=100, value=10, id='slider-sim', tooltip = {'always_visible': True}),
                 dbc.Button("Simulate!", size="lg", className="me-1", id='exec-btn_1', color="info"),
             ],
             className="d-grid gap-2",
         )
         ],
        className="h-100 p-5 bg-light border rounded-3",
    ),
    md=6,
)

jumbotron_header = dbc.Row(
    [left_jumbotron, right_jumbotron],
    className="align-items-md-stretch",
)

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

offcanvas = html.Div([html.Div([dbc.Button("Information regarding the modeling", id="rec-infocanvas-pred", n_clicks=0, style={'background-color': '#17408b', 'color': 'white', 'margin-bottom': '10px'})],
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

simulation_trade = html.Div([
    dbc.Container([
        dcc.Loading([dcc.Graph(id='prediction-output-graph_trade', clear_on_unhover=True),
                     dcc.Graph(id='prediction-output-graph_trade3', clear_on_unhover=True),
                     dash_table.DataTable(id='prediction-output-table_trade',
                                          filter_action="native",
                                          sort_action="native",
                                          sort_mode="multi",
                                          page_action="native",
                                          page_current=0,
                                          page_size=30,
                                          style_cell={'textAlign': 'center'},
                                          style_as_list_view=True
                                          )], fullscreen=False, type='dot',
                    color="#119DFF")]), html.H2(id="prediction-mae"), dash_table.DataTable(id="prediction-validation-table",
                                            filter_action="native",
                                            sort_action="native",
                                            sort_mode="multi",
                                            page_action="native",
                                            page_current=0,
                                            page_size=30,
                                            style_cell={'textAlign': 'center'},
                                            style_as_list_view=True
                                            )])

validation_trade = html.Div([
    dbc.Container([
        dcc.Loading([dcc.Graph(id='prediction-output-graph_trade-v', clear_on_unhover=True),
                     dcc.Graph(id='prediction-output-graph_error-v', clear_on_unhover=True),
                     dash_table.DataTable(id='prediction-output-table_trade-v',
                                          filter_action="native",
                                          sort_action="native",
                                          sort_mode="multi",
                                          page_action="native",
                                          page_current=0,
                                          page_size=30,
                                          style_cell={'textAlign': 'center'},
                                          style_as_list_view=True
                                          )], fullscreen=False, type='dot',
                    color="#119DFF")]), html.H2(id="prediction-mae-v"), dash_table.DataTable(id="prediction-validation-table-v",
                                            filter_action="native",
                                            sort_action="native",
                                            sort_mode="multi",
                                            page_action="native",
                                            page_current=0,
                                            page_size=30,
                                            style_cell={'textAlign': 'center'},
                                            style_as_list_view=True
                                            )])

output = dcc.Tabs(id='tabs-pred', value='tab-1', children=[
        dcc.Tab(label="Simulation", value='tab-1', children=[simulation_trade]),
        dcc.Tab(label='Validation', value='tab-2', children=[validation_trade])
])


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
    children=[
              offcanvas,
              jumbotron_header,
              pred_player_images,
              pred_player_text,
              buttons_trade,
              output])
