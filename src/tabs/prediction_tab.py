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


instructions_sim = """

Pick your team on the left side and pick trade partners for the presented players using the dropdown menu's below.

A higher number of simulations leads to less variance in the predictions i.e. more narrow boxplots below, but comes at the cost of longer computing time. Ten simulations take 30 seconds, while 1,000 simulations take roughly five minutes.

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


text_offcanvas = """Choose between validation mode and simulation mode: 
**Validation mode** uses the real data game outcomes from the season 2021/22 until mid february to evaluate the performance of our bayesian linear regression model. 
The **simulation tab** allows you to test out different trades and get a prediction for the trades' effect on your team's performance and changes around the league.
The boxplots displays a distribution over different season outcomes after simulating the upcoming season many times. The barplot below directly illustrates changes in the number of wins
due to your trade for each team. Below we show the outcome of the simulated season 2021/22 in a league-wide scoreboard. Configure your trade with the dropdown menu below, the respective players of your selected team. After configuring a trade hit the *Simulate!* button above."""

offcanvas = html.Div([html.Div([dbc.Button("Information regarding the modeling", id="rec-infocanvas-pred", n_clicks=0, style={'background-color': '#17408b', 'color': 'white', 'margin-bottom': '10px'})],
                                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                      html.Div([dbc.Offcanvas([dcc.Markdown(text_offcanvas)],
                                backdrop=True,
                                scrollable=True,
                                id="infocanvas-pred",
                                title = 'Information regarding Season Prediction',
                                is_open=False,
                                placement='end')],
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
                    color="#119DFF")]), dcc.Markdown(id="prediction-mae"), html.H2('Predicted Season Scoreboard'),dash_table.DataTable(id="prediction-validation-table",
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
                    color="#119DFF")]), dcc.Markdown(id="prediction-mae-v"), html.H2('Predicted Season Scoreboard'), dash_table.DataTable(id="prediction-validation-table-v",
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



prediction = html.Div(
    children=[
              offcanvas,
              jumbotron_header,
              pred_player_images,
              pred_player_text,
              buttons_trade,
              output])
