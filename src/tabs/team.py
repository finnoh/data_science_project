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
            placement='top'
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
                          offcanvas,
                          html.Hr(className="my-2"),
                          dbc.Container([
                              dcc.Graph(id='teamselect-capspace-graph')
                          ])], md=6, className="h-100 p-5 bg-light border rounded-3")

jumbotron = html.Div([dbc.Row(
    [left_jumbotron, right_part], className="align-items-md-stretch"),
    html.Div(id='team-table')]
)
